from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
    NamedVector
)
from typing import List, Dict, Any
from uuid import UUID
import numpy as np
import config
from models import Chunk
import logging

logger = logging.getLogger(__name__)


class VectorDatabase:
    
    def __init__(self):
        logger.info(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        self.collection_name = config.QDRANT_COLLECTION
        self.ensure_collection()
    
    def ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=768,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams()
                        )
                    }
                )
                logger.info("Collection created")
            else:
                logger.info(f"Collection {self.collection_name} exists")
                
        except Exception as e:
            logger.error(f"Collection setup error: {str(e)}")
            raise
    
    def index_chunks(self, chunks: List[Chunk], dense_embeddings: np.ndarray, sparse_vectors: List[Dict[int, float]]):
        if len(chunks) != len(dense_embeddings) or len(chunks) != len(sparse_vectors):
            raise ValueError("Chunks, embeddings, and sparse vectors count mismatch")
        
        child_chunks = [c for c in chunks if c.parent_id is not None]
        child_dense = [dense_embeddings[i] for i, c in enumerate(chunks) if c.parent_id is not None]
        child_sparse = [sparse_vectors[i] for i, c in enumerate(chunks) if c.parent_id is not None]
        
        if not child_chunks:
            logger.warning("No child chunks to index")
            return
        
        logger.info(f"Indexing {len(child_chunks)} child chunks")
        
        points = []
        for chunk, dense_emb, sparse_vec in zip(child_chunks, child_dense, child_sparse):
            point = PointStruct(
                id=str(chunk.id),
                vector={
                    "dense": dense_emb.tolist(),
                    "sparse": SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values())
                    )
                },
                payload={
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "parent_id": str(chunk.parent_id),
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type.value,
                    "format_type": chunk.format_type.value,
                    "page_number": chunk.page_number,
                    "bbox": chunk.bbox.dict() if chunk.bbox else None,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Indexed {len(points)} chunks")
            
        except Exception as e:
            logger.error(f"Indexing error: {str(e)}")
            raise
    
    def search(
        self, 
        dense_embedding: np.ndarray, 
        sparse_vector: Dict[int, float],
        document_id: UUID, 
        top_k: int = 50,
        use_dense: bool = True,
        use_sparse: bool = True
    ) -> List[Dict[str, Any]]:
        
        logger.info(f"Searching for top {top_k} chunks")
        
        try:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id))
                    )
                ]
            )
            
            if use_dense and use_sparse:
                # Hybrid search using Reciprocal Rank Fusion
                logger.info("Using hybrid search (dense + sparse)")
                
                # Dense search
                dense_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_embedding.tolist(),
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k * 2,
                    with_payload=True
                ).points
                
                # Sparse search
                sparse_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=SparseVector(
                        indices=list(sparse_vector.keys()),
                        values=list(sparse_vector.values())
                    ),
                    using="sparse",
                    query_filter=query_filter,
                    limit=top_k * 2,
                    with_payload=True
                ).points
                
                # Reciprocal Rank Fusion
                results = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
                
            elif use_dense:
                logger.info("Using dense-only search")
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_embedding.tolist(),
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                ).points
            
            elif use_sparse:
                logger.info("Using sparse-only search")
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=SparseVector(
                        indices=list(sparse_vector.keys()),
                        values=list(sparse_vector.values())
                    ),
                    using="sparse",
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                ).points
            else:
                logger.warning("Both dense and sparse disabled, returning empty results")
                return []
            
            return self.format_results(results)
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    def _reciprocal_rank_fusion(self, dense_results, sparse_results, top_k: int, k: int = 60):
        """Combine dense and sparse results using Reciprocal Rank Fusion"""
        scores = {}
        
        # Score dense results
        for rank, point in enumerate(dense_results, start=1):
            point_id = point.id
            scores[point_id] = scores.get(point_id, 0) + 1 / (k + rank)
        
        # Score sparse results
        for rank, point in enumerate(sparse_results, start=1):
            point_id = point.id
            scores[point_id] = scores.get(point_id, 0) + 1 / (k + rank)
        
        # Combine all unique points
        all_points = {p.id: p for p in dense_results}
        all_points.update({p.id: p for p in sparse_results})
        
        # Sort by RRF score
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        # Return points in ranked order with RRF scores
        ranked_points = []
        for point_id in ranked_ids:
            point = all_points[point_id]
            point.score = scores[point_id]  # Replace original score with RRF score
            ranked_points.append(point)
        
        return ranked_points
    
    def format_results(self, results) -> List[Dict[str, Any]]:
        search_results = []
        for result in results:
            search_results.append({
                "chunk_id": result.payload["chunk_id"],
                "parent_id": result.payload["parent_id"],
                "text": result.payload["text"],
                "chunk_type": result.payload["chunk_type"],
                "page_number": result.payload["page_number"],
                "bbox": result.payload["bbox"],
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })
        
        logger.info(f"Found {len(search_results)} results")
        return search_results
    
    def delete_document(self, document_id: UUID):
        logger.info(f"Deleting document {document_id}")
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id))
                        )
                    ]
                )
            )
            logger.info("Document deleted")
            
        except Exception as e:
            logger.error(f"Deletion error: {str(e)}")
            raise


vector_db = VectorDatabase()