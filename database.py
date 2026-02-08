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
    SparseVector
)
from typing import List, Dict, Any, Optional
from uuid import UUID
import numpy as np
import config
from models import Chunk
import logging
import sys

logger = logging.getLogger(__name__)


class EnhancedVectorDatabase:
    
    def __init__(self):
        logger.info(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        
        try:
            if config.QDRANT_API_KEY:
                self.client = QdrantClient(
                    url=f"https://{config.QDRANT_HOST}",
                    api_key=config.QDRANT_API_KEY
                )
            else:
                self.client = QdrantClient(
                    host=config.QDRANT_HOST,
                    port=config.QDRANT_PORT,
                    timeout=60
                )
            
            self.collection_name = config.QDRANT_COLLECTION
            self.embedding_dimension = config.EMBEDDING_DIMENSION
            
            self._ensure_collection()
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            logger.error("\n" + "="*80)
            logger.error("QDRANT CONNECTION ERROR")
            logger.error("="*80)
            logger.error("Qdrant vector database is not running or not accessible.")
            logger.error("\nTo fix this, you have 3 options:")
            logger.error("\n1. LOCAL DOCKER (Recommended):")
            logger.error("   docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant")
            logger.error("\n2. LOCAL BINARY:")
            logger.error("   Download from: https://github.com/qdrant/qdrant/releases")
            logger.error("   Run: ./qdrant")
            logger.error("\n3. QDRANT CLOUD (Free tier available):")
            logger.error("   Sign up at: https://cloud.qdrant.io")
            logger.error("   Then set in your key.env file:")
            logger.error("   QDRANT_HOST=your-cluster.cloud.qdrant.io")
            logger.error("   QDRANT_API_KEY=your-api-key")
            logger.error("\n4. IN-MEMORY MODE (For testing only, no persistence):")
            logger.error("   Set QDRANT_HOST=:memory: in key.env")
            logger.error("="*80 + "\n")
            
            # Try in-memory mode as fallback
            if config.QDRANT_HOST != ":memory:":
                logger.warning("Attempting fallback to in-memory mode...")
                try:
                    self.client = QdrantClient(":memory:")
                    self.collection_name = config.QDRANT_COLLECTION
                    self.embedding_dimension = config.EMBEDDING_DIMENSION
                    self._ensure_collection()
                    logger.warning("⚠️  Running in IN-MEMORY mode - data will NOT persist!")
                    logger.warning("⚠️  For production use, please set up a proper Qdrant instance.")
                except Exception as e2:
                    logger.error(f"Even in-memory mode failed: {str(e2)}")
                    raise RuntimeError(
                        "Cannot initialize vector database. Please set up Qdrant or use in-memory mode."
                    ) from e
            else:
                raise
    
    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self._create_collection()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"Collection info: {collection_info}")
                
        except Exception as e:
            logger.error(f"Collection setup error: {str(e)}")
            raise
    
    def _create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                    on_disk=False
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False
                    )
                )
            }
        )
        logger.info(f"Collection '{self.collection_name}' created successfully")
    
    def index_chunks(
        self,
        chunks: List[Chunk],
        dense_embeddings: np.ndarray,
        sparse_vectors: List[Dict[int, float]]
    ):
        if len(chunks) != len(dense_embeddings) or len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks, {len(dense_embeddings)} embeddings, "
                f"{len(sparse_vectors)} sparse vectors"
            )
        
        child_indices = [i for i, c in enumerate(chunks) if c.is_child]
        child_chunks = [chunks[i] for i in child_indices]
        child_dense = [dense_embeddings[i] for i in child_indices]
        child_sparse = [sparse_vectors[i] for i in child_indices]
        
        if not child_chunks:
            logger.warning("No child chunks to index")
            return
        
        logger.info(f"Indexing {len(child_chunks)} child chunks (out of {len(chunks)} total)")
        
        points = []
        for chunk, dense_emb, sparse_vec in zip(child_chunks, child_dense, child_sparse):
            point = self._create_point(chunk, dense_emb, sparse_vec)
            points.append(point)
        
        try:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.debug(f"Indexed batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")
            
            logger.info(f"Successfully indexed {len(points)} chunks")
            
        except Exception as e:
            logger.error(f"Indexing error: {str(e)}")
            raise
    
    def _create_point(
        self,
        chunk: Chunk,
        dense_embedding: np.ndarray,
        sparse_vector: Dict[int, float]
    ) -> PointStruct:
        if isinstance(dense_embedding, np.ndarray):
            dense_list = dense_embedding.tolist()
        else:
            dense_list = list(dense_embedding)
        
        sparse_indices = list(sparse_vector.keys())
        sparse_values = list(sparse_vector.values())
        
        point = PointStruct(
            id=str(chunk.id),
            vector={
                "dense": dense_list,
                "sparse": SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            },
            payload={
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type.value,
                "format_type": chunk.format_type.value,
                "page_number": chunk.page_number,
                "bbox": chunk.bbox.dict() if chunk.bbox else None,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata
            }
        )
        
        return point
    
    def search(
        self,
        dense_embedding: np.ndarray,
        sparse_vector: Dict[int, float],
        document_id: UUID,
        top_k: int = 50,
        use_dense: bool = True,
        use_sparse: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"Searching for top {top_k} chunks")
        logger.info(f"Search mode: dense={use_dense}, sparse={use_sparse}")
        
        try:
            query_filter = self._build_filter(document_id, metadata_filter)
            
            if use_dense and use_sparse:
                results = self._hybrid_search(
                    dense_embedding, sparse_vector, query_filter, top_k
                )
            elif use_dense:
                results = self._dense_search(
                    dense_embedding, query_filter, top_k
                )
            elif use_sparse:
                results = self._sparse_search(
                    sparse_vector, query_filter, top_k
                )
            else:
                logger.warning("Both dense and sparse disabled, returning empty results")
                return []
            
            formatted = self._format_results(results)
            logger.info(f"Found {len(formatted)} results")
            
            return formatted
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    def _build_filter(
        self,
        document_id: UUID,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Filter:
        conditions = [
            FieldCondition(
                key="document_id",
                match=MatchValue(value=str(document_id))
            )
        ]
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions)
    
    def _dense_search(
        self,
        dense_embedding: np.ndarray,
        query_filter: Filter,
        top_k: int
    ) -> List:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_embedding.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        ).points
        
        return results
    
    def _sparse_search(
        self,
        sparse_vector: Dict[int, float],
        query_filter: Filter,
        top_k: int
    ) -> List:
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
        
        return results
    
    def _hybrid_search(
        self,
        dense_embedding: np.ndarray,
        sparse_vector: Dict[int, float],
        query_filter: Filter,
        top_k: int
    ) -> List:
        logger.debug("Performing hybrid search with RRF")
        
        fetch_k = min(top_k * 3, 150)
        
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_embedding.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=fetch_k,
            with_payload=True
        ).points
        
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values())
            ),
            using="sparse",
            query_filter=query_filter,
            limit=fetch_k,
            with_payload=True
        ).points
        
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            top_k,
            k=config.RRF_K_PARAM
        )
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List,
        sparse_results: List,
        top_k: int,
        k: int = 60
    ) -> List:
        scores = {}
        
        dense_weight = config.DENSE_WEIGHT
        sparse_weight = config.SPARSE_WEIGHT
        
        for rank, point in enumerate(dense_results, start=1):
            point_id = point.id
            rrf_score = 1.0 / (k + rank)
            scores[point_id] = scores.get(point_id, 0) + (dense_weight * rrf_score)
        
        for rank, point in enumerate(sparse_results, start=1):
            point_id = point.id
            rrf_score = 1.0 / (k + rank)
            scores[point_id] = scores.get(point_id, 0) + (sparse_weight * rrf_score)
        
        all_points = {p.id: p for p in dense_results}
        all_points.update({p.id: p for p in sparse_results})
        
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        ranked_points = []
        for point_id in ranked_ids:
            point = all_points[point_id]
            point.score = scores[point_id]
            ranked_points.append(point)
        
        logger.debug(f"RRF fused {len(dense_results)} dense + {len(sparse_results)} sparse "
                    f"-> {len(ranked_points)} results")
        
        return ranked_points
    
    def _format_results(self, results: List) -> List[Dict[str, Any]]:
        formatted = []
        
        for result in results:
            formatted.append({
                "chunk_id": result.payload["chunk_id"],
                "parent_id": result.payload.get("parent_id"),
                "text": result.payload["text"],
                "chunk_type": result.payload["chunk_type"],
                "format_type": result.payload.get("format_type", "plain"),
                "page_number": result.payload["page_number"],
                "bbox": result.payload.get("bbox"),
                "token_count": result.payload.get("token_count", 0),
                "score": float(result.score) if hasattr(result, 'score') else 0.0,
                "metadata": result.payload.get("metadata", {})
            })
        
        return formatted
    
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
            logger.info(f"Document {document_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Deletion error: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "total_points": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


# Create the singleton instance with proper error handling
try:
    vector_db = EnhancedVectorDatabase()
except RuntimeError as e:
    logger.error(f"Failed to initialize vector database: {str(e)}")
    logger.error("Please set up Qdrant before running the application.")
    sys.exit(1)