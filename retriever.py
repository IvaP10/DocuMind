from typing import List, Dict, Any
from uuid import UUID
from sentence_transformers import CrossEncoder
import config
from database import vector_db
from embedder import embedder
import logging

logger = logging.getLogger(__name__)


class ImprovedRetriever:
    
    def __init__(self):
        logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)
        logger.info("Reranker loaded")
    
    def retrieve_context(
        self, 
        document_id: UUID, 
        query: str, 
        chunks_metadata: List[Dict[str, Any]],
        mode: str = "hybrid_full"
    ) -> Dict[str, Any]:
        
        experiment = config.EXPERIMENT_MODES.get(mode, config.EXPERIMENT_MODES["hybrid_full"])
        
        logger.info(f"Retrieving context: '{query[:50]}...'")
        logger.info(f"Mode: {experiment['name']}")
        
        # Step 1: Initial search
        candidates = self.search(
            document_id, 
            query,
            use_dense=experiment["use_dense"],
            use_sparse=experiment["use_sparse"]
        )
        
        if not candidates:
            logger.warning("No candidates found")
            return {
                "context": "", 
                "sources": [],
                "retrieved_chunks": [],
                "rerank_scores": []
            }
        
        # Step 2: Apply score threshold if enabled
        if experiment.get("apply_score_threshold", False):
            candidates = self.filter_by_score(candidates, config.DENSE_SCORE_THRESHOLD)
            logger.info(f"After score filtering: {len(candidates)} candidates")
        
        # Step 3: Reranking
        if experiment["use_reranker"]:
            top_chunks, rerank_scores = self.rerank(query, candidates)
            
            # Filter by rerank threshold
            if experiment.get("apply_score_threshold", False):
                filtered_chunks = []
                filtered_scores = []
                for chunk, score in zip(top_chunks, rerank_scores):
                    if score >= config.RERANK_SCORE_THRESHOLD:
                        filtered_chunks.append(chunk)
                        filtered_scores.append(score)
                
                if filtered_chunks:
                    top_chunks = filtered_chunks
                    rerank_scores = filtered_scores
                    logger.info(f"After rerank filtering: {len(top_chunks)} chunks")
        else:
            top_chunks = candidates[:config.TOP_K_RERANK]
            rerank_scores = [c.get("score", 0.0) for c in top_chunks]
        
        # Step 4: Parent-child or keep children
        if experiment["use_parent_child"]:
            context_chunks = self.get_parent_chunks(top_chunks, chunks_metadata)
        else:
            # Keep the precise child chunks that matched
            context_chunks = top_chunks
        
        # Step 5: Assemble context
        context_data = self.assemble_context(context_chunks)
        context_data["retrieved_chunks"] = top_chunks
        context_data["rerank_scores"] = rerank_scores
        context_data["mode"] = experiment["name"]
        
        return context_data
    
    def search(
        self, 
        document_id: UUID, 
        query: str,
        use_dense: bool = True,
        use_sparse: bool = True
    ) -> List[Dict[str, Any]]:
        
        logger.info(f"Searching (dense={use_dense}, sparse={use_sparse})")
        
        dense_embedding = embedder.embed_query(query)
        sparse_vector = embedder.create_sparse_vector(query)
        
        results = vector_db.search(
            dense_embedding=dense_embedding,
            sparse_vector=sparse_vector,
            document_id=document_id,
            top_k=config.TOP_K_INITIAL,
            use_dense=use_dense,
            use_sparse=use_sparse
        )
        
        logger.info(f"Found {len(results)} candidates")
        return results
    
    def filter_by_score(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter candidates by minimum score threshold"""
        filtered = [c for c in candidates if c.get("score", 0.0) >= threshold]
        logger.info(f"Filtered {len(candidates)} -> {len(filtered)} by score threshold {threshold}")
        return filtered if filtered else candidates[:max(1, config.TOP_K_RERANK)]  # Keep at least some results
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        
        logger.info(f"Reranking {len(candidates)} candidates to top {config.TOP_K_RERANK}")
        
        if len(candidates) <= config.TOP_K_RERANK:
            scores = [1.0] * len(candidates)
            return candidates, scores
        
        pairs = [[query, candidate["text"]] for candidate in candidates]
        scores = self.reranker.predict(pairs)
        
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top K
        top_chunks = [candidate for candidate, score in ranked[:config.TOP_K_RERANK]]
        top_scores = [float(score) for candidate, score in ranked[:config.TOP_K_RERANK]]
        
        logger.info(f"Reranked to {len(top_chunks)} chunks, scores: {[f'{s:.3f}' for s in top_scores]}")
        return top_chunks, top_scores
    
    def get_parent_chunks(
        self, 
        child_chunks: List[Dict[str, Any]], 
        all_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        logger.info("Getting parent chunks")
        
        parent_ids = set(chunk["parent_id"] for chunk in child_chunks)
        
        parent_chunks = []
        for chunk_meta in all_chunks:
            if chunk_meta.get("id") in parent_ids and chunk_meta.get("is_parent"):
                parent_chunks.append(chunk_meta)
        
        seen_parents = set()
        ordered_parents = []
        
        for child in child_chunks:
            parent_id = child["parent_id"]
            if parent_id not in seen_parents:
                parent = next((p for p in parent_chunks if p["id"] == parent_id), None)
                if parent:
                    ordered_parents.append(parent)
                    seen_parents.add(parent_id)
        
        logger.info(f"Got {len(ordered_parents)} parent chunks")
        return ordered_parents
    
    def assemble_context(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Assembling context")
        
        context_parts = []
        sources = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = chunk.get("token_count", 0)
            
            if total_tokens + chunk_tokens > config.MAX_CONTEXT_TOKENS:
                logger.info(f"Context limit reached at {total_tokens} tokens")
                break
            
            context_parts.append(self.format_chunk(chunk))
            total_tokens += chunk_tokens
            
            sources.append({
                "chunk_id": chunk.get("id") or chunk.get("chunk_id"),
                "text": chunk["text"][:200] + "...",
                "page": chunk.get("page_number"),
                "bbox": chunk.get("bbox"),
                "type": chunk.get("chunk_type")
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Context: {total_tokens} tokens, {len(sources)} sources")
        
        return {
            "context": context,
            "sources": sources,
            "total_tokens": total_tokens
        }
    
    def format_chunk(self, chunk: Dict[str, Any]) -> str:
        page = chunk.get("page_number", "?")
        chunk_type = chunk.get("chunk_type", "text")
        text = chunk.get("text", "")
        
        header = f"[Page {page} - {chunk_type.upper()}]"
        return f"{header}\n{text}"


retriever = ImprovedRetriever()