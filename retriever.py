from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from sentence_transformers import CrossEncoder
import config
from database import vector_db
from embedder import embedder
import logging
import numpy as np
import math
import time

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    
    def __init__(self):
        logger.info("Initializing Enhanced Retriever")
        
        if config.USE_COHERE_RERANK:
            self._init_cohere_reranker()
        else:
            self._init_crossencoder_reranker()
        
        logger.info("Retriever initialized successfully")
    
    def _init_crossencoder_reranker(self):
        logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
        try:
            self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)
            self.reranker_type = "crossencoder"
            logger.info("CrossEncoder reranker loaded")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            logger.info("Falling back to default reranker")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker_type = "crossencoder"
    
    def _init_cohere_reranker(self):
        try:
            import cohere
            self.cohere_client = cohere.Client(config.COHERE_API_KEY)
            self.reranker_type = "cohere"
            logger.info(f"Cohere reranker initialized: {config.COHERE_RERANK_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere: {e}")
            logger.info("Falling back to CrossEncoder")
            self._init_crossencoder_reranker()
    
    def retrieve_context(
        self,
        document_id: UUID,
        query: str,
        chunks_metadata: List[Dict[str, Any]],
        mode: str = "precision_optimized"
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        experiment = config.EXPERIMENT_MODES.get(mode, config.EXPERIMENT_MODES["precision_optimized"])
        logger.info(f"=" * 80)
        logger.info(f"RETRIEVAL: {experiment['name']}")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"=" * 80)
        
        processed_query = query
        if experiment.get("use_query_rewriting", False):
            processed_query = self._rewrite_query(query)
            logger.info(f"Rewritten query: {processed_query}")
        
        candidates = self._search(
            document_id,
            processed_query,
            use_dense=experiment["use_dense"],
            use_sparse=experiment["use_sparse"]
        )
        
        initial_count = len(candidates)
        logger.info(f"Initial candidates: {initial_count}")
        
        if not candidates:
            logger.warning("No candidates found")
            return self._empty_context_data(mode)
        
        if experiment.get("dynamic_threshold", False):
            candidates = self._dynamic_threshold_filter(candidates, experiment)
            logger.info(f"After dynamic threshold: {len(candidates)} candidates")
        elif experiment.get("apply_score_threshold", False):
            threshold = self._get_score_threshold(experiment)
            candidates = self._filter_by_score(candidates, threshold)
            logger.info(f"After score filtering: {len(candidates)} candidates")
        
        if experiment["use_reranker"]:
            target_k = experiment.get("top_k_rerank", config.TOP_K_RERANK)
            candidates, rerank_scores = self._rerank(query, candidates, target_k)
            
            if experiment.get("adaptive_threshold", False):
                candidates, rerank_scores = self._adaptive_rerank_filter(
                    candidates, rerank_scores, experiment
                )
        else:
            candidates = candidates[:config.TOP_K_RERANK]
            rerank_scores = [c.get("score", 0.0) for c in candidates]
        
        logger.info(f"After reranking: {len(candidates)} chunks")
        
        if config.ENABLE_DEDUPLICATION:
            dedup_method = experiment.get("dedup_method", "semantic")
            threshold = experiment.get("dedup_threshold", config.DEDUP_SIMILARITY_THRESHOLD)
            candidates = self._deduplicate(candidates, method=dedup_method, threshold=threshold)
            logger.info(f"After deduplication: {len(candidates)} chunks")
        
        if experiment.get("use_mmr", False) and len(candidates) > 1:
            lambda_param = experiment.get("mmr_lambda", config.MMR_LAMBDA)
            candidates = self._apply_mmr(query, candidates, lambda_param)
            logger.info(f"After MMR: {len(candidates)} diverse chunks")
        
        context_strategy = experiment.get("context_strategy", "child_with_parent_context")
        context_chunks = self._assemble_smart_context(
            candidates, chunks_metadata, strategy=context_strategy
        )
        
        context_data = self._build_context_data(context_chunks)
        context_data["retrieved_chunks"] = candidates
        context_data["rerank_scores"] = rerank_scores
        context_data["mode"] = experiment["name"]
        
        retrieval_time = (time.time() - start_time) * 1000
        context_data["metrics"] = {
            "total_candidates": initial_count,
            "after_filtering": len(candidates),
            "final_chunks": len(context_chunks),
            "retrieval_time_ms": retrieval_time,
            "mode": mode,
            "avg_rerank_score": np.mean(rerank_scores) if rerank_scores else 0.0,
            "max_rerank_score": max(rerank_scores) if rerank_scores else 0.0,
        }
        
        logger.info(f"Final context: {len(context_chunks)} chunks, {context_data['total_tokens']} tokens")
        logger.info(f"Retrieval time: {retrieval_time:.2f}ms")
        logger.info(f"=" * 80)
        
        return context_data
    
    def _rewrite_query(self, query: str) -> str:
        if len(query.split()) < 4:
            return query
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            prompt = f"""Rewrite this query to be more effective for document retrieval.
Make it more specific and explicit. Keep it concise (max 20 words).

Original query: {query}

Rewritten query:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            if len(rewritten.split()) > 25:
                return query
            
            return rewritten
            
        except Exception as e:
            logger.debug(f"Query rewriting failed: {e}, using original")
            return query
    
    def _search(
        self,
        document_id: UUID,
        query: str,
        use_dense: bool = True,
        use_sparse: bool = True
    ) -> List[Dict[str, Any]]:
        logger.debug(f"Search: dense={use_dense}, sparse={use_sparse}")
        
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
        
        return results
    
    def _dynamic_threshold_filter(
        self,
        candidates: List[Dict[str, Any]],
        experiment: Dict
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        
        scores = [c.get("score", 0.0) for c in candidates]
        
        if len(scores) < 2:
            return candidates
        
        top_score = scores[0]
        percentile_score = np.percentile(scores, config.DYNAMIC_THRESHOLD_PERCENTILE * 100)
        
        threshold = top_score * config.DYNAMIC_THRESHOLD_MULTIPLIER
        threshold = min(threshold, percentile_score)
        
        filtered = [c for c in candidates if c.get("score", 0.0) >= threshold]
        
        if not filtered:
            filtered = candidates[:max(3, config.TOP_K_RERANK // 2)]
            logger.warning(f"No candidates above dynamic threshold {threshold:.3f}, keeping top {len(filtered)}")
        
        logger.info(f"Dynamic threshold: {threshold:.3f} (top={top_score:.3f}, percentile={percentile_score:.3f})")
        logger.debug(f"Score filtering: {len(candidates)} -> {len(filtered)}")
        
        return filtered
    
    def _get_score_threshold(self, experiment: Dict) -> float:
        if experiment.get("strict_thresholds", False):
            return 0.25
        return 0.12
    
    def _filter_by_score(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        filtered = [c for c in candidates if c.get("score", 0.0) >= threshold]
        
        if not filtered and candidates:
            filtered = candidates[:max(3, config.TOP_K_RERANK // 2)]
            logger.warning(f"No candidates above threshold {threshold}, keeping top {len(filtered)}")
        
        logger.debug(f"Score filtering: {len(candidates)} -> {len(filtered)} (threshold={threshold})")
        return filtered
    
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        target_k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        logger.info(f"Reranking {len(candidates)} candidates to top {target_k}")
        
        if len(candidates) <= target_k:
            scores = [c.get("score", 1.0) for c in candidates]
            return candidates, scores
        
        try:
            if self.reranker_type == "cohere":
                ranked, scores = self._rerank_cohere(query, candidates, target_k)
            else:
                ranked, scores = self._rerank_crossencoder(query, candidates, target_k)
            
            for c, s in zip(ranked, scores):
                c["rerank_score"] = s
            
            logger.info(f"Reranked to {len(ranked)} chunks, top scores: {scores[:min(5, len(scores))]}")
            return ranked, scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning top candidates")
            return candidates[:target_k], [1.0] * min(target_k, len(candidates))
    
    def _rerank_crossencoder(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        pairs = [[query, c["text"]] for c in candidates]
        
        scores = self.reranker.predict(pairs)
        
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_ranked = ranked[:top_k]
        top_candidates = [c for c, _ in top_ranked]
        top_scores = [float(s) for _, s in top_ranked]
        
        return top_candidates, top_scores
    
    def _rerank_cohere(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        documents = [c["text"] for c in candidates]
        
        results = self.cohere_client.rerank(
            model=config.COHERE_RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_k
        )
        
        ranked_candidates = []
        ranked_scores = []
        
        for result in results.results:
            idx = result.index
            ranked_candidates.append(candidates[idx])
            ranked_scores.append(float(result.relevance_score))
        
        return ranked_candidates, ranked_scores
    
    def _adaptive_rerank_filter(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        experiment: Dict
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        if not scores or len(scores) < 2:
            return candidates, scores
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        dynamic_threshold = mean_score - 0.5 * std_score
        
        min_threshold = config.RERANK_SCORE_THRESHOLD
        if experiment.get("min_rerank_score"):
            min_threshold = experiment["min_rerank_score"]
        
        threshold = max(dynamic_threshold, min_threshold)
        
        filtered_candidates = []
        filtered_scores = []
        for c, s in zip(candidates, scores):
            if s >= threshold:
                filtered_candidates.append(c)
                filtered_scores.append(s)
        
        if not filtered_candidates:
            filtered_candidates = candidates[:3]
            filtered_scores = scores[:3]
        
        logger.info(f"Adaptive threshold: {threshold:.3f} (mean={mean_score:.3f}, std={std_score:.3f})")
        logger.info(f"Filtered: {len(candidates)} -> {len(filtered_candidates)} chunks")
        
        return filtered_candidates, filtered_scores
    
    def _deduplicate(
        self,
        chunks: List[Dict[str, Any]],
        method: str = "semantic",
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        if len(chunks) <= 1:
            return chunks
        
        if method == "semantic":
            return self._deduplicate_semantic(chunks, threshold)
        else:
            return self._deduplicate_jaccard(chunks, threshold)
    
    def _deduplicate_semantic(
        self,
        chunks: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        if len(chunks) <= 1:
            return chunks
        
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(texts)
        
        similarities = np.dot(embeddings, embeddings.T)
        
        unique_chunks = []
        used_indices = set()
        
        for i in range(len(chunks)):
            if i in used_indices:
                continue
            
            unique_chunks.append(chunks[i])
            used_indices.add(i)
            
            for j in range(i + 1, len(chunks)):
                if j not in used_indices and similarities[i, j] >= threshold:
                    used_indices.add(j)
        
        logger.debug(f"Semantic dedup: {len(chunks)} -> {len(unique_chunks)} (threshold={threshold})")
        return unique_chunks
    
    def _deduplicate_jaccard(
        self,
        chunks: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        def jaccard_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0
        
        unique_chunks = []
        for chunk in chunks:
            is_duplicate = False
            for unique in unique_chunks:
                sim = jaccard_similarity(chunk["text"], unique["text"])
                if sim >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        logger.debug(f"Jaccard dedup: {len(chunks)} -> {len(unique_chunks)} (threshold={threshold})")
        return unique_chunks
    
    def _apply_mmr(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        lambda_param: float
    ) -> List[Dict[str, Any]]:
        query_emb = embedder.embed_query(query)
        
        texts = [c["text"] for c in candidates]
        candidate_embs = embedder.embed_texts(texts)
        
        selected = []
        selected_embs = []
        remaining_indices = list(range(len(candidates)))
        
        target_k = min(config.TOP_K_RERANK, len(candidates))
        
        for _ in range(target_k):
            if not remaining_indices:
                break
            
            mmr_scores = []
            for idx in remaining_indices:
                relevance = float(np.dot(query_emb, candidate_embs[idx]))
                
                if selected_embs:
                    similarities = [
                        float(np.dot(candidate_embs[idx], sel_emb))
                        for sel_emb in selected_embs
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0.0
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))
            
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected.append(candidates[best_idx])
            selected_embs.append(candidate_embs[best_idx])
            remaining_indices.remove(best_idx)
        
        logger.debug(f"MMR: selected {len(selected)} diverse chunks (lambda={lambda_param})")
        return selected
    
    def _assemble_smart_context(
        self,
        child_chunks: List[Dict[str, Any]],
        all_chunks_metadata: List[Dict[str, Any]],
        strategy: str = "child_with_parent_context"
    ) -> List[Dict[str, Any]]:
        logger.info(f"Assembling context with strategy: {strategy}")
        
        if strategy == "child_only":
            return child_chunks
        
        elif strategy == "parent_only":
            return self._get_parent_chunks(child_chunks, all_chunks_metadata)
        
        elif strategy == "child_with_parent_context":
            return self._add_parent_context_window(child_chunks, all_chunks_metadata)
        
        elif strategy == "smart":
            return self._smart_hybrid_context(child_chunks, all_chunks_metadata)
        
        else:
            logger.warning(f"Unknown strategy '{strategy}', using child_only")
            return child_chunks
    
    def _get_parent_chunks(
        self,
        child_chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        parent_ids = set(c["parent_id"] for c in child_chunks if c.get("parent_id"))
        
        parent_chunks = []
        for chunk_meta in all_chunks:
            if chunk_meta.get("id") in parent_ids and chunk_meta.get("is_parent"):
                parent_chunks.append(chunk_meta)
        
        ordered = []
        seen = set()
        for child in child_chunks:
            parent_id = child.get("parent_id")
            if parent_id and parent_id not in seen:
                parent = next((p for p in parent_chunks if p["id"] == parent_id), None)
                if parent:
                    ordered.append(parent)
                    seen.add(parent_id)
        
        logger.debug(f"Got {len(ordered)} parent chunks")
        return ordered
    
    def _add_parent_context_window(
        self,
        child_chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        enriched_chunks = []
        
        parent_lookup = {
            c["id"]: c for c in all_chunks if c.get("is_parent")
        }
        
        for child in child_chunks:
            parent_id = child.get("parent_id")
            if not parent_id or parent_id not in parent_lookup:
                enriched_chunks.append(child)
                continue
            
            parent = parent_lookup[parent_id]
            parent_text = parent["text"]
            child_text = child["text"]
            
            child_pos = parent_text.find(child_text)
            
            if child_pos == -1:
                enriched_chunks.append(child)
                continue
            
            window_tokens = config.PARENT_CONTEXT_WINDOW
            
            before_text = parent_text[:child_pos]
            after_text = parent_text[child_pos + len(child_text):]
            
            before_words = before_text.split()
            after_words = after_text.split()
            
            window_size = window_tokens // 2
            context_before = " ".join(before_words[-window_size:]) if before_words else ""
            context_after = " ".join(after_words[:window_size]) if after_words else ""
            
            enriched_text = " ".join([
                context_before,
                f"[FOCUS: {child_text}]",
                context_after
            ]).strip()
            
            enriched_chunk = child.copy()
            enriched_chunk["text"] = enriched_text
            enriched_chunk["metadata"] = enriched_chunk.get("metadata", {})
            enriched_chunk["metadata"]["enriched_with_parent_context"] = True
            enriched_chunk["metadata"]["original_child_text"] = child_text
            
            enriched_chunks.append(enriched_chunk)
        
        logger.debug(f"Enriched {len(enriched_chunks)} chunks with parent context")
        return enriched_chunks
    
    def _smart_hybrid_context(
        self,
        child_chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        threshold = 0.7
        
        context_chunks = []
        parent_lookup = {
            c["id"]: c for c in all_chunks if c.get("is_parent")
        }
        
        for i, child in enumerate(child_chunks):
            score = child.get("rerank_score", child.get("score", 0.0))
            
            if score >= threshold:
                parent_id = child.get("parent_id")
                if parent_id and parent_id in parent_lookup:
                    context_chunks.append(parent_lookup[parent_id])
                else:
                    context_chunks.append(child)
            else:
                context_chunks.append(child)
        
        logger.debug(f"Smart hybrid: mixed parent/child contexts")
        return context_chunks
    
    def _build_context_data(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_parts = []
        sources = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = chunk.get("token_count", 0)
            
            if total_tokens + chunk_tokens > config.MAX_CONTEXT_TOKENS:
                logger.info(f"Context limit reached at {total_tokens} tokens")
                break
            
            formatted = self._format_chunk(chunk)
            context_parts.append(formatted)
            total_tokens += chunk_tokens
            
            sources.append({
                "chunk_id": chunk.get("id") or chunk.get("chunk_id"),
                "text": chunk["text"][:200] + "...",
                "page": chunk.get("page_number"),
                "bbox": chunk.get("bbox"),
                "type": chunk.get("chunk_type"),
                "score": chunk.get("rerank_score", chunk.get("score", 0.0))
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Context: {total_tokens} tokens, {len(sources)} sources")
        
        return {
            "context": context,
            "sources": sources,
            "total_tokens": total_tokens
        }
    
    def _format_chunk(self, chunk: Dict[str, Any]) -> str:
        page = chunk.get("page_number", "?")
        chunk_type = chunk.get("chunk_type", "text")
        text = chunk.get("text", "")
        
        header = f"[Page {page} - {chunk_type.upper()}]"
        
        return f"{header}\n{text}"
    
    def _empty_context_data(self, mode: str) -> Dict[str, Any]:
        return {
            "context": "",
            "sources": [],
            "total_tokens": 0,
            "retrieved_chunks": [],
            "rerank_scores": [],
            "mode": mode,
            "metrics": {
                "total_candidates": 0,
                "after_filtering": 0,
                "final_chunks": 0,
                "retrieval_time_ms": 0,
                "mode": mode
            }
        }


retriever = EnhancedRetriever()
