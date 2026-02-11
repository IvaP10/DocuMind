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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    
    def __init__(self):
        self._init_crossencoder_reranker()
    
    def _init_crossencoder_reranker(self):
        try:
            self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)
            self.reranker_type = "crossencoder"
        except Exception as e:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker_type = "crossencoder"
    
    def _extract_query_numbers(self, query: str) -> List[str]:
        numeric_patterns = [
            r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb](?:illion)?)?',
            r'-?\d+(?:\.\d+)?%',
            r'\b(?:19|20)\d{2}\b',
            r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        ]
        
        numbers = []
        for pattern in numeric_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            numbers.extend(matches)
        
        return list(set(numbers))
    
    def _classify_query_type(self, query: str) -> str:
        query_lower = query.lower()
        
        keyword_indicators = [
            'specific', 'code', 'number', 'id', 'identifier', 'exact',
            'name', 'date', 'year', 'percentage', 'dollar', 'price'
        ]
        
        conceptual_indicators = [
            'how', 'why', 'what is', 'explain', 'describe', 'summary',
            'overview', 'concept', 'understand', 'meaning', 'process'
        ]
        
        has_numbers = bool(re.search(r'\d', query))
        
        keyword_score = sum(1 for kw in keyword_indicators if kw in query_lower)
        conceptual_score = sum(1 for kw in conceptual_indicators if kw in query_lower)
        
        if has_numbers or keyword_score > conceptual_score:
            return "keyword"
        else:
            return "conceptual"
    
    def _get_adaptive_weights(self, query: str) -> Tuple[float, float]:
        query_type = self._classify_query_type(query)
        
        if query_type == "keyword":
            return 0.3, 0.7
        else:
            return 0.8, 0.2
    
    def retrieve_context(
        self,
        document_id: UUID,
        query: str,
        chunks_metadata: List[Dict[str, Any]],
        mode: str = "hybrid_quality"
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        experiment = config.EXPERIMENT_MODES.get(mode, config.EXPERIMENT_MODES["hybrid_quality"])
        
        processed_query = query
        if experiment.get("use_query_rewriting", False):
            processed_query = self._rewrite_query(query)
        
        query_numbers = self._extract_query_numbers(query)
        
        candidates = self._search(
            document_id,
            processed_query,
            use_dense=experiment["use_dense"],
            use_sparse=experiment["use_sparse"],
            experiment=experiment,
            query_original=query
        )
        
        initial_count = len(candidates)
        
        if not candidates:
            return self._empty_context_data(mode)
        
        if query_numbers and experiment.get("numeric_boost_factor"):
            candidates = self._boost_numeric_matches(candidates, query_numbers, chunks_metadata, experiment)
        
        if experiment.get("dynamic_threshold", False):
            candidates = self._dynamic_threshold_filter(candidates, experiment)
        elif experiment.get("apply_score_threshold", False):
            threshold = self._get_score_threshold(experiment)
            candidates = self._filter_by_score(candidates, threshold)
        
        if experiment["use_reranker"]:
            target_k = experiment.get("top_k_rerank", config.TOP_K_RERANK)
            candidates, rerank_scores = self._rerank(query, candidates, target_k)
            
            if experiment.get("use_llm_listwise_rerank", False) and len(candidates) >= 5:
                candidates, rerank_scores = self._llm_listwise_rerank(query, candidates[:10])
            
            if experiment.get("adaptive_threshold", False):
                candidates, rerank_scores = self._adaptive_rerank_filter(
                    candidates, rerank_scores, experiment
                )
        else:
            target_k = experiment.get("top_k_rerank", config.TOP_K_RERANK)
            candidates = candidates[:target_k]
            rerank_scores = [c["score"] for c in candidates]
        
        if experiment.get("use_mmr", False):
            mmr_lambda = experiment.get("mmr_lambda", config.MMR_LAMBDA)
            candidates = self._apply_mmr(query, candidates, mmr_lambda)
        
        if experiment.get("dedup_method"):
            dedup_threshold = experiment.get("dedup_threshold", config.DEDUP_SIMILARITY_THRESHOLD)
            candidates = self._deduplicate(candidates, experiment["dedup_method"], dedup_threshold)
        
        context_data = self._build_context_data(
            candidates,
            chunks_metadata,
            experiment,
            rerank_scores,
            mode
        )
        
        retrieval_time = time.time() - start_time
        context_data["metrics"]["retrieval_time_ms"] = retrieval_time * 1000
        
        return context_data
    
    def _boost_numeric_matches(
        self,
        candidates: List[Dict[str, Any]],
        query_numbers: List[str],
        chunks_metadata: List[Dict[str, Any]],
        experiment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        
        boost_factor = experiment.get("numeric_boost_factor", 2.0)
        
        metadata_map = {chunk["id"]: chunk for chunk in chunks_metadata}
        
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            chunk_meta = metadata_map.get(chunk_id)
            
            if not chunk_meta:
                continue
            
            chunk_numbers = chunk_meta.get("metadata", {}).get("numbers", [])
            if not chunk_numbers:
                continue
            
            chunk_number_values = [num["value"] for num in chunk_numbers]
            
            has_match = any(qnum in chunk_number_values for qnum in query_numbers)
            
            if has_match:
                candidate["score"] *= boost_factor
                candidate["numeric_boost"] = True
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return candidates
    
    def _llm_listwise_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            snippet_map = {}
            for i, candidate in enumerate(candidates):
                snippet_map[i] = {
                    "text": candidate["text"][:300],
                    "candidate": candidate
                }
            
            snippets_text = "\n\n".join([
                f"ID {i}: {snippet_map[i]['text']}"
                for i in range(len(candidates))
            ])
            
            prompt = f"""Rank these document snippets by relevance to the query. Return ONLY a comma-separated list of IDs in order from most to least relevant.

Query: {query}

Snippets:
{snippets_text}

Return format: 0,3,1,4,2 (just the IDs, no explanation)"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=50,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            ranking_text = response.choices[0].message.content.strip()
            
            ranked_ids = [int(x.strip()) for x in ranking_text.split(',') if x.strip().isdigit()]
            
            reranked = []
            for rank, idx in enumerate(ranked_ids):
                if idx in snippet_map:
                    candidate = snippet_map[idx]["candidate"]
                    candidate["llm_rank"] = rank
                    candidate["rerank_score"] = 1.0 - (rank / len(ranked_ids))
                    reranked.append(candidate)
            
            for i, candidate in enumerate(candidates):
                if candidate not in reranked:
                    candidate["llm_rank"] = len(reranked) + i
                    candidate["rerank_score"] = 0.0
                    reranked.append(candidate)
            
            rerank_scores = [c["rerank_score"] for c in reranked]
            
            return reranked, rerank_scores
            
        except Exception as e:
            logger.error(f"LLM listwise reranking failed: {e}")
            return candidates, [c.get("rerank_score", c["score"]) for c in candidates]
    
    def _rewrite_query(self, query: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Rewrite queries to be more specific and retrieval-friendly. Keep the core intent but expand with relevant terms. Return only the rewritten query."},
                    {"role": "user", "content": f"Rewrite: {query}"}
                ]
            )
            
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else query
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
    
    def _search(
        self,
        document_id: UUID,
        query: str,
        use_dense: bool = True,
        use_sparse: bool = True,
        experiment: Dict[str, Any] = None,
        query_original: str = None
    ) -> List[Dict[str, Any]]:
        
        if use_dense and use_sparse:
            return self._hybrid_search_parallel(document_id, query, experiment, query_original)
        elif use_dense:
            return self._dense_search(document_id, query)
        elif use_sparse:
            return self._sparse_search(document_id, query)
        else:
            return []
    
    def _hybrid_search_parallel(
        self,
        document_id: UUID,
        query: str,
        experiment: Optional[Dict[str, Any]],
        query_original: Optional[str]
    ) -> List[Dict[str, Any]]:
        
        dense_weight = config.DENSE_WEIGHT
        sparse_weight = config.SPARSE_WEIGHT
        
        if experiment and experiment.get("query_adaptive_weights") and query_original:
            dense_weight, sparse_weight = self._get_adaptive_weights(query_original)
        
        dense_embedding = None
        sparse_vector = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(embedder.embed_query, query)
            future_sparse = executor.submit(embedder.create_sparse_vector, query, "query")
            
            dense_embedding = future_dense.result()
            sparse_vector = future_sparse.result()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense_search = executor.submit(
                vector_db.dense_search,
                document_id,
                dense_embedding,
                config.TOP_K_INITIAL
            )
            future_sparse_search = executor.submit(
                vector_db.sparse_search,
                document_id,
                sparse_vector,
                config.TOP_K_INITIAL
            )
            
            dense_results = future_dense_search.result()
            sparse_results = future_sparse_search.result()
        
        results = self._fuse_results(dense_results, sparse_results, dense_weight, sparse_weight)
        
        return results
    
    def _fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Dict[str, Any]]:
        
        k = config.RRF_K_PARAM
        
        scores = {}
        all_results = {}
        
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + (dense_weight * rrf_score)
            all_results[chunk_id] = result
        
        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + (sparse_weight * rrf_score)
            if chunk_id not in all_results:
                all_results[chunk_id] = result
        
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        fused = []
        for chunk_id in ranked_ids:
            result = all_results[chunk_id]
            result["score"] = scores[chunk_id]
            fused.append(result)
        
        return fused
    
    def _dense_search(self, document_id: UUID, query: str) -> List[Dict[str, Any]]:
        dense_embedding = embedder.embed_query(query)
        
        results = vector_db.dense_search(
            document_id=document_id,
            dense_vector=dense_embedding,
            limit=config.TOP_K_INITIAL
        )
        
        return results
    
    def _sparse_search(self, document_id: UUID, query: str) -> List[Dict[str, Any]]:
        sparse_vector = embedder.create_sparse_vector(query, mode="query")
        
        results = vector_db.sparse_search(
            document_id=document_id,
            sparse_vector=sparse_vector,
            limit=config.TOP_K_INITIAL
        )
        
        return results
    
    def _dynamic_threshold_filter(
        self,
        candidates: List[Dict[str, Any]],
        experiment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        
        if not candidates:
            return candidates
        
        scores = [c["score"] for c in candidates]
        
        percentile = experiment.get("dynamic_threshold_percentile", config.DYNAMIC_THRESHOLD_PERCENTILE)
        multiplier = experiment.get("dynamic_threshold_multiplier", config.DYNAMIC_THRESHOLD_MULTIPLIER)
        
        threshold_score = np.percentile(scores, percentile * 100)
        dynamic_threshold = threshold_score * multiplier
        
        filtered = [c for c in candidates if c["score"] >= dynamic_threshold]
        
        return filtered if filtered else candidates[:5]
    
    def _get_score_threshold(self, experiment: Dict[str, Any]) -> float:
        return experiment.get("min_rerank_score", config.RERANK_SCORE_THRESHOLD)
    
    def _filter_by_score(self, candidates: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        filtered = [c for c in candidates if c["score"] >= threshold]
        return filtered if filtered else candidates[:3]
    
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        
        if not candidates:
            return [], []
        
        pairs = [[query, candidate["text"]] for candidate in candidates]
        
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)
        
        candidates_sorted = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        top_candidates = candidates_sorted[:top_k]
        top_scores = [c["rerank_score"] for c in top_candidates]
        
        return top_candidates, top_scores
    
    def _adaptive_rerank_filter(
        self,
        candidates: List[Dict[str, Any]],
        rerank_scores: List[float],
        experiment: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        
        if not rerank_scores:
            return candidates, rerank_scores
        
        mean_score = np.mean(rerank_scores)
        std_score = np.std(rerank_scores)
        
        threshold = mean_score - 0.5 * std_score
        
        filtered_candidates = []
        filtered_scores = []
        
        for candidate, score in zip(candidates, rerank_scores):
            if score >= threshold:
                filtered_candidates.append(candidate)
                filtered_scores.append(score)
        
        return filtered_candidates, filtered_scores
    
    def _apply_mmr(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        lambda_param: float
    ) -> List[Dict[str, Any]]:
        
        if len(candidates) <= 1:
            return candidates
        
        query_embedding = embedder.embed_query(query)
        
        texts = [c["text"] for c in candidates]
        doc_embeddings = embedder.embed_texts(texts)
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        first_idx = 0
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        while remaining_indices and len(selected_indices) < len(candidates):
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                relevance = float(np.dot(query_embedding, doc_embeddings[idx]))
                
                max_similarity = max([
                    float(np.dot(doc_embeddings[idx], doc_embeddings[sel_idx]))
                    for sel_idx in selected_indices
                ])
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        reranked_candidates = [candidates[i] for i in selected_indices]
        
        return reranked_candidates
    
    def _deduplicate(
        self,
        candidates: List[Dict[str, Any]],
        method: str,
        threshold: float
    ) -> List[Dict[str, Any]]:
        
        if not candidates or method not in ["semantic", "jaccard"]:
            return candidates
        
        if method == "semantic":
            return self._deduplicate_semantic(candidates, threshold)
        else:
            return self._deduplicate_jaccard(candidates, threshold)
    
    def _deduplicate_semantic(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        
        embeddings = embedder.embed_texts([c["text"] for c in candidates])
        
        keep = []
        for i, candidate in enumerate(candidates):
            is_duplicate = False
            
            for j in keep:
                similarity = float(np.dot(embeddings[i], embeddings[j]))
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(i)
        
        return [candidates[i] for i in keep]
    
    def _deduplicate_jaccard(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        
        def jaccard_similarity(text1: str, text2: str) -> float:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            
            return intersection / union if union > 0 else 0.0
        
        keep = []
        for i, candidate in enumerate(candidates):
            is_duplicate = False
            
            for j in keep:
                similarity = jaccard_similarity(candidate["text"], candidates[j]["text"])
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(i)
        
        return [candidates[i] for i in keep]
    
    def _build_context_data(
        self,
        candidates: List[Dict[str, Any]],
        chunks_metadata: List[Dict[str, Any]],
        experiment: Dict[str, Any],
        rerank_scores: List[float],
        mode: str
    ) -> Dict[str, Any]:
        
        context_strategy = experiment.get("context_strategy", "child_only")
        
        if context_strategy == "child_with_parent_context":
            context, sources, total_tokens = self._build_parent_child_context(
                candidates, chunks_metadata, experiment
            )
        else:
            context, sources, total_tokens = self._build_child_only_context(candidates)
        
        avg_rerank = np.mean(rerank_scores) if rerank_scores else 0.0
        max_rerank = max(rerank_scores) if rerank_scores else 0.0
        min_rerank = min(rerank_scores) if rerank_scores else 0.0
        
        metrics = {
            "total_candidates": len(candidates),
            "final_chunks": len(candidates),
            "avg_rerank_score": float(avg_rerank),
            "max_rerank_score": float(max_rerank),
            "min_rerank_score": float(min_rerank),
            "mode": mode
        }
        
        return {
            "context": context,
            "sources": sources,
            "total_tokens": total_tokens,
            "retrieved_chunks": candidates,
            "rerank_scores": rerank_scores,
            "mode": mode,
            "metrics": metrics
        }
    
    def _build_parent_child_context(
        self,
        candidates: List[Dict[str, Any]],
        chunks_metadata: List[Dict[str, Any]],
        experiment: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], int]:
        
        metadata_map = {chunk["id"]: chunk for chunk in chunks_metadata}
        
        context_parts = []
        sources = []
        total_tokens = 0
        
        parent_context_window = experiment.get("parent_context_window", config.PARENT_CONTEXT_WINDOW)
        
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            chunk_metadata = metadata_map.get(chunk_id)
            
            if not chunk_metadata:
                continue
            
            parent_id = chunk_metadata.get("parent_id")
            
            if parent_id:
                parent_metadata = metadata_map.get(parent_id)
                if parent_metadata:
                    parent_text = parent_metadata["text"]
                    child_text = chunk_metadata["text"]
                    
                    child_start = parent_text.find(child_text)
                    
                    if child_start >= 0:
                        context_start = max(0, child_start - parent_context_window)
                        context_end = min(len(parent_text), child_start + len(child_text) + parent_context_window)
                        
                        expanded_text = parent_text[context_start:context_end]
                    else:
                        expanded_text = child_text
                    
                    context_text = expanded_text
                else:
                    context_text = chunk_metadata["text"]
            else:
                context_text = chunk_metadata["text"]
            
            page = chunk_metadata.get("page_number", 1)
            
            context_parts.append(f"[Page {page}]\n{context_text}")
            
            sources.append({
                "chunk_id": chunk_id,
                "text": context_text,
                "page": page,
                "score": candidate.get("rerank_score", candidate.get("score", 0.0))
            })
            
            tokens = chunk_metadata.get("token_count", len(context_text.split()))
            total_tokens += tokens
            
            if total_tokens >= config.MAX_CONTEXT_TOKENS:
                break
        
        context = "\n\n".join(context_parts)
        
        return context, sources, total_tokens
    
    def _build_child_only_context(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], int]:
        
        context_parts = []
        sources = []
        total_tokens = 0
        
        for candidate in candidates:
            text = candidate["text"]
            page = candidate.get("page_number", 1)
            
            context_parts.append(f"[Page {page}]\n{text}")
            
            sources.append({
                "chunk_id": candidate["chunk_id"],
                "text": text,
                "page": page,
                "score": candidate.get("rerank_score", candidate.get("score", 0.0))
            })
            
            tokens = len(text.split())
            total_tokens += tokens
            
            if total_tokens >= config.MAX_CONTEXT_TOKENS:
                break
        
        context = "\n\n".join(context_parts)
        
        return context, sources, total_tokens
    
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
                "final_chunks": 0,
                "avg_rerank_score": 0.0,
                "max_rerank_score": 0.0,
                "min_rerank_score": 0.0,
                "mode": mode
            }
        }


retriever = EnhancedRetriever()