from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from sentence_transformers import CrossEncoder
import config
from database import vector_db
from embedder import embedder
import logging
import numpy as np
import time
import re
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class EnhancedRetriever:

    def __init__(self):
        try:
            self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)
        except Exception:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _extract_query_numbers(self, query: str) -> List[str]:
        patterns = [
            r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb](?:illion)?)?',
            r'-?\d+(?:\.\d+)?%',
            r'\b(?:19|20)\d{2}\b',
            r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        ]
        numbers = []
        for p in patterns:
            numbers.extend(re.findall(p, query, re.IGNORECASE))
        return list(set(numbers))

    def _get_adaptive_weights(self, query: str) -> Tuple[float, float]:
        q = query.lower()
        keyword_score = sum(1 for kw in ['specific','code','number','id','exact','name','date','year','percentage','dollar','price'] if kw in q)
        conceptual_score = sum(1 for kw in ['how','why','what is','explain','describe','summary','overview','concept','understand'] if kw in q)
        has_numbers = bool(re.search(r'\d', query))
        if has_numbers or keyword_score > conceptual_score:
            return 0.3, 0.7
        return 0.8, 0.2

    def retrieve_context(self, document_id: UUID, query: str, chunks_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.time()
        exp = config.MODE

        query_numbers = self._extract_query_numbers(query)

        candidates, query_embedding = self._hybrid_search(document_id, query)

        if not candidates:
            return self._empty_context()

        if query_numbers:
            candidates = self._boost_numeric_matches(candidates, query_numbers, chunks_metadata)

        candidates, rerank_scores = self._rerank(query, candidates, exp["top_k_rerank"])

        candidates, rerank_scores = self._adaptive_rerank_filter(candidates, rerank_scores)

        candidates = self._apply_mmr(query_embedding, candidates, exp["mmr_lambda"])

        candidates = self._deduplicate_fast(candidates, exp["dedup_threshold"])

        context_data = self._build_context(candidates, chunks_metadata, rerank_scores)
        context_data["metrics"]["retrieval_time_ms"] = (time.time() - start_time) * 1000
        return context_data

    def _hybrid_search(self, document_id: UUID, query: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        dense_w, sparse_w = self._get_adaptive_weights(query)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_dense = ex.submit(embedder.embed_query, query)
            f_sparse = ex.submit(embedder.create_sparse_vector, query, "query")
            dense_emb = f_dense.result()
            sparse_vec = f_sparse.result()

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_ds = ex.submit(vector_db.dense_search, document_id, dense_emb, config.TOP_K_INITIAL)
            f_ss = ex.submit(vector_db.sparse_search, document_id, sparse_vec, config.TOP_K_INITIAL)
            dense_res = f_ds.result()
            sparse_res = f_ss.result()

        return self._fuse_results(dense_res, sparse_res, dense_w, sparse_w), dense_emb

    def _fuse_results(self, dense: List, sparse: List, dw: float, sw: float) -> List[Dict[str, Any]]:
        k = config.RRF_K_PARAM
        scores: Dict[str, float] = {}
        pool: Dict[str, Dict] = {}

        for rank, r in enumerate(dense, 1):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + dw / (k + rank)
            pool[cid] = r

        for rank, r in enumerate(sparse, 1):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + sw / (k + rank)
            pool.setdefault(cid, r)

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        for cid in ranked:
            pool[cid]["score"] = scores[cid]
        return [pool[cid] for cid in ranked]

    def _boost_numeric_matches(self, candidates: List[Dict], query_numbers: List[str], chunks_metadata: List[Dict]) -> List[Dict]:
        meta_map = {c["id"]: c for c in chunks_metadata}
        boost = config.MODE["numeric_boost_factor"]
        for c in candidates:
            meta = meta_map.get(c["chunk_id"])
            if not meta:
                continue
            chunk_nums = [n["value"] for n in meta.get("metadata", {}).get("numbers", [])]
            if any(qn in chunk_nums for qn in query_numbers):
                c["score"] *= boost
                c["numeric_boost"] = True
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> Tuple[List[Dict], List[float]]:
        if not candidates:
            return [], []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        sorted_cands = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        return sorted_cands, [c["rerank_score"] for c in sorted_cands]

    def _adaptive_rerank_filter(self, candidates: List[Dict], scores: List[float]) -> Tuple[List[Dict], List[float]]:
        if not scores:
            return candidates, scores
        mean = np.mean(scores)
        filtered = [(c, s) for c, s in zip(candidates, scores) if s >= mean]
        if not filtered:
            return candidates[:3], scores[:3]
        fc, fs = zip(*filtered)
        return list(fc), list(fs)

    def _apply_mmr(self, query_embedding: np.ndarray, candidates: List[Dict], lam: float) -> List[Dict]:
        if len(candidates) <= 1:
            return candidates
        doc_embs = embedder.embed_texts([c["text"] for c in candidates])
        selected = [0]
        remaining = list(range(1, len(candidates)))
        while remaining:
            best_score = -float('inf')
            best_idx = None
            for idx in remaining:
                rel = float(np.dot(query_embedding, doc_embs[idx]))
                max_sim = max(float(np.dot(doc_embs[idx], doc_embs[s])) for s in selected)
                mmr = lam * rel - (1 - lam) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [candidates[i] for i in selected]

    def _deduplicate_fast(self, candidates: List[Dict], threshold: float) -> List[Dict]:
        if not candidates:
            return candidates
        keep = []
        seen_texts = []
        for c in candidates:
            text = c["text"].lower().strip()
            words = set(text.split())
            is_dup = False
            for seen in seen_texts:
                intersection = len(words & seen)
                union = len(words | seen)
                if union > 0 and intersection / union >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(c)
                seen_texts.append(words)
        return keep

    def _build_context(self, candidates: List[Dict], chunks_metadata: List[Dict], rerank_scores: List[float]) -> Dict[str, Any]:
        meta_map = {c["id"]: c for c in chunks_metadata}
        context_parts = []
        sources = []
        total_tokens = 0
        pw = config.PARENT_CONTEXT_WINDOW

        for candidate in candidates:
            cid = candidate["chunk_id"]
            meta = meta_map.get(cid)
            if not meta:
                continue
            parent_id = meta.get("parent_id")
            if parent_id:
                parent = meta_map.get(parent_id)
                if parent:
                    pt = parent["text"]
                    ct = meta["text"]
                    start = pt.find(ct)
                    if start >= 0:
                        expanded = pt[max(0, start - pw): min(len(pt), start + len(ct) + pw)]
                    else:
                        expanded = ct
                else:
                    expanded = meta["text"]
            else:
                expanded = meta["text"]

            page = meta.get("page_number", 1)
            context_parts.append(f"[Page {page}]\n{expanded}")
            sources.append({"chunk_id": cid, "text": expanded, "page": page, "score": candidate.get("rerank_score", candidate.get("score", 0.0))})
            total_tokens += meta.get("token_count", len(expanded.split()))
            if total_tokens >= config.MAX_CONTEXT_TOKENS:
                break

        avg_r = float(np.mean(rerank_scores)) if rerank_scores else 0.0
        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "total_tokens": total_tokens,
            "retrieved_chunks": candidates,
            "rerank_scores": rerank_scores,
            "metrics": {
                "total_candidates": len(candidates),
                "final_chunks": len(candidates),
                "avg_rerank_score": avg_r,
                "max_rerank_score": float(max(rerank_scores)) if rerank_scores else 0.0,
                "min_rerank_score": float(min(rerank_scores)) if rerank_scores else 0.0,
            }
        }

    def _empty_context(self) -> Dict[str, Any]:
        return {
            "context": "", "sources": [], "total_tokens": 0,
            "retrieved_chunks": [], "rerank_scores": [],
            "metrics": {"total_candidates": 0, "final_chunks": 0, "avg_rerank_score": 0.0, "max_rerank_score": 0.0, "min_rerank_score": 0.0}
        }


retriever = EnhancedRetriever()
