from pathlib import Path
from uuid import uuid4
import json
import time
from typing import List, Dict, Any
import numpy as np
import logging

from parser import parser
from chunker import chunker
from embedder import embedder
from database import vector_db
from retriever import retriever
from generator import generator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RELEVANT_PAGES_KEYS = ["page_numbers","relevant_pages","pages","source_pages","ground_truth_pages","answer_pages"]


class RAGEvaluator:

    def load_test_dataset(self, path: str) -> List[Dict[str, Any]]:
        with open(path) as f:
            return json.load(f)

    def process_document(self, pdf_path: str):
        doc_id = uuid4()
        print("Parsing document...")
        total_pages, elements = parser.parse(pdf_path)

        print("Creating chunks...")
        chunks = chunker.create_chunks(str(doc_id), elements)

        chunks_metadata = [
            {
                "id": str(c.id),
                "parent_id": str(c.parent_id) if c.parent_id else None,
                "text": c.text,
                "chunk_type": c.chunk_type.value,
                "page_number": c.page_number,
                "bbox": c.bbox.model_dump() if c.bbox else None,
                "token_count": c.token_count,
                "is_parent": c.parent_id is None,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        child_chunks = [c for c in chunks if c.parent_id is not None]
        texts = [c.text for c in child_chunks]

        print("Generating embeddings...")
        dense_embeddings = embedder.embed_texts(texts)
        sparse_vectors = embedder.create_sparse_vectors_batch(texts)

        print("Indexing...")
        vector_db.index_chunks(child_chunks, dense_embeddings, sparse_vectors)
        print(f"✓ {len(chunks)} chunks\n")
        return doc_id, chunks_metadata

    def _extract_relevant_pages(self, gt: Dict) -> List[int]:
        for key in RELEVANT_PAGES_KEYS:
            pages = gt.get(key)
            if pages is not None:
                return [int(p) for p in pages]
        return []

    def calc_retrieval_metrics(self, retrieved: List[Dict], gt_pages: List[int]) -> Dict[str, float]:
        if not gt_pages:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}
        gt_set = set(gt_pages)
        seen = set()
        uniq_pages = []
        for c in retrieved:
            p = c["page_number"]
            if p not in seen:
                seen.add(p)
                uniq_pages.append(p)
        rel = len(set(uniq_pages) & gt_set)
        precision = rel / len(uniq_pages) if uniq_pages else 0.0
        recall = rel / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        mrr = next((1.0 / (rank + 1) for rank, p in enumerate(uniq_pages) if p in gt_set), 0.0)
        return {"precision": precision, "recall": recall, "f1": f1, "mrr": mrr}

    def calc_answer_similarity(self, generated: str, expected: str) -> float:
        if not generated or not expected:
            return 0.0
        try:
            g = embedder.embed_single(generated)
            e = embedder.embed_single(expected)
            return max(0.0, min(1.0, float(np.dot(g, e))))
        except Exception:
            gw, ew = set(generated.lower().split()), set(expected.lower().split())
            u = len(gw | ew)
            return len(gw & ew) / u if u else 0.0

    def evaluate_query(self, doc_id, chunks_metadata, query: str, gt: Dict) -> Dict[str, Any]:
        t0 = time.time()
        context_data = retriever.retrieve_context(document_id=doc_id, query=query, chunks_metadata=chunks_metadata)
        t1 = time.time()
        answer_data = generator.generate_answer(query=query, context_data=context_data)
        t2 = time.time()

        ret_metrics = self.calc_retrieval_metrics(context_data.get("retrieved_chunks", []), self._extract_relevant_pages(gt))
        ans_sim = self.calc_answer_similarity(answer_data["answer"], gt.get("expected_answer", ""))

        return {
            "precision": ret_metrics["precision"],
            "recall": ret_metrics["recall"],
            "f1": ret_metrics["f1"],
            "mrr": ret_metrics["mrr"],
            "answer_similarity": ans_sim,
            "confidence": answer_data["confidence"],
            "verified": answer_data["verified"],
            "citation_recall": answer_data.get("citation_metrics", {}).get("citation_recall", 0.0),
            "citation_precision": answer_data.get("citation_metrics", {}).get("citation_precision", 0.0),
            "retrieval_time": t1 - t0,
            "generation_time": t2 - t1,
            "total_time": t2 - t0,
            "num_sources": len(answer_data.get("sources", [])),
        }

    def aggregate(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {}
        mn = lambda k: float(np.mean([r[k] for r in results]))
        return {
            "precision": mn("precision"), "recall": mn("recall"), "f1": mn("f1"), "mrr": mn("mrr"),
            "answer_accuracy": mn("answer_similarity"), "confidence": mn("confidence"),
            "verification_rate": float(np.mean([1.0 if r["verified"] else 0.0 for r in results])),
            "citation_recall": mn("citation_recall"), "citation_precision": mn("citation_precision"),
            "avg_retrieval_time": mn("retrieval_time"), "avg_generation_time": mn("generation_time"),
            "avg_total_time": mn("total_time"), "avg_sources": mn("num_sources"),
            "total_queries": len(results),
        }

    def print_results(self, results: List[Dict]):
        m = self.aggregate(results)
        print(f"\n{'='*80}\nEVALUATION RESULTS\n{'='*80}")
        print(f"Retrieval  — Precision: {m['precision']:.1%}  Recall: {m['recall']:.1%}  F1: {m['f1']:.1%}  MRR: {m['mrr']:.3f}")
        print(f"Generation — Accuracy: {m['answer_accuracy']:.1%}  Confidence: {m['confidence']:.1%}  Verified: {m['verification_rate']:.1%}")
        print(f"Citation   — Recall: {m['citation_recall']:.1%}  Precision: {m['citation_precision']:.1%}")
        print(f"Speed      — Avg: {m['avg_total_time']:.2f}s  (Ret: {m['avg_retrieval_time']:.2f}s  Gen: {m['avg_generation_time']:.2f}s)")
        print(f"Sources    — Avg: {m['avg_sources']:.1f}  Queries: {int(m['total_queries'])}")
        print(f"{'='*80}\n")

    def save_results(self, results: List[Dict], output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.aggregate(results), f, indent=2)
        print(f"Saved: {output_path}")


def main():
    print("\nRAG EVALUATION\n")
    pdf_path = input("PDF path: ").strip().strip('"').strip("'")
    test_path = input("Test dataset JSON: ").strip().strip('"').strip("'")

    if not Path(pdf_path).exists():
        print("Error: PDF not found")
        return
    if not Path(test_path).exists():
        print("Error: Dataset not found")
        return

    ev = RAGEvaluator()
    doc_id, chunks_metadata = ev.process_document(pdf_path)
    test_data = ev.load_test_dataset(test_path)
    print(f"Evaluating {len(test_data)} queries...\n")

    results = []
    for i, tc in enumerate(test_data, 1):
        print(f"[{i}/{len(test_data)}] {tc['query'][:60]}...")
        results.append(ev.evaluate_query(doc_id, chunks_metadata, tc["query"], tc))

    ev.print_results(results)
    ev.save_results(results, "evaluation_results.json")


if __name__ == "__main__":
    main()
