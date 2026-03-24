from pathlib import Path
from uuid import uuid4
import json
import time
from typing import List, Dict, Any
import numpy as np
import logging

from pdf_parser import parser
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

    def process_document(self, path_str: str):
        from main import ingest_folder, ingest_single_pdf
        path = Path(path_str)
        if path.is_dir():
            doc_ids, chunks_metadata = ingest_folder(path_str)
            return None, chunks_metadata
        elif path.suffix.lower() == '.pdf':
            doc_ids, chunks_metadata = ingest_single_pdf(path_str)
            return doc_ids[0] if doc_ids else None, chunks_metadata
        else:
            print("Error: Unsupported path.")
            import sys
            sys.exit(1)

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

    def evaluate_all(self, test_data: List[Dict], doc_id, chunks_metadata):
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
        
        data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        
        for i, tc in enumerate(test_data, 1):
            print(f"[{i}/{len(test_data)}] {tc['query'][:60]}...")
            query = tc.get("query", "")
            gt = tc.get("expected_answer", "")
            
            context_data = retriever.retrieve_context(document_id=doc_id, query=query, chunks_metadata=chunks_metadata)
            answer_data = generator.generate_answer(query=query, context_data=context_data)
            
            contexts = [c.get("text", "") for c in context_data.get("retrieved_chunks", [])]
            
            data["question"].append(query)
            data["answer"].append(answer_data["answer"])
            data["contexts"].append(contexts)
            data["ground_truth"].append(gt)
            
        dataset = Dataset.from_dict(data)
        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        
        print("\nRunning RAGAS evaluation...")
        result = evaluate(dataset, metrics=metrics)
        
        return result


def main():
    print("\nRAGAS EVALUATION\n")
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
    
    result = ev.evaluate_all(test_data, doc_id, chunks_metadata)
    
    print(f"\n{'='*80}\nRAGAS EVALUATION RESULTS\n{'='*80}")
    print(result)
    print(f"{'='*80}\n")
    
    df = result.to_pandas()
    df.to_csv("ragas_evaluation_results.csv", index=False)
    print("Saved detailed results to ragas_evaluation_results.csv")

if __name__ == "__main__":
    main()
