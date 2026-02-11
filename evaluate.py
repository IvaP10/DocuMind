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
import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.results = {}
    
    def load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    def process_document(self, pdf_path: str):
        doc_id = uuid4()
        
        print("Parsing document...")
        total_pages, elements = parser.parse(pdf_path)
        
        print("Creating chunks...")
        chunks = chunker.create_chunks(str(doc_id), elements)
        
        chunks_metadata = [
            {
                "id": str(chunk.id),
                "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type.value,
                "page_number": chunk.page_number,
                "bbox": chunk.bbox.model_dump() if chunk.bbox else None,
                "token_count": chunk.token_count,
                "is_parent": chunk.parent_id is None,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        
        child_chunks = [c for c in chunks if c.parent_id is not None]
        texts = [chunk.text for chunk in child_chunks]

        print("Generating embeddings...")
        dense_embeddings = embedder.embed_texts(texts)
        sparse_vectors = embedder.create_sparse_vectors_batch(texts)
        
        print("Indexing into database...")
        vector_db.index_chunks(child_chunks, dense_embeddings, sparse_vectors)
        
        print(f"✓ Document processed: {len(chunks)} chunks\n")
        
        return doc_id, chunks_metadata
    
    def calculate_retrieval_metrics(self, retrieved_chunks: List[Dict], ground_truth_pages: List[int]) -> Dict[str, float]:
        if not ground_truth_pages:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}
        
        retrieved_pages = [chunk["page_number"] for chunk in retrieved_chunks]
        relevant_retrieved = len(set(retrieved_pages) & set(ground_truth_pages))
        
        precision = relevant_retrieved / len(retrieved_pages) if retrieved_pages else 0.0
        recall = relevant_retrieved / len(ground_truth_pages) if ground_truth_pages else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        mrr = 0.0
        for rank, page in enumerate(retrieved_pages, 1):
            if page in ground_truth_pages:
                mrr = 1.0 / rank
                break
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr
        }
    
    def calculate_answer_similarity(self, generated: str, expected: str) -> float:
        if not generated or not expected:
            return 0.0
        
        try:
            gen_embedding = embedder.embed_single(generated)
            exp_embedding = embedder.embed_single(expected)
            
            similarity = float(np.dot(gen_embedding, exp_embedding))
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            gen_words = set(generated.lower().split())
            exp_words = set(expected.lower().split())
            
            if not gen_words or not exp_words:
                return 0.0
            
            intersection = len(gen_words & exp_words)
            union = len(gen_words | exp_words)
            
            return intersection / union if union > 0 else 0.0
    
    def evaluate_query(self, doc_id, chunks_metadata, query: str, ground_truth: Dict[str, Any], mode: str = "hybrid_quality") -> Dict[str, Any]:
        start_time = time.time()
        
        context_data = retriever.retrieve_context(
            document_id=doc_id,
            query=query,
            chunks_metadata=chunks_metadata,
            mode=mode
        )
        
        retrieval_time = time.time() - start_time
        
        answer_data = generator.generate_answer(
            query=query,
            context_data=context_data
        )
        
        total_time = time.time() - start_time
        generation_time = total_time - retrieval_time
        
        retrieval_metrics = self.calculate_retrieval_metrics(
            context_data.get("retrieved_chunks", []),
            ground_truth.get("relevant_pages", [])
        )
        
        answer_similarity = self.calculate_answer_similarity(
            answer_data["answer"],
            ground_truth.get("expected_answer", "")
        )
        
        return {
            "precision": retrieval_metrics["precision"],
            "recall": retrieval_metrics["recall"],
            "f1": retrieval_metrics["f1"],
            "mrr": retrieval_metrics["mrr"],
            "answer_similarity": answer_similarity,
            "confidence": answer_data["confidence"],
            "verified": answer_data["verified"],
            "citation_recall": answer_data.get("citation_metrics", {}).get("citation_recall", 0.0),
            "citation_precision": answer_data.get("citation_metrics", {}).get("citation_precision", 0.0),
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "num_sources": len(answer_data.get("sources", []))
        }
    
    def evaluate_all_modes(self, doc_id, chunks_metadata, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        modes = ["hybrid_quality", "fast_response"]
        mode_results = {mode: [] for mode in modes}
        
        for i, test_case in enumerate(test_data, 1):
            query = test_case["query"]
            print(f"[{i}/{len(test_data)}] {query[:60]}...")
            
            for mode in modes:
                result = self.evaluate_query(
                    doc_id, 
                    chunks_metadata, 
                    query, 
                    test_case,
                    mode
                )
                mode_results[mode].append(result)
        
        return mode_results
    
    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        
        metrics = {
            "precision": np.mean([r["precision"] for r in results]),
            "recall": np.mean([r["recall"] for r in results]),
            "f1": np.mean([r["f1"] for r in results]),
            "mrr": np.mean([r["mrr"] for r in results]),
            "answer_accuracy": np.mean([r["answer_similarity"] for r in results]),
            "confidence": np.mean([r["confidence"] for r in results]),
            "verification_rate": np.mean([1.0 if r["verified"] else 0.0 for r in results]),
            "citation_recall": np.mean([r["citation_recall"] for r in results]),
            "citation_precision": np.mean([r["citation_precision"] for r in results]),
            "avg_retrieval_time": np.mean([r["retrieval_time"] for r in results]),
            "avg_generation_time": np.mean([r["generation_time"] for r in results]),
            "avg_total_time": np.mean([r["total_time"] for r in results]),
            "avg_sources": np.mean([r["num_sources"] for r in results]),
            "total_queries": len(results)
        }
        
        return metrics
    
    def print_results(self, mode_results: Dict[str, List[Dict[str, Any]]]):
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}\n")
        
        for mode, results in mode_results.items():
            metrics = self.aggregate_metrics(results)
            mode_name = config.EXPERIMENT_MODES[mode]["name"]
            
            print(f"{mode_name}:")
            print(f"  Precision: {metrics['precision']:.1%}  Recall: {metrics['recall']:.1%}  F1: {metrics['f1']:.1%}")
            print(f"  MRR: {metrics['mrr']:.3f}  Accuracy: {metrics['answer_accuracy']:.1%}  Confidence: {metrics['confidence']:.1%}")
            print(f"  Citation: {metrics['citation_recall']:.1%} recall, {metrics['citation_precision']:.1%} precision")
            print(f"  Time: {metrics['avg_total_time']:.2f}s\n")
        
        print(f"{'='*80}\n")
    
    def save_results(self, mode_results: Dict[str, List[Dict[str, Any]]], output_path: str):
        aggregated = {}
        for mode, results in mode_results.items():
            aggregated[mode] = self.aggregate_metrics(results)
        
        with open(output_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"Saved: {output_path}")


def main():
    print("\nRAG EVALUATION\n")
    
    pdf_path = input("PDF path: ").strip().strip('"').strip("'")
    test_data_path = input("Test dataset JSON: ").strip().strip('"').strip("'")
    
    if not Path(pdf_path).exists():
        print("Error: PDF not found")
        return
    
    if not Path(test_data_path).exists():
        print("Error: Dataset not found")
        return
    
    evaluator = RAGEvaluator()
    
    doc_id, chunks_metadata = evaluator.process_document(pdf_path)
    
    test_data = evaluator.load_test_dataset(test_data_path)
    print(f"Evaluating {len(test_data)} queries...\n")
    
    mode_results = evaluator.evaluate_all_modes(doc_id, chunks_metadata, test_data)
    
    evaluator.print_results(mode_results)
    
    evaluator.save_results(mode_results, "evaluation_results.json")


if __name__ == "__main__":
    main()