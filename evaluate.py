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

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        self.results = {
            "retrieval_metrics": {},
            "generation_metrics": {},
            "overall_metrics": {},
            "mode_comparison": {}
        }
    
    def load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    def process_document(self, pdf_path: str):
        print(f"📄 Processing document: {pdf_path}")
        
        doc_id = uuid4()
        
        print("  → Parsing PDF...")
        total_pages, elements = parser.parse(pdf_path)
        print(f"  ✓ Parsed {total_pages} pages, {len(elements)} elements")
        
        print("  → Creating chunks...")
        chunks = chunker.create_chunks(str(doc_id), elements)
        print(f"  ✓ Created {len(chunks)} chunks")
        
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
        print(f"  → Indexing {len(child_chunks)} child chunks...")

        dense_embeddings = embedder.embed_texts(texts)
        sparse_vectors = embedder.create_sparse_vectors_batch(texts)
        vector_db.index_chunks(child_chunks, dense_embeddings, sparse_vectors)
        print(f"  ✓ Indexed successfully")
        
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
            "mrr": mrr,
            "relevant_retrieved": relevant_retrieved,
            "total_retrieved": len(retrieved_pages),
            "total_relevant": len(ground_truth_pages)
        }
    
    def calculate_answer_similarity(self, generated: str, expected: str) -> float:
        """Calculate semantic similarity using embeddings instead of Jaccard"""
        if not generated or not expected:
            return 0.0
        
        try:
            # Use embedder for semantic similarity
            from embedder import embedder
            import numpy as np
            
            gen_embedding = embedder.embed_single(generated)
            exp_embedding = embedder.embed_single(expected)
            
            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(gen_embedding, exp_embedding))
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            # Fallback to improved word overlap
            gen_words = set(generated.lower().split())
            exp_words = set(expected.lower().split())
            
            if not gen_words or not exp_words:
                return 0.0
            
            intersection = len(gen_words & exp_words)
            union = len(gen_words | exp_words)
            
            jaccard = intersection / union if union > 0 else 0.0
            return jaccard
    
    def evaluate_query(self, doc_id, chunks_metadata, query: str, ground_truth: Dict[str, Any], mode: str = "hybrid_full") -> Dict[str, Any]:
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
            "query": query,
            "mode": mode,
            "retrieval_metrics": retrieval_metrics,
            "answer_similarity": answer_similarity,
            "confidence": answer_data["confidence"],
            "verified": answer_data["verified"],
            "citation_metrics": answer_data.get("citation_metrics", {}),
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "num_sources": len(answer_data.get("sources", []))
        }
    
    def evaluate_all_modes(self, doc_id, chunks_metadata, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        # UPDATED: Include precision_optimized mode
        modes = ["hybrid_full", "dense_only", "no_rerank", "no_parent", "precision_optimized"]
        
        # Filter modes to only those that exist in config
        available_modes = [m for m in modes if m in config.EXPERIMENT_MODES]
        mode_results = {mode: [] for mode in available_modes}
        
        print(f"\n{'='*100}")
        print(f"EVALUATING {len(test_data)} QUERIES ACROSS {len(available_modes)} MODES")
        print(f"{'='*100}\n")
        
        for i, test_case in enumerate(test_data, 1):
            query = test_case["query"]
            print(f"[{i}/{len(test_data)}] {query[:70]}...")
            
            for mode in available_modes:
                mode_name = config.EXPERIMENT_MODES[mode]["name"]
                print(f"  → Testing: {mode_name}")
                
                result = self.evaluate_query(
                    doc_id, 
                    chunks_metadata, 
                    query, 
                    test_case,
                    mode
                )
                mode_results[mode].append(result)
                
                print(f"    P: {result['retrieval_metrics']['precision']:.3f} | " +
                      f"R: {result['retrieval_metrics']['recall']:.3f} | " +
                      f"F1: {result['retrieval_metrics']['f1']:.3f} | " +
                      f"Sim: {result['answer_similarity']:.3f}")
            print()
        
        return mode_results
    
    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        
        metrics = {
            "avg_precision": np.mean([r["retrieval_metrics"]["precision"] for r in results]),
            "avg_recall": np.mean([r["retrieval_metrics"]["recall"] for r in results]),
            "avg_f1": np.mean([r["retrieval_metrics"]["f1"] for r in results]),
            "avg_mrr": np.mean([r["retrieval_metrics"]["mrr"] for r in results]),
            "avg_answer_similarity": np.mean([r["answer_similarity"] for r in results]),
            "avg_confidence": np.mean([r["confidence"] for r in results]),
            "verification_rate": np.mean([1.0 if r["verified"] else 0.0 for r in results]),
            "avg_citation_recall": np.mean([r["citation_metrics"].get("citation_recall", 0.0) for r in results]),
            "avg_citation_precision": np.mean([r["citation_metrics"].get("citation_precision", 0.0) for r in results]),
            "avg_retrieval_time": np.mean([r["retrieval_time"] for r in results]),
            "avg_generation_time": np.mean([r["generation_time"] for r in results]),
            "avg_total_time": np.mean([r["total_time"] for r in results]),
            "avg_sources_used": np.mean([r["num_sources"] for r in results])
        }
        
        return metrics
    
    def print_results(self, mode_results: Dict[str, List[Dict[str, Any]]]):
        print("\n" + "="*100)
        print("EVALUATION RESULTS")
        print("="*100 + "\n")
        
        for mode, results in mode_results.items():
            metrics = self.aggregate_metrics(results)
            mode_name = config.EXPERIMENT_MODES[mode]["name"]
            
            print(f"\n{'─'*100}")
            print(f"MODE: {mode_name}")
            print(f"{'─'*100}")
            
            print("\nRETRIEVAL METRICS:")
            print(f"  Precision:        {metrics['avg_precision']:.4f} ({metrics['avg_precision']:.1%})")
            print(f"  Recall:           {metrics['avg_recall']:.4f} ({metrics['avg_recall']:.1%})")
            print(f"  F1 Score:         {metrics['avg_f1']:.4f} ({metrics['avg_f1']:.1%})")
            print(f"  MRR:              {metrics['avg_mrr']:.4f}")
            
            print("\nGENERATION METRICS:")
            print(f"  Answer Similarity: {metrics['avg_answer_similarity']:.4f}")
            print(f"  Confidence:        {metrics['avg_confidence']:.4f}")
            print(f"  Verification Rate: {metrics['verification_rate']:.4f}")
            print(f"  Citation Recall:   {metrics['avg_citation_recall']:.4f}")
            print(f"  Citation Precision: {metrics['avg_citation_precision']:.4f}")
            
            print("\nPERFORMANCE METRICS:")
            print(f"  Retrieval Time:    {metrics['avg_retrieval_time']:.3f}s")
            print(f"  Generation Time:   {metrics['avg_generation_time']:.3f}s")
            print(f"  Total Time:        {metrics['avg_total_time']:.3f}s")
            print(f"  Avg Sources:       {metrics['avg_sources_used']:.2f}")
        
        # NEW: Precision comparison table
        self.print_precision_comparison(mode_results)
        
        print("\n" + "="*100)
        print("MODE COMPARISON (Best Values)")
        print("="*100 + "\n")
        
        comparison_metrics = ["avg_precision", "avg_recall", "avg_f1", "avg_answer_similarity", "avg_total_time"]
        
        for metric in comparison_metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            mode_values = []
            for mode, results in mode_results.items():
                metrics = self.aggregate_metrics(results)
                mode_values.append((mode, metrics[metric]))
            
            if "time" in metric:
                mode_values.sort(key=lambda x: x[1])
            else:
                mode_values.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (mode, value) in enumerate(mode_values, 1):
                mode_name = config.EXPERIMENT_MODES[mode]["name"]
                if "time" in metric:
                    print(f"  {rank}. {mode_name:40} {value:.3f}s")
                else:
                    print(f"  {rank}. {mode_name:40} {value:.4f} ({value:.1%})")
        
        print("\n" + "="*100 + "\n")
    
    def print_precision_comparison(self, mode_results: Dict[str, List[Dict[str, Any]]]):
        """NEW: Print detailed precision comparison"""
        print("\n" + "="*100)
        print("PRECISION IMPROVEMENT ANALYSIS")
        print("="*100 + "\n")
        
        # Get baseline (hybrid_full)
        if "hybrid_full" in mode_results:
            baseline_metrics = self.aggregate_metrics(mode_results["hybrid_full"])
            baseline_precision = baseline_metrics["avg_precision"]
            baseline_recall = baseline_metrics["avg_recall"]
            baseline_f1 = baseline_metrics["avg_f1"]
            
            print("BASELINE (hybrid_full):")
            print(f"  Precision: {baseline_precision:.1%}")
            print(f"  Recall:    {baseline_recall:.1%}")
            print(f"  F1 Score:  {baseline_f1:.1%}")
            print()
            
            # Compare other modes
            for mode, results in mode_results.items():
                if mode == "hybrid_full":
                    continue
                
                metrics = self.aggregate_metrics(results)
                mode_name = config.EXPERIMENT_MODES[mode]["name"]
                
                precision_delta = metrics["avg_precision"] - baseline_precision
                recall_delta = metrics["avg_recall"] - baseline_recall
                f1_delta = metrics["avg_f1"] - baseline_f1
                
                print(f"{mode_name}:")
                print(f"  Precision: {metrics['avg_precision']:.1%} ({precision_delta:+.1%})")
                print(f"  Recall:    {metrics['avg_recall']:.1%} ({recall_delta:+.1%})")
                print(f"  F1 Score:  {metrics['avg_f1']:.1%} ({f1_delta:+.1%})")
                
                # Highlight if precision improved significantly
                if precision_delta > 0.10:
                    print(f"  ✓ SIGNIFICANT PRECISION IMPROVEMENT!")
                elif precision_delta > 0.05:
                    print(f"  ✓ Moderate precision improvement")
                
                print()
    
    def save_results(self, mode_results: Dict[str, List[Dict[str, Any]]], output_path: str):
        aggregated = {}
        for mode, results in mode_results.items():
            aggregated[mode] = {
                "metrics": self.aggregate_metrics(results),
                "detailed_results": results
            }
        
        with open(output_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
    
    def generate_resume_metrics(self, mode_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        # NEW: Use precision_optimized if available, otherwise hybrid_full
        best_mode = "precision_optimized" if "precision_optimized" in mode_results else "hybrid_full"
        best_metrics = self.aggregate_metrics(mode_results[best_mode])
        
        resume_metrics = {
            "model_architecture": f"Hybrid RAG with Parent-Child Chunking ({best_mode} mode)",
            "retrieval_performance": {
                "precision": f"{best_metrics['avg_precision']:.2%}",
                "recall": f"{best_metrics['avg_recall']:.2%}",
                "f1_score": f"{best_metrics['avg_f1']:.2%}",
                "mrr": f"{best_metrics['avg_mrr']:.4f}"
            },
            "generation_quality": {
                "answer_accuracy": f"{best_metrics['avg_answer_similarity']:.2%}",
                "confidence_score": f"{best_metrics['avg_confidence']:.2%}",
                "verification_rate": f"{best_metrics['verification_rate']:.2%}",
                "citation_recall": f"{best_metrics['avg_citation_recall']:.2%}"
            },
            "system_performance": {
                "avg_query_time": f"{best_metrics['avg_total_time']:.2f}s",
                "retrieval_latency": f"{best_metrics['avg_retrieval_time']:.2f}s",
                "generation_latency": f"{best_metrics['avg_generation_time']:.2f}s"
            },
            "key_achievements": [
                f"Achieved {best_metrics['avg_f1']:.1%} F1 score in document retrieval",
                f"Maintained {best_metrics['verification_rate']:.1%} answer verification rate",
                f"Average query response time: {best_metrics['avg_total_time']:.2f}s",
                f"Precision: {best_metrics['avg_precision']:.1%}"
            ],
            "optimization_mode": best_mode
        }
        
        return resume_metrics


def main():
    print("\n" + "="*100)
    print("RAG MODEL EVALUATION SYSTEM")
    print("="*100 + "\n")
    
    pdf_path = input("Enter PDF path: ").strip()
    test_data_path = input("Enter test dataset JSON path: ").strip()
    
    if not Path(pdf_path).exists():
        print("✗ PDF file not found")
        return
    
    if not Path(test_data_path).exists():
        print("✗ Test dataset not found")
        return
    
    evaluator = RAGEvaluator()
    
    print("\n" + "="*100)
    print("STEP 1: DOCUMENT PROCESSING")
    print("="*100 + "\n")
    doc_id, chunks_metadata = evaluator.process_document(pdf_path)
    
    print("\n" + "="*100)
    print("STEP 2: LOADING TEST DATASET")
    print("="*100 + "\n")
    test_data = evaluator.load_test_dataset(test_data_path)
    print(f"✓ Loaded {len(test_data)} test queries")
    
    print("\n" + "="*100)
    print("STEP 3: RUNNING EVALUATIONS")
    print("="*100)
    mode_results = evaluator.evaluate_all_modes(doc_id, chunks_metadata, test_data)
    
    print("\n" + "="*100)
    print("STEP 4: GENERATING RESULTS")
    print("="*100 + "\n")
    evaluator.print_results(mode_results)
    
    output_path = "evaluation_results.json"
    evaluator.save_results(mode_results, output_path)
    
    resume_metrics = evaluator.generate_resume_metrics(mode_results)
    with open("resume_metrics.json", 'w') as f:
        json.dump(resume_metrics, f, indent=2)
    print(f"✓ Resume metrics saved to resume_metrics.json")
    
    print("\n" + "="*100)
    print("RESUME-READY METRICS")
    print("="*100)
    print(json.dumps(resume_metrics, indent=2))
    print("="*100 + "\n")


if __name__ == "__main__":
    main()