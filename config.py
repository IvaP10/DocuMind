import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import warnings

warnings.filterwarnings('ignore')
for _lg in ['docling','docling_core','docling.document_converter','docling.datamodel',
            'docling.models','docling.pipeline','docling.utils','datasets',
            'sentence_transformers','httpx']:
    logging.getLogger(_lg).setLevel(logging.ERROR)

load_dotenv("key.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documind_v2")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHUNK_SIZE_PARENT = int(os.getenv("CHUNK_SIZE_PARENT", 600))
CHUNK_SIZE_CHILD = int(os.getenv("CHUNK_SIZE_CHILD", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 75))

RESPECT_SENTENCE_BOUNDARIES = os.getenv("RESPECT_SENTENCE_BOUNDARIES", "true").lower() == "true"
RESPECT_PARAGRAPH_BOUNDARIES = os.getenv("RESPECT_PARAGRAPH_BOUNDARIES", "true").lower() == "true"

TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", 20))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 8))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 2500))

DYNAMIC_THRESHOLD_PERCENTILE = float(os.getenv("DYNAMIC_THRESHOLD_PERCENTILE", 0.70))
DYNAMIC_THRESHOLD_MULTIPLIER = float(os.getenv("DYNAMIC_THRESHOLD_MULTIPLIER", 0.7))

RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", 0.08))

RRF_K_PARAM = int(os.getenv("RRF_K_PARAM", 100))

DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.70))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.30))

ENABLE_DEDUPLICATION = True
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", 0.90))

ENABLE_MMR = True
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", 0.8))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

if "EMBEDDING_DIMENSION" in os.environ:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
else:
    EMBEDDING_DIMENSION = EMBEDDING_MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 3072)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 2000))

ABSTENTION_THRESHOLD = float(os.getenv("ABSTENTION_THRESHOLD", 0.20))

CONFIDENCE_CALIBRATION = {
    "use_calibration": True,
    "verification_weight": 0.30,
    "source_quality_weight": 0.25,
    "citation_quality_weight": 0.25,
    "retrieval_confidence_weight": 0.20,
}

MODE = {
    "name": "Hybrid Quality-First",
    "use_dense": True,
    "use_sparse": True,
    "use_reranker": True,
    "apply_score_threshold": False,
    "dynamic_threshold": False,
    "use_query_rewriting": False,
    "use_mmr": True,
    "mmr_lambda": 0.8,
    "adaptive_threshold": False,
    "min_rerank_score": 0.08,
    "dedup_method": "fast",
    "dedup_threshold": 0.90,
    "context_strategy": "child_with_parent_context",
    "top_k_initial": 20,
    "top_k_rerank": 8,
    "abstention_enabled": False,
    "atomic_fact_verification": True,
    "hallucination_penalty": 0.98,
    "numeric_verification": True,
    "numeric_exact_match": True,
    "extract_numeric_context": True,
    "citation_quality_weight": 0.25,
    "require_citations_for_claims": True,
    "citation_recall_threshold": 0.65,
    "rerank_by_query_similarity": True,
    "parent_context_window": 100,
    "dense_weight": 0.70,
    "sparse_weight": 0.30,
    "use_llm_listwise_rerank": False,
    "query_adaptive_weights": True,
    "numeric_boost_factor": 2.0,
}

ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
LOG_RETRIEVAL_SCORES = os.getenv("LOG_RETRIEVAL_SCORES", "true").lower() == "true"

PARENT_CONTEXT_WINDOW = int(os.getenv("PARENT_CONTEXT_WINDOW", 100))

assert CHUNK_SIZE_PARENT > CHUNK_SIZE_CHILD
assert CHUNK_OVERLAP < CHUNK_SIZE_CHILD
assert abs(DENSE_WEIGHT + SPARSE_WEIGHT - 1.0) < 0.01
assert TOP_K_RERANK <= TOP_K_INITIAL
assert EMBEDDING_DIMENSION > 0
