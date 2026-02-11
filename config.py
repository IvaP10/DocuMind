import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import warnings

warnings.filterwarnings('ignore')

logging.getLogger('docling').setLevel(logging.ERROR)
logging.getLogger('docling_core').setLevel(logging.ERROR)
logging.getLogger('docling.document_converter').setLevel(logging.ERROR)
logging.getLogger('docling.datamodel').setLevel(logging.ERROR)
logging.getLogger('docling.models').setLevel(logging.ERROR)
logging.getLogger('docling.pipeline').setLevel(logging.ERROR)
logging.getLogger('docling.utils').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)

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

TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", 30))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 10))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 3000))

DYNAMIC_THRESHOLD_PERCENTILE = float(os.getenv("DYNAMIC_THRESHOLD_PERCENTILE", 0.70))
DYNAMIC_THRESHOLD_MULTIPLIER = float(os.getenv("DYNAMIC_THRESHOLD_MULTIPLIER", 0.7))

RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", 0.10))
ADAPTIVE_THRESHOLD_ENABLED = os.getenv("ADAPTIVE_THRESHOLD", "true").lower() == "true"

RRF_K_PARAM = int(os.getenv("RRF_K_PARAM", 100))

DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.65))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.35))

ENABLE_QUERY_REWRITING = os.getenv("ENABLE_QUERY_REWRITING", "true").lower() == "true"

ENABLE_DEDUPLICATION = os.getenv("ENABLE_DEDUPLICATION", "true").lower() == "true"
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", 0.85))
DEDUP_METHOD = os.getenv("DEDUP_METHOD", "semantic")

ENABLE_MMR = os.getenv("ENABLE_MMR", "true").lower() == "true"
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", 0.7))

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

USE_OPENAI_EMBEDDINGS = True
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

LLM_PROVIDER = "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 2000))

ATOMIC_FACT_VERIFICATION = os.getenv("ATOMIC_FACT_VERIFICATION", "true").lower() == "true"
ABSTENTION_THRESHOLD = float(os.getenv("ABSTENTION_THRESHOLD", 0.20))

CONFIDENCE_CALIBRATION = {
    "use_calibration": os.getenv("USE_CONFIDENCE_CALIBRATION", "true").lower() == "true",
    "verification_weight": 0.25,
    "source_quality_weight": 0.25,
    "citation_quality_weight": 0.30,
    "retrieval_confidence_weight": 0.20,
}

EXPERIMENT_MODES = {
    "hybrid_quality": {
        "name": "Hybrid Quality-First",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": "child_with_parent_context",
        "apply_score_threshold": False,
        "dynamic_threshold": False,
        "use_query_rewriting": True,
        "use_mmr": True,
        "mmr_lambda": 0.6,
        "adaptive_threshold": False,
        "min_rerank_score": 0.08,
        "dedup_method": "semantic",
        "dedup_threshold": 0.82,
        "context_strategy": "child_with_parent_context",
        "top_k_initial": 25,
        "top_k_rerank": 12,
        "abstention_enabled": False,
        "atomic_fact_verification": True,
        "hallucination_penalty": 0.98,
        "numeric_verification": True,
        "numeric_exact_match": True,
        "extract_numeric_context": True,
        "citation_quality_weight": 0.30,
        "require_citations_for_claims": True,
        "citation_recall_threshold": 0.65,
        "rerank_by_query_similarity": True,
        "parent_context_window": 150,
        "dense_weight": 0.65,
        "sparse_weight": 0.35,
        "use_llm_listwise_rerank": True,
        "query_adaptive_weights": True,
        "numeric_boost_factor": 2.0,
    },
    
    "fast_response": {
        "name": "Fast Response",
        "use_dense": True,
        "use_sparse": False,
        "use_reranker": False,
        "use_parent_child": "child_only",
        "apply_score_threshold": False,
        "use_query_rewriting": False,
        "use_mmr": False,
        "adaptive_threshold": False,
        "dedup_method": "jaccard",
        "dedup_threshold": 0.88,
        "context_strategy": "child_only",
        "top_k_initial": 3,
        "top_k_rerank": 3,
        "abstention_enabled": False,
        "atomic_fact_verification": False,
        "numeric_verification": False,
        "min_rerank_score": 0.05,
        "use_llm_listwise_rerank": False,
        "query_adaptive_weights": False,
    }
}

DEFAULT_MODE = os.getenv("DEFAULT_MODE", "hybrid_quality")

ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(exist_ok=True)

ENABLE_BATCH_PROCESSING = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"
LOG_RETRIEVAL_SCORES = os.getenv("LOG_RETRIEVAL_SCORES", "true").lower() == "true"

PARENT_CONTEXT_WINDOW = int(os.getenv("PARENT_CONTEXT_WINDOW", 100))

ENABLE_METADATA_FILTERING = os.getenv("ENABLE_METADATA_FILTERING", "true").lower() == "true"

def validate_config():
    assert CHUNK_SIZE_PARENT > CHUNK_SIZE_CHILD, "Parent chunks must be larger than child chunks"
    assert CHUNK_OVERLAP < CHUNK_SIZE_CHILD, "Overlap should be less than child chunk size"
    assert 0 <= DENSE_WEIGHT <= 1, "Dense weight must be between 0 and 1"
    assert 0 <= SPARSE_WEIGHT <= 1, "Sparse weight must be between 0 and 1"
    assert abs(DENSE_WEIGHT + SPARSE_WEIGHT - 1.0) < 0.01, "Weights should sum to 1"
    assert 0 <= MMR_LAMBDA <= 1, "MMR lambda must be between 0 and 1"
    assert TOP_K_RERANK <= TOP_K_INITIAL, "Rerank K should not exceed initial K"
    assert EMBEDDING_DIMENSION > 0, "Embedding dimension must be positive"

validate_config()
