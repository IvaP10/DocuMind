import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("key.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documind_v2")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHUNK_SIZE_PARENT = int(os.getenv("CHUNK_SIZE_PARENT", 600))
CHUNK_SIZE_CHILD = int(os.getenv("CHUNK_SIZE_CHILD", 200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))

RESPECT_SENTENCE_BOUNDARIES = os.getenv("RESPECT_SENTENCE_BOUNDARIES", "true").lower() == "true"
RESPECT_PARAGRAPH_BOUNDARIES = os.getenv("RESPECT_PARAGRAPH_BOUNDARIES", "true").lower() == "true"

TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", 50))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 3500))

DYNAMIC_THRESHOLD_PERCENTILE = float(os.getenv("DYNAMIC_THRESHOLD_PERCENTILE", 0.75))
DYNAMIC_THRESHOLD_MULTIPLIER = float(os.getenv("DYNAMIC_THRESHOLD_MULTIPLIER", 0.6))

RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", 0.15))
ADAPTIVE_THRESHOLD_ENABLED = os.getenv("ADAPTIVE_THRESHOLD", "true").lower() == "true"

RRF_K_PARAM = int(os.getenv("RRF_K_PARAM", 60))

DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.7))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.3))

ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "false").lower() == "true"
QUERY_EXPANSION_METHOD = os.getenv("QUERY_EXPANSION_METHOD", "llm")
QUERY_EXPANSION_TERMS = int(os.getenv("QUERY_EXPANSION_TERMS", 3))

ENABLE_QUERY_REWRITING = os.getenv("ENABLE_QUERY_REWRITING", "true").lower() == "true"

ENABLE_DEDUPLICATION = os.getenv("ENABLE_DEDUPLICATION", "true").lower() == "true"
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", 0.85))
DEDUP_METHOD = os.getenv("DEDUP_METHOD", "semantic")

ENABLE_MMR = os.getenv("ENABLE_MMR", "true").lower() == "true"
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", 0.7))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

EMBEDDING_MODEL_DIMENSIONS = {
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

if "EMBEDDING_DIMENSION" in os.environ:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
else:
    EMBEDDING_DIMENSION = EMBEDDING_MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 768)
    print(f"Auto-detected embedding dimension: {EMBEDDING_DIMENSION} for model: {EMBEDDING_MODEL}")

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_COHERE_RERANK = os.getenv("USE_COHERE_RERANK", "false").lower() == "true"
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1500))

ATOMIC_FACT_VERIFICATION = os.getenv("ATOMIC_FACT_VERIFICATION", "true").lower() == "true"
ABSTENTION_THRESHOLD = float(os.getenv("ABSTENTION_THRESHOLD", 0.25))

CONFIDENCE_CALIBRATION = {
    "use_calibration": os.getenv("USE_CONFIDENCE_CALIBRATION", "true").lower() == "true",
    "verification_weight": 0.35,
    "source_quality_weight": 0.25,
    "citation_quality_weight": 0.30,
    "retrieval_confidence_weight": 0.10,
}

EXPERIMENT_MODES = {
    "precision_optimized": {
        "name": "Precision Optimized (High Quality)",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": "child_with_parent_context",
        "apply_score_threshold": True,
        "dynamic_threshold": True,
        "use_query_rewriting": True,
        "use_mmr": True,
        "mmr_lambda": 0.75,
        "adaptive_threshold": True,
        "strict_thresholds": True,
        "min_rerank_score": 0.20,
        "dedup_method": "semantic",
        "dedup_threshold": 0.85,
        "context_strategy": "child_with_parent_context",
        "top_k_rerank": 4,
        "abstention_enabled": True,
    },
    
    "hybrid_optimized": {
        "name": "Hybrid Optimized (Balanced)",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": "child_with_parent_context",
        "apply_score_threshold": True,
        "dynamic_threshold": True,
        "use_query_rewriting": True,
        "use_mmr": True,
        "mmr_lambda": 0.7,
        "adaptive_threshold": True,
        "dedup_method": "semantic",
        "dedup_threshold": 0.85,
        "context_strategy": "child_with_parent_context",
        "top_k_rerank": 5,
        "abstention_enabled": False,
    },
    
    "hybrid_full": {
        "name": "Hybrid Full (Complete Pipeline)",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": "smart",
        "apply_score_threshold": True,
        "use_query_rewriting": True,
        "use_mmr": True,
        "mmr_lambda": 0.6,
        "adaptive_threshold": True,
        "dedup_method": "semantic",
        "context_strategy": "child_with_parent_context",
        "top_k_rerank": 8,
    },
    
    "dense_only": {
        "name": "Dense Only + Rerank + MMR",
        "use_dense": True,
        "use_sparse": False,
        "use_reranker": True,
        "use_parent_child": "smart",
        "apply_score_threshold": True,
        "use_query_rewriting": True,
        "use_mmr": True,
        "mmr_lambda": 0.65,
        "adaptive_threshold": True,
        "dedup_method": "semantic",
        "context_strategy": "child_with_parent_context",
    },
    
    "no_rerank": {
        "name": "No Reranking",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": False,
        "use_parent_child": "smart",
        "apply_score_threshold": True,
        "use_query_rewriting": False,
        "use_mmr": False,
        "adaptive_threshold": False,
        "dedup_method": "semantic",
        "context_strategy": "child_with_parent_context",
    },
    
    "no_parent": {
        "name": "No Parent Context",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": "child_only",
        "apply_score_threshold": True,
        "use_query_rewriting": True,
        "use_mmr": True,
        "mmr_lambda": 0.6,
        "adaptive_threshold": True,
        "dedup_method": "semantic",
        "context_strategy": "child_only",
    },
    
    "recall_max": {
        "name": "Maximum Recall (Comprehensive Coverage)",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": "parent_only",
        "apply_score_threshold": False,
        "use_query_rewriting": True,
        "use_mmr": False,
        "adaptive_threshold": False,
        "dedup_method": "jaccard",
        "dedup_threshold": 0.95,
        "context_strategy": "parent_only",
    },
    
    "latency_optimized": {
        "name": "Low Latency (Fast Responses)",
        "use_dense": True,
        "use_sparse": False,
        "use_reranker": False,
        "use_parent_child": "child_only",
        "apply_score_threshold": True,
        "use_query_rewriting": False,
        "use_mmr": False,
        "adaptive_threshold": False,
        "dedup_method": "jaccard",
        "context_strategy": "child_only",
    }
}

DEFAULT_MODE = os.getenv("DEFAULT_MODE", "precision_optimized")

ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(exist_ok=True)

ENABLE_BATCH_PROCESSING = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"
LOG_RETRIEVAL_SCORES = os.getenv("LOG_RETRIEVAL_SCORES", "true").lower() == "true"

ENABLE_HYDE = os.getenv("ENABLE_HYDE", "false").lower() == "true"

PARENT_CONTEXT_WINDOW = int(os.getenv("PARENT_CONTEXT_WINDOW", 100))

ENABLE_METADATA_FILTERING = os.getenv("ENABLE_METADATA_FILTERING", "true").lower() == "true"

ENABLE_LATE_INTERACTION = os.getenv("ENABLE_LATE_INTERACTION", "false").lower() == "true"

def validate_config():
    assert CHUNK_SIZE_PARENT > CHUNK_SIZE_CHILD, "Parent chunks must be larger than child chunks"
    assert CHUNK_OVERLAP < CHUNK_SIZE_CHILD, "Overlap should be less than child chunk size"
    assert 0 <= DENSE_WEIGHT <= 1, "Dense weight must be between 0 and 1"
    assert 0 <= SPARSE_WEIGHT <= 1, "Sparse weight must be between 0 and 1"
    assert abs(DENSE_WEIGHT + SPARSE_WEIGHT - 1.0) < 0.01, "Weights should sum to 1"
    assert 0 <= MMR_LAMBDA <= 1, "MMR lambda must be between 0 and 1"
    assert TOP_K_RERANK <= TOP_K_INITIAL, "Rerank K should not exceed initial K"
    assert EMBEDDING_DIMENSION > 0, "Embedding dimension must be positive"
    
    if EMBEDDING_MODEL in EMBEDDING_MODEL_DIMENSIONS:
        expected_dim = EMBEDDING_MODEL_DIMENSIONS[EMBEDDING_MODEL]
        if EMBEDDING_DIMENSION != expected_dim:
            print(f"⚠️  WARNING: EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION}) doesn't match "
                  f"expected dimension ({expected_dim}) for model {EMBEDDING_MODEL}")
            print(f"   This may cause errors. Consider using EMBEDDING_DIMENSION={expected_dim}")

validate_config()
