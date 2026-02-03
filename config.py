import os
from dotenv import load_dotenv

load_dotenv("key.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documind")

CHUNK_SIZE_PARENT = int(os.getenv("CHUNK_SIZE_PARENT", 1500))
CHUNK_SIZE_CHILD = int(os.getenv("CHUNK_SIZE_CHILD", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", 50))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 5000))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

EXPERIMENT_MODES = {
    "hybrid_full": {
        "name": "Hybrid + Rerank + Parent-Child",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": True
    },
    "dense_only": {
        "name": "Dense Only",
        "use_dense": True,
        "use_sparse": False,
        "use_reranker": True,
        "use_parent_child": True
    },
    "no_rerank": {
        "name": "No Reranking",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": False,
        "use_parent_child": True
    },
    "no_parent": {
        "name": "No Parent-Child",
        "use_dense": True,
        "use_sparse": True,
        "use_reranker": True,
        "use_parent_child": False
    }
}