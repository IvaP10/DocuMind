from typing import List, Dict, Optional, Set
import numpy as np
from collections import Counter
import re
import math
import logging
from pathlib import Path
import pickle
import hashlib
import json
from functools import lru_cache

try:
    import config
except ImportError:
    class Config:
        OPENAI_API_KEY = ""
        OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
        EMBEDDING_BATCH_SIZE = 2048
        OPENAI_EMBEDDING_MAX_TOKENS_PER_REQUEST = 300000
        CACHE_DIR = Path("./cache")
        ENABLE_EMBEDDING_CACHE = True
    config = Config()

logger = logging.getLogger(__name__)

class EnhancedEmbedder:
    
    def __init__(self):
        self._init_openai_embedder()
        
        self.k1 = 1.5
        self.b = 0.75
        
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.corpus_size: int = 0
        
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.next_token_id: int = 0
        
        self._compile_tokenization_assets()
        
        self.cache_dir = config.CACHE_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = config.ENABLE_EMBEDDING_CACHE
        
        import tiktoken
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _compile_tokenization_assets(self):
        self.re_percent = re.compile(r'(\d+)%')
        self.re_dollar = re.compile(r'\$(\d+)')
        self.re_units = [
            (re.compile(r'(\d+)k\b', re.IGNORECASE), r'\1 thousand'),
            (re.compile(r'(\d+)m\b', re.IGNORECASE), r'\1 million'),
            (re.compile(r'(\d+)b\b', re.IGNORECASE), r'\1 billion'),
        ]
        self.re_clean = re.compile(r'[^\w\s\'-]')
        self.re_whitespace = re.compile(r'\s+')
        
        self.stopwords: Set[str] = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them'
        }
    
    def _init_openai_embedder(self):
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.embedding_type = "openai"
            
            dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self.dimension = dimension_map.get(config.OPENAI_EMBEDDING_MODEL, 3072)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise

    def _get_cache_key(self, texts: List[str]) -> str:
        content = json.dumps(texts, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_cached_embeddings(self, cache_key: str) -> Optional[np.ndarray]:
        if not self.enable_cache:
            return None
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cached_embeddings(self, cache_key: str, embeddings: np.ndarray):
        if not self.enable_cache:
            return
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, embeddings)

    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.array([])
        
        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE
        
        cache_key = self._get_cache_key(texts)
        cached = self._load_cached_embeddings(cache_key)
        if cached is not None:
            return cached
        
        try:
            result = self._embed_with_openai(texts, batch_size)
            self._save_cached_embeddings(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def _embed_with_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        max_tokens_per_input = 8191
        max_tokens_per_request = 250000
        max_items_per_request = 2048
        
        if batch_size > max_items_per_request:
            batch_size = max_items_per_request
            
        truncated_texts = []
        for text in texts:
            tokens = self._encoding.encode(text)
            if len(tokens) > max_tokens_per_input:
                tokens = tokens[:max_tokens_per_input]
                text = self._encoding.decode(tokens)
            truncated_texts.append(text)
            
        token_counts = [len(self._encoding.encode(t)) for t in truncated_texts]
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for i, (text, tc) in enumerate(zip(truncated_texts, token_counts)):

            if (current_tokens + tc > max_tokens_per_request) or (len(current_batch) >= batch_size):
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(text)
            current_tokens += tc
            
        if current_batch:
            batches.append(current_batch)
            
        all_embeddings = []
        logger.info(f"Embedding {len(texts)} texts in {len(batches)} batches.")
        
        for i, batch in enumerate(batches):
            batch_tokens = sum(len(self._encoding.encode(t)) for t in batch)
            logger.info(f"Batch {i+1}/{len(batches)}: {len(batch)} items, {batch_tokens} tokens")
            
            try:
                response = self.openai_client.embeddings.create(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in batch {i+1}: {e}")
                raise
                
        embeddings = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return embeddings / norms

    @lru_cache(maxsize=1024)
    def embed_single(self, text: str) -> np.ndarray:
        return self.embed_texts([text], show_progress=False)[0]
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_single(query)

    def build_bm25_index(self, texts: List[str]):
        self.doc_lengths = []
        self.doc_freqs = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_token_id = 0
        self.corpus_size = len(texts)
        
        for text in texts:
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.next_token_id
                    self.id_to_token[self.next_token_id] = token
                    self.next_token_id += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0

    def create_sparse_vector(self, text: str, mode: str = "query") -> Dict[int, float]:
        if self.corpus_size == 0:
            return self._create_tfidf_vector(text)

        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        sparse_vector = {}
        
        if mode == "document":
            doc_len = len(tokens)
            for token, tf in token_counts.items():
                if token in self.token_to_id:
                    token_id = self.token_to_id[token]
                    
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                    tf_norm = numerator / denominator
                    
                    sparse_vector[token_id] = float(tf_norm)
                    
        else:
            for token in tokens:
                if token in self.token_to_id:
                    token_id = self.token_to_id[token]
                    
                    df = self.doc_freqs.get(token, 0)
                    idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
                    
                    sparse_vector[token_id] = sparse_vector.get(token_id, 0.0) + float(idf)

        return sparse_vector
    
    def create_sparse_vectors_batch(self, texts: List[str], mode: str = "document") -> List[Dict[int, float]]:
        return [self.create_sparse_vector(text, mode=mode) for text in texts]

    def _create_tfidf_vector(self, text: str) -> Dict[int, float]:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        total = len(tokens)
        vec = {}
        for t, c in counts.items():
            t_id = hash(t) % 100000 
            vec[t_id] = (c / total) if total > 0 else 0.0
        return vec

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        
        text = self.re_percent.sub(r'\1 percent', text)
        text = self.re_dollar.sub(r'dollar \1', text)
        for pattern, repl in self.re_units:
            text = pattern.sub(repl, text)
        
        text = self.re_clean.sub(' ', text)
        text = self.re_whitespace.sub(' ', text)
        
        raw_tokens = text.split()
        filtered = []
        
        for t in raw_tokens:
            if t in self.stopwords:
                continue
            if len(t) > 2 or (len(t) == 2 and any(c.isdigit() for c in t)):
                filtered.append(t)
                
        return filtered

    def save_index(self, path: Path):
        path = Path(path)
        data = {
            'vocab': self.token_to_id,
            'doc_freqs': self.doc_freqs,
            'stats': {
                'avg_len': self.avg_doc_length,
                'corpus_size': self.corpus_size
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path: Path):
        path = Path(path)
        if not path.exists():
            return
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.token_to_id = data['vocab']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.doc_freqs = data['doc_freqs']
        self.avg_doc_length = data['stats']['avg_len']
        self.corpus_size = data['stats']['corpus_size']
        self.next_token_id = len(self.token_to_id)

embedder = EnhancedEmbedder()
