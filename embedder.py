from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from collections import Counter
import re
import config
import logging

logger = logging.getLogger(__name__)


class Embedder:
    
    def __init__(self):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Single embedding error: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_single(query)
    
    def create_sparse_vector(self, text: str) -> Dict[int, float]:
        tokens = self.tokenize(text.lower())
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        sparse_vector = {}
        for token, count in token_counts.items():
            token_id = hash(token) % 100000
            tf = count / total_tokens if total_tokens > 0 else 0
            sparse_vector[token_id] = float(tf)
        
        return sparse_vector
    
    def create_sparse_vectors_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        return [self.create_sparse_vector(text) for text in texts]
    
    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]


embedder = Embedder()