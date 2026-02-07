from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from collections import Counter
import re
import math
import config
import logging

logger = logging.getLogger(__name__)


class ImprovedEmbedder:
    
    def __init__(self):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        
        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freqs = {}
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.corpus_size = 0
    
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
    
    def build_bm25_index(self, texts: List[str]):
        """Build BM25 statistics from corpus"""
        logger.info(f"Building BM25 index for {len(texts)} documents")
        
        self.corpus_size = len(texts)
        self.doc_lengths = []
        self.doc_freqs = {}
        
        # Calculate document frequencies and lengths
        for text in texts:
            tokens = self.tokenize(text.lower())
            self.doc_lengths.append(len(tokens))
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        logger.info(f"BM25 index built: avg_doc_length={self.avg_doc_length:.1f}, vocab_size={len(self.doc_freqs)}")
    
    def create_sparse_vector_bm25(self, text: str, doc_index: int = None) -> Dict[int, float]:
        """Create sparse vector using BM25 scoring instead of simple TF"""
        tokens = self.tokenize(text.lower())
        token_counts = Counter(tokens)
        
        # Get document length for this text
        if doc_index is not None and doc_index < len(self.doc_lengths):
            doc_length = self.doc_lengths[doc_index]
        else:
            doc_length = len(tokens)
        
        sparse_vector = {}
        
        for token, tf in token_counts.items():
            # Calculate IDF
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue  # Skip terms not in corpus
            
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            
            # Calculate BM25 score
            norm_tf = tf / doc_length if doc_length > 0 else 0
            length_norm = doc_length / self.avg_doc_length if self.avg_doc_length > 0 else 1.0
            
            bm25_score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * length_norm))
            
            # Use deterministic hash for token ID
            token_id = abs(hash(token)) % 100000
            sparse_vector[token_id] = float(bm25_score)
        
        return sparse_vector
    
    def create_sparse_vector(self, text: str) -> Dict[int, float]:
        """Fallback to simple TF for query (when BM25 index not available)"""
        if self.corpus_size > 0:
            # Use BM25 for queries too
            return self.create_sparse_vector_bm25(text)
        
        # Fallback to TF-based scoring
        tokens = self.tokenize(text.lower())
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        sparse_vector = {}
        for token, count in token_counts.items():
            token_id = abs(hash(token)) % 100000
            tf = count / total_tokens if total_tokens > 0 else 0
            sparse_vector[token_id] = float(tf)
        
        return sparse_vector
    
    def create_sparse_vectors_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """Create sparse vectors for a batch of texts using BM25"""
        # Build BM25 index first
        self.build_bm25_index(texts)
        
        # Create BM25 vectors
        vectors = []
        for idx, text in enumerate(texts):
            vector = self.create_sparse_vector_bm25(text, idx)
            vectors.append(vector)
        
        return vectors
    
    def tokenize(self, text: str) -> List[str]:
        """Improved tokenization with better preprocessing"""
        # Remove punctuation but keep hyphens in words
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        tokens = text.split()
        
        # Filter: length > 2, not pure numbers, not stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'this', 'that', 'these', 'those'}
        
        filtered_tokens = []
        for t in tokens:
            if len(t) > 2 and not t.isdigit() and t not in stopwords:
                filtered_tokens.append(t)
        
        return filtered_tokens


embedder = ImprovedEmbedder()