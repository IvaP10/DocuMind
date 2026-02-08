"""
Data Models for RAG System (2026)
Enhanced with better typing and validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Document processing status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ChunkType(str, Enum):
    """Type of content in chunk"""
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    CODE = "code"
    EQUATION = "equation"


class FormatType(str, Enum):
    """Format of chunk content"""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"


class BoundingBox(BaseModel):
    """Bounding box for layout element"""
    x: float
    y: float
    width: float
    height: float
    
    def dict(self, *args, **kwargs):
        """Override to ensure compatibility"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }


class Chunk(BaseModel):
    """Individual chunk of document content"""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    parent_id: Optional[UUID] = None
    text: str
    chunk_type: ChunkType = ChunkType.TEXT
    format_type: FormatType = FormatType.PLAIN
    page_number: int
    bbox: Optional[BoundingBox] = None
    token_count: int
    embedding_id: Optional[str] = None  # For vector DB reference
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v
    
    @property
    def is_parent(self) -> bool:
        """Check if this is a parent chunk"""
        return self.parent_id is None
    
    @property
    def is_child(self) -> bool:
        """Check if this is a child chunk"""
        return self.parent_id is not None


class Document(BaseModel):
    """Document metadata"""
    id: UUID = Field(default_factory=uuid4)
    filename: str
    file_path: str
    status: ProcessingStatus = ProcessingStatus.UPLOADING
    total_pages: int = 0
    total_chunks: int = 0
    total_parent_chunks: int = 0
    total_child_chunks: int = 0
    upload_time: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    file_size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Citation for answer source"""
    chunk_id: UUID
    text: str
    page: int
    bbox: Optional[BoundingBox] = None
    chunk_type: ChunkType = ChunkType.TEXT
    score: Optional[float] = None  # Relevance score
    rank: Optional[int] = None     # Ranking position


class SearchResult(BaseModel):
    """Individual search result from vector DB"""
    chunk_id: str
    parent_id: Optional[str] = None
    text: str
    chunk_type: str
    page_number: int
    bbox: Optional[Dict] = None
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Additional fields for analysis
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    mmr_score: Optional[float] = None


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval performance"""
    total_candidates: int
    after_filtering: int
    after_reranking: int
    after_deduplication: int
    final_chunks: int
    retrieval_time_ms: float
    mode: str
    
    # Score statistics
    avg_dense_score: Optional[float] = None
    avg_rerank_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None


class ContextData(BaseModel):
    """Retrieved context for generation"""
    context: str
    sources: List[Dict[str, Any]]
    total_tokens: int
    retrieved_chunks: List[Dict[str, Any]]
    rerank_scores: List[float] = Field(default_factory=list)
    mode: str
    metrics: Optional[RetrievalMetrics] = None


class CitationMetrics(BaseModel):
    """Metrics for citation quality"""
    citation_recall: float
    citation_precision: float
    citation_f1: float
    total_claims: int
    supported_claims: int
    unsupported_claims: List[str] = Field(default_factory=list)
    hallucinated_sentences: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Request to query a document"""
    document_id: UUID
    query: str
    mode: Optional[str] = "hybrid_optimized"
    top_k: Optional[int] = None
    include_metadata: bool = False
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()


class QueryResponse(BaseModel):
    """Response to query request"""
    answer: str
    citations: List[Citation]
    confidence: float
    processing_time: float
    verified: bool
    retrieval_metrics: Optional[RetrievalMetrics] = None
    citation_metrics: Optional[CitationMetrics] = None
    mode: str
    context_used: Optional[str] = None  # For debugging
    
    @field_validator('confidence')
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: UUID
    status: ProcessingStatus
    message: str
    total_chunks: Optional[int] = None


class DocumentStatusResponse(BaseModel):
    """Status of document processing"""
    document_id: UUID
    status: ProcessingStatus
    progress: int = Field(ge=0, le=100)  # 0-100
    current_phase: str
    error_message: Optional[str] = None
    time_elapsed: Optional[float] = None


class EvaluationResult(BaseModel):
    """Results from RAG evaluation"""
    query: str
    ground_truth: Optional[str] = None
    generated_answer: str
    
    # Retrieval metrics
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    retrieval_f1: Optional[float] = None
    mrr: Optional[float] = None  # Mean Reciprocal Rank
    ndcg: Optional[float] = None  # Normalized Discounted Cumulative Gain
    
    # Generation metrics
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    
    # Combined
    ragas_score: Optional[float] = None
    
    processing_time: float
    mode: str


class BatchEvaluationResult(BaseModel):
    """Aggregated results from batch evaluation"""
    total_queries: int
    successful: int
    failed: int
    
    # Aggregated metrics
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_retrieval_f1: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_ragas_score: float
    
    avg_processing_time: float
    mode: str
    
    individual_results: List[EvaluationResult]