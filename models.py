from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"


class FormatType(str, Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    parent_id: Optional[UUID] = None
    text: str
    chunk_type: ChunkType
    format_type: FormatType
    page_number: int
    bbox: Optional[BoundingBox] = None
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    file_path: str
    status: ProcessingStatus
    total_pages: int = 0
    total_chunks: int = 0
    upload_time: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


class Citation(BaseModel):
    chunk_id: UUID
    text: str
    page: int
    bbox: Optional[BoundingBox] = None
    chunk_type: ChunkType


class QueryRequest(BaseModel):
    document_id: UUID
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    processing_time: float
    verified: bool


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    status: ProcessingStatus
    message: str


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    status: ProcessingStatus
    progress: int
    current_phase: str
    error_message: Optional[str] = None