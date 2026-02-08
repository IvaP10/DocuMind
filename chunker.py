"""
Enhanced Contextual Chunker (2026)
Improved chunking with semantic boundary preservation
"""
from typing import List, Optional, Dict, Any
from uuid import uuid4
import tiktoken
import re
from models import Chunk, ChunkType, FormatType
from parser import LayoutElement
import config
import logging

logger = logging.getLogger(__name__)


class EnhancedContextualChunker:
    """
    Advanced chunking strategy with:
    - Semantic boundary preservation
    - Smart parent-child relationships
    - Configurable overlap strategies
    - Better handling of tables and special content
    """
    
    def __init__(self):
        """Initialize chunker with tokenizer and config"""
        logger.info("Initializing Enhanced Contextual Chunker")
        
        # Use latest OpenAI tokenizer (o200k_base for GPT-4o/GPT-4)
        self.encoding = tiktoken.get_encoding("o200k_base")
        
        # Load configuration
        self.parent_size = config.CHUNK_SIZE_PARENT
        self.child_size = config.CHUNK_SIZE_CHILD
        self.overlap = config.CHUNK_OVERLAP
        self.respect_sentences = config.RESPECT_SENTENCE_BOUNDARIES
        self.respect_paragraphs = config.RESPECT_PARAGRAPH_BOUNDARIES
        
        logger.info(f"Chunk sizes - Parent: {self.parent_size}, Child: {self.child_size}, Overlap: {self.overlap}")
    
    def create_chunks(self, document_id: str, elements: List[LayoutElement]) -> List[Chunk]:
        """
        Create parent and child chunks from layout elements
        
        Args:
            document_id: Document identifier
            elements: List of layout elements from parser
            
        Returns:
            List of all chunks (parents and children)
        """
        logger.info(f"Creating chunks for document {document_id}")
        logger.info(f"Processing {len(elements)} layout elements")
        
        all_chunks = []
        
        # Group elements into logical sections for parent chunks
        sections = self._group_into_sections(elements)
        logger.info(f"Created {len(sections)} sections")
        
        # Process each section
        for section_idx, section in enumerate(sections):
            # Create parent chunk for this section
            parent_chunk = self._create_parent_chunk(document_id, section, section_idx)
            all_chunks.append(parent_chunk)
            
            # Create child chunks from this section
            child_chunks = self._create_child_chunks(document_id, parent_chunk, section)
            all_chunks.extend(child_chunks)
        
        # Log statistics
        parent_count = sum(1 for c in all_chunks if c.is_parent)
        child_count = sum(1 for c in all_chunks if c.is_child)
        logger.info(f"Created {len(all_chunks)} total chunks: {parent_count} parents, {child_count} children")
        
        return all_chunks
    
    def _group_into_sections(self, elements: List[LayoutElement]) -> List[List[LayoutElement]]:
        """
        Group elements into sections for parent chunks
        Uses semantic boundaries when possible
        """
        sections = []
        current_section = []
        current_tokens = 0
        
        for element in elements:
            element_tokens = self._count_tokens(element.content)
            
            # Check if adding this element would exceed parent size
            would_exceed = current_tokens + element_tokens > self.parent_size
            
            # Decide whether to start new section
            should_split = False
            
            if would_exceed and current_section:
                should_split = True
            
            # Always start new section for tables (keep them atomic when possible)
            if element.element_type == ChunkType.TABLE and current_section:
                # Only if table fits in parent size
                if element_tokens <= self.parent_size:
                    should_split = True
            
            # Start new section for major headers
            if element.element_type == ChunkType.HEADER and current_section:
                if current_tokens > self.parent_size * 0.5:  # Only if section is substantial
                    should_split = True
            
            if should_split:
                # Save current section
                sections.append(current_section)
                
                # Start new section with overlap if configured
                if self.overlap > 0 and current_section:
                    overlap_elements = self._get_overlap_elements(current_section)
                    current_section = overlap_elements + [element]
                    current_tokens = sum(self._count_tokens(e.content) for e in current_section)
                else:
                    current_section = [element]
                    current_tokens = element_tokens
            else:
                current_section.append(element)
                current_tokens += element_tokens
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _get_overlap_elements(self, section: List[LayoutElement]) -> List[LayoutElement]:
        """Get elements from end of section for overlap"""
        if not section:
            return []
        
        overlap_elements = []
        overlap_tokens = 0
        
        # Take elements from end until we reach overlap size
        for element in reversed(section):
            element_tokens = self._count_tokens(element.content)
            if overlap_tokens + element_tokens > self.overlap:
                break
            overlap_elements.insert(0, element)
            overlap_tokens += element_tokens
        
        return overlap_elements
    
    def _create_parent_chunk(
        self, 
        document_id: str, 
        section: List[LayoutElement],
        section_idx: int
    ) -> Chunk:
        """Create a parent chunk from a section of elements"""
        
        # Combine element texts
        combined_text = self._combine_elements(section)
        
        # Get metadata from first element
        first_elem = section[0]
        
        # Collect element types
        element_types = [elem.element_type.value for elem in section]
        
        # Determine primary chunk type (most common type)
        type_counts = {}
        for elem_type in element_types:
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        primary_type = max(type_counts, key=type_counts.get)
        
        # Create parent chunk
        parent_chunk = Chunk(
            id=uuid4(),
            document_id=document_id,
            parent_id=None,
            text=combined_text,
            chunk_type=ChunkType(primary_type),
            format_type=first_elem.format_type,
            page_number=first_elem.page,
            bbox=first_elem.bbox,
            token_count=self._count_tokens(combined_text),
            metadata={
                "is_parent": True,
                "section_index": section_idx,
                "num_elements": len(section),
                "element_types": element_types,
                "pages": list(set(elem.page for elem in section)),
                "has_table": any(e.element_type == ChunkType.TABLE for e in section),
                "has_code": any(e.element_type == ChunkType.CODE for e in section)
            }
        )
        
        return parent_chunk
    
    def _create_child_chunks(
        self, 
        document_id: str, 
        parent: Chunk, 
        section: List[LayoutElement]
    ) -> List[Chunk]:
        """Create child chunks from section elements"""
        
        children = []
        
        for elem_idx, element in enumerate(section):
            element_tokens = self._count_tokens(element.content)
            
            # If element fits in child size, create single child chunk
            if element_tokens <= self.child_size:
                child = self._create_single_child(document_id, parent, element, elem_idx)
                children.append(child)
            else:
                # Element too large, need to split
                sub_chunks = self._split_large_element(document_id, parent, element, elem_idx)
                children.extend(sub_chunks)
        
        return children
    
    def _create_single_child(
        self,
        document_id: str,
        parent: Chunk,
        element: LayoutElement,
        elem_idx: int
    ) -> Chunk:
        """Create a single child chunk from an element"""
        
        return Chunk(
            id=uuid4(),
            document_id=document_id,
            parent_id=parent.id,
            text=element.content,
            chunk_type=element.element_type,
            format_type=element.format_type,
            page_number=element.page,
            bbox=element.bbox,
            token_count=self._count_tokens(element.content),
            metadata={
                "is_parent": False,
                "parent_id": str(parent.id),
                "element_index": elem_idx,
                **element.metadata
            }
        )
    
    def _split_large_element(
        self,
        document_id: str,
        parent: Chunk,
        element: LayoutElement,
        elem_idx: int
    ) -> List[Chunk]:
        """Split a large element into multiple child chunks"""
        
        chunks = []
        text = element.content
        
        # For tables, try to split by rows if markdown
        if element.element_type == ChunkType.TABLE and '|' in text:
            sub_chunks = self._split_table(text)
        else:
            # Split by sentences
            sub_chunks = self._split_by_sentences(text)
        
        # Create chunks from splits
        for sub_idx, chunk_text in enumerate(sub_chunks):
            if not chunk_text.strip():
                continue
            
            chunk = Chunk(
                id=uuid4(),
                document_id=document_id,
                parent_id=parent.id,
                text=chunk_text,
                chunk_type=element.element_type,
                format_type=element.format_type,
                page_number=element.page,
                bbox=element.bbox,
                token_count=self._count_tokens(chunk_text),
                metadata={
                    "is_parent": False,
                    "parent_id": str(parent.id),
                    "element_index": elem_idx,
                    "sub_index": sub_idx,
                    **element.metadata
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_table(self, table_text: str) -> List[str]:
        """Split markdown table into smaller chunks by rows"""
        lines = table_text.split('\n')
        
        # Find header and separator
        header_lines = []
        data_lines = []
        
        for i, line in enumerate(lines):
            if i == 0 or (i == 1 and '|---' in line or '|-' in line):
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        # Group data rows into chunks
        chunks = []
        current_chunk = header_lines.copy()
        current_tokens = sum(self._count_tokens(line) for line in current_chunk)
        
        for line in data_lines:
            line_tokens = self._count_tokens(line)
            
            if current_tokens + line_tokens > self.child_size and len(current_chunk) > len(header_lines):
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                # Start new chunk with header
                current_chunk = header_lines.copy() + [line]
                current_tokens = sum(self._count_tokens(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if len(current_chunk) > len(header_lines):
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [table_text]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks by sentences with overlap"""
        
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding sentence would exceed limit
            if current_tokens + sentence_tokens > self.child_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last 1-2 sentences)
                if self.overlap > 0:
                    overlap_count = min(2, len(current_chunk))
                    overlap_sentences = current_chunk[-overlap_count:] if overlap_count > 0 else []
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = self._count_tokens(' '.join(current_chunk))
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _combine_elements(self, elements: List[LayoutElement]) -> str:
        """Combine multiple elements into single text"""
        parts = []
        
        for element in elements:
            # Add element content
            parts.append(element.content)
        
        # Join with double newline to preserve structure
        return "\n\n".join(parts)
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with improved handling
        Preserves abbreviations and special cases
        """
        # Protect abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Inc|Ltd|Co|Corp|vs|etc|e\.g|i\.e|Jr|Sr|Prof|Rev)\.\s',
                     r'\1<ABBREV> ', text, flags=re.IGNORECASE)
        
        # Protect numbers with decimals
        text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
        
        # Mark sentence boundaries
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1<SENT>\2', text)
        
        # Restore abbreviation markers and decimals
        text = text.replace('<ABBREV>', '.')
        text = text.replace('<DECIMAL>', '.')
        
        # Split on sentence markers
        sentences = text.split('<SENT>')
        
        # Clean and merge very short sentences
        merged = []
        buffer = ""
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Merge very short sentences (< 6 words) with previous
            if len(sent.split()) < 6 and buffer:
                buffer += " " + sent
            else:
                if buffer:
                    merged.append(buffer)
                buffer = sent
        
        if buffer:
            merged.append(buffer)
        
        return [s.strip() for s in merged if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.debug(f"Token counting failed: {e}, using word-based estimate")
            # Fallback: estimate tokens as 1.3x word count
            return int(len(text.split()) * 1.3)


# Global chunker instance
chunker = EnhancedContextualChunker()