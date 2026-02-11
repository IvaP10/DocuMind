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
    
    def __init__(self):
        logger.info("Initializing Enhanced Contextual Chunker")
        
        self.encoding = tiktoken.get_encoding("o200k_base")
        
        self.parent_size = config.CHUNK_SIZE_PARENT
        self.child_size = config.CHUNK_SIZE_CHILD
        self.overlap = config.CHUNK_OVERLAP
        self.respect_sentences = config.RESPECT_SENTENCE_BOUNDARIES
        self.respect_paragraphs = config.RESPECT_PARAGRAPH_BOUNDARIES
        
        self._compile_numeric_patterns()
        
        logger.info(f"Chunk sizes - Parent: {self.parent_size}, Child: {self.child_size}, Overlap: {self.overlap}")
    
    def _compile_numeric_patterns(self):
        self.numeric_patterns = [
            (re.compile(r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb](?:illion)?)?'), 'number'),
            (re.compile(r'-?\d+(?:\.\d+)?%'), 'percentage'),
            (re.compile(r'\b(?:19|20)\d{2}\b'), 'year'),
            (re.compile(r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b', re.IGNORECASE), 'fiscal_period'),
            (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), 'date'),
        ]
    
    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        numbers = []
        seen = set()
        
        for pattern, num_type in self.numeric_patterns:
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                if matched_text not in seen:
                    seen.add(matched_text)
                    numbers.append({
                        "value": matched_text,
                        "type": num_type,
                        "position": match.start()
                    })
        
        numbers.sort(key=lambda x: x["position"])
        return numbers
    
    def create_chunks(self, document_id: str, elements: List[LayoutElement]) -> List[Chunk]:
        logger.info(f"Creating chunks for document {document_id}")
        logger.info(f"Processing {len(elements)} layout elements")
        
        all_chunks = []
        
        sections = self._group_into_sections(elements)
        logger.info(f"Created {len(sections)} sections")
        
        for section_idx, section in enumerate(sections):
            parent_chunk = self._create_parent_chunk(document_id, section, section_idx)
            all_chunks.append(parent_chunk)
            
            child_chunks = self._create_child_chunks(document_id, parent_chunk, section)
            all_chunks.extend(child_chunks)
        
        parent_count = sum(1 for c in all_chunks if c.is_parent)
        child_count = sum(1 for c in all_chunks if c.is_child)
        logger.info(f"Created {len(all_chunks)} total chunks: {parent_count} parents, {child_count} children")
        
        return all_chunks
    
    def _group_into_sections(self, elements: List[LayoutElement]) -> List[List[LayoutElement]]:
        sections = []
        current_section = []
        current_tokens = 0
        
        for element in elements:
            element_tokens = self._count_tokens(element.content)
            
            would_exceed = current_tokens + element_tokens > self.parent_size
            
            should_split = False
            
            if would_exceed and current_section:
                should_split = True
            
            if element.element_type == ChunkType.TABLE and current_section:
                if element_tokens <= self.parent_size:
                    should_split = True
            
            if element.element_type == ChunkType.HEADER and current_section:
                if current_tokens > self.parent_size * 0.5:
                    should_split = True
            
            if should_split:
                sections.append(current_section)
                
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
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _get_overlap_elements(self, section: List[LayoutElement]) -> List[LayoutElement]:
        if not section:
            return []
        
        overlap_elements = []
        overlap_tokens = 0
        
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
        
        combined_text = self._combine_elements(section)
        
        first_elem = section[0]
        
        element_types = [elem.element_type.value for elem in section]
        
        type_counts = {}
        for elem_type in element_types:
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        primary_type = max(type_counts, key=type_counts.get)
        
        numbers = self._extract_numbers(combined_text)
        
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
                "has_code": any(e.element_type == ChunkType.CODE for e in section),
                "numbers": numbers,
            }
        )
        
        return parent_chunk
    
    def _create_child_chunks(
        self, 
        document_id: str, 
        parent: Chunk, 
        section: List[LayoutElement]
    ) -> List[Chunk]:
        
        children = []
        
        for elem_idx, element in enumerate(section):
            element_tokens = self._count_tokens(element.content)
            
            if element_tokens <= self.child_size:
                child = self._create_single_child(document_id, parent, element, elem_idx)
                children.append(child)
            else:
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
        
        numbers = self._extract_numbers(element.content)
        
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
                "numbers": numbers,
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
        
        chunks = []
        text = element.content
        
        if element.element_type == ChunkType.TABLE and '|' in text:
            sub_chunks = self._split_table(text)
        else:
            sub_chunks = self._split_by_sentences(text)
        
        for sub_idx, chunk_text in enumerate(sub_chunks):
            if not chunk_text.strip():
                continue
            
            numbers = self._extract_numbers(chunk_text)
            
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
                    "numbers": numbers,
                    **element.metadata
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_table(self, table_text: str) -> List[str]:
        lines = table_text.split('\n')
        
        header_lines = []
        data_lines = []
        
        for i, line in enumerate(lines):
            if i == 0 or (i == 1 and '|---' in line or '|-' in line):
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        chunks = []
        current_chunk = header_lines.copy()
        current_tokens = sum(self._count_tokens(line) for line in current_chunk)
        
        for line in data_lines:
            line_tokens = self._count_tokens(line)
            
            if current_tokens + line_tokens > self.child_size and len(current_chunk) > len(header_lines):
                chunks.append('\n'.join(current_chunk))
                current_chunk = header_lines.copy() + [line]
                current_tokens = sum(self._count_tokens(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        if len(current_chunk) > len(header_lines):
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [table_text]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.child_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
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
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _combine_elements(self, elements: List[LayoutElement]) -> str:
        parts = []
        
        for element in elements:
            parts.append(element.content)
        
        return "\n\n".join(parts)
    
    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Inc|Ltd|Co|Corp|vs|etc|e\.g|i\.e|Jr|Sr|Prof|Rev)\.\s',
                     r'\1<ABBREV> ', text, flags=re.IGNORECASE)
        
        text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
        
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1<SENT>\2', text)
        
        text = text.replace('<ABBREV>', '.')
        text = text.replace('<DECIMAL>', '.')
        
        sentences = text.split('<SENT>')
        
        merged = []
        buffer = ""
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
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
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.debug(f"Token counting failed: {e}, using word-based estimate")
            return int(len(text.split()) * 1.3)


chunker = EnhancedContextualChunker()
