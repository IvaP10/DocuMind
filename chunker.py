from typing import List
from uuid import uuid4
import tiktoken
import re
from models import Chunk
from parser import LayoutElement
import config
import logging

logger = logging.getLogger(__name__)

class ContextualChunker:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.parent_size = config.CHUNK_SIZE_PARENT
        self.child_size = config.CHUNK_SIZE_CHILD
        self.overlap = config.CHUNK_OVERLAP
    
    def create_chunks(self, document_id: str, elements: List[LayoutElement]) -> List[Chunk]:
        logger.info(f"Creating chunks for document {document_id}")
        all_chunks = []
        sections = self.group_sections(elements)
        
        for section in sections:
            parent_chunk = self.create_parent(document_id, section)
            all_chunks.append(parent_chunk)
            
            child_chunks = self.create_children(document_id, parent_chunk, section)
            all_chunks.extend(child_chunks)
        
        logger.info(f"Created {len(all_chunks)} total chunks")
        return all_chunks
    
    def group_sections(self, elements: List[LayoutElement]) -> List[List[LayoutElement]]:
        sections = []
        current_section = []
        current_tokens = 0
        
        for element in elements:
            element_tokens = self.count_tokens(element.content)
            
            if current_tokens + element_tokens > self.parent_size and current_section:
                sections.append(current_section)
                current_section = [element]
                current_tokens = element_tokens
            else:
                current_section.append(element)
                current_tokens += element_tokens
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def create_parent(self, document_id: str, section: List[LayoutElement]) -> Chunk:
        combined_text = "\n\n".join([elem.content for elem in section])
        first_elem = section[0]
        
        return Chunk(
            id=uuid4(),
            document_id=document_id,
            parent_id=None,
            text=combined_text,
            chunk_type=first_elem.element_type,
            format_type=first_elem.format_type,
            page_number=first_elem.page,
            bbox=first_elem.bbox,
            token_count=self.count_tokens(combined_text),
            metadata={
                "is_parent": True,
                "num_elements": len(section),
                "element_types": [elem.element_type.value for elem in section]
            }
        )
    
    def create_children(self, document_id: str, parent: Chunk, section: List[LayoutElement]) -> List[Chunk]:
        children = []
        
        for element in section:
            if self.count_tokens(element.content) <= self.child_size:
                child = Chunk(
                    id=uuid4(),
                    document_id=document_id,
                    parent_id=parent.id,
                    text=element.content,
                    chunk_type=element.element_type,
                    format_type=element.format_type,
                    page_number=element.page,
                    bbox=element.bbox,
                    token_count=self.count_tokens(element.content),
                    metadata={
                        "is_parent": False,
                        "parent_id": str(parent.id)
                    }
                )
                children.append(child)
            else:
                sub_chunks = self.split_element(element, parent.id, document_id)
                children.extend(sub_chunks)
        
        return children
    
    def split_element(self, element: LayoutElement, parent_id: str, document_id: str) -> List[Chunk]:
        chunks = []
        text = element.content
        sentences = self.split_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.child_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        id=uuid4(),
                        document_id=document_id,
                        parent_id=parent_id,
                        text=chunk_text,
                        chunk_type=element.element_type,
                        format_type=element.format_type,
                        page_number=element.page,
                        bbox=element.bbox,
                        token_count=self.count_tokens(chunk_text),
                        metadata={"is_parent": False, "parent_id": str(parent_id)}
                    )
                )
                
                overlap_sentences = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_sentences + [sentence]
                current_tokens = self.count_tokens(" ".join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    id=uuid4(),
                    document_id=document_id,
                    parent_id=parent_id,
                    text=chunk_text,
                    chunk_type=element.element_type,
                    format_type=element.format_type,
                    page_number=element.page,
                    bbox=element.bbox,
                    token_count=self.count_tokens(chunk_text),
                    metadata={"is_parent": False, "parent_id": str(parent_id)}
                )
            )
        
        return chunks
    
    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def count_tokens(self, text: str) -> int:
        try:
            return len(self.encoding.encode(text))
        except:
            return int(len(text.split()) * 1.3)


chunker = ContextualChunker()