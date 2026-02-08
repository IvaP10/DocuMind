"""
Enhanced PDF Parser using Docling (2026)
Improved layout analysis and content extraction
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.labels import DocItemLabel
from models import ChunkType, FormatType, BoundingBox

logger = logging.getLogger(__name__)


class LayoutElement:
    """Represents a single layout element from the document"""
    
    def __init__(
        self,
        content: str,
        element_type: ChunkType,
        page: int,
        bbox: Optional[BoundingBox] = None,
        format_type: FormatType = FormatType.PLAIN,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.element_type = element_type
        self.page = page
        self.bbox = bbox
        self.format_type = format_type
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"LayoutElement(type={self.element_type}, page={self.page}, len={len(self.content)})"


class EnhancedPDFParser:
    """
    Enhanced PDF parser using Docling with improved:
    - Layout detection
    - Table extraction
    - Equation handling
    - Multi-column support
    """
    
    def __init__(self):
        """Initialize parser with optimized Docling settings"""
        logger.info("Initializing Enhanced PDF Parser")
        
        # Configure PDF pipeline options for better extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned documents
        pipeline_options.do_table_structure = True  # Enhanced table extraction
        
        # Initialize converter with options
        self.docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        logger.info("PDF Parser initialized successfully")
    
    def parse(self, pdf_path: str) -> Tuple[int, List[LayoutElement]]:
        """
        Parse PDF and extract structured layout elements
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (total_pages, list of LayoutElements)
        """
        logger.info(f"Parsing PDF: {pdf_path}")
        
        # Validate file exists
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Convert document
            result = self.docling_converter.convert(pdf_path)
            doc = result.document
            
            # Get total pages
            total_pages = self._get_total_pages(doc)
            logger.info(f"Document has {total_pages} pages")
            
            # Extract elements
            elements = self._extract_elements(doc)
            
            # Post-process elements
            elements = self._post_process_elements(elements)
            
            logger.info(f"Extracted {len(elements)} layout elements")
            
            # Log element type distribution
            type_counts = {}
            for elem in elements:
                type_counts[elem.element_type] = type_counts.get(elem.element_type, 0) + 1
            logger.info(f"Element distribution: {type_counts}")
            
            return total_pages, elements
            
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            raise
    
    def _get_total_pages(self, doc) -> int:
        """Extract total page count from document"""
        try:
            if hasattr(doc, 'pages'):
                if isinstance(doc.pages, dict):
                    return len(doc.pages)
                elif isinstance(doc.pages, list):
                    return len(doc.pages)
            
            # Fallback: count page numbers in elements
            max_page = 0
            for item, _ in doc.iterate_items():
                if hasattr(item, 'prov') and item.prov:
                    page_num = item.prov[0].page_no
                    max_page = max(max_page, page_num)
            
            return max_page if max_page > 0 else 1
            
        except Exception as e:
            logger.warning(f"Could not determine page count: {e}")
            return 1
    
    def _extract_elements(self, doc) -> List[LayoutElement]:
        """Extract all layout elements from document"""
        elements = []
        
        for item, _ in doc.iterate_items():
            element = self._process_item(doc, item)
            if element:
                elements.append(element)
        
        return elements
    
    def _process_item(self, doc, item) -> Optional[LayoutElement]:
        """Process a single document item into a LayoutElement"""
        
        # Determine element type and format
        element_type, format_type = self._classify_item(item)
        
        # Extract text content
        text_content = self._extract_text(doc, item, element_type, format_type)
        
        # Skip empty content
        if not text_content or not text_content.strip():
            return None
        
        # Extract page number
        page_num = self._extract_page_number(item)
        
        # Extract bounding box
        bbox = self._extract_bbox(item)
        
        # Create metadata
        metadata = self._create_metadata(item)
        
        return LayoutElement(
            content=text_content,
            element_type=element_type,
            page=page_num,
            bbox=bbox,
            format_type=format_type,
            metadata=metadata
        )
    
    def _classify_item(self, item) -> Tuple[ChunkType, FormatType]:
        """Classify item type and format"""
        
        # Build label mapping dynamically - only include attributes that exist
        label_mapping = {}
        
        # Always available labels
        if hasattr(DocItemLabel, 'TABLE'):
            label_mapping[DocItemLabel.TABLE] = (ChunkType.TABLE, FormatType.MARKDOWN)
        
        if hasattr(DocItemLabel, 'CODE'):
            label_mapping[DocItemLabel.CODE] = (ChunkType.CODE, FormatType.PLAIN)
        
        if hasattr(DocItemLabel, 'LIST_ITEM'):
            label_mapping[DocItemLabel.LIST_ITEM] = (ChunkType.LIST, FormatType.PLAIN)
        
        if hasattr(DocItemLabel, 'SECTION_HEADER'):
            label_mapping[DocItemLabel.SECTION_HEADER] = (ChunkType.HEADER, FormatType.PLAIN)
        
        if hasattr(DocItemLabel, 'PAGE_HEADER'):
            label_mapping[DocItemLabel.PAGE_HEADER] = (ChunkType.HEADER, FormatType.PLAIN)
        
        if hasattr(DocItemLabel, 'PAGE_FOOTER'):
            label_mapping[DocItemLabel.PAGE_FOOTER] = (ChunkType.FOOTER, FormatType.PLAIN)
        
        # Optional labels (may not exist in all Docling versions)
        if hasattr(DocItemLabel, 'EQUATION'):
            label_mapping[DocItemLabel.EQUATION] = (ChunkType.EQUATION, FormatType.LATEX)
        
        if hasattr(DocItemLabel, 'FORMULA'):
            label_mapping[DocItemLabel.FORMULA] = (ChunkType.EQUATION, FormatType.LATEX)
        
        # Check if item has a label and it's in our mapping
        if hasattr(item, 'label') and item.label in label_mapping:
            return label_mapping[item.label]
        
        # Check for equations by content pattern (fallback)
        if hasattr(item, 'text'):
            text = item.text.strip()
            # Simple heuristic for math content
            if text.startswith('$') and text.endswith('$'):
                return ChunkType.EQUATION, FormatType.LATEX
        
        # Default to text
        return ChunkType.TEXT, FormatType.PLAIN
    
    def _extract_text(
        self, 
        doc, 
        item, 
        element_type: ChunkType, 
        format_type: FormatType
    ) -> str:
        """Extract text content from item based on type"""
        
        # For tables, export as markdown
        if element_type == ChunkType.TABLE:
            try:
                if hasattr(doc, "export_to_markdown"):
                    return doc.export_to_markdown(item)
                elif hasattr(item, 'export_to_markdown'):
                    return item.export_to_markdown()
            except Exception as e:
                logger.debug(f"Markdown export failed: {e}, using text fallback")
        
        # For equations, try to preserve LaTeX
        if element_type == ChunkType.EQUATION:
            if hasattr(item, 'latex'):
                return f"$${item.latex}$$"
            elif hasattr(item, 'text'):
                text = item.text.strip()
                # If it looks like an equation, wrap it
                if not text.startswith('$'):
                    return f"$${text}$$"
        
        # Default: use text attribute
        if hasattr(item, 'text'):
            return item.text
        
        return ""
    
    def _extract_page_number(self, item) -> int:
        """Extract page number from item"""
        try:
            if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
                return item.prov[0].page_no
        except Exception as e:
            logger.debug(f"Could not extract page number: {e}")
        
        return 1  # Default to page 1
    
    def _extract_bbox(self, item) -> Optional[BoundingBox]:
        """Extract bounding box coordinates from item"""
        try:
            if hasattr(item, 'prov') and item.prov and len(item.prov) > 0:
                bbox = item.prov[0].bbox
                return BoundingBox(
                    x=float(bbox.l),
                    y=float(bbox.t),
                    width=float(bbox.r - bbox.l),
                    height=float(bbox.b - bbox.t)
                )
        except Exception as e:
            logger.debug(f"Could not extract bounding box: {e}")
        
        return None
    
    def _create_metadata(self, item) -> Dict[str, Any]:
        """Create metadata dictionary from item"""
        metadata = {}
        
        # Add confidence scores if available
        if hasattr(item, 'confidence'):
            metadata['confidence'] = item.confidence
        
        # Add label information
        if hasattr(item, 'label'):
            metadata['original_label'] = str(item.label)
        
        # Add any custom attributes
        if hasattr(item, 'metadata'):
            metadata.update(item.metadata)
        
        return metadata
    
    def _post_process_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Post-process extracted elements"""
        
        processed = []
        
        for element in elements:
            # Clean up text content
            element.content = self._clean_text(element.content)
            
            # Skip if content became empty after cleaning
            if not element.content or not element.content.strip():
                continue
            
            # Merge consecutive headers/footers (optional)
            if element.element_type in [ChunkType.HEADER, ChunkType.FOOTER]:
                # You can implement header/footer merging logic here
                pass
            
            processed.append(element)
        
        # Sort by page and position
        processed = self._sort_elements(processed)
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize unicode
        text = text.strip()
        
        return text
    
    def _sort_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Sort elements by page number and vertical position"""
        
        def sort_key(elem: LayoutElement):
            # Primary sort: page number
            page = elem.page
            
            # Secondary sort: vertical position (if bbox available)
            y_pos = elem.bbox.y if elem.bbox else 0
            
            return (page, y_pos)
        
        return sorted(elements, key=sort_key)


# Global parser instance
parser = EnhancedPDFParser()