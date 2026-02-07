import logging
from typing import List, Optional, Dict, Any
from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel
from models import ChunkType, FormatType, BoundingBox

logger = logging.getLogger(__name__)

class LayoutElement:
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

class PDFParser:
    
    def __init__(self):
        self.docling_converter = DocumentConverter()
    
    def parse(self, pdf_path: str) -> tuple[int, List[LayoutElement]]:
        logger.info(f"Parsing PDF: {pdf_path}")
        
        try:
            result = self.docling_converter.convert(pdf_path)
            elements = []
            doc = result.document
            
            total_pages = 0
            if hasattr(doc, 'pages') and isinstance(doc.pages, dict):
                total_pages = len(doc.pages)
            elif hasattr(doc, 'pages') and isinstance(doc.pages, list):
                total_pages = len(doc.pages)
            
            for item, _ in doc.iterate_items():
                text_content = ""
                element_type = ChunkType.TEXT
                format_type = FormatType.PLAIN
                
                if item.label == DocItemLabel.TABLE:
                    element_type = ChunkType.TABLE
                    format_type = FormatType.MARKDOWN
                    if hasattr(doc, "export_to_markdown"):
                        text_content = doc.export_to_markdown(item)
                    else:
                        text_content = item.text
                elif hasattr(item, 'text') and item.text.strip():
                    text_content = item.text
                
                if not text_content:
                    continue
                    
                page_num = 1
                if hasattr(item, 'prov') and item.prov:
                    page_num = item.prov[0].page_no
                
                element = LayoutElement(
                    content=text_content,
                    element_type=element_type,
                    page=page_num,
                    bbox=self._extract_bbox(item),
                    format_type=format_type
                )
                elements.append(element)
            
            logger.info(f"Parsed {total_pages} pages, {len(elements)} elements")
            return total_pages, elements
            
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            raise
    
    def _extract_bbox(self, item) -> Optional[BoundingBox]:
        try:
            if hasattr(item, 'prov') and item.prov:
                bbox = item.prov[0].bbox
                return BoundingBox(
                    x=bbox.l,
                    y=bbox.t,
                    width=bbox.r - bbox.l,
                    height=bbox.b - bbox.t
                )
        except Exception:
            pass
        return None

parser = PDFParser()