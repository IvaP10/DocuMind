from typing import List, Optional, Dict, Any
from docling.document_converter import DocumentConverter
from models import ChunkType, FormatType, BoundingBox
import logging

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
            total_pages = len(result.document.pages)
            
            for page in result.document.pages:
                page_num = page.page_no
                
                for text_block in page.text:
                    element = LayoutElement(
                        content=text_block.text,
                        element_type=ChunkType.TEXT,
                        page=page_num,
                        bbox=self._extract_bbox(text_block),
                        format_type=FormatType.PLAIN
                    )
                    elements.append(element)
                
                for table in page.tables:
                    table_content = self._format_table(table)
                    element = LayoutElement(
                        content=table_content,
                        element_type=ChunkType.TABLE,
                        page=page_num,
                        bbox=self._extract_bbox(table),
                        format_type=self._determine_table_format(table)
                    )
                    elements.append(element)
            
            logger.info(f"Parsed {total_pages} pages, {len(elements)} elements")
            return total_pages, elements
            
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            raise
    
    def _extract_bbox(self, element) -> Optional[BoundingBox]:
        try:
            if hasattr(element, 'bbox'):
                bbox = element.bbox
                return BoundingBox(
                    x=bbox.l,
                    y=bbox.t,
                    width=bbox.r - bbox.l,
                    height=bbox.b - bbox.t
                )
        except:
            pass
        return None
    
    def _format_table(self, table) -> str:
        try:
            if self._has_merged_cells(table):
                return self._table_to_html(table)
            else:
                return self._table_to_markdown(table)
        except:
            return str(table)
    
    def _determine_table_format(self, table) -> FormatType:
        if self._has_merged_cells(table):
            return FormatType.HTML
        return FormatType.MARKDOWN
    
    def _has_merged_cells(self, table) -> bool:
        try:
            if hasattr(table, 'cells'):
                for cell in table.cells:
                    if hasattr(cell, 'rowspan') and cell.rowspan > 1:
                        return True
                    if hasattr(cell, 'colspan') and cell.colspan > 1:
                        return True
        except:
            pass
        return False
    
    def _table_to_markdown(self, table) -> str:
        try:
            rows = []
            if hasattr(table, 'data'):
                for row in table.data:
                    rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
                
                if rows:
                    header_sep = "| " + " | ".join("---" for _ in table.data[0]) + " |"
                    rows.insert(1, header_sep)
                    
            return "\n".join(rows)
        except:
            return str(table)
    
    def _table_to_html(self, table) -> str:
        try:
            html = ["<table>"]
            
            if hasattr(table, 'data'):
                for i, row in enumerate(table.data):
                    html.append("<tr>")
                    tag = "th" if i == 0 else "td"
                    for cell in row:
                        html.append(f"<{tag}>{cell}</{tag}>")
                    html.append("</tr>")
                    
            html.append("</table>")
            return "".join(html)
        except:
            return str(table)


parser = PDFParser()