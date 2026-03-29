from pathlib import Path
from uuid import uuid4, UUID
import logging
import sys
import time
import asyncio
import json
import re
from collections import defaultdict
from threading import Lock

from typing import Dict, Any, List, Optional

_import_start = time.time()

from pdf_parser import parser
from chunker import chunker
from embedder import embedder
from database import vector_db
from retriever import retriever
from generator import generator
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('rag_system.log')]
)
logger = logging.getLogger(__name__)
logger.info(f"Modules loaded in {time.time() - _import_start:.2f}s")


# ── Shared state between background ingestion and query loop ──────────────────

class IngestionState:
    """Thread-safe shared state that grows as each PDF finishes indexing."""

    def __init__(self):
        self._lock = Lock()
        self.doc_ids: List[UUID] = []
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.total_pdfs = 0
        self.processed_pdfs = 0
        self.is_complete = False
        self.current_file = ""
        self.errors: List[str] = []
        self._pending_messages: List[str] = []
        self.query_active = False          # suppresses ingestion prints during streaming

    def add_document(self, doc_id: UUID, chunks_meta: List[Dict[str, Any]]):
        with self._lock:
            self.doc_ids.append(doc_id)
            self.chunks_metadata.extend(chunks_meta)
            self.processed_pdfs += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "doc_ids": list(self.doc_ids),
                "chunks_metadata": list(self.chunks_metadata),
                "processed": self.processed_pdfs,
                "total": self.total_pdfs,
                "is_complete": self.is_complete,
            }

    def post_message(self, msg: str):
        """Print immediately, or buffer if a query is streaming."""
        with self._lock:
            if self.query_active:
                self._pending_messages.append(msg)
            else:
                print(msg)

    def flush_messages(self):
        with self._lock:
            for msg in self._pending_messages:
                print(msg)
            self._pending_messages.clear()

    @property
    def status_line(self) -> str:
        with self._lock:
            if self.is_complete:
                return f"✓ All {self.total_pdfs} PDF(s) indexed ({len(self.chunks_metadata)} chunks)"
            if self.total_pdfs == 0:
                return "Initializing..."
            return f"Indexing [{self.processed_pdfs}/{self.total_pdfs}] — {self.current_file}"


# ── Document processing (unchanged logic, same helpers) ──────────────────────

def process_document(pdf_path: str, doc_id: Optional[UUID] = None) -> tuple[UUID, List[Dict[str, Any]], List[str]]:
    start_time = time.time()
    if doc_id is None:
        doc_id = uuid4()

    filename = Path(pdf_path).name

    total_pages, elements = parser.parse(pdf_path)
    chunks = chunker.create_chunks(str(doc_id), elements)

    for chunk in chunks:
        chunk.metadata["source_filename"] = filename

    chunks_metadata = [
        {
            "id": str(c.id),
            "parent_id": str(c.parent_id) if c.parent_id else None,
            "text": c.text,
            "chunk_type": c.chunk_type.value,
            "format_type": c.format_type.value,
            "page_number": c.page_number,
            "bbox": c.bbox.dict() if c.bbox else None,
            "token_count": c.token_count,
            "is_parent": c.is_parent,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    all_texts = [c.text for c in chunks]

    logger.info(f"Parsed {len(chunks)} chunks from {filename} in {time.time()-start_time:.2f}s | doc_id={doc_id}")
    return doc_id, chunks_metadata, all_texts, chunks


# ── Background ingestion task ─────────────────────────────────────────────────

async def _ingest_one_pdf(pdf_path: str, state: IngestionState):
    """Process → embed → index a single PDF, then atomically add it to the queryable pool."""
    filename = Path(pdf_path).name
    state.current_file = filename

    doc_id, chunks_meta, texts, chunks = await asyncio.to_thread(
        process_document, pdf_path
    )

    dense_embeddings = await asyncio.to_thread(embedder.embed_texts, texts, False)
    sparse_vectors = await asyncio.to_thread(embedder.create_sparse_vectors_batch, texts)
    await asyncio.to_thread(vector_db.index_chunks, chunks, dense_embeddings, sparse_vectors)

    # Atomic: this PDF is now queryable
    state.add_document(doc_id, chunks_meta)


async def ingest_async(input_path: str, state: IngestionState):
    """Background task — ingests all PDFs and updates shared state progressively."""
    path = Path(input_path)

    await asyncio.to_thread(vector_db.reset_collection)

    if path.is_dir():
        pdf_files = sorted(path.glob("*.pdf"))
        if not pdf_files:
            print(f"Error: No PDF files found in {input_path}")
            state.is_complete = True
            return
        state.total_pdfs = len(pdf_files)

        for pdf_file in pdf_files:
            try:
                await _ingest_one_pdf(str(pdf_file), state)
            except Exception as e:
                state.errors.append(f"{pdf_file.name}: {e}")
                logger.error(f"Failed to ingest {pdf_file.name}: {e}", exc_info=True)
                with state._lock:
                    state.processed_pdfs += 1
    else:
        state.total_pdfs = 1
        try:
            await _ingest_one_pdf(input_path, state)
        except Exception as e:
            state.errors.append(f"{path.name}: {e}")
            logger.error(f"Failed to ingest {path.name}: {e}", exc_info=True)

    state.is_complete = True


# ── Citation consolidation ────────────────────────────────────────────────────

def consolidate_citations(text: str) -> str:
    """Merge all [[Source: X | Page: N]] into one citation per file at the end."""
    pattern = r'\[\[Source:\s*(.+?)\s*\|\s*Page:\s*([\d,\s]+)\s*\]\]'

    sources: Dict[str, set] = defaultdict(set)
    for m in re.finditer(pattern, text):
        filename = m.group(1).strip()
        for p in m.group(2).split(','):
            p = p.strip()
            if p.isdigit():
                sources[filename].add(int(p))

    # Strip all inline citations
    cleaned = re.sub(pattern, '', text)
    cleaned = re.sub(r'[,\s]*\.', '.', cleaned)   # fix trailing comma before period
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = cleaned.strip()

    if sources:
        parts = []
        for fname, pages in sorted(sources.items()):
            page_str = ','.join(str(p) for p in sorted(pages))
            parts.append(f"[Source: {fname} | Page: {page_str}]")
        cleaned += '\n' + ' '.join(parts)

    return cleaned


# ── Async query pipeline (no nested asyncio.run) ─────────────────────────────

async def answer_query_async(
    chunks_metadata: List[Dict[str, Any]],
    query: str,
    state: IngestionState,
    document_id: Optional[UUID] = None,
) -> Dict[str, Any]:
    start_time = time.time()

    context_data = await retriever.retrieve_context(
        query=query, chunks_metadata=chunks_metadata, document_id=document_id
    )

    # Stream tokens word-by-word, silently capturing citations
    state.query_active = True
    final_payload = None
    citation_sources: Dict[str, set] = defaultdict(set)
    cite_buffer = ""       # accumulates text inside [[ ... ]]
    inside_cite = False     # True while we're inside a [[ ]] block
    full_answer = ""

    try:
        async for chunk in generator.generate_answer_stream(query, context_data):
            data = json.loads(chunk)
            if data.get("type") == "token":
                full_answer += data["content"]
                for ch in data["content"]:
                    if inside_cite:
                        cite_buffer += ch
                        # Check if citation closed
                        if cite_buffer.endswith("]]"):
                            # Parse the captured citation
                            m = re.match(r'\[\[Source:\s*(.+?)\s*\|\s*Page:\s*([\d,\s]+)\s*\]\]', cite_buffer)
                            if m:
                                fname = m.group(1).strip()
                                for p in m.group(2).split(','):
                                    p = p.strip()
                                    if p.isdigit():
                                        citation_sources[fname].add(int(p))
                            else:
                                # Not a citation pattern — flush buffer as regular text
                                print(cite_buffer, end="", flush=True)
                            cite_buffer = ""
                            inside_cite = False
                    else:
                        if ch == '[' and cite_buffer == "":
                            cite_buffer = "["
                        elif ch == '[' and cite_buffer == "[":
                            cite_buffer = "[["
                            inside_cite = True
                        else:
                            if cite_buffer:
                                print(cite_buffer, end="", flush=True)
                                cite_buffer = ""
                            print(ch, end="", flush=True)
            elif data.get("type") in ("verification", "error"):
                final_payload = data
    finally:
        # Flush any remaining buffer
        if cite_buffer and not inside_cite:
            print(cite_buffer, end="", flush=True)
        state.query_active = False
        state.flush_messages()

    # Print consolidated citation on a new line
    if citation_sources:
        parts = []
        for fname, pages in sorted(citation_sources.items()):
            page_str = ','.join(str(p) for p in sorted(pages))
            parts.append(f"[Source: {fname} | Page: {page_str}]")
        print(f"\n{' '.join(parts)}")
    else:
        print()

    if final_payload and final_payload.get("type") == "verification":
        final_payload["processing_time"] = time.time() - start_time
        print(f"confidence: {final_payload['confidence']:.1%}\n")
    elif final_payload and final_payload.get("type") == "error":
        print(f"  {final_payload.get('content', 'Unknown error')}\n")

    return final_payload or {}


# ── Interactive query loop (runs concurrently with ingestion) ─────────────────

async def interactive_mode_async(state: IngestionState):

    while True:
        try:
            snap = state.snapshot()
            if state.is_complete or snap['total'] == 0:
                prompt = "Query> "
            else:
                prompt = f"[{snap['processed']}/{snap['total']}] Query> "

            user_input = (await asyncio.to_thread(input, prompt)).strip()

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            elif user_input.lower() == 'status':
                print(f"  {state.status_line}")
                if state.errors:
                    for err in state.errors:
                        print(f"  ✗ {err}")
                print()
            elif user_input.lower() == 'stats':
                snap = state.snapshot()
                cm = snap["chunks_metadata"]
                parents = sum(1 for c in cm if c.get("is_parent"))
                children = sum(1 for c in cm if not c.get("is_parent"))
                tokens = sum(c.get("token_count", 0) for c in cm)
                pages = len(set(c.get("page_number") for c in cm))
                filenames = set(c.get("metadata", {}).get("source_filename", "?") for c in cm)
                print(f"\nDocuments: {len(snap['doc_ids'])} | Files: {', '.join(filenames)}")
                print(f"Chunks: {len(cm)} ({parents} parents, {children} children) | Tokens: {tokens:,} | Pages: {pages}")
                print(f"Ingestion: {state.status_line}\n")
            else:
                # If no docs indexed yet, wait for the first one before executing
                if not state.chunks_metadata:
                    print("⏳ Waiting for first document to finish indexing...")
                    while not state.chunks_metadata and not state.is_complete:
                        await asyncio.sleep(0.3)
                    if not state.chunks_metadata:
                        print("No documents were indexed. Cannot answer.\n")
                        continue
                    print()

                try:
                    snap = state.snapshot()
                    await answer_query_async(snap["chunks_metadata"], user_input, state)
                except Exception as e:
                    print(f"Error: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


# ── Entrypoint ────────────────────────────────────────────────────────────────

async def async_main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter PDF file or folder path: ").strip()
    input_path = input_path.strip('"').strip("'")
    path = Path(input_path)

    if not path.exists():
        print(f"Error: Path not found: {input_path}")
        return 1

    if not path.is_dir() and path.suffix.lower() != '.pdf':
        print(f"Error: Unsupported path. Provide a PDF file or a folder containing PDFs.")
        return 1

    print(f"Processing: {input_path}\n")

    state = IngestionState()

    try:
        # Launch ingestion as a background task — does NOT block the query loop
        ingestion_task = asyncio.create_task(ingest_async(input_path, state))

        # Start accepting queries immediately (waits only for the first PDF)
        await interactive_mode_async(state)

        # If user quits before ingestion finishes, cancel gracefully
        if not ingestion_task.done():
            ingestion_task.cancel()
            try:
                await ingestion_task
            except asyncio.CancelledError:
                logger.info("Ingestion cancelled by user exit")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Application error: {e}", exc_info=True)
        return 1

    return 0


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
