from pathlib import Path
from uuid import uuid4, UUID
import logging
import sys
import time
from typing import Dict, Any, List

from parser import parser
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


def process_document(pdf_path: str) -> tuple[UUID, List[Dict[str, Any]]]:
    start_time = time.time()
    doc_id = uuid4()

    total_pages, elements = parser.parse(pdf_path)
    chunks = chunker.create_chunks(str(doc_id), elements)

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
    dense_embeddings = embedder.embed_texts(all_texts, show_progress=True)
    sparse_vectors = embedder.create_sparse_vectors_batch(all_texts)
    vector_db.index_chunks(chunks, dense_embeddings, sparse_vectors)

    logger.info(f"Processed {len(chunks)} chunks in {time.time()-start_time:.2f}s | doc_id={doc_id}")
    return doc_id, chunks_metadata


def answer_query(doc_id: UUID, chunks_metadata: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    start_time = time.time()
    context_data = retriever.retrieve_context(document_id=doc_id, query=query, chunks_metadata=chunks_metadata)
    answer_data = generator.generate_answer(query=query, context_data=context_data)
    answer_data["processing_time"] = time.time() - start_time

    print(f"\n{'='*80}\nANSWER:\n{'='*80}")
    print(answer_data['answer'])
    print(f"\n{'='*80}")
    print(f"Confidence: {answer_data['confidence']:.1%} | Verified: {answer_data['verified']} | Sources: {len(answer_data['sources'])}")
    print(f"{'='*80}\n")
    return answer_data


def interactive_mode(doc_id: UUID, chunks_metadata: List[Dict[str, Any]]):
    print(f"\n{'='*80}\nINTERACTIVE MODE — doc_id: {doc_id}")
    print("Commands: <query>  |  stats  |  quit\n")

    while True:
        try:
            user_input = input("Query> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'stats':
                parents = sum(1 for c in chunks_metadata if c.get("is_parent"))
                children = sum(1 for c in chunks_metadata if not c.get("is_parent"))
                tokens = sum(c.get("token_count", 0) for c in chunks_metadata)
                pages = len(set(c.get("page_number") for c in chunks_metadata))
                print(f"\nChunks: {len(chunks_metadata)} ({parents} parents, {children} children) | Tokens: {tokens:,} | Pages: {pages}\n")
            else:
                try:
                    answer_query(doc_id, chunks_metadata, user_input)
                except Exception as e:
                    print(f"Error: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    print(f"\n{'='*80}")
    print(f"RAG SYSTEM | Model: {config.LLM_MODEL} | Embeddings: {config.EMBEDDING_MODEL}")
    print(f"{'='*80}\n")

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter PDF path: ").strip()
    pdf_path = pdf_path.strip('"').strip("'")

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return 1
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: Only PDF files supported.")
        return 1

    print(f"Processing: {pdf_path}\n")
    try:
        doc_id, chunks_metadata = process_document(pdf_path)
        print(f"\nDocument ready — {len(chunks_metadata)} chunks\n")
        interactive_mode(doc_id, chunks_metadata)
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
