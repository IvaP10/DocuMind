from pathlib import Path
from uuid import uuid4, UUID
import logging
import sys
import time
from typing import Dict, Any, List, Optional

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


def ingest_folder(folder_path: str) -> tuple[List[UUID], List[Dict[str, Any]]]:
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"Error: No PDF files found in {folder_path}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {folder_path}\n")

    vector_db.reset_collection()

    all_doc_ids: List[UUID] = []
    all_chunks_metadata: List[Dict[str, Any]] = []
    all_corpus_texts: List[str] = []
    all_chunks_by_doc: List[tuple] = []

    for pdf_file in pdf_files:
        print(f"  Parsing: {pdf_file.name}...")
        doc_id, chunks_metadata, texts, chunks = process_document(str(pdf_file))
        all_doc_ids.append(doc_id)
        all_chunks_metadata.extend(chunks_metadata)
        all_corpus_texts.extend(texts)
        all_chunks_by_doc.append((doc_id, chunks, texts))

    print(f"\nBuilding global BM25 index over {len(all_corpus_texts)} chunks...")
    embedder.build_bm25_index(all_corpus_texts)
    logger.info(f"Built BM25 index: corpus_size={embedder.corpus_size}, vocab_size={len(embedder.token_to_id)}")

    for doc_id, chunks, texts in all_chunks_by_doc:
        print(f"  Embedding & indexing doc {doc_id}...")
        dense_embeddings = embedder.embed_texts(texts, show_progress=True)
        sparse_vectors = embedder.create_sparse_vectors_batch(texts)
        vector_db.index_chunks(chunks, dense_embeddings, sparse_vectors)

    print(f"\nIngestion complete: {len(all_doc_ids)} documents, {len(all_chunks_metadata)} total chunks")
    return all_doc_ids, all_chunks_metadata


def ingest_single_pdf(pdf_path: str) -> tuple[List[UUID], List[Dict[str, Any]]]:
    vector_db.reset_collection()
    doc_id, chunks_metadata, texts, chunks = process_document(pdf_path)

    print(f"Building BM25 index over {len(texts)} chunks...")
    embedder.build_bm25_index(texts)
    logger.info(f"Built BM25 index: corpus_size={embedder.corpus_size}, vocab_size={len(embedder.token_to_id)}")

    print("Embedding & indexing...")
    dense_embeddings = embedder.embed_texts(texts, show_progress=True)
    sparse_vectors = embedder.create_sparse_vectors_batch(texts)
    vector_db.index_chunks(chunks, dense_embeddings, sparse_vectors)

    print(f"\nIngestion complete: 1 document, {len(chunks_metadata)} chunks")
    return [doc_id], chunks_metadata


def answer_query(chunks_metadata: List[Dict[str, Any]], query: str, document_id: Optional[UUID] = None) -> Dict[str, Any]:
    start_time = time.time()
    context_data = retriever.retrieve_context(query=query, chunks_metadata=chunks_metadata, document_id=document_id)
    answer_data = generator.generate_answer(query=query, context_data=context_data)
    answer_data["processing_time"] = time.time() - start_time

    print(f"\n{'='*80}\nANSWER:\n{'='*80}")
    print(answer_data['answer'])
    print(f"\n{'='*80}")
    print(f"Confidence: {answer_data['confidence']:.1%} | Verified: {answer_data['verified']} | Sources: {len(answer_data['sources'])}")
    print(f"{'='*80}\n")
    return answer_data


def interactive_mode(doc_ids: List[UUID], chunks_metadata: List[Dict[str, Any]]):
    print(f"\n{'='*80}\nINTERACTIVE MODE — {len(doc_ids)} document(s) loaded")
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
                filenames = set(c.get("metadata", {}).get("source_filename", "?") for c in chunks_metadata)
                print(f"\nDocuments: {len(doc_ids)} | Files: {', '.join(filenames)}")
                print(f"Chunks: {len(chunks_metadata)} ({parents} parents, {children} children) | Tokens: {tokens:,} | Pages: {pages}\n")
            else:
                try:
                    answer_query(chunks_metadata, user_input)
                except Exception as e:
                    print(f"Error: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    print(f"\n{'='*80}")
    print(f"RAG SYSTEM | Model: {config.LLM_MODEL} | Embeddings: {config.EMBEDDING_MODEL}")
    print(f"{'='*80}\n")

    input_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter PDF file or folder path: ").strip()
    input_path = input_path.strip('"').strip("'")
    path = Path(input_path)

    if not path.exists():
        print(f"Error: Path not found: {input_path}")
        return 1

    print(f"Processing: {input_path}\n")
    try:
        if path.is_dir():
            doc_ids, chunks_metadata = ingest_folder(input_path)
        elif path.suffix.lower() == '.pdf':
            doc_ids, chunks_metadata = ingest_single_pdf(input_path)
        else:
            print(f"Error: Unsupported path. Provide a PDF file or a folder containing PDFs.")
            return 1

        interactive_mode(doc_ids, chunks_metadata)
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
