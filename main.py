from pathlib import Path
from uuid import uuid4
import logging

from parser import parser
from chunker import chunker
from embedder import embedder
from database import vector_db
from retriever import retriever
from generator import generator

logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

chunks_db = {}

def process_document(pdf_path: str):
    doc_id = uuid4()
    total_pages, elements = parser.parse(pdf_path)
    chunks = chunker.create_chunks(str(doc_id), elements)
    
    chunks_metadata = [
        {
            "id": str(chunk.id),
            "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
            "text": chunk.text,
            "chunk_type": chunk.chunk_type.value,
            "page_number": chunk.page_number,
            "bbox": chunk.bbox.dict() if chunk.bbox else None,
            "token_count": chunk.token_count,
            "is_parent": chunk.parent_id is None,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    child_chunks = [c for c in chunks if c.parent_id is not None]
    texts = [chunk.text for chunk in child_chunks]
    
    dense_embeddings = embedder.embed_texts(texts)
    sparse_vectors = embedder.create_sparse_vectors_batch(texts)
    vector_db.index_chunks(chunks, dense_embeddings, sparse_vectors)
    
    return doc_id, chunks_metadata


def answer_query(doc_id, chunks_metadata, query, mode="hybrid_full"):
    context_data = retriever.retrieve_context(
        document_id=doc_id,
        query=query,
        chunks_metadata=chunks_metadata,
        mode=mode
    )
    
    answer_data = generator.generate_answer(
        query=query,
        context_data=context_data
    )
    
    print(f"\n{answer_data['answer']}\n")
    return answer_data


def main():
    pdf_path = input("Enter path: ").strip()
    
    if not Path(pdf_path).exists():
        print("File not found")
        return
    
    if not pdf_path.lower().endswith('.pdf'):
        print("Only PDF files supported")
        return
    
    print("Processing...")
    
    try:
        doc_id, chunks_metadata = process_document(pdf_path)
        chunks_db[str(doc_id)] = chunks_metadata
    except Exception as e:
        print(f"Error: {e}")
        return
    
    mode = "hybrid_full"
    
    while True:
        query = input("Enter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q', 'stop']:
            break
        
        if not query:
            continue
        
        try:
            answer_query(doc_id, chunks_metadata, query, mode)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()