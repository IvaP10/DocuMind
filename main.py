from pathlib import Path
from uuid import uuid4, UUID
import logging
import sys
import time
from typing import Optional, Dict, Any, List

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
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_system.log')
    ]
)
logger = logging.getLogger(__name__)

chunks_db: Dict[str, List[Dict[str, Any]]] = {}


def process_document(pdf_path: str) -> tuple[UUID, List[Dict[str, Any]]]:
    logger.info("=" * 80)
    logger.info("DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 80)
    
    start_time = time.time()
    doc_id = uuid4()
    
    try:
        logger.info("Step 1/5: Parsing PDF...")
        parse_start = time.time()
        total_pages, elements = parser.parse(pdf_path)
        parse_time = time.time() - parse_start
        logger.info(f"✓ Parsed {total_pages} pages, {len(elements)} elements ({parse_time:.2f}s)")
        
        logger.info("Step 2/5: Creating chunks...")
        chunk_start = time.time()
        chunks = chunker.create_chunks(str(doc_id), elements)
        chunk_time = time.time() - chunk_start
        
        parent_count = sum(1 for c in chunks if c.is_parent)
        child_count = sum(1 for c in chunks if c.is_child)
        logger.info(f"✓ Created {len(chunks)} chunks: {parent_count} parents, {child_count} children ({chunk_time:.2f}s)")
        
        logger.info("Step 3/5: Creating metadata...")
        chunks_metadata = _create_chunks_metadata(chunks)
        logger.info(f"✓ Metadata created for {len(chunks_metadata)} chunks")
        
        logger.info("Step 4/5: Generating embeddings...")
        embed_start = time.time()
        
        all_texts = [chunk.text for chunk in chunks]
        
        dense_embeddings = embedder.embed_texts(all_texts, show_progress=True)
        sparse_vectors = embedder.create_sparse_vectors_batch(all_texts)
        
        embed_time = time.time() - embed_start
        logger.info(f"✓ Generated {len(dense_embeddings)} dense + {len(sparse_vectors)} sparse embeddings ({embed_time:.2f}s)")
        
        logger.info("Step 5/5: Indexing into vector database...")
        index_start = time.time()
        vector_db.index_chunks(chunks, dense_embeddings, sparse_vectors)
        index_time = time.time() - index_start
        logger.info(f"✓ Indexed into Qdrant ({index_time:.2f}s)")
        
        total_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"✓ PROCESSING COMPLETE")
        logger.info(f"  Document ID: {doc_id}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Breakdown: Parse={parse_time:.1f}s, Chunk={chunk_time:.1f}s, "
                   f"Embed={embed_time:.1f}s, Index={index_time:.1f}s")
        logger.info("=" * 80)
        
        return doc_id, chunks_metadata
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}", exc_info=True)
        raise


def _create_chunks_metadata(chunks: List) -> List[Dict[str, Any]]:
    metadata = []
    
    for chunk in chunks:
        metadata.append({
            "id": str(chunk.id),
            "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
            "text": chunk.text,
            "chunk_type": chunk.chunk_type.value,
            "format_type": chunk.format_type.value,
            "page_number": chunk.page_number,
            "bbox": chunk.bbox.dict() if chunk.bbox else None,
            "token_count": chunk.token_count,
            "is_parent": chunk.is_parent,
            "metadata": chunk.metadata
        })
    
    return metadata


def answer_query(
    doc_id: UUID,
    chunks_metadata: List[Dict[str, Any]],
    query: str,
    mode: str = "hybrid_quality"
) -> Dict[str, Any]:
    logger.info("\n" + "=" * 80)
    logger.info("QUERY ANSWERING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Query: {query}")
    logger.info(f"Mode: {mode}")
    
    start_time = time.time()
    
    try:
        logger.info("\nStep 1/2: Retrieving context...")
        context_data = retriever.retrieve_context(
            document_id=doc_id,
            query=query,
            chunks_metadata=chunks_metadata,
            mode=mode
        )
        
        logger.info("\nStep 2/2: Generating answer...")
        answer_data = generator.generate_answer(
            query=query,
            context_data=context_data
        )
        
        total_time = time.time() - start_time
        answer_data["processing_time"] = total_time
        
        logger.info("=" * 80)
        logger.info("✓ ANSWER GENERATED")
        logger.info(f"  Confidence: {answer_data['confidence']:.1%}")
        logger.info(f"  Verified: {answer_data['verified']}")
        logger.info(f"  Sources: {len(answer_data['sources'])}")
        logger.info(f"  Processing time: {total_time:.2f}s")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer_data['answer'])
        print("\n" + "=" * 80)
        print(f"Confidence: {answer_data['confidence']:.1%} | "
              f"Verified: {answer_data['verified']} | "
              f"Sources: {len(answer_data['sources'])}")
        print("=" * 80 + "\n")
        
        return answer_data
        
    except Exception as e:
        logger.error(f"Query answering failed: {str(e)}", exc_info=True)
        raise


def interactive_mode(doc_id: UUID, chunks_metadata: List[Dict[str, Any]], initial_mode: str):
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print(f"Document ID: {doc_id}")
    print(f"Available modes: {', '.join(config.EXPERIMENT_MODES.keys())}")
    print(f"Current mode: {initial_mode}")
    print("\nCommands:")
    print("  <query>           - Ask a question")
    print("  mode <name>       - Change retrieval mode")
    print("  stats             - Show document stats")
    print("  quit/exit         - Exit interactive mode")
    print("=" * 80 + "\n")
    
    current_mode = initial_mode
    
    while True:
        try:
            user_input = input("Query> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower().startswith('mode '):
                new_mode = user_input[5:].strip()
                if new_mode in config.EXPERIMENT_MODES:
                    current_mode = new_mode
                    print(f"✓ Mode changed to: {config.EXPERIMENT_MODES[new_mode]['name']}")
                else:
                    print(f"✗ Invalid mode. Available: {', '.join(config.EXPERIMENT_MODES.keys())}")
                continue
            
            elif user_input.lower() == 'stats':
                _print_stats(doc_id, chunks_metadata)
                continue
            
            try:
                answer_query(doc_id, chunks_metadata, user_input, current_mode)
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                logger.error(f"Query error: {e}", exc_info=True)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except EOFError:
            print("\nGoodbye!")
            break


def _print_stats(doc_id: UUID, chunks_metadata: List[Dict[str, Any]]):
    parent_count = sum(1 for c in chunks_metadata if c.get("is_parent"))
    child_count = sum(1 for c in chunks_metadata if not c.get("is_parent"))
    total_tokens = sum(c.get("token_count", 0) for c in chunks_metadata)
    pages = set(c.get("page_number") for c in chunks_metadata)
    
    try:
        db_stats = vector_db.get_collection_stats()
    except:
        db_stats = {}
    
    print("\n" + "=" * 80)
    print("DOCUMENT STATISTICS")
    print("=" * 80)
    print(f"Document ID:     {doc_id}")
    print(f"Total chunks:    {len(chunks_metadata)}")
    print(f"  - Parents:     {parent_count}")
    print(f"  - Children:    {child_count}")
    print(f"Total tokens:    {total_tokens:,}")
    print(f"Pages:           {len(pages)}")
    if db_stats:
        print(f"\nVector Database:")
        print(f"  - Total points: {db_stats.get('total_points', 'N/A')}")
        print(f"  - Status:       {db_stats.get('status', 'N/A')}")
    print("=" * 80 + "\n")


def main():
    print("\n" + "=" * 80)
    print("RAG SYSTEM - Enhanced 2026 Edition")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Embedding model: {config.EMBEDDING_MODEL}")
    print(f"  - LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
    print(f"  - Reranker: {config.RERANKER_MODEL}")
    print("=" * 80 + "\n")
    
    print("Available Modes:")
    print("  1. hybrid_quality  - Hybrid Quality-First (Precision, Groundedness, Numeric Accuracy)")
    print("  2. fast_response   - Fast Response (Speed optimized)")
    print()
    
    mode_choice = input("Select mode (1 or 2) [1]: ").strip() or "1"
    
    if mode_choice == "1":
        selected_mode = "hybrid_quality"
    elif mode_choice == "2":
        selected_mode = "fast_response"
    else:
        print("Invalid choice, using hybrid_quality")
        selected_mode = "hybrid_quality"
    
    print(f"\n✓ Selected mode: {config.EXPERIMENT_MODES[selected_mode]['name']}\n")
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter PDF path: ").strip()
    
    pdf_path = pdf_path.strip('"').strip("'")
    
    if not Path(pdf_path).exists():
        print(f"✗ Error: File not found: {pdf_path}")
        return 1
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"✗ Error: Only PDF files supported: {pdf_path}")
        return 1
    
    print(f"\nProcessing: {pdf_path}")
    print("This may take a minute...\n")
    
    try:
        doc_id, chunks_metadata = process_document(pdf_path)
        chunks_db[str(doc_id)] = chunks_metadata
        
        print(f"\n✓ Document processed successfully!")
        print(f"  Document ID: {doc_id}")
        print(f"  Total chunks: {len(chunks_metadata)}")
        
        interactive_mode(doc_id, chunks_metadata, selected_mode)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())