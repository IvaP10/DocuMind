# DocuMind RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with advanced retrieval techniques, parent-child chunking, and multi-modal optimization strategies.

## 🎯 Overview

DocuMind is an enterprise-grade RAG system designed for accurate question-answering over PDF documents. It combines state-of-the-art document processing, hybrid retrieval (dense + sparse), intelligent reranking, and LLM-powered generation with built-in verification.

### Key Features

- **Advanced Document Processing**: Docling-based PDF parsing with layout-aware extraction
- **Parent-Child Chunking**: Hierarchical chunking strategy for better context preservation
- **Hybrid Retrieval**: Combines dense (semantic) and sparse (BM25) search
- **Intelligent Reranking**: Cross-encoder reranking with Cohere API support
- **Multiple Optimization Modes**: 8 pre-configured modes for different use cases
- **Answer Verification**: Atomic fact verification with confidence scoring
- **Citation Tracking**: Automatic source attribution with quality metrics
- **Comprehensive Evaluation**: Built-in evaluation framework with multiple metrics

## 📊 Performance Metrics

Based on evaluation results (`resume_metrics.json`):

| Metric | Score |
|--------|-------|
| **Retrieval F1** | 63.01% |
| **Precision** | 55.38% |
| **Recall** | 83.87% |
| **Answer Accuracy** | 61.01% |
| **Verification Rate** | 83.87% |
| **Avg Query Time** | 3.14s |
| **MRR** | 0.7796 |

## 🏗️ Architecture

```
┌─────────────┐
│   PDF Doc   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Parser         │  ← Docling (layout-aware)
│  (parser.py)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Chunker        │  ← Parent-child hierarchical chunking
│  (chunker.py)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Embedder       │  ← Dense + Sparse embeddings
│  (embedder.py)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Vector DB      │  ← Qdrant (hybrid search)
│  (database.py)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Retriever      │  ← Hybrid search + Reranking + MMR
│  (retriever.py) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Generator      │  ← LLM + Verification
│  (generator.py) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│    Answer       │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt

# Install Qdrant (vector database)
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Configuration

Create a `key.env` file with your API keys:

```env
# Required for embeddings (choose one)
OPENAI_API_KEY=sk-...
# or use local models (no key needed)

# Required for LLM generation
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...

# Optional: For reranking
COHERE_API_KEY=...

# Qdrant settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documind_v2
```

### Basic Usage

```python
from main import process_document, answer_query

# 1. Process a PDF document
doc_id, chunks_metadata = process_document("path/to/document.pdf")

# 2. Ask questions
result = answer_query(
    doc_id=doc_id,
    chunks_metadata=chunks_metadata,
    query="What is the main conclusion?",
    mode="precision_optimized"
)

print(result['answer'])
print(f"Confidence: {result['confidence']:.1%}")
```

### Command Line Interface

```bash
# Interactive mode
python main.py path/to/document.pdf

# Or let it prompt you
python main.py
```

## 📋 Configuration Options

### Retrieval Modes

The system includes 8 optimization modes in `config.py`:

| Mode | Description | Use Case |
|------|-------------|----------|
| `precision_optimized` | High quality, strict filtering | When accuracy is critical |
| `hybrid_optimized` | Balanced quality/speed | General purpose |
| `hybrid_full` | Complete pipeline | Comprehensive analysis |
| `dense_only` | Semantic search only | Conceptual queries |
| `no_rerank` | Skip reranking step | Speed optimization |
| `no_parent` | Child chunks only | Fine-grained retrieval |
| `recall_max` | Maximum coverage | Exploratory queries |
| `latency_optimized` | Fastest responses | Real-time applications |

### Key Configuration Parameters

```python
# Chunking
CHUNK_SIZE_PARENT = 600      # Parent chunk size (tokens)
CHUNK_SIZE_CHILD = 200       # Child chunk size (tokens)
CHUNK_OVERLAP = 20           # Overlap between chunks

# Retrieval
TOP_K_INITIAL = 50           # Initial candidates
TOP_K_RERANK = 5             # After reranking
MAX_CONTEXT_TOKENS = 3500    # Max context for LLM

# Hybrid Search Weights
DENSE_WEIGHT = 0.7           # Dense search weight
SPARSE_WEIGHT = 0.3          # Sparse search weight

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
USE_OPENAI_EMBEDDINGS = False

# LLM
LLM_PROVIDER = "openai"      # or "anthropic"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
```

## 🧪 Evaluation

The system includes a comprehensive evaluation framework:

```bash
python evaluate.py
```

### Evaluation Metrics

**Retrieval Metrics:**
- Precision: Relevance of retrieved chunks
- Recall: Coverage of relevant information
- F1 Score: Harmonic mean of precision and recall
- MRR: Mean Reciprocal Rank

**Generation Metrics:**
- Answer Accuracy: Semantic similarity to ground truth
- Confidence: Model's confidence in the answer
- Verification Rate: Percentage of verified answers
- Citation Quality: Precision and recall of citations

### Creating Test Datasets

```json
[
  {
    "query": "What is the main finding?",
    "expected_answer": "The study found that...",
    "relevant_pages": [1, 3, 5]
  }
]
```

## 📁 Project Structure

```
.
├── config.py           # Configuration and experiment modes
├── models.py           # Pydantic data models
├── parser.py           # PDF parsing (Docling)
├── chunker.py          # Parent-child chunking
├── embedder.py         # Dense + sparse embeddings
├── database.py         # Qdrant vector database
├── retriever.py        # Hybrid retrieval + reranking
├── generator.py        # LLM generation + verification
├── main.py             # Main application entry point
├── evaluate.py         # Evaluation framework
└── README.md           # This file
```

## 🔧 Advanced Features

### Parent-Child Chunking

The system uses hierarchical chunking:
- **Parent chunks** (600 tokens): Provide broader context
- **Child chunks** (200 tokens): Enable precise retrieval
- Strategy: Retrieve children, expand to parents for context

### Hybrid Search

Combines two search paradigms:
```python
# Dense (semantic): Captures meaning
dense_score = cosine_similarity(query_embedding, chunk_embedding)

# Sparse (BM25): Captures keywords
sparse_score = BM25(query_tokens, chunk_tokens)

# Combined with RRF (Reciprocal Rank Fusion)
final_score = combine_with_rrf([dense_results, sparse_results])
```

### Reranking

Two-stage retrieval:
1. Initial retrieval: Fast, get top-K candidates
2. Reranking: Slow but accurate, cross-encoder scoring

### MMR (Maximal Marginal Relevance)

Reduces redundancy while maintaining relevance:
```python
MMR = λ * relevance - (1-λ) * max_similarity_to_selected
```

### Answer Verification

Multi-level verification:
1. **Atomic Fact Extraction**: Break answer into claims
2. **Citation Verification**: Check each claim against sources
3. **Confidence Calibration**: Weighted scoring
4. **Abstention**: Refuse low-confidence answers

## 🎛️ API Reference

### Core Functions

#### `process_document(pdf_path: str)`
Process a PDF document through the entire pipeline.

**Returns:** `(doc_id: UUID, chunks_metadata: List[Dict])`

#### `answer_query(doc_id, chunks_metadata, query, mode)`
Answer a query using the RAG pipeline.

**Parameters:**
- `doc_id`: Document identifier
- `chunks_metadata`: All chunks metadata
- `query`: User question (string)
- `mode`: Retrieval mode (default: "precision_optimized")

**Returns:** `Dict` with answer, confidence, sources, etc.

### Retriever API

```python
from retriever import retriever

context_data = retriever.retrieve_context(
    document_id=doc_id,
    query="What is X?",
    chunks_metadata=chunks_metadata,
    mode="hybrid_optimized"
)
```

### Generator API

```python
from generator import generator

answer_data = generator.generate_answer(
    query="What is X?",
    context_data=context_data
)
```

## 🔍 Troubleshooting

### Common Issues

**1. Qdrant Connection Error**
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**2. Out of Memory**
```python
# Reduce batch size
EMBEDDING_BATCH_SIZE = 16  # Default: 32
BATCH_SIZE = 16
```

**3. Slow Processing**
```python
# Use faster mode
DEFAULT_MODE = "latency_optimized"

# Or reduce reranking
TOP_K_RERANK = 3  # Default: 5
```

**4. Low Quality Answers**
```python
# Use precision mode
DEFAULT_MODE = "precision_optimized"

# Increase context
MAX_CONTEXT_TOKENS = 5000  # Default: 3500
```

## 📊 Benchmarking

Run benchmarks across all modes:

```python
from evaluate import RAGEvaluator

evaluator = RAGEvaluator()
doc_id, chunks = evaluator.process_document("doc.pdf")
test_data = evaluator.load_test_dataset("test.json")

results = evaluator.evaluate_all_modes(doc_id, chunks, test_data)
evaluator.print_results(results)
evaluator.save_results(results, "results.json")
```

## 🛠️ Customization

### Adding a New Retrieval Mode

In `config.py`:

```python
EXPERIMENT_MODES["my_custom_mode"] = {
    "name": "My Custom Mode",
    "use_dense": True,
    "use_sparse": True,
    "use_reranker": True,
    "use_parent_child": "child_with_parent_context",
    "apply_score_threshold": True,
    "mmr_lambda": 0.7,
    # ... other settings
}
```

### Using a Different LLM

```python
# In config.py
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-3-5-sonnet-20241022"

# Or OpenAI
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o"
```

### Custom Embedding Models

```python
# In config.py
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Or OpenAI
USE_OPENAI_EMBEDDINGS = True
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
```

## 📈 Performance Optimization

### For Speed
- Use `latency_optimized` mode
- Reduce `TOP_K_INITIAL` and `TOP_K_RERANK`
- Disable query rewriting: `ENABLE_QUERY_REWRITING = False`
- Use smaller embedding model

### For Quality
- Use `precision_optimized` mode
- Increase `TOP_K_INITIAL` and `MAX_CONTEXT_TOKENS`
- Enable all features: reranking, MMR, verification
- Use larger embedding and LLM models

### For Scale
- Enable batch processing: `ENABLE_BATCH_PROCESSING = True`
- Increase `EMBEDDING_BATCH_SIZE`
- Use Qdrant cloud for distributed deployment
- Implement caching: `ENABLE_EMBEDDING_CACHE = True`

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Multi-document support**: Query across multiple PDFs
2. **Streaming responses**: Real-time answer generation
3. **Advanced evaluation**: RAGAS, TruLens integration
4. **UI/Frontend**: Web interface for the system
5. **Additional retrieval strategies**: Query decomposition, HyDE
6. **Multimedia support**: Images, tables as separate modalities

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Docling**: Layout-aware PDF parsing
- **Qdrant**: Vector database
- **Sentence Transformers**: Embedding models
- **OpenAI/Anthropic**: LLM providers
- **Cohere**: Reranking API

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the evaluation metrics for optimization guidance

## 🔄 Version History

**v2.0** (Current)
- Parent-child chunking
- Hybrid retrieval (dense + sparse)
- Multi-mode optimization
- Answer verification
- Comprehensive evaluation

**v1.0**
- Basic RAG pipeline
- Dense retrieval only
- Single-mode operation

---

**Built with ❤️ for accurate, reliable document Q&A**
