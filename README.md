# DocuMind

> A production-grade Retrieval-Augmented Generation (RAG) system for intelligent PDF question answering — with hybrid search, parent-child chunking, atomic fact verification, and calibrated confidence scoring.

---

## Overview

DocuMind is a full-stack RAG pipeline that ingests PDF documents and answers natural language queries with grounded, citation-backed responses. It combines dense semantic embeddings and sparse BM25 retrieval into a hybrid search engine, reranks candidates with a cross-encoder, and generates answers using GPT-4o-mini — all while verifying numeric accuracy and measuring hallucination at the atomic fact level.

Built for precision. Designed for reliability.

---

## Features

### Retrieval
- **Hybrid Search** — Fuses dense (OpenAI `text-embedding-3-large`) and sparse (BM25) retrieval via Reciprocal Rank Fusion (RRF)
- **Parent-Child Chunking** — Child chunks are indexed for precision retrieval; parent context windows expand the answer context for richer generation
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` reranks initial candidates for relevance
- **LLM Listwise Reranking** — Optional GPT-4o-mini listwise reranking pass for maximum precision
- **Query-Adaptive Weights** — Automatically shifts dense/sparse balance based on whether the query is keyword-style or conceptual
- **Query Rewriting** — Optionally expands and refines queries before retrieval
- **MMR Diversification** — Maximal Marginal Relevance reduces redundancy in retrieved chunks
- **Semantic & Jaccard Deduplication** — Removes near-duplicate passages before generation

### Generation
- **Atomic Fact Verification** — Decomposes answers into individual claims and verifies each against retrieved context
- **Numeric Accuracy Checking** — Detects and flags numbers in the answer that don't appear in the source context
- **Citation Analysis** — Measures citation recall, precision, and F1; penalizes hallucinated page references
- **Calibrated Confidence Scoring** — Multi-factor confidence score combining verification, source quality, citation metrics, and retrieval confidence
- **Abstention** — Refuses to answer when retrieval confidence falls below a configurable threshold

### Infrastructure
- **Qdrant Vector Database** — Stores both dense and sparse vectors with metadata filtering
- **Docling PDF Parser** — Layout-aware extraction of text, tables, code, headers, equations, and footers
- **OpenAI Embeddings** — Batched, cached, and normalized dense vectors
- **Evaluation Suite** — Automated precision, recall, F1, MRR, answer similarity, and citation quality across multiple modes

---

## Architecture

```
PDF
 │
 ▼
┌─────────────────┐
│   parser.py     │  Docling layout extraction → LayoutElements
│  (EnhancedPDFParser) │  (text, tables, headers, code, equations)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   chunker.py    │  Parent-child chunking with sentence-aware splitting
│ (EnhancedContextualChunker) │  Numeric extraction, overlap handling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   embedder.py   │  Dense: OpenAI text-embedding-3-large (3072-dim)
│ (EnhancedEmbedder)   │  Sparse: BM25 with custom tokenizer + IDF weighting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   database.py   │  Qdrant — hybrid dense+sparse collection
│(EnhancedVectorDatabase)│  RRF fusion, metadata filtering
└────────┬────────┘
         │
    Query Time
         │
         ▼
┌─────────────────┐
│  retriever.py   │  Hybrid search → reranking → MMR → dedup
│(EnhancedRetriever)   │  Parent context expansion, numeric boosting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  generator.py   │  GPT-4o-mini generation with quote-first prompting
│(EnhancedGenerator)   │  Atomic verification, numeric check, citation scoring
└────────┬────────┘
         │
         ▼
      Answer + Confidence + Citations
```

---

## Retrieval Modes

DocuMind ships with two pre-configured experiment modes:

| Setting | `hybrid_quality` | `fast_response` |
|---|---|---|
| Dense retrieval | ✅ | ✅ |
| Sparse retrieval | ✅ | ❌ |
| Cross-encoder rerank | ✅ | ❌ |
| LLM listwise rerank | ✅ | ❌ |
| Query rewriting | ✅ | ❌ |
| MMR diversification | ✅ | ❌ |
| Atomic fact verification | ✅ | ❌ |
| Numeric verification | ✅ | ❌ |
| Parent-child context | ✅ | ❌ |
| Initial top-k | 25 | 3 |
| Rerank top-k | 12 | 3 |

---

## Installation

**Requirements:** Python 3.10+, a running Qdrant instance (or in-memory mode), an OpenAI API key.

```bash
git clone https://github.com/your-username/documind.git
cd documind
pip install -r requirements.txt
```

**Dependencies include:**
```
openai
qdrant-client
docling
docling-core
sentence-transformers
tiktoken
pydantic
python-dotenv
numpy
```

---

## Configuration

Copy `key.env` and fill in your credentials:

```env
OPENAI_API_KEY=sk-...

# Qdrant (leave blank for in-memory mode)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documind_v2
QDRANT_API_KEY=

# Chunking
CHUNK_SIZE_PARENT=600
CHUNK_SIZE_CHILD=300
CHUNK_OVERLAP=75

# Retrieval
TOP_K_INITIAL=30
TOP_K_RERANK=10
DENSE_WEIGHT=0.65
SPARSE_WEIGHT=0.35

# Models
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o-mini
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Mode
DEFAULT_MODE=hybrid_quality
```

All config values have sensible defaults and are validated at startup.

---

## Usage

### Interactive Q&A

```bash
python main.py path/to/document.pdf
```

You'll be prompted to select a retrieval mode, then enter the interactive query shell:

```
Query> What was the revenue in Q3 2024?
Query> mode fast_response
Query> stats
Query> quit
```

### Programmatic Usage

```python
from main import process_document, answer_query

doc_id, chunks_metadata = process_document("report.pdf")

result = answer_query(doc_id, chunks_metadata, "What are the key risks?")

print(result["answer"])
print(f"Confidence: {result['confidence']:.1%}")
print(f"Verified: {result['verified']}")
```

### Evaluation

Prepare a JSON test dataset:

```json
[
  {
    "query": "What was net income in FY2023?",
    "expected_answer": "Net income was $1.2 billion in FY2023.",
    "relevant_pages": [4, 5]
  }
]
```

Run evaluation:

```bash
python evaluate.py
```

DocuMind will benchmark both `hybrid_quality` and `fast_response` modes and report:

```
Hybrid Quality-First:
  Precision: 87.3%  Recall: 91.2%  F1: 89.2%
  MRR: 0.847  Accuracy: 84.1%  Confidence: 79.3%
  Citation: 82.4% recall, 88.6% precision
  Time: 4.21s

Fast Response:
  Precision: 71.0%  Recall: 74.5%  F1: 72.7%
  MRR: 0.703  Accuracy: 70.2%  Confidence: 68.1%
  Citation: 65.1% recall, 71.2% precision
  Time: 0.89s
```

---

## Project Structure

```
documind/
├── main.py          # Entry point — document processing + interactive Q&A
├── parser.py        # Docling-based PDF layout parser
├── chunker.py       # Parent-child chunking with numeric extraction
├── embedder.py      # Dense (OpenAI) + sparse (BM25) embeddings
├── database.py      # Qdrant vector DB — indexing and hybrid search
├── retriever.py     # Full retrieval pipeline — search, rerank, MMR, dedup
├── generator.py     # GPT-4o-mini generation with verification
├── evaluate.py      # Benchmarking suite for retrieval and generation
├── models.py        # Pydantic data models
├── config.py        # Centralized configuration with validation
└── key.env          # API keys and environment variables
```

---

## How Confidence Scoring Works

DocuMind computes a multi-factor calibrated confidence score for every answer:

```
confidence = 0.25 × verification_score
           + 0.25 × source_quality_score
           + 0.30 × citation_f1
           + 0.20 × retrieval_confidence

× atomic_fact_support_rate
× numeric_accuracy_penalty (if numbers mismatch)
× hallucination_penalty^(n_hallucinated_sentences)
```

This means confidence degrades measurably when the answer contains unverifiable numbers, unsupported claims, or page citations that don't match the context — not just when retrieval scores are low.

---

## Qdrant Setup

**Local (Docker):**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Cloud:** Set `QDRANT_HOST`, `QDRANT_PORT`, and `QDRANT_API_KEY` in `key.env`.

**In-memory (no setup):** Leave `QDRANT_HOST=localhost` with no running instance — DocuMind will automatically fall back to an in-memory Qdrant client.

---

## Limitations

- PDF only — no DOCX, HTML, or image-only scans without OCR configured in Docling
- Single-document queries — cross-document retrieval is not currently supported
- Qdrant required — other vector databases are not pluggable without modifying `database.py`
- OpenAI dependency — both embeddings and generation use OpenAI; swapping providers requires changes to `embedder.py` and `generator.py`

---

## License

MIT
