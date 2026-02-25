# DocuMind

Retrieval-Augmented Generation for PDF question answering. Hybrid search, parent-child chunking, atomic fact verification, and calibrated confidence scoring.

---

## How it works

DocuMind parses PDFs with Docling, chunks them into parent-child pairs, embeds with OpenAI + BM25, indexes into Qdrant, and answers queries with GPT-4o-mini. Every answer is verified at the atomic fact level and scored for confidence.

```
PDF → Parse → Chunk → Embed → Index → [Query] → Hybrid Search → Rerank → Generate → Verify
```

---

## Features

**Retrieval**
- Hybrid dense (OpenAI `text-embedding-3-large`) + sparse (BM25) search via RRF
- Query-adaptive weights — shifts to 40/60 dense/sparse for numeric/keyword queries
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) with adaptive score filtering
- Parent-child chunking — child chunks retrieved, parent context used for generation
- Numeric boosting — chunks matching query numbers ranked higher
- Jaccard deduplication before generation

**Generation**
- GPT-4o-mini with mandatory `[[Source: file | Page: N]]` citation per claim
- Atomic fact verification via a secondary LLM call
- Numeric accuracy check — flags numbers in the answer absent from source context
- Multi-factor confidence score (verification + source quality + citation F1 + retrieval confidence)
- Heuristic word-overlap fallback if LLM verification fails

**Infrastructure**
- Qdrant for hybrid dense+sparse vector storage (auto falls back to in-memory)
- SHA-256 disk cache for embeddings — zero API calls on repeated runs
- Multi-document ingestion — entire folders with a shared global BM25 index
- Evaluation suite — precision, recall, F1, MRR, answer similarity, citation quality

---

## Quickstart

```bash
git clone https://github.com/your-username/documind.git
cd documind
pip install -r requirements.txt
cp key.env.example key.env  # add your OPENAI_API_KEY
```

**Single PDF**
```bash
python main.py report.pdf
```

**Folder of PDFs**
```bash
python main.py reports/
```

**Interactive shell**
```
Query> What was revenue in Q3 2024?
Query> stats
Query> quit
```

---

## Configuration

`key.env`:

```env
OPENAI_API_KEY=sk-...

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documind_v2
QDRANT_API_KEY=           # leave blank for local/in-memory

CHUNK_SIZE_PARENT=600
CHUNK_SIZE_CHILD=300
CHUNK_OVERLAP=75

TOP_K_INITIAL=30
TOP_K_RERANK=15
DENSE_WEIGHT=0.70
SPARSE_WEIGHT=0.30
MAX_CONTEXT_TOKENS=4000

EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o-mini
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

ENABLE_EMBEDDING_CACHE=true
```

---

## Programmatic usage

```python
from main import ingest_single_pdf, answer_query

doc_ids, chunks_metadata = ingest_single_pdf("report.pdf")
result = answer_query(chunks_metadata, "What are the key risks?")

print(result["answer"])
print(f"Confidence: {result['confidence']:.1%}")
print(f"Verified:   {result['verified']}")
```

---

## Evaluation

Prepare a JSON dataset:

```json
[
  {
    "query": "What was net income in FY2023?",
    "expected_answer": "Net income was $1.2 billion in FY2023.",
    "relevant_pages": [4, 5]
  }
]
```

```bash
python evaluate.py
```

Output:

```
Retrieval  — Precision: 87.3%  Recall: 91.2%  F1: 89.2%  MRR: 0.847
Generation — Accuracy: 84.1%  Confidence: 79.3%  Verified: 91.0%
Citation   — Recall: 82.4%  Precision: 88.6%
Speed      — Avg: 4.21s
```

---

## Confidence scoring

```
confidence = 0.30 × verification_score
           + 0.25 × source_quality_score
           + 0.25 × citation_f1
           + 0.20 × retrieval_confidence

           × atomic_fact_support_rate
           × 0.85  (if numeric mismatches)
           × 0.98ⁿ (n = hallucinated sentences)
```

---

## Project structure

```
documind/
├── main.py        # Ingestion + interactive Q&A
├── parser.py      # Docling PDF parser
├── chunker.py     # Parent-child chunking
├── embedder.py    # OpenAI dense + BM25 sparse embeddings
├── database.py    # Qdrant indexing and search
├── retriever.py   # Hybrid search, reranking, dedup
├── generator.py   # Generation + verification + confidence
├── evaluate.py    # Evaluation suite
├── models.py      # Pydantic models
├── config.py      # Configuration
└── key.env        # API keys
```

---

## Qdrant

```bash
# Local
docker run -p 6333:6333 qdrant/qdrant

# Cloud — set QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY in key.env

# In-memory — leave QDRANT_HOST=localhost with no running instance
```

---

## Requirements

Python 3.10+ · OpenAI API key · Qdrant (optional, falls back to in-memory)

```
openai>=1.0.0
qdrant-client>=1.7.0
sentence-transformers>=2.2.0
tiktoken>=0.5.0
docling>=1.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

---

## Limitations

- PDF only — no DOCX or image-only scans without Docling OCR configured
- OpenAI only — swapping embedding/generation providers requires changes to `embedder.py` and `generator.py`
- BM25 index rebuilds on every ingestion run — incremental updates not supported
- Single collection — collection is reset on each ingestion

