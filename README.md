# DocuMind: A Hybrid Retrieval-Augmented Generation (RAG) Pipeline for Complex Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
Standard Retrieval-Augmented Generation (RAG) systems often struggle with long-form, structurally complex academic and financial documents, leading to hallucination, poor context retrieval, and loss of tabular data fidelity. **DocuMind** is an advanced, production-grade RAG pipeline designed to overcome these limitations. It introduces a multi-modal parsing strategy, semantic parent-child chunking, and a hybrid dense-sparse retrieval engine fortified by CrossEncoder reranking and numeric verification to guarantee highly calibrated, reliable, and verifiable AI-generated answers.

---

## Methodology & Architecture

DocuMind is built upon a modular, five-stage architecture prioritizing accuracy and trace-ability over naive retrieval.

### 1. Intelligent Multi-Modal Parsing (`pdf_parser.py`)
Documents are heterogeneous; a single parsing strategy is insufficient. Our parser dynamically routes pages based on a heuristic density profile:
- **PyMuPDF**: Extracts text layers and infers semantic roles (e.g., Headers, Lists, Paragraphs) via font metrics.
- **pdfplumber**: Isolates tables, converts them into strict Markdown, and subtracts them from the text flow to prevent spatial distortion.
- **Docling**: Acts as a fallback Optical Character Recognition (OCR) engine for scanned or image-dense pages.

### 2. Semantic Contextual Chunking (`chunker.py`)
Instead of fixed-size slicing, DocuMind employs a **Parent-Child Chunking Strategy**:
- **Child Chunks**: Small, semantically cohesive units (e.g., individual sentences, table rows) optimized for high retrieval precision and token limit management.
- **Parent Chunks**: Larger contextual windows that wrap the child chunks. During generation, the model is fed the parent context to ensure the semantic neighborhood is preserved, mitigating context fragmentation.

### 3. Hybrid Retrieval Engine (`embedder.py`, `retriever.py`, `database.py`)
Relying solely on dense embeddings fails on exact keyword matching (e.g., SKU numbers, specific acronyms). We use a dual-index approach via **Qdrant**:
- **Dense Vectors**: Semantic search powered by OpenAI's `text-embedding-3-large`.
- **Sparse Vectors (BM25)**: Lexical exact-match search generated natively over the corpus.
- **Fusion**: Search results are merged using **Reciprocal Rank Fusion (RRF)**, ensuring the best of both retrieval methodologies.

### 4. CrossEncoder Reranking
Initial retrieval produces a broad set of candidates. A CrossEncoder model (`ms-marco-MiniLM-L-6-v2`) evaluates the query-document pair, re-scoring and filtering out low-relevance chunks before they reach the generation layer.

### 5. Calibrated Generation (`generator.py`)
The LLM (`gpt-4o-mini`) is constrained by strict meta-prompts:
- **Numeric Verification**: Enforces exact continuous matching for statistics and financial figures.
- **Attribution**: Requires rigorous in-line citations mapped back to the source document and page number.
- **Confidence Scoring**: Outputs a calibrated confidence probability based on retrieval density and source verification, penalizing hallucination.

---

## Project Structure

```text
.
├── chunker.py           # Implements Semantic Parent-Child chunking
├── config.py            # Global hyperparameters, thresholds, and weights
├── database.py          # Qdrant Vector DB client and index management
├── embedder.py          # Dense/Sparse vector generation and caching
├── evaluate.py          # robust evaluation suite for QA benchmarking
├── generator.py         # LLM abstraction with citation and verification logic
├── main.py              # CLI entry point for ingestion and interactive querying
├── models.py            # Pydantic data schemas representing the layout
├── pdf_parser.py        # Multi-backend document parsing logic
├── retriever.py         # Hybrid retrieval and RRF reranking logic
└── requirements.txt     # Dependency graph
```

---

## Setup & Installation

**Prerequisites:** Python 3.12+

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/DocuMind-RAG.git
   cd DocuMind-RAG
   ```

2. **Create a Virtual Environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `key.env` file in the root directory. You will need API keys for OpenAI (and optionally Qdrant Cloud if not running locally):
   ```env
   OPENAI_API_KEY="sk-your-openai-key-here"
   QDRANT_HOST="localhost"
   QDRANT_PORT=6333
   ```

---

## Usage

### Ingestion & Interactive Mode
You can point the pipeline at a single PDF or an entire directory of PDFs. The system will build the BM25 index, generate dense embeddings, and start an interactive REPL shell.

```bash
python main.py path/to/your/financial_reports/
```

Inside the interactive shell:
```text
Query> What was the total revenue for FY2023?
Query> stats
Query> quit
```

## Evaluation Metrics

The system is rigorously evaluated against ground-truth datasets (e.g., FinanceBench) using the industry-standard **RAGAS framework**. Run `evaluate.py` to calculate:
- **Faithfulness**: Measures the factual consistency of the generated answer against the retrieved context. Penalizes hallucination.
- **Answer Relevancy**: Computes the extent to which the generated answer directly addresses the initial user query.
- **Context Precision**: Determines whether the most relevant chunks of context were highly ranked in the retrieval stage.
- **Context Recall**: Measures whether all aspects of the ground truth were successfully retrieved in the contextual windows.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [Qdrant](https://qdrant.tech/) for their high-performance vector database.
- [OpenAI](https://openai.com/) for generating dense embeddings and LLM reasoning.
- [Docling](https://github.com/DS4SD/docling) and [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for robust document processing.
