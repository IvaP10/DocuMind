# DocuMind: Multimodal RAG & Document Intelligence

A sophisticated Retrieval-Augmented Generation (RAG) system designed to extract and answer questions from complex PDF documents while preserving layout structure and minimizing hallucinations.

## 🎯 What Does This Do?

DocuMind helps you:
- Upload PDF documents and ask questions about them
- Get accurate answers with verifiable citations
- Handle complex layouts including tables and multi-column text
- Reduce AI hallucinations through built-in verification

## 🏗️ System Architecture

### Parent-Child Chunking Strategy
- **Parent Chunks**: Large contextual blocks (1500 tokens) for comprehensive understanding
- **Child Chunks**: Smaller searchable units (300 tokens) for precise retrieval
- This hierarchical approach balances search precision with context preservation

### Hybrid Search
- **Dense Vectors**: Semantic understanding using sentence transformers
- **Sparse Vectors**: Keyword-based matching for exact term retrieval
- **Reranking**: Cross-encoder model refines top results for accuracy

### Verification System
- Audits generated answers against retrieved evidence
- Checks for unsupported claims and hallucinations
- Provides confidence scores and citation metrics

## 📊 Performance Metrics

- **MRR (Mean Reciprocal Rank)**: 0.71
- **Recall@10**: 80%
- **End-to-end QA Accuracy**: 72%
- **Answer Faithfulness**: 76%
- **Hallucination Rate**: 14%

## 🚀 Quick Start

### Installation

1. Clone the repository
2. Create a `key.env` file with your OpenAI API key
3. Install dependencies: `pip install -r requirements.txt`
4. Start Qdrant vector database (Docker recommended)

### Usage

Run `python main.py` to start the interactive mode. Enter your PDF path and ask questions about the document.

## 📁 Project Structure

```
documind/
│
├── main.py              # Main entry point for interactive queries
├── evaluate.py          # Evaluation system with multiple test modes
├── parser.py            # PDF parsing with Docling (handles tables & layout)
├── chunker.py           # Parent-child chunking strategy
├── embedder.py          # Dense + sparse vector generation
├── database.py          # Qdrant vector database operations
├── retriever.py         # Hybrid search + reranking
├── generator.py         # LLM answer generation + verification
├── models.py            # Pydantic data models
└── config.py            # Configuration settings
```

## ⚙️ Configuration

Key settings in `config.py`:
- Chunk sizes: Parent (1500 tokens), Child (300 tokens)
- Retrieval: Top-50 initial candidates, Top-5 after reranking
- Models: GPT-4-mini, all-mpnet-base-v2, ms-marco-MiniLM

## 🧪 Experiment Modes

The system supports four retrieval configurations:
1. **hybrid_full** - All features enabled (best performance)
2. **dense_only** - Dense vectors with reranking
3. **no_rerank** - Hybrid search without reranking
4. **no_parent** - No parent-child retrieval

Run `python evaluate.py` to compare all modes.

## 🔍 How It Works

1. **Document Processing**
   - Parse PDF with Docling to preserve tables and layout
   - Create parent chunks (large context) and child chunks (searchable units)
   - Generate dense embeddings and sparse vectors for child chunks

2. **Query Processing**
   - Embed user query with dense and sparse representations
   - Search vector database for top-k child chunks
   - Rerank results using cross-encoder
   - Retrieve parent chunks for full context

3. **Answer Generation**
   - Feed context to LLM with strict grounding instructions
   - Generate answer with page citations
   - Verify answer against retrieved evidence
   - Calculate confidence score and citation metrics

4. **Verification Loop**
   - Check each claim against source material
   - Identify unsupported claims and potential hallucinations
   - Provide transparency through citation analysis

## 📈 Evaluation Metrics

The system tracks comprehensive metrics:

**Retrieval Metrics:**
- Precision, Recall, F1 Score
- Mean Reciprocal Rank (MRR)

**Generation Metrics:**
- Answer similarity to ground truth
- Confidence scores
- Verification rate
- Citation precision and recall

**Performance Metrics:**
- Retrieval latency
- Generation latency
- Total query time

## 🛠️ Tech Stack

- **Python** - Core language
- **Qdrant** - Vector database for hybrid search
- **Docling** - PDF parsing with layout preservation
- **SentenceTransformers** - Dense embeddings
- **OpenAI API** - LLM for answer generation
- **CrossEncoder** - Reranking model

---

**Note**: This project was developed as part of research into layout-aware document retrieval systems. The parent-child chunking strategy and verification pipeline are designed to balance retrieval precision with context preservation while minimizing hallucinations in document Q&A systems.
