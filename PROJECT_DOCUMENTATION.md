# PDF RAG Chatbot

## Overview

This project is a Retrieval-Augmented Generation chatbot that answers questions from PDF textbooks. The current demo uses three OpenStax textbooks: Biology 2e, Chemistry 2e, and College Physics 2e.

The pipeline extracts PDF text, uses OCR for scanned pages, creates page-aware chunks, stores embeddings in Qdrant, retrieves relevant passages, and generates answers with a local Ollama model.

## Architecture

```text
PDFs -> PyMuPDF -> Tesseract OCR fallback -> page-aware chunks
     -> BAAI/bge-small-en-v1.5 -> Qdrant -> semantic retrieval
     -> Ollama qwen2.5:3b -> FastAPI -> Streamlit
```

## Technology Stack

- Python
- PyMuPDF and pytesseract
- Tesseract OCR
- `BAAI/bge-small-en-v1.5` embeddings
- Qdrant vector database
- Ollama with `qwen2.5:3b`
- FastAPI backend
- Streamlit frontend
- Docker Desktop

## Repository Structure

```text
app/
├── generation/answer.py   # Context prompt and Ollama generation
├── ingestion/extract.py   # Native extraction and OCR fallback
├── ingestion/chunk.py     # Page-aware chunking and JSONL export
├── retrieval/index.py     # Embeddings and Qdrant upload
├── retrieval/search.py    # Semantic question retrieval
└── main.py                # FastAPI endpoints
streamlit_app.py          # Chat interface
requirements.txt          # Python dependencies
```

## Prerequisites

Install Python, Docker Desktop, Tesseract OCR, and Ollama. Verify them from PowerShell:

```powershell
python --version
docker --version
tesseract --version
ollama --version
```

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

Start Qdrant:

```powershell
docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant
```

## Ingestion and Indexing

Put legally usable PDFs in `data/pdfs/`. Process them:

```powershell
Get-ChildItem data\pdfs -Filter *.pdf | ForEach-Object {
    $output = "data\processed\$($_.BaseName).jsonl"
    python -m app.ingestion.chunk $_.FullName --output $output
}
```

Build the Qdrant index:

```powershell
python -m app.retrieval.index --recreate --batch-size 64
```

Each indexed payload preserves the document ID, filename, page number, source type, chunk text, and vector.

## Run the Application

Use two PowerShell terminals.

Terminal 1:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

Terminal 2:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open the UI at:

```text
http://localhost:8501
```

FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

## API

Health check:

```text
GET /health
```

Ask a question:

```text
POST /query
```

Example request:

```json
{
  "question": "What is cellular respiration?",
  "top_k": 5
}
```

The backend retrieves relevant passages and sends them to Ollama with instructions to answer only from the supplied context. The frontend displays the answer without exposing technical retrieval metadata.

## Example Questions

- What is cellular respiration?
- What is photosynthesis?
- What is the difference between ionic and covalent bonds?
- What is Newton's second law?
- Explain kinetic energy.
- How are cellular respiration and photosynthesis related?

## Current Validation

```text
Documents: 3
Indexed chunks: 4,964
Vector collection: pdf_chunks
Embedding dimension: 384
```

Verified features include native extraction, OCR fallback, deterministic chunking, embedding generation, Qdrant search, FastAPI health checks, Streamlit startup, and Ollama generation.

## Limitations

- The original challenge asks for at least 10 PDFs; this demo currently uses 3 PDFs.
- CPU-only generation currently takes longer than the original 2-5 second target.
- The local PDF corpus and generated JSONL files are excluded from GitHub.
- Qdrant and Ollama must be running locally.
- OCR quality depends on scan quality and page layout.

For production or judging deployment, use a GPU or faster generation model and measure p50 and p95 latency with a fixed evaluation set.

## GitHub

Repository: https://github.com/kapil9828/Rag

The repository contains source code and setup instructions, but not the virtual environment, PDFs, processed files, model caches, or secrets.
