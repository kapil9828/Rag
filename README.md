# PDF RAG Chatbot

A local retrieval-augmented generation chatbot for PDF textbooks.

## Stack

- PyMuPDF and Tesseract OCR for PDF extraction
- `BAAI/bge-small-en-v1.5` for embeddings
- Qdrant for vector search
- Ollama with `qwen2.5:3b` for answer generation
- FastAPI and Streamlit for the application

## Run Locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Start Qdrant:

   ```powershell
   docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant
   ```

4. Download the Ollama model:

   ```powershell
   ollama pull qwen2.5:3b
   ```

5. Put PDFs in `data/pdfs/`, process them, and build the index:

   ```powershell
   Get-ChildItem data\pdfs -Filter *.pdf | ForEach-Object {
       $output = "data\processed\$($_.BaseName).jsonl"
       python -m app.ingestion.chunk $_.FullName --output $output
   }
   python -m app.retrieval.index --recreate
   ```

6. Run the backend and frontend in separate terminals:

   ```powershell
   python -m uvicorn app.main:app --reload --port 8000
   streamlit run streamlit_app.py
   ```

Open http://localhost:8501.

## Data

The local PDF corpus and generated JSONL files are intentionally excluded from Git. Add your own legally usable PDFs to `data/pdfs/` and regenerate the processed files and Qdrant index.