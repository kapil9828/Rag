from __future__ import annotations

import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.generation.answer import answer_question
from app.retrieval.search import Retriever


app = FastAPI(title="PDF RAG Chatbot", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    model: str = "qwen2.5:3b"


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


@app.get("/health")
def health() -> dict[str, object]:
    client = QdrantClient(url="http://localhost:6333")
    collection = client.get_collection("pdf_chunks")
    return {"status": "ok", "collection": "pdf_chunks", "points": collection.points_count}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, object]:
    started = time.perf_counter()
    try:
        results = get_retriever().search(request.question, request.top_k)
        generated = answer_question(request.question, results, request.model)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "answer": generated.answer,
        "sources": generated.sources,
        "retrieved_chunks": [
            {
                "score": round(result.score, 4),
                "filename": result.filename,
                "page_number": result.page_number,
                "text": result.text,
                "source_type": result.source_type,
            }
            for result in results
        ],
        "latency_ms": latency_ms,
    }