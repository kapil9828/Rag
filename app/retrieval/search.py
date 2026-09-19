from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.retrieval.index import COLLECTION_NAME, MODEL_NAME


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk_id: str
    filename: str
    page_number: int
    text: str
    source_type: str


class Retriever:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = COLLECTION_NAME,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def search(self, question: str, limit: int = 5) -> list[SearchResult]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")

        vector = self.model.encode(question, normalize_embeddings=True).tolist()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    score=float(point.score),
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    filename=str(payload.get("filename", "unknown")),
                    page_number=int(payload.get("page_number", 0)),
                    text=str(payload.get("text", "")),
                    source_type=str(payload.get("source_type", "unknown")),
                )
            )
        return results