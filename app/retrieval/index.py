from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "pdf_chunks"
VECTOR_SIZE = 384
CHUNK_NAMESPACE = uuid.UUID("f4b7f60e-e50b-43c4-a4c2-32a6d8e6c2a2")


def load_chunks(processed_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(processed_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
                if not chunk.get("text") or not chunk.get("chunk_id"):
                    continue
                chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No usable JSONL chunks found in {processed_dir}")
    return chunks


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def index_chunks(
    processed_dir: Path,
    qdrant_url: str = "http://localhost:6333",
    batch_size: int = 64,
    recreate: bool = False,
) -> int:
    chunks = load_chunks(processed_dir)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    client = QdrantClient(url=qdrant_url)
    ensure_collection(client, recreate=recreate)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = model.encode(
            [chunk["text"] for chunk in batch],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        points = [
            models.PointStruct(
                id=point_id(chunk["chunk_id"]),
                vector=vector.tolist(),
                payload=chunk,
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        print(f"Indexed {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed processed chunks and upload them to Qdrant.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    count = index_chunks(
        args.processed_dir,
        args.qdrant_url,
        args.batch_size,
        args.recreate,
    )
    print(f"Indexed {count} chunks in Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()