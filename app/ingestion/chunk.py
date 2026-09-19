from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ingestion.extract import PageText, extract_pdf


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    source_type: str
    token_count: int


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def chunk_page(
    page: PageText,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[TextChunk]:
    """Split one page into deterministic overlapping token windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    tokens = tokenize(page.text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunk_number = len(chunks) + 1
        chunks.append(
            TextChunk(
                chunk_id=f"{page.document_id}-p{page.page_number}-c{chunk_number}",
                document_id=page.document_id,
                filename=page.filename,
                page_number=page.page_number,
                text=" ".join(window),
                source_type=page.source_type,
                token_count=len(window),
            )
        )
        if start + chunk_size >= len(tokens):
            break
    return chunks


def chunk_pdf(
    pdf_path: Path,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in extract_pdf(pdf_path):
        chunks.extend(chunk_page(page, chunk_size, overlap))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk a PDF into JSON records.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    chunks = chunk_pdf(args.pdf, args.chunk_size, args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()