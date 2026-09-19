from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


@dataclass(frozen=True)
class PageText:
    document_id: str
    filename: str
    page_number: int
    text: str
    source_type: str


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_page(page: pymupdf.Page) -> str:
    """Render a scanned page and extract text with the native Tesseract binary."""
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image)


def extract_pdf(pdf_path: Path, min_native_chars: int = 100) -> list[PageText]:
    """Extract one searchable record per page, using OCR when native text is absent."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    document_id = pdf_path.stem
    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            native_text = clean_text(page.get_text("text"))
            if len(native_text) >= min_native_chars:
                text = native_text
                source_type = "native"
            else:
                text = clean_text(ocr_page(page))
                source_type = "ocr"

            if text:
                pages.append(
                    PageText(
                        document_id=document_id,
                        filename=pdf_path.name,
                        page_number=index,
                        text=text,
                        source_type=source_type,
                    )
                )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from a PDF with OCR fallback.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--min-native-chars", type=int, default=100)
    args = parser.parse_args()

    for page in extract_pdf(args.pdf, args.min_native_chars):
        print(asdict(page))


if __name__ == "__main__":
    main()