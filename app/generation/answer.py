from __future__ import annotations

from dataclasses import dataclass

import ollama

from app.retrieval.search import SearchResult


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sources: list[dict[str, object]]


SYSTEM_PROMPT = """You are a document-grounded assistant.
Use only the supplied context to answer the user's question.
If the context does not contain the answer, say exactly:
I could not find that information in the provided documents.
Do not invent facts or citations. Keep the answer concise and answer directly
without displaying source names, page numbers, or citation markers."""


def build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"SOURCE: {result.filename}, page {result.page_number}\n{result.text}"
        for result in results
    )


def answer_question(
    question: str,
    results: list[SearchResult],
    model: str = "qwen2.5:3b",
) -> GeneratedAnswer:
    if not results:
        return GeneratedAnswer(
            answer="I could not find that information in the provided documents.",
            sources=[],
        )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{build_context(results)}\n\nQuestion: {question}",
            },
        ],
        options={"temperature": 0, "num_predict": 256},
    )
    sources = [
        {
            "filename": result.filename,
            "page_number": result.page_number,
            "score": round(result.score, 4),
        }
        for result in results
    ]
    return GeneratedAnswer(answer=response["message"]["content"].strip(), sources=sources)