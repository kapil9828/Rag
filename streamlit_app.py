from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="PDF Research Assistant", page_icon="📚", layout="wide")


def get_health() -> dict[str, object] | None:
    try:
        with urlopen(f"{API_URL}/health", timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def ask_api(question: str, top_k: int) -> dict[str, object]:
    body = json.dumps({"question": question, "top_k": top_k}).encode("utf-8")
    request = Request(
        f"{API_URL}/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


st.title("PDF Research Assistant")
st.caption("Ask questions grounded in the indexed textbook collection.")

with st.sidebar:
    st.header("Index status")
    health = get_health()
    if health:
        st.success("API and Qdrant online")
        st.metric("Indexed chunks", health.get("points", 0))
        st.caption(f"Collection: {health.get('collection', 'unknown')}")
    else:
        st.error("API unavailable")
        st.caption(f"Start FastAPI at {API_URL}")
    top_k = st.slider("Retrieved passages", min_value=1, max_value=10, value=5)

    st.header("FAQs")
    st.caption("Try one of these questions")
    faq_questions = [
        "What is cellular respiration?",
        "What is photosynthesis?",
        "What is the difference between ionic and covalent bonds?",
        "What is Newton's second law?",
        "Explain kinetic energy.",
    ]
    for faq_question in faq_questions:
        st.button(
            faq_question,
            key=f"faq_{faq_question}",
            use_container_width=True,
            on_click=lambda question=faq_question: st.session_state.update(
                pending_question=question
            ),
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("data"):
            data = message["data"]
            st.caption(
                f"{data.get('retrieved_chunks', []) and len(data['retrieved_chunks']) or 0} passages "
                f"retrieved | {data.get('latency_ms', '?')} ms"
            )

question = st.session_state.pop("pending_question", None)
question = question or st.chat_input("Ask a question about the indexed PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the document collection..."):
            try:
                data = ask_api(question, top_k)
                st.markdown(data["answer"])
                st.caption(
                    f"{len(data.get('retrieved_chunks', []))} passages retrieved "
                    f"| {data.get('latency_ms', '?')} ms"
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "data": data}
                )
            except Exception as error:
                message = f"Request failed: {error}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})