import asyncio
from pathlib import Path
import time
import uuid

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="AI Automation for Business Q&A", page_icon="🤖", layout="centered")
st.session_state.setdefault("session_id", str(uuid.uuid4()))


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=os.getenv("INNGEST_DEV", "1") == "0")


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_file",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


# Off on the public server (docker-compose sets ALLOW_UPLOADS=0): every upload lands in the one shared index.
if os.getenv("ALLOW_UPLOADS", "1") == "1":
    with st.sidebar:
        st.header("Documents")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

        if uploaded is not None:
            with st.spinner("Uploading and triggering ingestion..."):
                path = save_uploaded_pdf(uploaded)
                # Kick off the event and block until the send completes
                asyncio.run(send_rag_ingest_event(path))
                # Small pause for user feedback continuity
                time.sleep(0.3)
            st.success(f"Triggered ingestion for: {path.name}")
            st.caption("You can upload another PDF if you like.")

st.markdown("<h2 style='text-align:center'>AI Automation for Business</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;opacity:0.6'>Ask a question. Answers come only from the indexed PDFs.</p>",
    unsafe_allow_html=True,
)


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

question = st.chat_input("Ask anything about AI automation...", max_chars=500)

if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question.strip())

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # session_id keys the per-session limits in main.py
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question.strip(), "session_id": st.session_state.session_id},
                    timeout=60,
                )
            except requests.RequestException:
                st.error("Can't reach the server right now. Please try again in a moment.")
                st.stop()
            if resp.status_code == 429:
                st.error(resp.json()["detail"])
                st.stop()
            if not resp.ok:
                st.error("Something went wrong while answering. Please try again in a moment.")
                st.stop()
            output = resp.json()
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.markdown(answer or "(No answer)")
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")
