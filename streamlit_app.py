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


async def send_rag_query_event(question: str) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                # Keys for the throttle and rate limit in main.py; without them all users share one bucket.
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.session_id,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    key = os.getenv("INNGEST_SIGNING_KEY")
    # Inngest caches this endpoint ~15s per URL; a changing param makes each poll see fresh status.
    resp = requests.get(url, params={"t": time.time_ns()}, headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            # Inngest flips status to Completed slightly before it stores the output; keep polling until it's there.
            if status in ("Completed", "Succeeded", "Success", "Finished") and run.get("output"):
                return run["output"]
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        elif time.time() - start > 45:
            # Must stay well past Inngest's ~15s API cache: self-hosted Inngest can hide a new run that long.
            raise TimeoutError("Couldn't get an answer. The server may be busy or down, or you've hit the limit of 4 questions per 4 hours. Try again in a moment.")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


question = st.chat_input("Ask anything about AI automation...", max_chars=500)

if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question.strip())

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip()))
            # Poll the local Inngest API for the run's output
            try:
                output = wait_for_run_output(event_id)
            except (TimeoutError, RuntimeError) as e:
                st.error(str(e))
                st.stop()
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.markdown(answer or "(No answer)")
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")
