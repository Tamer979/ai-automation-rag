import logging
import threading
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
from custom_types import RAGUpertResult,RAGChunkAndSrc

from data_loader import client as openai_client, load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

inngest_client = inngest.Inngest(
        app_id = "rag app",
        logger=logging.getLogger("uvicorn"),
        is_production=os.getenv("INNGEST_DEV", "1") == "0",  # dev server unless INNGEST_DEV=0 (docker-compose sets it)
        serializer = inngest.PydanticSerializer()
)

CHAT_MODEL = "gpt-4o-mini"
MAX_QUESTION_CHARS = 500  # caps cost per request; streamlit_app.py enforces the same limit in the UI
TOP_K = 4  # chunks retrieved per question; picked by the eval (evals/RESULTS.md), not settable from the UI


# Shared with evals/run_eval.py so the eval tests the exact prompt the app uses.
def build_messages(question: str, contexts: list[str]) -> list[dict]:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    return [
        {"role": "system", "content": "You are an AI assistant for an AI automation agency. Answer the question using the context provided."},
        {"role": "user", "content": user_content}
    ]


@inngest_client.create_function(
    fn_id = "RAG: ingest File",
    trigger = inngest.TriggerEvent(event="rag/ingest_file")
)
async def rag_ingest_file(ctx: inngest.Context):
    def _load(ctx: inngest.Context)-> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks = chunks, source_id = source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name = f'{source_id}:{i}')) for i in range (len(chunks))]
        payloads = [{"source": source_id, "text" : chunks [i]} for i in range (len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpertResult(ingested = len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type = RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type = RAGUpertResult)
    return ingested.model_dump()

# Question limits, to cap OpenAI cost on the public demo.
SESSION_LIMIT = 4                # questions per session...
SESSION_WINDOW_S = 4 * 3600      # ...per 4 hours
MIN_GAP_S = 5                    # seconds between questions in one session
DAILY_LIMIT = 200                # questions per UTC day across all users

# shortcut: in-memory, so limits reset when the backend restarts and only work with one backend process; move to Redis/Postgres if it ever runs on several.
_lock = threading.Lock()
_session_times: dict[str, list[float]] = defaultdict(list)
_daily = {"day": "", "count": 0}


def check_limits(session_id: str, now: float) -> str | None:
    """Record the question and return None if allowed, else a message saying why not."""
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    with _lock:
        if _daily["day"] != day:
            _daily.update(day=day, count=0)
        if _daily["count"] >= DAILY_LIMIT:
            return "The demo has reached its question limit for today. Please try again tomorrow."
        times = [t for t in _session_times.get(session_id, []) if now - t < SESSION_WINDOW_S]
        if times and now - times[-1] < MIN_GAP_S:
            return "Please wait a few seconds between questions."
        if len(times) >= SESSION_LIMIT:
            minutes = int((times[0] + SESSION_WINDOW_S - now) // 60) + 1
            return f"You've used your {SESSION_LIMIT} questions for now. You can ask again in about {minutes} minutes."
        times.append(now)
        _session_times[session_id] = times
        _daily["count"] += 1
        return None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = Field(min_length=1, max_length=100)


app = FastAPI()


# Not exposed to the internet: Caddy only proxies Streamlit, which calls this inside the Docker network.
@app.post("/ask")
def ask(req: AskRequest) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "Question is empty")
    if (reason := check_limits(req.session_id, time.time())) is not None:
        raise HTTPException(429, reason)
    found = QdrantStorage().search(embed_texts([question])[0], TOP_K)
    res = openai_client.chat.completions.create(
        model=CHAT_MODEL, max_tokens=1024, temperature=0.2, messages=build_messages(question, found["contexts"])
    )
    return {"answer": res.choices[0].message.content.strip(), "sources": found["sources"]}


inngest.fast_api.serve (app, inngest_client, functions=[rag_ingest_file])

