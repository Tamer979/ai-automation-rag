import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid 
import os
import datetime 
from custom_types import RAGQueryResult,RAGSearchResult,RAGUpertResult,RAGChunkAndSrc

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

inngest_client = inngest.Inngest(
        app_id = "rag app",
        logger=logging.getLogger("uvicorn"),
        is_production=os.getenv("INNGEST_DEV", "1") == "0",  # dev server unless INNGEST_DEV=0 (docker-compose sets it)
        serializer = inngest.PydanticSerializer()
)

CHAT_MODEL = "gpt-4o-mini"


# Shared with evals/run_eval.py so the eval tests the exact prompt the app uses.
def build_messages(question: str, contexts: list[str]) -> list[dict]:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer conciesly using the context above."
    )
    return [
        {"role": "system", "content": "You are a Ai assistant for an AI automation agency. Answer the question using the context provided."},
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

@inngest_client.create_function(
    fn_id = "RAG: Query PEF",
    trigger= inngest.TriggerEvent(event = "rag/query_pdf_ai"),
    throttle=inngest.Throttle(
        limit=1,
        period=datetime.timedelta(seconds=5),
        burst=2,
        key="event.data.user_id",
    ),
    rate_limit=inngest.RateLimit(
        limit=4,
        period=datetime.timedelta(hours=4),
        key="event.data.session_id",
    ),
    concurrency=[inngest.Concurrency(limit=10)],
    
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 4) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts = found["contexts"], sources = found["sources"])

    question = ctx.event.data["question"]
    top_k = int (ctx.event.data.get("top_k", 4))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type = RAGSearchResult) 

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model = CHAT_MODEL
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter = adapter,
        body={"max_tokens": 1024, "temperature": 0.2, "messages": build_messages(question, found.contexts)}
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer,"sources": found.sources,"num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve (app, inngest_client, functions=[rag_ingest_file, rag_query_pdf_ai])

