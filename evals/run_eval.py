"""Run the RAG eval set against the live Qdrant collection.

Usage (repo root, Qdrant running):  uv run python -m evals.run_eval [evals/op_ai_business.jsonl]
Calls the pipeline directly, not through Inngest, so the app's query rate limit doesn't apply.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from data_loader import embed_texts
from main import CHAT_MODEL, build_messages
from vector_db import QdrantStorage

# shortcut: hardcoded gpt-4o-mini prices (USD per 1M tokens, input/output), update if you change models.
PRICE_IN, PRICE_OUT = 0.15, 0.60
REFUSALS = ["don't know", "do not know", "not provided", "not mention", "no information", "not specified",
            "does not", "doesn't", "not available", "not contain", "not included", "unable", "cannot"]


def norm(s: str) -> str:
    return " ".join(s.lower().replace(",", "").replace("’", "'").split())


def passes(text: str, must: list[str]) -> bool:
    # Each "must" item passes if any of its |-separated alternatives appears in the text.
    t = norm(text)
    return all(any(alt in t for alt in item.split("|")) for item in must)


def grade(case: dict, answer: str, contexts: list[str]) -> tuple[bool, bool | None]:
    if case.get("unanswerable"):
        return any(r in norm(answer) for r in REFUSALS), None
    return passes(answer, case["must"]), passes(" ".join(contexts), case["must"])


def main(path: str = "evals/op_ai_business.jsonl", top_k: int = 4) -> None:
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    client, store = OpenAI(), QdrantStorage()
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        contexts = store.search(embed_texts([case["q"]])[0], top_k)["contexts"]
        res = client.chat.completions.create(model=CHAT_MODEL, max_tokens=1024, temperature=0.2,
                                             messages=build_messages(case["q"], contexts))
        latency = time.perf_counter() - t0
        answer = res.choices[0].message.content.strip()
        ok, retrieved = grade(case, answer, contexts)
        cost = (res.usage.prompt_tokens * PRICE_IN + res.usage.completion_tokens * PRICE_OUT) / 1e6
        rows.append({**case, "answer": answer, "pass": ok, "retrieved": retrieved, "latency_s": round(latency, 2),
                     "prompt_tokens": res.usage.prompt_tokens, "completion_tokens": res.usage.completion_tokens,
                     "cost_usd": cost})
        print(f"{'PASS' if ok else 'FAIL'}  ret={retrieved!s:5}  {latency:5.2f}s  {case['q'][:70]}")

    answerable = [r for r in rows if not r.get("unanswerable")]
    print(f"\nModel: {CHAT_MODEL}  top_k={top_k}")
    print(f"Accuracy:            {sum(r['pass'] for r in rows)}/{len(rows)}")
    print(f"  answerable:        {sum(r['pass'] for r in answerable)}/{len(answerable)}")
    print(f"  refused correctly: {sum(r['pass'] for r in rows if r.get('unanswerable'))}/{len(rows) - len(answerable)}")
    print(f"Retrieval hit rate:  {sum(r['retrieved'] for r in answerable)}/{len(answerable)}")
    print(f"Avg latency:         {sum(r['latency_s'] for r in rows) / len(rows):.2f}s")
    print(f"Total cost:          ${sum(r['cost_usd'] for r in rows):.4f}")

    out = Path("evals/results") / f"{CHAT_MODEL}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2], *map(int, sys.argv[2:3]))
