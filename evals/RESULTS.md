## Evaluation

I measured answer quality with a labeled eval set instead of judging by eye. I ran 11 configurations, varying chunk size, number of retrieved chunks (top_k), and the amount of unrelated "noise" content in the index.

### Setup

| | |
|---|---|
| Source document | *How to Build a One Person AI Business in 2026* (8-page PDF) |
| Eval set | 28 questions: 25 with known answers + 3 the document cannot answer |
| Chat model | `gpt-4o-mini`, temperature 0.2 |
| Embeddings | `text-embedding-3-large` (3072 dims), cosine similarity in Qdrant |
| Chunking | LlamaIndex `SentenceSplitter`, sizes in tokens |
| Noise PDFs | Frozen yogurt, NBA, United States (Wikipedia exports) |

**What each metric means**
- **Answer accuracy:** the answer contains every required fact, e.g. both ends of a price range.
- **Correct refusals:** for the 3 unanswerable questions, the model says the information isn't in the context instead of inventing it.
- **Retrieval hit rate:** the required facts appear in the retrieved chunks. This separates search failures from model failures.
- **Cost:** chat tokens for all 28 questions at gpt-4o-mini prices. Embedding cost is excluded; it's negligible.

The eval calls the same retrieval and prompt code the app uses (`build_messages` in `main.py`), so it tests the real pipeline.

### Results

| # | Chunk size / overlap | Chunks in index | Noise PDFs | top_k | Accuracy | Correct refusals | Retrieval hits | Avg latency | Cost per run |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1000 / 200 | 8 | no | 5 | **25/25** | 3/3 | 25/25 | 1.09s | $0.0156 |
| 2 | 1000 / 200 | 8 | no | 3 | 24/25 | 3/3 | 24/25 | 1.06s | $0.0095 |
| 3 | 1000 / 200 | 8 | no | 2 | 24/25 | 3/3 | 24/25 | 1.20s | $0.0065 |
| 4 | 256 / 50 | 31 | no | 2 | 22/25 | 3/3 | 21/25 | 1.17s | $0.0024 |
| 5 | 256 / 50 | 31 | no | 4 | 23/25 | 3/3 | 24/25 | 1.17s | $0.0042 |
| 6 | 256 / 50 | 31 | no | 5 | 23/25 | 3/3 | 24/25 | 1.11s | $0.0051 |
| 7 | 512 / 100 | 16 | no | 2 | 24/25 | 3/3 | 23/25 | 1.07s | $0.0038 |
| 8 | 512 / 100 | 16 | no | 4 | **25/25** | 3/3 | 25/25 | 1.14s | $0.0073 |
| 9 | 512 / 100 | 16 | no | 5 | **25/25** | 3/3 | 25/25 | 1.16s | $0.0090 |
| 10 | 512 / 100 | 45 | yes | 2 | 24/25 | 3/3 | 23/25 | 1.29s | $0.0038 |
| 11 | **512 / 100** | **45** | **yes** | **4** | **25/25** | **3/3** | **25/25** | **1.20s** | **$0.0073** |

Row 11 is the configuration the app ships with.

### What I learned

1. **Chunks that are too large hide small facts.** At 1000 tokens each PDF page became one chunk. A two-sentence statistic (Capgemini: 13% of AI projects reach production) shared a chunk with a whole page about another topic, ranked 4th of 8, and was missed whenever top_k was under 4 (rows 2–3).
2. **Chunks that are too small break list answers, and the failures are worse.** At 256 tokens, questions whose answer spans a section ("What are the three outcomes…", "the four rungs…") failed. Instead of saying the information was missing, the model returned **confident wrong answers**: a made-up list of outcomes and a ladder with a rung missing (rows 4–6).
3. **512 tokens with top_k=4 was the best tradeoff:** 100% accuracy at **53% lower cost** than the first perfect setup (row 8 vs row 1).
4. **Unrelated noise didn't affect retrieval.** Adding 3 off-topic PDFs nearly tripled the index (16 → 45 chunks) with identical results (rows 7–8 vs 10–11).
5. **Every miss at 512 and 1000 tokens failed safely.** The model said "the context does not include…" rather than inventing an answer. All 11 configurations got 3/3 on the unanswerable questions.

### Bugs found along the way

- **Ingestion crashed once pages had to be split.** NLTK refuses to open hardlinked data files, and `uv` installs packages as hardlinks by default. With 1000-token chunks each page fit in one chunk, so this code path never ran. Fixed with `link-mode = "copy"` in `pyproject.toml`.
- **The eval's own grader had a bug.** PDF line breaks inside phrases like "morning brief" made correct retrievals count as misses. Fixed by normalizing whitespace, then covered by a unit test (`evals/test_run_eval.py`).

### Limitations

- One source document and one run per configuration. With 25 questions, a 1-question difference is within noise.
- Grading is keyword-based. It can pass a wrong answer that happens to contain the right number. I read the failing and borderline answers by hand.
- The noise PDFs are off-topic. Related AI documents will compete for the same chunks and are the harder test, and the next step.

### Reproduce

```bash
uv run python -m evals.test_run_eval                          # grader unit test
uv run python -m evals.run_eval                               # default: top_k=4
uv run python -m evals.run_eval evals/op_ai_business.jsonl 2  # custom top_k
```

Each run saves per-question answers, latency, tokens and cost to `evals/results/`.
