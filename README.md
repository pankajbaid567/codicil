# Codicil

Ask about Indian banking rules in English, Hindi or Hinglish. Codicil answers from 17 current RBI Directions, quotes the exact clause with its document, path and "updated as on" date, and says "not found in these Directions" when it cannot support an answer. It runs with no API key and no GPU.

> Informational only. Not legal or financial advice. The RBI texts are the authority.

## Measured results

All numbers come from `docs/results.md`, which is generated from raw files in `eval/results/`. 148 questions (author-labelled, AI-drafted, not yet reviewed by the author), 17 documents, 2,277 chunks.

| | Held-out test split (46 answerable) |
|---|---|
| BM25 only | nDCG@10 0.663, R@10 0.804 |
| Dense only (multilingual-e5-small) | nDCG@10 0.682, R@10 0.870 |
| Hybrid (RRF) + English cross-encoder rerank (final) | **nDCG@10 0.787, R@10 0.957** |

- **Abstention works for English, not for Hinglish.** English uses the cross-encoder score with a threshold chosen on dev only: test precision 1.00, recall 0.81 on 16 unanswerable questions (the first version, a dense-cosine threshold, scored 0.71 / 0.31 and missed the targets). Hinglish has no usable threshold (AUC 0.63 to 0.76), so it never abstains on score; Hindi has 4 unanswerable questions per split. Unanswerable Hinglish and Hindi questions are translations of English ones. Details and caveats in `docs/results.md`.
- **Extractive answers** contain the gold sentence in about 77% of served answerable test questions; the cited clause is always shown in full.
- **Local latency** (Apple M3, closed loop, quiet machine, measured before the tokenizer change below): p50 338 ms, p95 432 ms. **Memory**: peak 402 MB maximum resident after loading and answering (was 568 MB at startup until the 290 MB Hugging Face tokenizer of e5 was replaced by SentencePiece, which gives identical token ids; ADR-023). Free-tier hosting speed is not benchmarked.
- **Tests**: 63 tests, `mypy --strict` and `ruff` clean, and a mutation check where 15 of 15 injected bugs are caught (10 of 14 on the first run; the 4 survivors exposed test gaps that were then closed).

Confidence intervals are wide (n is small); differences under about 0.05 nDCG are ties. See `docs/evaluation.md` for threats to validity.

## How it works

```
question ─► detect language / Hinglish glossary ─► embed ─┐
                                                          ├─► BM25 + dense ─► RRF ─► (English) cross-encoder rerank
corpus ─► clause-tree parser ─► small-to-big chunks ─► immutable index artifact     │
                                                                                    ▼
              cited answer or abstention ◄─ sentence support check ◄─ extractive / optional LLM ◄─ parent-clause context
```

- **Structure-aware chunking.** RBI text is a tree (chapter, section, numbered paragraph, `(1)`, `(a)`, `(i)`, provisos). The parser rebuilds it, chunks follow clauses, children repeat their stem, and retrieval ranks small chunks but shows the enclosing paragraph. Chunking is lossless on all 17 documents (tested).
- **Hybrid retrieval.** Own BM25 with a header field, `multilingual-e5-small` vectors (exact search; 2,277 vectors), Reciprocal Rank Fusion. Devanagari queries skip BM25; the English-only reranker skips non-English queries. Both rules come from the ablation, not taste.
- **Grounding is checked, not requested.** Each answer sentence must be supported by its cited chunk (numbers must match exactly). Unsupported sentences are dropped; none left means abstain.
- **Immutable, content-addressed index.** The service is stateless; every response and cache key carries the index hash; a tampered index refuses to load; an index built with another embedder refuses to serve.
- **Optional LLM.** Any OpenAI-compatible endpoint (Groq, OpenRouter, Gemini, Ollama) via `CODICIL_LLM_*`, with timeout, retry and a circuit breaker. If it fails the service degrades to extractive mode. LLM answer quality is **not** measured and no claim is made about it.
- Hexagonal layout (`domain` / `application` / `adapters` / `api`), an architecture test enforces the dependency rule. Decisions are in `docs/DESIGN.md` (ADR-000 to ADR-020).

## Run it

```bash
make setup            # uv venv + dependencies
make models           # pinned, checksum-verified ONNX models into data/models
.venv/bin/python -m uvicorn codicil.api.main:app --port 8000    # then open http://localhost:8000
```

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question":"How soon must a bank reverse an unauthorised electronic transaction?","debug":true}'
```

Endpoints: `POST /v1/ask` (JSON or `Accept: text/event-stream`), `POST /v1/search`, `GET /v1/documents`, `GET /v1/index`, `/healthz`, `/metrics`; errors are `application/problem+json`.

Rebuild everything: `make corpus` (polite fetch of the RBI HTML pages, hashes in `data/corpus/manifest.json`), `make index`, `make eval`, `make results`, `make mutation`.

## Deploy (free)

**Live demo:** https://codicil.onrender.com (Render free plan, Singapore). After the tokenizer change it started and stayed up through the checks on 2026-11-12: an English question answered with a citation, a Hinglish question answered with the right clause, and an out-of-scope question (repo rate) abstained. On the 0.1 CPU free plan English questions that go through the cross-encoder took 8 to 10 s in those first calls (cold), Hinglish under 1 s; these are single observations, not a benchmark.

`Dockerfile` and `render.yaml` are provided (one container, models baked in at build, index committed). The first build on Render's free 512 MB plan compiled fine but was killed for memory at startup (about 568 MB peak); replacing the e5 tokenizer (ADR-023) brought the peak to about 402 MB. Whether the free plan then stays up under load is checked after redeploy and noted in `docs/deployment.md`. The free plan sleeps after about 15 minutes idle, so the first request after a pause takes tens of seconds.

## Limits

One issuer, 17 documents, 148 author-labelled questions; abstention thresholds rest on 30 English negatives (12 Hinglish, 8 Hindi, translated from English); extractive sentence selection is crude; Hindi has 8 questions in total; the corpus is a snapshot (`manifest.json` has the dates) and is not kept current automatically; one Direction (Commercial Banks, Credit Facilities) is excluded because RBI publishes it only as a PDF behind a bot challenge, which this project does not try to bypass.

Source documents © Reserve Bank of India; see `NOTICE.md`. Code: MIT.
