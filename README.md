# Codicil

Ask about Indian banking rules in English, Hindi or Hinglish. Codicil answers from 17 current RBI Directions, quotes the exact clause with its document, path and "updated as on" date, and says "not found in these Directions" when it cannot support an answer. It runs with no API key and no GPU.

> Informational only. Not legal or financial advice. The RBI texts are the authority.

## Measured results

All numbers come from `docs/results.md`, which is generated from raw files in `eval/results/`. 148 questions (author-labelled, AI-drafted, not yet reviewed by the author), 17 documents, 2,223 chunks. The test split has been run twice (second look after a corpus fix; see `docs/results.md`).

| | Held-out test split (46 answerable) |
|---|---|
| BM25 only | nDCG@10 0.663, R@10 0.804 |
| Dense only (multilingual-e5-small) | nDCG@10 0.698, R@10 0.913 |
| Hybrid (RRF) + English cross-encoder rerank (final) | **nDCG@10 0.800, R@10 0.978** |

- **Abstention works for English, not for Hinglish.** English uses the cross-encoder score with a threshold chosen on dev only: test precision 1.00, recall 0.81 on 16 unanswerable questions (the first version, a dense-cosine threshold, scored 0.71 / 0.31 and missed the targets). Hinglish has no usable threshold (AUC 0.63 to 0.76), so it never abstains on score; Hindi has 4 unanswerable questions per split. Unanswerable Hinglish and Hindi questions are translations of English ones. Details and caveats in `docs/results.md`.
- **Extractive answers** contain the gold sentence in about 77% of served answerable test questions (88% on dev, where list completion was developed). When a quoted sentence introduces a list ("...in the following events:"), the items that follow are quoted too, so answers no longer stop at a colon; on dev this cut cut-off answers from 13 of 51 to 2 of 51, on test the gain is one question and within noise. The cited clause is always shown in full.
- **Latency**: on the EC2 demo host, p50 about 1.3 s and p95 1.5 to 1.7 s end to end over HTTPS for English questions (see Deploy). Locally on an Apple M3 (closed loop, quiet machine, measured before the tokenizer change below): p50 338 ms, p95 432 ms. **Memory**: peak 402 MB maximum resident after loading and answering (was 568 MB at startup until the 290 MB Hugging Face tokenizer of e5 was replaced by SentencePiece, which gives identical token ids; ADR-023). Free-tier hosting speed is not benchmarked.
- **Tests**: 88 tests, `mypy --strict` and `ruff` clean, and a mutation check where 25 of 25 injected bugs are caught (10 of 14 on the first run; the survivors exposed test gaps that were then closed).

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
- **Hybrid retrieval.** Own BM25 with a header field, `multilingual-e5-small` vectors (exact search; 2,223 vectors), Reciprocal Rank Fusion. Devanagari queries skip BM25; the English-only reranker skips non-English queries. Both rules come from the ablation, not taste.
- **Grounding is checked, not requested.** Each answer sentence must be supported by its cited chunk (numbers must match exactly). Unsupported sentences are dropped; none left means abstain.
- **Immutable, content-addressed index.** The service is stateless; every response and cache key carries the index hash; a tampered index refuses to load; an index built with another embedder refuses to serve.
- **Optional LLM.** Any OpenAI-compatible endpoint (Groq, OpenRouter, Gemini, Ollama) via `CODICIL_LLM_*`, with timeout, retry and a circuit breaker. On Amazon Bedrock it uses the instance role instead of a key (`CODICIL_LLM_PROVIDER=bedrock`). When an LLM is configured it is the default answer mode and *Extractive* stays one click away; if the LLM fails the service degrades to extractive mode. An AI-graded comparison of the two modes is in `docs/results.md` (summary under Deploy).
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

## Deploy

**Live demo (primary):** https://codicil.3-109-36-90.sslip.io: one container on an AWS EC2 `t4g.small` (2 GB, Graviton, Mumbai) behind Caddy with a Let's Encrypt certificate. Answer mode in the UI is *LLM* (the default, because the server has one configured) or *Extractive* (quotes the clause word for word; the choice is remembered in the browser); LLM mode runs `openai.gpt-oss-120b` on **Amazon Bedrock** through the Converse API with the instance role (no API key anywhere): `CODICIL_LLM_PROVIDER=bedrock`, `CODICIL_LLM_MODEL`, `CODICIL_LLM_REGION`. The model rewrites the quoted sentences in plain words, and every sentence is still checked against the cited clause; if Bedrock fails or times out the answer falls back to extractive. Measured on 20 fresh English questions: all 20 answered by the LLM with no fallback, p50 1.7 s and p95 2.0 s end to end (extractive: about 1.3 s). **Answer quality, AI-graded** (96 answerable dev and test questions, each answered in both modes; a different model, `qwen3-32b`, judges against the authoritative clause with both answer orders): the LLM answer is preferred in 64 pairs, tied in 20 and loses in 7 (net +0.63 per pair, 95% CI +0.49 to +0.75; test split alone +0.68), and is judged correct for 91% of pairs against 68% for extractive. The judge disagreed with itself when the order was swapped on 18 of 91 pairs (counted as ties). It is one judge, not human grading, and the 'unsupported' rate overstates the gap because it counts a real but different clause as unsupported; details and the failure cases are in `docs/results.md`. The older Gemini sample used a verbatim-coverage proxy that cannot see phrasing and showed no gain. **LLM is the default** on this server because of that result, with *Extractive* available in the UI and with `"use_llm": false` in the API; the cost is about 0.4 s more per answer and the risk that a rewritten sentence is wrong in a way the grounding check does not catch. A server with no LLM configured answers extractively. Measured on 2026-11-19 from a laptop in India, through the HTTPS API (so network time is included): **p50 1.3 s, p95 1.5 to 1.7 s** for English questions (the cross-encoder rerank is about 1.2 s of that), in three runs of 20 to 30 fresh questions each, container memory about 300 MB. One run on one machine; not a load test.

**The UI has no separate frontend**: the same container serves the API and the web page, so the URL above is the app, and the page talks to its own `/v1/*` endpoints.

**Vector database on AWS (optional, off by default).** Dense search can run on **Amazon S3 Vectors** instead of the in-process matrix: `CODICIL_S3VECTORS_BUCKET=<vector bucket>` plus `python scripts/upload_s3vectors.py`. It uses the instance role (no stored key); the vector index is named after the index build (`codicil-<sha12>`), and each query checks that the vectors belong to the build the server runs. The demo has it enabled: pick *Amazon S3 Vectors* under "Search backend" in the UI, or send `"backend": "s3vectors"` to `/v1/ask`. Measured on the demo host with 30 fresh English questions per backend: p50 1.31 s on S3 Vectors against 1.31 s in-process (p95 1.58 s against 1.70 s), so there is no measurable end-to-end difference, because the cross-encoder rerank (about 1.2 s on this CPU) dominates. In a separate 24-question run, 23 returned identical citations to the in-process search (the remaining one was not investigated). The in-process matrix remains the default because at 2,223 vectors it needs no network call; S3 Vectors costs cents per month at this size and is the path that scales to many more documents. Qdrant is also supported (below).

**Earlier host:** https://codicil.onrender.com (Render free plan, Singapore, with the optional Gemini LLM mode). It runs on 0.1 CPU and sleeps after about 15 minutes idle, so English questions took 5 to 15 s in the checks of 2026-11-12 and 2026-11-19 (single observations, not a benchmark).

The `Dockerfile` builds one container (models baked in at build, index committed). To run it on a small host: `docker build -t codicil .` then `docker run -d --restart unless-stopped --network host --memory 800m -e PORT=8001 codicil`, and proxy a hostname to `127.0.0.1:8001`. The first Render build on a 512 MB plan was killed for memory at startup (about 568 MB peak); replacing the e5 tokenizer (ADR-023) brought the peak to about 402 MB, which is why a 1 GB instance was too small to share with another app and a 2 GB one was used.

## Limits

One issuer, 17 documents, 148 author-labelled questions; abstention thresholds rest on 30 English negatives (12 Hinglish, 8 Hindi, translated from English); extractive sentence selection is crude; Hindi has 8 questions in total; the corpus is a snapshot (`manifest.json` has the dates) and is not kept current automatically; one Direction (Commercial Banks, Credit Facilities) is excluded because RBI publishes it only as a PDF behind a bot challenge, which this project does not try to bypass.

Source documents © Reserve Bank of India; see `NOTICE.md`. Code: MIT.
