# Architecture

## Context
```
 browser ──HTTPS──► [ Codicil service ] ──► (optional) OpenAI-compatible LLM  (Groq / OpenRouter / Ollama)
   UI + API            one container             └ circuit-broken; absence degrades to extractive mode
                          │
                          └── reads once at start: IndexArtifact (immutable, content-addressed)
                                                   + ONNX models (embedder, reranker)

 build time (laptop or CI):  fetch ─► parse ─► chunk ─► embed ─► IndexArtifact ──► committed / baked into the image
 eval time:                  eval dataset + IndexArtifact ─► metrics, ablations, regression gate
```
The service is **stateless**. Everything it knows is in an index artifact whose SHA-256 is returned in every response and used in every cache key. A new corpus is a new artifact and a new deploy; nothing mutates in place, so there is no migration, no partial index and no "which version answered this" question.

## Layers (dependency rule: arrows point inward only)
```
 adapters (I/O)            application (use cases)         domain (pure)
 ───────────────           ───────────────────────         ──────────────────
 api/        FastAPI   ──► AskQuestion, Search       ──►   models.py   Document, Clause, Chunk, Query,
 cli/        build,eval    BuildIndex, RunEvaluation         Evidence, Answer, Citation, Abstention
 adapters/                       │                          structure.py  clause-tree parser
   pdf.py, fetch.py              │ depends on ports         chunking.py  strategies
   onnx_embedder.py              ▼                          bm25.py, fusion.py, tokens.py
   onnx_reranker.py         ports.py (Protocols)            verify.py, abstain.py, metrics.py
   openai_compat.py           Embedder, Reranker,           lang.py, glossary.py
   index_store.py             Generator, Clock, Fetcher
```
- **Domain** has no I/O, no network, no clock, no randomness; every function is deterministic and unit-testable. An architecture test fails the build if a domain module imports an adapter, `httpx`, `fastapi` or `onnxruntime`.
- **Ports** are `typing.Protocol`s. Each adapter has a deterministic fake used in tests (`HashingEmbedder`, `FakeGenerator`), and a **contract test suite** that runs against every implementation of a port.
- **Composition root** (`codicil/wiring.py`) is the only place that chooses implementations, driven by typed settings. No service locator, no globals.

## Patterns, and where they are used
| Pattern | Where | Why |
|---|---|---|
| Ports and adapters | embedder, reranker, generator, fetcher, store | Swap ONNX for Workers AI, or Groq for Ollama, without touching the pipeline |
| Pipeline (pipe and filter) | the answer path | Each stage typed and measurable; the trace is just the stage timings |
| Strategy | chunkers, fusion, generators | Ablations run the same code with a different strategy object |
| Immutable value objects | all domain models are frozen dataclasses | Safe to cache and share across threads |
| Content-addressed artifact | the index | Reproducibility and cache invalidation for free |
| Circuit breaker, timeout, retry with jitter | LLM adapter | A slow free-tier provider must not take the service down |
| Bulkhead | CPU-bound stages run in a bounded thread pool | One heavy request cannot starve health checks |
| Typed errors at the edge | domain raises `CodicilError` subclasses for fatal faults; expected outcomes (abstention, degraded mode) are values on `Answer` | Errors become `problem+json`; abstention is never an exception |
| Null object | `NoReranker`, `ExtractiveGenerator` | Optional stages need no `if` scattered through the code |
| Cache-aside with versioned keys | query embedding and answer cache | Keys include the index SHA-256, so a new index can never serve a stale answer |

## Runtime
- One process, ASGI (uvicorn). Models and index load at start (a few seconds), then `/healthz` is green. All CPU-bound work (embedding, rerank) runs in a `ThreadPoolExecutor(max_workers=2)`; ONNX Runtime releases the GIL.
- **Latency budget** (warm, extractive, free-tier CPU): normalise 2 ms, embed query 30 to 120 ms, BM25 5 ms, dense 5 ms, fuse 1 ms, rerank (optional) 150 to 400 ms, pack 2 ms, extract 20 ms, verify 5 ms. Per-stage timings are in the response trace and in `Server-Timing`.
- **Memory budget** (resident): ONNX Runtime and embedder about 120 MB, reranker about 60 MB, index about 40 MB, Python and FastAPI about 90 MB. Measured, not estimated, in `docs/results.md`.
- **Concurrency**: request-scoped state only; the index and models are read-only after start, so no locks on the hot path.

## Failure modes
| Failure | Behaviour |
|---|---|
| LLM key absent / provider down / slow | Extractive answer, `mode: "extractive"` and a reason in the trace |
| Embedder or reranker fails to load | Start refuses (fail fast); a reranker failure at request time skips rerank |
| Index and embedder mismatch | Start refuses with the two ids in the error |
| Latency deadline exceeded | Skip optional stages in order: rerank, then diversify; never skip verification |
| Retrieval weak | Abstain, with the best evidence shown as "closest passages" |
| Abusive client | Token-bucket rate limit per client, body size limit, strict schema validation |

## API (v1)
`POST /v1/ask` (JSON or `Accept: text/event-stream`), `POST /v1/search` (retrieval only, for debugging), `GET /v1/documents`, `GET /v1/index` (artifact id, embedder, counts), `GET /healthz`, `GET /metrics`. Errors are RFC 9457 `application/problem+json`. Every response carries `request_id`, `index_sha`, `mode`, and the trace when `debug=true`.

## Security and privacy
Public documents only. No accounts, no stored queries (logs record a salted hash and lengths, not text). Retrieved text is wrapped as data in prompts. Every answer carries the not-advice notice. Dependencies pinned and audited in CI.

## Repository layout
```
src/codicil/{domain,application,adapters,api,cli}/   tests/{unit,property,contract,architecture,eval}/
data/corpus/manifest.json      eval/{questions.jsonl, README.md}      docs/      scripts/
```
