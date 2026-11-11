# Design decisions

Each ADR states the decision, the alternatives rejected, and what would change the decision. "Pending experiment" means the choice is made by a pre-registered rule on the dev split, and the result is recorded here.

## ADR-000: Name
**Codicil**: a legal addendum that amends an earlier instrument, which is exactly the problem (rules are amended and "updated as on" a date). GitHub search 2026-11-11: 30 repositories with the name, none for this purpose.

## ADR-001: Python 3.12, FastAPI, standard library first
Python has the ML ecosystem and ONNX Runtime wheels; FastAPI gives typed, async, self-documenting HTTP. Code is typed (`mypy --strict`) and linted (`ruff`). Dependencies are few and justified: `numpy`, `onnxruntime`, `tokenizers`, `pypdf`, `pdfminer.six`, `snowballstemmer`, `fastapi`, `httpx`, `pydantic`. **Rejected**: LangChain/LlamaIndex (hide the pipeline this project exists to show; heavy for a 512 MB host); a vector database service (ADR-008).

## ADR-002: Hexagonal architecture with an enforced dependency rule
Pure domain, application use cases over ports, adapters at the edge, one composition root. A test parses imports and fails on a violation. **Why**: every ablation and every fake in the test suite depends on the stages being swappable.

## ADR-003: Corpus is the current RBI Directions, issuer-agnostic metadata
RBI's 2025-26 consolidation produced current, long, cross-referencing texts that postdate model training, so retrieval is genuinely required. v1 ingests 17 consumer- and payments-relevant Directions. Metadata (`issuer`, `entity`, `updated_as_on`, `supersedes`) is issuer-neutral so SEBI can be added without schema change. PDFs are not committed (a fetch script with hashes is); the derived index is, with attribution to RBI. **Alternative rejected**: a Wikipedia or arXiv corpus (no temporal structure, and models have seen it).

## ADR-004: The index is an immutable, content-addressed artifact; the service is stateless
No database to run on a free tier, no migrations, reproducible builds, trivial rollback, cache keys that cannot go stale. **Cost**: a corpus change is a rebuild and redeploy, which is acceptable for a corpus that changes quarterly. **Would change if**: documents had to be added at runtime (then a store port with a Postgres/pgvector adapter, already shaped by the `IndexStore` protocol).

## ADR-005: Structure-aware, small-to-big chunking with deterministic contextual headers
Legal meaning lives in the clause path and in provisos, so chunk boundaries follow clauses, retrieval ranks small children, and the reader gets the parent. Headers are deterministic, so rebuilding the index is reproducible and free. **Rejected**: fixed windows as the default (kept as the ablation baseline), embedding-similarity "semantic" chunking (nondeterministic, no better on structured text), LLM-written contextual summaries (cost and non-determinism).

## ADR-006: Hybrid retrieval fused with Reciprocal Rank Fusion
Regulatory text mixes exact terms ("Section 12.3", "KYC", "unauthorised electronic banking transaction") that BM25 nails with paraphrase that dense retrieval nails. RRF avoids comparing incomparable scores and has one parameter. **Pending experiment** for the weights and `k` on the dev split; hybrid must not be worse than the best single retriever or it is dropped.

## ADR-007: Embedding model chosen by pre-registered experiment
Candidates: `bge-small-en-v1.5` (English, with query normalisation) and `multilingual-e5-small` (native Hindi), both ONNX int8, 384-d. **Rule**: choose the higher nDCG@10 on the full dev set; if within 0.01, choose the one with better Hinglish recall@10. Result recorded in `docs/results.md`.

## ADR-008: Exact vector search; no vector database
At about 15k vectors, a float32 matrix product takes single-digit milliseconds. An ANN index adds recall loss and a dependency for no benefit here. The `VectorIndex` protocol leaves room for HNSW above roughly 100k vectors.

## ADR-009: Reranking is optional, deadline-aware and earns its place by ablation
A cross-encoder usually improves precision at the top but costs hundreds of milliseconds on a free-tier CPU. It ships on only if the ablation shows a meaningful nDCG@10 gain within the latency budget; it is skipped, and the skip is traced, when the request deadline is nearly spent.

## ADR-010: Hinglish handled by detection plus a domain glossary, LLM translation optional
Most users of Indian banking questions write romanised Hindi. A glossary of a few hundred banking terms is deterministic and free; an LLM translation path improves it when a key exists. The lift (or lack of one) is measured on the Hinglish subset.

## ADR-011: Temporal validity is first-class
Each document has `updated_as_on`, and where the source states it, `supersedes` and `superseded_by`. Retrieval defaults to in-force documents; every answer prints the date of each cited document; the corpus deliberately includes one superseded document so the behaviour is testable.

## ADR-012: Extractive by default, LLM optional behind a provider-neutral port
The project must work, and be evaluable, with no key and no cost. The extractive generator is also the honest baseline an LLM answer has to beat. One OpenAI-compatible adapter covers Groq, OpenRouter, Gemini's compatible endpoint and local Ollama.

## ADR-013: Grounding is verified, not requested
Asking a model to cite is not evidence it did. Each sentence is checked against its cited chunk (numbers and named entities must be present; content-word recall above a threshold); failures are dropped; an empty answer is an abstention. Abstention on weak retrieval uses a threshold fitted on the dev split to a target precision.

## ADR-014: Resilience around the only network dependency
Timeout, bounded retry with jitter, circuit breaker; failure degrades to extractive, never to an error page. CPU-bound work runs in a bounded pool so health checks stay responsive.

## ADR-015: Cache keys include the index hash
An answer cache keyed by `(index_sha, normalised query, parameters)` can never serve an answer from an old corpus. No semantic cache: a near-match returning another question's answer is a worse failure than a miss.

## ADR-016: Evaluation is part of the design, not an afterthought
Gold labels are **evidence quotes** (document, quote), not chunk ids, so the same dataset scores any chunking strategy. Dev and test are split once; thresholds are tuned on dev; test is run once per release. Confidence intervals by bootstrap. A regression gate in CI uses cached vectors and fails on a drop beyond the stated tolerance.

## ADR-017: Free-tier deployment as a single container
One Render free web service (512 MB) serves API and UI; models and index are baked into the image so cold start does no downloads; the UI is static files served by the same app. Cloudflare Pages can host the UI separately. Cold starts are real and documented; no claims are made from the free instance's latency.

## ADR-018: Not advice, no personal data
Every answer carries a notice that it is not legal or financial advice. No accounts; no query text is stored.

## ADR-019: Test strategy
Unit tests for pure functions; property-based tests (Hypothesis) for invariants (chunks cover all text without overlap beyond the declared amount, RRF is order-stable, tokenizer round trips, metric bounds); contract tests per port; an architecture test for the dependency rule; golden tests for metrics with hand-computed values; a mutation check that injects bugs the suite must catch.

## ADR-020: Source format is RBI's HTML pages; documents without an HTML body are excluded
RBI's PDF host sits behind an interactive bot challenge, which this project does not try to get around. The same Directions are published as full-text HTML pages on rbi.org.in, which also preserve chapter, paragraph and amendment-note structure better than text extracted from a PDF layout. One selected document (Commercial Banks, Credit Facilities) has no HTML body (the page only links the PDF), so it is excluded; the corpus is 17 documents. Fetching is sequential with a 2 s pause and an identifying User-Agent, and the script stops rather than retry if it receives a challenge page.

## ADR-021: Qdrant is an optional dense-search backend; the file index stays the default
Qdrant (cloud or self-hosted) can serve the dense half of retrieval through the `DenseStore` port (`adapters/qdrant_store.py`); BM25, reranking and verification are unchanged, and chunk text still comes from the immutable index artifact, so a point id is the chunk's position in that index and every Qdrant payload carries the index sha. It is enabled only when `CODICIL_QDRANT_URL` is set; without it nothing changes and CI needs no cluster. Honest trade-offs: at 2,277 chunks an in-memory matrix product is as fast as a network call and cannot be slower than it, so the expected gains are a smaller resident set on a 512 MB host and server-side payload filters, not latency. Both are to be measured on the same eval questions before any claim is made (`docs/results.md`); the first check is that results match the numpy search (`tests/unit/test_qdrant_store.py`, run against Qdrant's local mode). Credentials are read from the environment only and never stored in the repo.

**Amendment (same day):** Qdrant is not the process-wide default. A request chooses it with `backend: "qdrant"` (the web UI has a Search backend selector, enabled only when `CODICIL_QDRANT_URL` is set on the server); the default stays the in-process matrix, and any Qdrant failure falls back to it and says so in the answer's timings. The abstention signal `top_dense` is still computed from the in-process vectors, so abstention behaves the same on both backends.

## ADR-022: Abstention is routed per language, on the strongest signal measured for each
The first abstention rule (top dense cosine below 0.84, fitted on dev) did not transfer to test (precision 0.71, recall 0.31). Comparing signals on pooled dev and test showed the English cross-encoder's top score separates answerable from unanswerable better than the cosine (AUC 0.95 against 0.92), and a threshold chosen on dev alone (-0.18: highest recall at precision 0.9) gave test precision 1.00 and recall 0.81. English therefore abstains on the reranker score. For Hinglish neither signal had a threshold that kept answerable questions served while catching negatives (AUC 0.63 for cosine, 0.76 for the cross-encoder score), so Hinglish never abstains on score and relies on the sentence-support check and on showing the cited clause. Hindi keeps the cosine rule (AUC 0.80, four negatives per split: anecdotal). Costs: the English threshold is tied to the MiniLM cross-encoder, not portable to another reranker; the English test split was examined twice, although no threshold was fitted on it; the unanswerable Hinglish and Hindi questions are translations of English ones, drafted by an AI assistant; the language detector tags 5 of 34 Hinglish questions as English, so those take the English path.

## ADR-023: multilingual-e5 is tokenized with SentencePiece, not the Hugging Face tokenizer.json
The first deployment attempt on a 512 MB host was killed at startup ("Out of memory", peak about 568 MB locally). Measured: loading the e5 `tokenizer.json` (250k-piece vocabulary) keeps about 290 MB resident, more than the 113 MB int8 model. `adapters/sp_tokenizer.py` reproduces the XLM-R id layout on the original `sentencepiece.bpe.model` (pinned by checksum in `data/models.lock.json`, fetched from `intfloat/multilingual-e5-small`); ids are identical to the Hugging Face tokenizer on all 2,277 indexed chunks (with truncation) and all eval questions, and `tests/unit/test_sp_tokenizer.py` pins hand-checked cases. Peak after loading and answering fell to about 402 MB. Side effect found on the way: the earlier batched embeddings padded with id 0 (the Hugging Face default) instead of the model's pad id 1; padding is masked but the int8 model is not exactly invariant to it (cosine between the two paddings down to 0.997). The stored index vectors were built with the old padding and are kept (the service embeds one query at a time, so it is unaffected); evaluation tables were regenerated with the corrected padding. Recall@10 did not change anywhere; nDCG@10 moved by up to 0.014, which is inside the confidence intervals, and abstention signals come from single-query runs and were not affected.
