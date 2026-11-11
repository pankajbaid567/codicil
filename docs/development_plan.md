# Development plan

Checklist style. An item is ticked only when its exit criterion is met by a command that can be re-run.

## Phase 0: design (this folder)
- [x] PRD, RAG design, architecture, ADRs, evaluation design, testing, deployment

## Phase 1: foundations
- [x] Scaffold: `pyproject.toml`, `uv`, ruff, mypy strict, pytest, Makefile, CI
- [x] Domain models, settings, error types, ports and fakes
- [x] Architecture test for the dependency rule

## Phase 2: corpus and structure
- [x] Fetch script with manifest and hashes; PDF extraction with page offsets
- [x] Header/footer removal, de-hyphenation; golden tests on real pages
- [x] Clause-tree parser; property tests; parse 16 real documents with a report of unparsed lines

## Phase 3: indexing and retrieval
- [x] Chunker (clause, fixed-window, sentence-window) with property tests
- [x] BM25 (own implementation) with golden tests
- [x] ONNX embedder adapter, exact vector index, RRF fusion, filters, entity boost
- [x] Index artifact build and load, content hash, embedder-mismatch guard
- [x] Optional ONNX cross-encoder reranker (English only, by ablation)

## Phase 4: answering
- [x] Query normalisation (language detection, glossary)
- [x] Context packing with parent expansion
- [x] Extractive generator; OpenAI-compatible generator with breaker, retry, timeout
- [x] Support verifier and abstention; temporal citations

## Phase 5: service
- [x] FastAPI app: ask, search, documents, index, health, metrics; SSE; problem+json; rate limit
- [x] Static UI with citations, retrieval trace and abstention states

## Phase 6: evaluation
- [x] Question set with evidence quotes; quote verifier; split
- [x] Metrics with bootstrap CIs; ablation runner; tau and weight tuning on dev
- [x] Regression gate; `docs/results.md` generated from raw JSON

## Phase 7: proof and ship
- [x] Mutation check; latency and memory run
- [x] Dockerfile, `render.yaml`, CI written (not yet built or run: Docker daemon unavailable locally; CI unrun until pushed); README with measured numbers and limits
- [ ] Resume-claims register; user review of eval labels; first push, CI green, Render deploy and smoke test
