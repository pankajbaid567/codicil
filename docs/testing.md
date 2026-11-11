# Testing

| Layer | Tool | What it proves |
|---|---|---|
| Unit | pytest | Pure functions: tokenizer, clause parser, chunker, BM25, RRF, verifier, metrics |
| Property | Hypothesis | Chunks cover the text and respect the token budget; RRF is permutation-stable and monotone; metrics stay in [0, 1]; the glossary never loses the original query |
| Golden | pytest | nDCG, MRR and recall against values computed by hand in the test |
| Contract | pytest, one suite per port | Every `Embedder` returns unit vectors of the declared dimension, is deterministic and handles empty input; every `Generator` honours the JSON contract and timeouts; every `IndexStore` round-trips |
| Architecture | pytest + `ast` | The domain imports no adapter, `httpx`, `fastapi` or `onnxruntime` |
| API | httpx ASGI client | Schemas, problem+json errors, rate limit, SSE framing, degradation when the LLM is down (a local fake OpenAI-compatible server) |
| Eval | `make eval-gate` | No retrieval regression against the committed baseline |
| Mutation | `scripts/mutation_check.py` | Injected bugs (off-by-one in RRF, verification disabled, abstention threshold ignored, chunker drops the last clause, wrong embedder prefix, cache key without the index hash) are all caught |
| Static | `ruff`, `mypy --strict` | Style and types |
Coverage is reported, not worshipped: the target is the domain and application layers at or above 90%.
