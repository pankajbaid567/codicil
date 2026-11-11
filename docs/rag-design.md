# RAG design

The pipeline is a chain of small, separately testable stages. Each stage has a typed input and output, a latency budget, and an ablation that shows whether it earns its place.

```
 offline (build the index)                         online (answer a question)
 ───────────────────────────                       ───────────────────────────────────────────────
 fetch ─► parse ─► structure ─► chunk ─► embed     question ─► normalise ─► embed query ─┐
            │         tree       (child+     │                  (lang, glossary)         │
            │                    parent)     ▼                                           ▼
            └────────── manifest      IndexArtifact ◄── BM25 + vectors ──► hybrid retrieve (RRF)
                        (sha256)       (content-addressed)                       │
                                                                  filter (issuer/in force)
                                                                                 │
                                                                    rerank (optional, deadline)
                                                                                 │
                                                          expand to parent, pack to token budget
                                                                                 │
                                                       generate (extractive | LLM, strict contract)
                                                                                 │
                                                       verify support per sentence ─► abstain?
                                                                                 │
                                                                       answer + citations + trace
```

## 1. Fetch and parse
- A **manifest** (`data/corpus/manifest.json`) lists each document: title, issuer, URL, retrieval date, SHA-256, "updated as on" date, and `supersedes` / `superseded_by` where the source says so. PDFs are not committed; the fetch script is polite (sequential, identifying User-Agent, delay) and verifies hashes.
- Text is extracted per page with layout-aware reading order, then cleaned: running headers and footers removed by frequency across pages, hyphenated line breaks repaired, whitespace normalised. Every cleaned character keeps an offset back to its page, so a citation can say "page 41".

## 2. Structure (the part most RAG demos skip)
Regulatory text is a tree: Chapter > Section > numbered paragraph (`12.3.1`) > lettered item `(a)` > roman item `(i)`, plus provisos ("Provided that ...") and explanations that modify the sentence before them. A parser recognises these markers and builds a **clause tree**. A paragraph path like `Chapter III > 12 > 12.3 > (b)` becomes part of every chunk's identity, because the rule in `(b)` is meaningless without the stem in `12.3`.

## 3. Chunking: small-to-big with contextual headers
- **Child chunks** (retrieval units, about 120 to 220 tokens) are built from whole clauses; a clause longer than the budget is split at sentence boundaries; short sibling clauses are merged only within the same parent. Clauses are never split mid-sentence and provisos stay with the sentence they modify.
- **Contextual header**, deterministic and free: `RBI | Credit Cards and Debit Cards: Issuance and Conduct Directions, 2025 | updated 1 Oct 2026 | para 12.3 (b)`. It is embedded with the chunk and indexed in the sparse field, which is the cheap, reproducible form of "contextual retrieval".
- **Parent** = the enclosing numbered paragraph (or section). Retrieval ranks children; the reader sees the parent so a quoted rule keeps its stem and provisos.
- Baselines kept for the ablation: fixed 256-token windows with 15% overlap, and sentence windows.

## 4. Indexing
- **Sparse**: own BM25 implementation (Okapi, k1 and b tuned on the dev split) over two fields, header and body, with a field boost; Unicode-aware tokenizer, English stop words, light stemming; Devanagari text tokenised on whitespace without stemming.
- **Dense**: ONNX embedding model, 384-d, int8, L2-normalised, stored as float16. E5-family models need `query:` and `passage:` prefixes; the adapter owns that so callers cannot get it wrong.
- **Index artifact**: `manifest.json` + `chunks.jsonl` + `vectors.f16` + `bm25.json`, addressed by a SHA-256 over all of them. The index records the embedder id and dimension and refuses to be queried by a mismatched embedder.

## 5. Query understanding
- **Language and script**: Devanagari, romanised Hindi (Hinglish) detected with a small lexicon of high-frequency Hindi function words and banking terms, otherwise English.
- **Normalisation**: a curated domain glossary maps Hinglish and Hindi banking terms to canonical English ("byaj" to "interest", "karz" and "loan", "khata" to "account", "dhokhadhadi" to "fraud", "chargeback", "EMI" stays) and the original query is kept alongside it. With an LLM configured, translation is delegated to it; without one the glossary route is the fallback. Whether this helps is an ablation, not an assumption.
- **Intent hints**: entity filters from the text ("NBFC", "payments bank", "credit card") become soft boosts, because Commercial Banks and NBFCs have near-identical clause text and the wrong entity is the most dangerous failure.

## 6. Retrieval
1. BM25 top 50 and dense top 50 over children.
2. **Reciprocal Rank Fusion**, `score = sum 1/(k + rank)`, `k = 60`. RRF is used because BM25 and cosine scores are not comparable and per-query score normalisation is brittle.
3. **Filters and priors**: in-force documents only by default; a superseded document is returned only when asked for or flagged; entity boost from the hints above.
4. **Rerank** top 20 with a cross-encoder when the latency budget allows (deadline-aware: skipped, and the skip recorded in the trace, if the budget is gone). Kept only if the ablation shows a real nDCG gain.
5. **Diversify**: drop near-duplicates (the same clause appears in several entity-wise Directions) with MMR-style suppression, but keep one citation per entity when the question names none, because the answer may differ by entity.
6. **Pack**: expand children to parents, then fill a token budget (default 1,500 tokens) in rank order, merging overlapping parents.

## 7. Generation
- **Extractive mode (default, no key)**: select the evidence sentences that best match the question (cross-encoder or lexical score), return them verbatim with citations. Deterministic, free, and the baseline every LLM answer must beat.
- **LLM mode (optional)**: any OpenAI-compatible endpoint (Groq, OpenRouter, Gemini's compatible endpoint, a local Ollama). Temperature 0, a strict JSON contract `{answer_sentences: [{text, cited_chunk_ids}], abstain: bool}`, evidence passed inside delimiters with an instruction that it is data, never instructions.
- **Resilience**: timeout, bounded retry with jitter, circuit breaker; on failure degrade to extractive and say so.

## 8. Verification and abstention
- **Sentence-level support check**: each generated sentence must be supported by at least one cited chunk, tested with a numeric-and-entity-aware lexical entailment (all numbers and named entities in the sentence must appear in the evidence; content-word recall above a threshold). Optional NLI model slot behind the same interface. Unsupported sentences are dropped; none left means abstain.
- **Abstain** when retrieval is weak. English: the cross-encoder's top score is below a threshold chosen on the dev split (ADR-022). Hindi: the top dense cosine is below 0.84. Hinglish: no score abstention, because no signal separated the classes. Thresholds, data and caveats are published in `docs/results.md`.
- **Temporal honesty**: every answer states the "updated as on" date of each cited document.

## 9. Things deliberately not done (and why)
Semantic chunking by embedding similarity (nondeterministic and slower to rebuild), HyDE and multi-query (extra LLM calls, tested in the ablation only when a key exists), a semantic answer cache (a wrong hit is worse than a miss), an ANN index (15k vectors fit in a brute-force matmul well under budget), fine-tuning (no labelled data worth the cost).
