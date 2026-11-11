# Codicil: product requirements

**One line.** Ask a question about Indian banking rules in English, Hindi or Hinglish; get an answer that quotes the exact clause of the current RBI Directions, says "as on <date>", and says "not found in these documents" when that is the truth.

## Why this problem
RBI consolidated its rulebook in 2025-26 into entity-wise "Directions" (Commercial Banks, NBFCs, Payments Banks and so on), many "Updated as on October 1, 2026". Those texts are newer than any language model's training data, long (the KYC Directions alone run to many dozens of pages), full of cross-references and provisos, and they change. That makes it a real retrieval problem rather than a demo: a model without retrieval cannot know the answer, and a model with sloppy retrieval will quote the wrong entity's rule or a superseded one with full confidence.

## Users and jobs
| User | Job |
|---|---|
| A salaried customer (VITRAM-style) | "Can my bank charge me X?", "How long do they have to fix a failed transaction?" in plain language, including Hinglish |
| A fintech developer or analyst | Find the governing clause for a product rule and cite it |
| A recruiter or interviewer | See a RAG system whose retrieval, abstention and citations are measured, not asserted |

## Goals
1. **Grounded**: every sentence of an answer is tied to a cited clause; unsupported sentences are removed; if nothing survives the answer is an explicit abstention.
2. **Measured**: retrieval (recall@k, MRR, nDCG), abstention (precision/recall), citation support and latency are reported with confidence intervals on a labelled set, with an ablation of every pipeline stage.
3. **Free to run and deploy**: works with no API key (extractive mode); an optional free-tier LLM only improves phrasing. Deploys on a free web tier within 512 MB.
4. **Engineered**: hexagonal architecture, immutable versioned index, typed code, property and contract tests, and a mutation check.

## Non-goals
Legal or financial advice (it says so on every answer); personalised recommendations; a general chatbot; documents beyond the RBI Directions in v1 (SEBI and others are designed for, not built); multi-turn memory; fine-tuning; user accounts.

## Success metrics (targets are set before measuring; results go in `docs/results.md`)
| Metric | Target | Why this number |
|---|---|---|
| Recall@10 on answerable questions (test split) | >= 0.85 | A reader can only be shown what was retrieved |
| nDCG@10, hybrid vs best single retriever | hybrid not worse, ideally +0.03 | Otherwise fusion is complexity without value |
| Abstention on unanswerable questions | precision >= 0.8, recall >= 0.7 | Saying "not found" correctly is the feature |
| Citation support rate of returned answers | >= 0.95 (lexical check) | Grounding must be checkable |
| p95 latency, extractive mode, warm | < 800 ms on the free tier | Usable without a loading bar |
| Memory | < 400 MB resident | Free tier is 512 MB |
Targets that are missed will be reported as missed.

## Constraints
Free tier only (Render, Cloudflare, Vercel, Streamlit or Hugging Face); no paid APIs; the author's machine has 8 GB RAM and no GPU; public documents only; every claim in the README must trace to a command.
