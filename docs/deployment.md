# Deployment

## Current: AWS EC2 `t4g.small` (2026-11-19)
One Docker container (`docker build`, models baked in) on the same instance as another app, Caddy in front (`codicil.<ip-with-dashes>.sslip.io`, automatic Let's Encrypt), `--memory 800m`, port 8001 on the host network, extractive mode. A 1 GB `t4g.micro` could not hold both apps (about 400 MB available), so the instance was resized to 2 GB (about 6 to 7 USD per month more, paid from AWS credits). Measured: p50 1.3 s, p95 1.5 to 1.7 s for English questions through the public HTTPS endpoint (rerank about 1.2 s of it), about 300 to 350 MB resident.

**LLM mode** uses Amazon Bedrock (`openai.gpt-oss-120b-1:0`, `ap-south-1`) through the Converse API with the instance role; the role already allowed `bedrock:InvokeModel` on that model, so no new permission was needed. Container environment: `CODICIL_LLM_PROVIDER=bedrock`, `CODICIL_LLM_MODEL`, `CODICIL_LLM_REGION=ap-south-1`, `CODICIL_LLM_REASONING_EFFORT=low`, `CODICIL_LLM_TIMEOUT_S=25`. Measured: 20 of 20 fresh English questions answered by the LLM, none fell back, p50 1.7 s, p95 2.0 s. LLM is the default answer mode on this host (the UI preselects it; Extractive is one click away and remembered per browser).

**Amazon S3 Vectors** is connected as an optional dense backend: vector bucket `codicil-vectors` in `ap-south-1` (SSE-S3), vector index `codicil-<sha12>` (cosine, 384 dimensions, 2,223 vectors), read-only IAM permissions on that bucket for the instance role (the write permission used for the upload was removed afterwards), and `CODICIL_S3VECTORS_BUCKET` / `CODICIL_S3VECTORS_REGION` in the container environment. After a corpus rebuild: build the image, run `scripts/upload_s3vectors.py` once (it creates the new `codicil-<sha12>` index), recreate the container. The UI is served by the same container, so there is no separate frontend deployment. After a corpus rebuild: `git pull`, `docker build`, recreate the container; a Qdrant collection from an older index is refused by the sha check and must be re-uploaded.

## Earlier: Render free

### One Docker web service on Render (free)
- **Why**: 512 MB RAM fits ONNX Runtime plus two small int8 models plus a 15k-vector index (budget in `architecture.md`); Docker lets the models and index be baked into the image so cold start downloads nothing; the same container serves API and UI.
- **Cold start**: the free instance sleeps after about 15 minutes idle and takes tens of seconds to wake. The UI says so and shows a "waking the service" state. No latency claims are made from the free instance.
- **Secrets**: the optional LLM key (`CODICIL_LLM_API_KEY`, with `CODICIL_LLM_BASE_URL` and `CODICIL_LLM_MODEL`) is set by the owner in the dashboard. Without it the service runs in extractive mode.
- **Build**: `docker build` runs `python -m codicil.cli fetch-models` (pinned model revisions with checksums) and copies the committed index artifact; nothing is fetched at runtime.

## Alternatives considered
| Option | Verdict |
|---|---|
| Cloudflare Workers + Workers AI + Vectorize | Free and fast, but a rewrite in TypeScript and the pipeline (clause-aware BM25, verification) is the point of the project |
| Vercel (Python functions) | Size and memory limits too tight for ONNX models |
| Streamlit Community Cloud | Easy, but a UI framework instead of a service: no API, SSE or metrics to demonstrate |
| Hugging Face Spaces (CPU) | Good fit and a fallback; 16 GB RAM would hide the memory discipline, so Render is the primary |
| Cloudflare Pages for the UI | Optional split: static UI on Pages, API on Render, CORS configured |

## Environment
`CODICIL_INDEX_DIR`, `CODICIL_EMBEDDER` (`bge-small-en` or `e5-small-multilingual`), `CODICIL_RERANK` (`on`, `off`, `auto`), `CODICIL_LLM_*`, `CODICIL_RATE_LIMIT_PER_MIN`, `CODICIL_CORS_ORIGINS`, `PORT`.
