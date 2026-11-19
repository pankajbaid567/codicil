#!/bin/bash
# (Re)create the Codicil container with its production configuration. Safe to run any time; used by deploys and by the weekly refresh.
# Serves the refreshed index when one exists (/opt/codicil-data/index/current), otherwise the index baked into the image.
set -euo pipefail
DATA=/opt/codicil-data
mkdir -p "$DATA"
INDEX_ENV=()
if [ -e "$DATA/index/current" ]; then INDEX_ENV=(-e CODICIL_INDEX_DIR=/data/index/current); fi
docker rm -f codicil >/dev/null 2>&1 || true
docker run -d --name codicil --restart unless-stopped --network host --memory 800m --memory-swap 1200m \
  -v "$DATA":/data:ro \
  -e PORT=8001 -e AWS_DEFAULT_REGION=ap-south-1 \
  -e CODICIL_FRESHNESS_FILE=/data/corpus/freshness.json \
  -e CODICIL_S3VECTORS_BUCKET=codicil-vectors -e CODICIL_S3VECTORS_REGION=ap-south-1 \
  -e CODICIL_LLM_PROVIDER=bedrock -e CODICIL_LLM_MODEL=openai.gpt-oss-120b-1:0 -e CODICIL_LLM_REGION=ap-south-1 \
  -e CODICIL_LLM_REASONING_EFFORT=low -e CODICIL_LLM_TIMEOUT_S=25 \
  "${INDEX_ENV[@]}" codicil:latest >/dev/null
for _ in $(seq 1 30); do
  if curl -fs -m 5 http://127.0.0.1:8001/healthz >/dev/null; then echo "codicil up: $(curl -s http://127.0.0.1:8001/healthz)"; exit 0; fi
  sleep 3
done
echo "codicil did not become healthy" >&2
exit 1
