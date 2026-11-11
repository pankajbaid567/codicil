# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# dependencies first: this layer is cached until pyproject.toml changes
COPY pyproject.toml ./
RUN python - <<'PY' > requirements.txt
import tomllib; p = tomllib.load(open("pyproject.toml","rb"))["project"]; print("\n".join(p["dependencies"] + p["optional-dependencies"]["qdrant"]))
PY
RUN pip install -r requirements.txt

# models are fetched at build time, pinned and checksum-verified, so the running container downloads nothing
COPY scripts/fetch_models.py scripts/fetch_models.py
COPY data/models.lock.json data/models.lock.json
RUN python scripts/fetch_models.py

COPY src ./src
COPY pyproject.toml ./
RUN pip install --no-deps .
# fail the build if the web UI did not make it into the installed package (it once did not)
RUN python -c "from codicil.api.app import STATIC; assert (STATIC / 'index.html').exists(), STATIC"
COPY index/multilingual-e5-small ./index/multilingual-e5-small

RUN useradd -m app && chown -R app /app
USER app
ENV CODICIL_INDEX_DIR=index/multilingual-e5-small PORT=10000
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "uvicorn codicil.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
