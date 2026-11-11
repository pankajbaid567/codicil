"""HTTP adapter. Thin: validation, rate limiting, caching, serialisation. All behaviour lives in the application layer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from codicil.application.answer import AnswerConfig, AnswerService
from codicil.domain.models import Answer
from codicil.settings import Settings
from codicil.wiring import build_service, default_answer_config

log = logging.getLogger("codicil")
STATIC = Path(__file__).parent / "static"


class AskBody(BaseModel):
    question: str = Field(min_length=3)
    use_llm: bool = True
    backend: Literal["local", "qdrant"] = "local"  # dense search: in-process matrix or Qdrant (falls back to local when unavailable)
    debug: bool = False


class SearchBody(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(10, ge=1, le=30)


def cache_key(index_sha: str, question: str, llm: bool, backend: str = "local") -> str:
    """The index hash is part of the key, so an answer computed against an old corpus can never be served after a redeploy."""
    return hashlib.sha256(f"{index_sha}|{question.strip().lower()}|{llm}|{backend}".encode()).hexdigest()


class TokenBucket:
    """Per-client token bucket. In-memory is right for one instance; a shared store would replace it behind the same interface."""

    def __init__(self, per_min: int, clock: Any = time.monotonic) -> None:
        self.rate, self.cap, self._clock = per_min / 60.0, float(per_min), clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._clock()
            tokens, last = self._state.get(key, (self.cap, now))
            tokens = min(self.cap, tokens + (now - last) * self.rate)
            ok = tokens >= 1.0
            self._state[key] = (tokens - 1.0 if ok else tokens, now)
            if len(self._state) > 10_000:  # bound memory under abuse
                self._state.clear()
            return bool(ok)


class LRU:
    def __init__(self, size: int) -> None:
        self.size, self._d, self._lock = size, OrderedDict[str, Any](), threading.Lock()

    def get(self, k: str) -> Any | None:
        with self._lock:
            if k in self._d:
                self._d.move_to_end(k)
                return self._d[k]
        return None

    def put(self, k: str, v: Any) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._d[k] = v
            self._d.move_to_end(k)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def problem(status: int, title: str, detail: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"type": "about:blank", "title": title, "status": status, "detail": detail, "request_id": request_id},
        status_code=status,
        media_type="application/problem+json",
    )


def answer_json(a: Answer, debug: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "abstained": a.abstained,
        "mode": a.mode,
        "index_sha": a.index_sha[:12],
        "notice": a.notice,
        "answer": [{"text": s.text, "citations": list(s.citation_ids)} for s in a.sentences],
        "citations": [
            {
                "id": c.chunk_id,
                "document": c.doc_title,
                "path": c.path,
                "quote": c.quote,
                "updated_as_on": c.updated_as_on.isoformat() if c.updated_as_on else None,
                "status": c.status.value,
            }
            for c in a.citations
        ],
    }
    if a.abstained:
        out["reason"] = a.abstain_reason
        out["closest"] = [
            {"document": c.doc_title, "path": c.path, "quote": c.quote[:300], "updated_as_on": c.updated_as_on.isoformat() if c.updated_as_on else None}
            for c in a.closest
        ]
    if debug:
        out["timings_ms"] = [{"stage": t.stage, "ms": round(t.ms, 1), "note": t.note} for t in a.timings]
        out["signals"] = dict(a.signals)
    return out


def create_app(settings: Settings | None = None, service: AnswerService | None = None) -> FastAPI:
    s = settings or Settings()
    svc = service or build_service(s)
    cfg: AnswerConfig = default_answer_config(s)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu")  # bulkhead: CPU-bound work cannot starve health checks
    bucket, cache = TokenBucket(s.rate_limit_per_min), LRU(s.cache_size)
    counters = {"requests": 0, "abstained": 0, "errors": 0, "cache_hits": 0, "rate_limited": 0}
    app = FastAPI(title="Codicil", version="0.1.0", docs_url="/docs")

    @app.middleware("http")
    async def wrap(request: Request, call_next: Any) -> Response:
        rid = uuid.uuid4().hex[:12]
        request.state.rid = rid
        t0 = time.perf_counter()
        if request.url.path.startswith("/v1/") and not bucket.allow(request.client.host if request.client else "anon"):
            counters["rate_limited"] += 1
            return problem(429, "Too many requests", "slow down and retry shortly", rid)
        try:
            resp: Response = await call_next(request)
        except Exception:
            counters["errors"] += 1
            log.exception("unhandled error rid=%s", rid)
            return problem(500, "Internal error", "unexpected failure", rid)
        ms = (time.perf_counter() - t0) * 1000
        resp.headers["X-Request-Id"] = rid
        resp.headers["Server-Timing"] = f"total;dur={ms:.1f}"
        log.info("rid=%s %s %s %d %.0fms", rid, request.method, request.url.path, resp.status_code, ms)  # no query text is ever logged
        return resp

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = ", ".join(".".join(str(x) for x in e["loc"][1:]) for e in exc.errors())
        return problem(422, "Invalid request", f"check: {fields}", getattr(request.state, "rid", "-"))

    if s.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins.split(","), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def run_ask(body: AskBody) -> Answer:
        backend = body.backend if svc.vector_store is not None else "local"
        key = cache_key(svc.index.sha, body.question, body.use_llm and svc.generator is not None, backend)
        hit = cache.get(key)
        if hit is not None:
            counters["cache_hits"] += 1
            return hit  # type: ignore[no-any-return]
        a = svc.ask(body.question.strip(), cfg, use_llm=body.use_llm, backend=backend)
        cache.put(key, a)
        return a

    @app.post("/v1/ask")
    async def ask(body: AskBody, request: Request) -> Response:
        if len(body.question) > s.max_question_chars:
            return problem(422, "Question too long", f"limit is {s.max_question_chars} characters", request.state.rid)
        counters["requests"] += 1
        a = await asyncio.get_running_loop().run_in_executor(pool, run_ask, body)
        counters["abstained"] += a.abstained
        if "text/event-stream" in request.headers.get("accept", ""):

            async def events() -> AsyncIterator[str]:
                payload = answer_json(a, body.debug)
                for sent in payload["answer"]:
                    yield f"event: sentence\ndata: {json.dumps(sent, ensure_ascii=False)}\n\n"
                yield f"event: done\ndata: {json.dumps({k: v for k, v in payload.items() if k != 'answer'}, ensure_ascii=False)}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return JSONResponse(answer_json(a, body.debug))

    @app.post("/v1/search")
    async def search(body: SearchBody) -> dict[str, Any]:
        def go() -> list[dict[str, Any]]:
            from codicil.domain.glossary import normalise_query

            q = normalise_query(body.query)
            hits, _ = svc.retrieve(q, cfg)
            out = []
            for h in hits[: body.top_k]:
                c = svc.index.chunks[svc.index.by_id[h.chunk_id]]
                out.append({"rank": h.rank, "score": round(h.score, 5), "document": svc.index.docs[c.doc_id].short_title, "path": c.path_str, "text": c.text})
            return out

        return {"language": normalise_query_lang(body.query), "results": await asyncio.get_running_loop().run_in_executor(pool, go)}

    def normalise_query_lang(text: str) -> str:
        from codicil.domain.glossary import detect_language

        return detect_language(text)

    @app.get("/v1/documents")
    async def documents() -> list[dict[str, Any]]:
        return [
            {
                "id": d.doc_id,
                "title": d.short_title,
                "entity": d.entity,
                "updated_as_on": d.updated_as_on.isoformat() if d.updated_as_on else None,
                "status": d.status.value,
                "url": d.url,
            }
            for d in svc.index.docs.values()
        ]

    @app.get("/v1/index")
    async def index_info() -> dict[str, Any]:
        i = svc.index
        return {
            "sha256": i.sha,
            "embedder": i.embedder_id,
            "chunks": len(i.chunks),
            "parents": len(i.parents),
            "documents": len(i.docs),
            "reranker": svc.reranker.model_id if svc.reranker else None,
            "llm": svc.generator.model_id if svc.generator else None,
            "qdrant": svc.vector_store is not None,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "index": svc.index.sha[:12]}

    @app.get("/metrics")
    async def metrics() -> Response:
        body = "".join(f"codicil_{k}_total {v}\n" for k, v in counters.items())
        return Response(body, media_type="text/plain; version=0.0.4")

    if STATIC.exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
    return app


def asdict_safe(x: Any) -> Any:
    return asdict(x)
