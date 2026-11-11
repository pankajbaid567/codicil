#!/usr/bin/env python3
"""Start the real server in a subprocess, send questions one at a time, report latency percentiles and resident memory.
Closed-loop, single client, local machine: this measures the pipeline, not capacity, and says nothing about the free hosting tier."""

from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else 0.0


def main() -> None:
    qs = [json.loads(line)["question"] for line in (ROOT / "eval/questions.jsonl").read_text().splitlines() if line]
    env = {**os.environ, "CODICIL_RATE_LIMIT_PER_MIN": "100000", "CODICIL_CACHE_SIZE": "0", "CODICIL_LOG_LEVEL": "WARNING"}
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "codicil.api.main:app", "--port", str(PORT), "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as c:
            while True:
                try:
                    if c.get("/healthz").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.2)
            startup = time.time() - t0
            rss_idle = rss_mb(proc.pid)
            for q in qs[:5]:
                c.post("/v1/ask", json={"question": q})
            lat, peak = [], rss_idle
            for q in qs * 2:
                t = time.perf_counter()
                c.post("/v1/ask", json={"question": q})
                lat.append((time.perf_counter() - t) * 1000)
                peak = max(peak, rss_mb(proc.pid))
        pct = lambda p: sorted(lat)[min(len(lat) - 1, int(p / 100 * len(lat)))]  # noqa: E731
        print(
            json.dumps(
                {
                    "requests": len(lat),
                    "startup_s": round(startup, 1),
                    "rss_idle_mb": round(rss_idle),
                    "rss_peak_mb": round(peak),
                    "p50_ms": round(pct(50)),
                    "p95_ms": round(pct(95)),
                    "p99_ms": round(pct(99)),
                    "mean_ms": round(st.mean(lat)),
                },
                indent=1,
            )
        )
    finally:
        proc.terminate()
        proc.wait(10)


if __name__ == "__main__":
    main()
