"""Generator for any OpenAI-compatible chat endpoint (Groq, OpenRouter, Gemini, Ollama).

Timeout, bounded retry with jitter, circuit breaker.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable

import httpx

from codicil.application.ports import GenerationError, GenerationRequest, GenerationResult


class CircuitBreaker:
    """Opens after `threshold` consecutive failures; lets one probe through after `cooldown` seconds."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold, self.cooldown, self._clock = threshold, cooldown, clock
        self._fails = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._clock() - self._opened_at >= self.cooldown:
                self._opened_at = self._clock()  # half-open: one probe, re-armed
                return True
            return False

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._fails, self._opened_at = 0, None
            else:
                self._fails += 1
                if self._fails >= self.threshold:
                    self._opened_at = self._clock()

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None


class OpenAICompatGenerator:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 12.0,
        reasoning_effort: str = "",
        send_temperature: bool = True,
        retries: int = 1,
        client: httpx.Client | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model_id, self._effort, self._send_temperature = model, reasoning_effort, send_temperature
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key, self._retries, self._sleep = api_key, retries, sleep
        self._client = client or httpx.Client(timeout=timeout)
        self.breaker = breaker or CircuitBreaker()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.breaker.allow():
            raise GenerationError("circuit open")
        body = {
            "model": self.model_id,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "system", "content": request.system}, {"role": "user", "content": request.user}],
        }
        if self._send_temperature:
            body["temperature"] = request.temperature
        if self._effort:
            body["reasoning_effort"] = self._effort
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                r = self._client.post(self._url, json=body, headers={"Authorization": f"Bearer {self._key}"})
                if r.status_code >= 500 or r.status_code == 429:
                    raise GenerationError(f"provider status {r.status_code}")
                if r.status_code >= 400:
                    raise GenerationError(f"provider status {r.status_code}: {r.text[:140]!r}")
                j = r.json()
                self.breaker.record(True)
                u = j.get("usage") or {}
                return GenerationResult(
                    j["choices"][0]["message"]["content"], j.get("model", self.model_id), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                )
            except (httpx.HTTPError, GenerationError, KeyError, ValueError, IndexError) as e:
                last = e
                if attempt < self._retries:
                    self._sleep(0.3 * (2**attempt) + random.random() * 0.2)
        self.breaker.record(False)
        raise GenerationError(f"generation failed: {last}" if isinstance(last, GenerationError) else f"generation failed: {type(last).__name__}") from last
