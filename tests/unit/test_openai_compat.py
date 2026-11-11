import httpx
import pytest

from codicil.adapters.openai_compat import CircuitBreaker, OpenAICompatGenerator
from codicil.application.ports import GenerationError, GenerationRequest

REQ = GenerationRequest("sys", "user")


def gen(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatGenerator("http://llm.test/v1", "k", "m", client=client, sleep=lambda s: None, **kw)


def ok(request):
    return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})


def test_success_and_auth_header():
    seen = {}

    def h(request):
        seen["auth"] = request.headers["authorization"]
        return ok(request)

    assert gen(h).generate(REQ).text == "hi" and seen["auth"] == "Bearer k"


def test_retries_then_succeeds():
    calls = []

    def h(request):
        calls.append(1)
        return httpx.Response(503) if len(calls) == 1 else ok(request)

    assert gen(h, retries=1).generate(REQ).text == "hi" and len(calls) == 2


def test_breaker_opens_and_recovers():
    now = [0.0]
    br = CircuitBreaker(threshold=2, cooldown=10, clock=lambda: now[0])
    g = gen(lambda r: httpx.Response(500), retries=0, breaker=br)
    for _ in range(2):
        with pytest.raises(GenerationError):
            g.generate(REQ)
    assert br.is_open
    with pytest.raises(GenerationError, match="circuit open"):
        g.generate(REQ)
    now[0] = 11.0
    assert br.allow()  # half-open probe


def test_malformed_response_is_a_generation_error():
    with pytest.raises(GenerationError):
        gen(lambda r: httpx.Response(200, json={"nope": 1}), retries=0).generate(REQ)
