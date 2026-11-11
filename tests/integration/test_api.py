import httpx
import pytest

from codicil.adapters.hashing_embedder import HashingEmbedder
from codicil.api.app import TokenBucket, create_app
from codicil.application.answer import AnswerService
from codicil.application.ports import GenerationError
from codicil.settings import Settings
from tests.fixtures.mini import mini_index


class DownLLM:
    model_id = "down"

    def generate(self, request):
        raise GenerationError("boom")


@pytest.fixture
def make_app(tmp_path):
    def build(**kw):
        svc = AnswerService(mini_index(tmp_path / "idx"), HashingEmbedder(), None, kw.pop("generator", None))
        return create_app(Settings(rate_limit_per_min=kw.pop("rate", 100), min_dense=kw.pop("min_dense", 0.0), **kw), svc)

    return build


async def post(app, path, **kw):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, **kw)


async def get(app, path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.get(path)


async def test_ask_answers_with_citation(make_app):
    r = await post(make_app(), "/v1/ask", json={"question": "within how many days must a bank credit an unauthorised electronic transaction?", "debug": True})
    j = r.json()
    assert r.status_code == 200 and not j["abstained"] and "10 working days" in j["answer"][0]["text"]
    assert j["citations"][0]["updated_as_on"] is None or isinstance(j["citations"][0]["updated_as_on"], str)
    assert r.headers["x-request-id"] and "server-timing" in r.headers and j["timings_ms"]


async def test_ask_abstains_when_nothing_matches(make_app):
    r = await post(make_app(min_dense=0.3), "/v1/ask", json={"question": "zebra quantum helicopter submarine"})
    j = r.json()
    assert j["abstained"] and j["answer"] == [] and j["reason"]


async def test_validation_is_problem_json(make_app):
    r = await post(make_app(), "/v1/ask", json={"question": "x"})
    assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"
    r = await post(make_app(), "/v1/ask", json={"question": "a" * 501})
    assert r.status_code == 422


async def test_rate_limit(make_app):
    app = make_app(rate=2)
    codes = [(await post(app, "/v1/search", json={"query": "interest"})).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes[2:]


async def test_llm_failure_degrades_to_extractive(make_app):
    r = await post(
        make_app(generator=DownLLM()),
        "/v1/ask",
        json={"question": "within how many days must a bank credit an unauthorised electronic transaction?", "debug": True},
    )
    j = r.json()
    assert r.status_code == 200 and j["mode"] == "extractive" and any("llm failed" in t["note"] for t in j["timings_ms"])


async def test_cache_and_metrics(make_app):
    app = make_app()
    body = {"question": "how is interest on savings deposits calculated?"}
    await post(app, "/v1/ask", json=body)
    await post(app, "/v1/ask", json=body)
    assert "codicil_cache_hits_total 1" in (await get(app, "/metrics")).text


async def test_sse_and_info_endpoints(make_app):
    app = make_app()
    r = await post(app, "/v1/ask", headers={"accept": "text/event-stream"}, json={"question": "how is interest on savings deposits calculated?"})
    assert "event: sentence" in r.text and "event: done" in r.text
    assert (await get(app, "/healthz")).json()["status"] == "ok"
    assert (await get(app, "/v1/index")).json()["chunks"] >= 4
    assert (await get(app, "/v1/documents")).json()[0]["id"] == "mini"


def test_token_bucket_refills():
    t = [0.0]
    b = TokenBucket(60, clock=lambda: t[0])
    assert all(b.allow("a") for _ in range(60)) and not b.allow("a")
    t[0] = 2.0
    assert b.allow("a") and b.allow("b")


async def test_abstention_threshold_is_enforced(make_app):
    q = {"question": "within how many days must a bank credit an unauthorised electronic transaction?"}
    assert not (await post(make_app(min_dense=0.0), "/v1/ask", json=q)).json()["abstained"]
    assert (await post(make_app(min_dense=0.99), "/v1/ask", json=q)).json()["abstained"]


async def test_web_ui_is_served_at_root(make_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app()), base_url="http://t") as c:
        r = await c.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "Codicil" in r.text


async def test_ui_offers_both_modes_and_index_reports_llm_availability(make_app):
    app = make_app()
    page = (await get(app, "/")).text
    assert 'value="extractive"' in page and 'value="llm"' in page and "use_llm" in page
    assert (await get(app, "/v1/index")).json()["llm"] is None  # the UI disables the LLM option in this case
    assert (await get(make_app(generator=DownLLM()), "/v1/index")).json()["llm"] == "down"
