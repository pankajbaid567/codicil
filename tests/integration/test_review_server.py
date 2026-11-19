import importlib.util
import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, rel: str):
    sys.path.insert(0, str(ROOT / "eval"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def server(tmp_path):
    mod = load("review_labels", "scripts/review_labels.py")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    items = [{"id": "q1", "question": "?", "gold": [], "nearest": []}, {"id": "q2", "question": "?", "gold": [], "nearest": []}]
    log = tmp_path / "reviews" / "reviews.jsonl"
    srv = ThreadingHTTPServer(("127.0.0.1", port), mod.make_handler(items, port, log))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", log
    srv.shutdown()
    srv.server_close()


def test_page_items_and_reviews_are_served_locally(server):
    base, _ = server
    assert "Codicil label review" in httpx.get(base + "/").text
    assert [i["id"] for i in httpx.get(base + "/api/items").json()] == ["q1", "q2"]
    assert httpx.get(base + "/api/reviews").json() == {}


def test_a_review_is_appended_and_the_last_verdict_wins(server):
    base, log = server
    h = {"X-Review": "1"}
    assert httpx.post(base + "/api/review", json={"id": "q1", "verdict": "wrong_evidence", "note": "n"}, headers=h).status_code == 200
    assert httpx.post(base + "/api/review", json={"id": "q1", "verdict": "ok"}, headers=h).status_code == 200
    assert len(log.read_text().splitlines()) == 2  # append-only
    assert httpx.get(base + "/api/reviews").json()["q1"]["verdict"] == "ok"


@pytest.mark.parametrize(
    "headers,body,code",
    [
        ({}, {"id": "q1", "verdict": "ok"}, 403),  # no X-Review header: a cross-site form post cannot set it
        ({"X-Review": "1"}, {"id": "q1", "verdict": "great"}, 400),
        ({"X-Review": "1"}, {"id": "nope", "verdict": "ok"}, 400),
        ({"X-Review": "1"}, ["not", "an", "object"], 400),
    ],
)
def test_bad_posts_are_refused_and_nothing_is_written(server, headers, body, code):
    base, log = server
    assert httpx.post(base + "/api/review", json=body, headers=headers).status_code == code
    assert not log.exists()


def test_requests_under_another_host_name_are_refused(server):
    base, _ = server
    assert httpx.get(base + "/api/items", headers={"Host": "evil.example"}).status_code == 403  # DNS-rebinding guard
    assert httpx.post(base + "/api/review", json={"id": "q1", "verdict": "ok"}, headers={"Host": "evil.example", "X-Review": "1"}).status_code == 403


def test_sensitivity_helpers_use_only_the_selected_questions():
    mod = load("review_sensitivity", "eval/review_sensitivity.py")
    rows = [{"id": "a", "ndcg@10": 1.0, "recall@10": 1.0, "mrr@10": 1.0}, {"id": "b", "ndcg@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0}]
    assert mod.retrieval_means(rows, None)["ndcg@10"] == 0.5 and mod.retrieval_means(rows, {"a"})["ndcg@10"] == 1.0
    assert mod.retrieval_means(rows, {"zzz"}) == {"n": 0}
    sig = [
        {"id": "x", "language": "en", "answerable": False, "signals": {"reranked": 1, "top_score": -3.0, "top_dense": 0.5, "coverage": 1}, "answer_hit": False},
        {"id": "y", "language": "en", "answerable": True, "signals": {"reranked": 1, "top_score": 5.0, "top_dense": 0.9, "coverage": 1}, "answer_hit": True},
        {
            "id": "z",
            "language": "hinglish",
            "answerable": True,
            "signals": {"reranked": 0, "top_score": 0, "top_dense": 0.9, "coverage": 1},
            "answer_hit": True,
        },
    ]
    a = mod.abstention(sig, None)
    assert a["n"] == 2 and a["n_unanswerable"] == 1 and a["abstain_precision"] == 1.0 and a["abstain_recall"] == 1.0
    assert mod.abstention(sig, {"y"}) == {"n": 1}  # nothing unanswerable confirmed: no rates to report
