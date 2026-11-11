from codicil.adapters.hashing_embedder import HashingEmbedder
from codicil.application.answer import AnswerConfig, AnswerService
from tests.fixtures.mini import mini_index

QUESTION = "What is the maximum interest rate?"


class FixedReranker:
    model_id = "fixed"

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, query, passages):
        return [self.value] * len(passages)


def ask(tmp_path, rerank_score, min_rerank):
    svc = AnswerService(mini_index(tmp_path), HashingEmbedder(), FixedReranker(rerank_score))
    return svc.ask(QUESTION, AnswerConfig(min_dense=0.0, min_rerank=min_rerank))


def test_low_rerank_score_abstains_for_english(tmp_path):
    a = ask(tmp_path, rerank_score=-3.0, min_rerank=-0.18)
    assert a.abstained and "closely enough" in a.abstain_reason


def test_high_rerank_score_does_not_abstain(tmp_path):
    assert not ask(tmp_path, rerank_score=4.0, min_rerank=-0.18).abstained


def test_threshold_unset_keeps_dense_rule_only(tmp_path):
    assert not ask(tmp_path, rerank_score=-9.0, min_rerank=None).abstained
