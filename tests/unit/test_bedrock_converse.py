import pytest

from codicil.adapters.bedrock_converse import BedrockConverseGenerator, extract_text
from codicil.adapters.openai_compat import CircuitBreaker
from codicil.application.ports import GenerationError, GenerationRequest


class _ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class _Fake:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def converse(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def _ok(text, i=11, o=7):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "usage": {"inputTokens": i, "outputTokens": o}}


REQ = GenerationRequest("sys", "question", max_tokens=300, temperature=0.0)
NOSLEEP = lambda s: None  # noqa: E731


def test_text_is_returned_without_reasoning_blocks_or_inline_reasoning():
    out = {"message": {"content": [{"reasoningContent": {"reasoningText": {"text": "thinking"}}}, {"text": '<reasoning>plan</reasoning>{"sentences":[]}'}]}}
    assert extract_text(out) == '{"sentences":[]}'


def test_request_shape_and_usage():
    fake = _Fake(_ok('{"sentences":[]}'))
    r = BedrockConverseGenerator("openai.gpt-oss-120b-1:0", fake, reasoning_effort="low", sleep=NOSLEEP).generate(REQ)
    assert (r.text, r.model, r.prompt_tokens, r.completion_tokens) == ('{"sentences":[]}', "openai.gpt-oss-120b-1:0", 11, 7)
    c = fake.calls[0]
    assert c["modelId"] == "openai.gpt-oss-120b-1:0" and c["system"] == [{"text": "sys"}]
    assert c["inferenceConfig"] == {"maxTokens": 300, "temperature": 0.0}
    assert c["additionalModelRequestFields"] == {"reasoning_effort": "low"}


def test_transient_error_is_retried_but_access_denied_is_not():
    fake = _Fake(_ClientError("ThrottlingException"), _ok("fine"))
    assert BedrockConverseGenerator("m", fake, retries=1, sleep=NOSLEEP).generate(REQ).text == "fine"
    denied = _Fake(_ClientError("AccessDeniedException"), _ok("never"))
    with pytest.raises(GenerationError):
        BedrockConverseGenerator("m", denied, retries=2, sleep=NOSLEEP).generate(REQ)
    assert len(denied.calls) == 1


def test_empty_output_and_open_circuit_raise():
    with pytest.raises(GenerationError, match="empty"):
        BedrockConverseGenerator("m", _Fake(_ok("<reasoning>x</reasoning>")), retries=0, sleep=NOSLEEP).generate(REQ)
    breaker = CircuitBreaker(threshold=1, cooldown=999)
    g = BedrockConverseGenerator("m", _Fake(_ClientError("ThrottlingException")), retries=0, breaker=breaker, sleep=NOSLEEP)
    with pytest.raises(GenerationError):
        g.generate(REQ)
    with pytest.raises(GenerationError, match="circuit open"):
        g.generate(REQ)
