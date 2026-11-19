"""Generator for Amazon Bedrock models through the Converse API.

Credentials come from the standard AWS chain (on EC2, the instance role: no API key is stored or minted).
Same contract as `OpenAICompatGenerator`: bounded retry with jitter and a circuit breaker; any failure is a
`GenerationError`, which makes the answer service fall back to extractive mode.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import Any

from codicil.adapters.openai_compat import CircuitBreaker
from codicil.application.ports import GenerationError, GenerationRequest, GenerationResult

# gpt-oss models can return their reasoning inline as <reasoning>...</reasoning>; it is not part of the answer
_REASONING = re.compile(r"<reasoning>.*?</reasoning>", re.S)
_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException", "InternalServerException", "ModelNotReadyException"}


def extract_text(output: dict[str, Any]) -> str:
    """The text blocks of a Converse response message, without reasoning blocks."""
    blocks = output["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b)
    return _REASONING.sub("", text).strip()


class BedrockConverseGenerator:
    def __init__(
        self,
        model: str,
        client: Any,
        reasoning_effort: str = "",
        retries: int = 1,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model_id, self._client, self._effort, self._retries, self._sleep = model, client, reasoning_effort, retries, sleep
        self.breaker = breaker or CircuitBreaker()

    @classmethod
    def connect(cls, model: str, region: str = "", timeout: float = 12.0, reasoning_effort: str = "") -> BedrockConverseGenerator:
        import boto3  # optional dependency, imported only when configured
        from botocore.config import Config

        client = boto3.client(
            "bedrock-runtime", region_name=region or None, config=Config(connect_timeout=3, read_timeout=timeout, retries={"max_attempts": 1})
        )
        return cls(model, client, reasoning_effort)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.breaker.allow():
            raise GenerationError("circuit open")
        args: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": request.system}],
            "messages": [{"role": "user", "content": [{"text": request.user}]}],
            "inferenceConfig": {"maxTokens": request.max_tokens, "temperature": request.temperature},
        }
        if self._effort:
            args["additionalModelRequestFields"] = {"reasoning_effort": self._effort}
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                res = self._client.converse(**args)
                text = extract_text(res["output"])
                if not text:
                    raise GenerationError("empty model output")
                self.breaker.record(True)
                u = res.get("usage") or {}
                return GenerationResult(text, self.model_id, u.get("inputTokens", 0), u.get("outputTokens", 0))
            except Exception as e:  # botocore errors, malformed responses; anything here means "no LLM answer"
                last = e
                code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
                if code and code not in _RETRYABLE:
                    break  # access denied, validation, unknown model: retrying cannot help
                if attempt < self._retries:
                    self._sleep(0.3 * (2**attempt) + random.random() * 0.2)
        self.breaker.record(False)
        raise GenerationError(f"generation failed: {last}" if isinstance(last, GenerationError) else f"generation failed: {type(last).__name__}") from last
