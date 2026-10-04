"""Shared fixtures for the BedrockBackend thinking-mode tests (no network, no boto3 needed).

Not a test module (no ``test_`` prefix), imported as ``tests.unit.bedrock_thinking_harness``
like ``review_workflow_harness``. Confirmed live on Bedrock Converse (us-east-1, 2026-10-04)
for us.anthropic.claude-sonnet-5 and us.anthropic.claude-sonnet-5-5:

* temperature=0.0 -> ValidationException "`temperature` is deprecated for this model."
* additionalModelRequestFields {"reasoning_config": {"type": "enabled", ...}} ->
  ValidationException with LIVE_ENABLED_THINKING_REJECTION below.
* additionalModelRequestFields {"thinking": {"type": "adaptive"}} -> OK, with or without
  temperature=1.0; an easy question came back with a text block and no reasoningContent.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage

LIVE_ENABLED_THINKING_REJECTION = (
    '"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" '
    'and "output_config.effort" to control thinking behavior.'
)

# A model the classifier has never seen: budget by default, so the first thinking request
# is the budget shape and only a live rejection moves it to adaptive.
UNCLASSIFIED_MODEL = "us.anthropic.claude-nova-7"

HAIKU_FALLBACK = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

MESSAGES = [ChatMessage(role="user", content="hi")]

OK_RESPONSE: dict[str, Any] = {
    "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1},
}

Responder = Callable[[dict[str, Any]], dict[str, Any]]

# The most requests one test may send through a recording client. The longest legitimate
# single call sends five (budget with temperature, budget without, adaptive, then budget and
# adaptive again on the fallback model); the eight-thread tests send sixteen between them.
MAX_REQUESTS = 25


class ValidationException(Exception):
    """Simulated botocore ValidationException (same stand-in the other Bedrock tests use)."""


def make_backend(model: str, fallback: str | None = None, budget: int = 2048) -> Any:
    from trelix.llm.providers.bedrock_backend import BedrockBackend

    extra: dict[str, Any] = {"bedrock_fallback_model": fallback} if fallback else {}
    config = LLMConfig(
        provider="bedrock",
        model=model,
        aws_region="us-east-1",
        thinking_budget_tokens=budget,
        _env_file=None,  # type: ignore[call-arg]
        **extra,
    )
    boto3_mock = MagicMock()
    boto3_mock.Session.return_value.client.return_value = MagicMock()
    botocore_config_mock = MagicMock()
    botocore_mock = MagicMock()
    botocore_mock.config = botocore_config_mock
    modules = {
        "boto3": boto3_mock,
        "botocore": botocore_mock,
        "botocore.config": botocore_config_mock,
    }
    with patch.dict("sys.modules", modules):
        return BedrockBackend(config)


def recording_client(
    responder: Responder | None = None, response: dict[str, Any] | None = None
) -> tuple[MagicMock, list[dict[str, Any]]]:
    """A fake bedrock-runtime client whose converse/converse_stream snapshot each request
    (the backend reuses and mutates one dict across its in-call retries), then defer to
    *responder(request)* when given, else return *response*."""
    client = MagicMock()
    seen: list[dict[str, Any]] = []

    def record(kwargs: dict[str, Any]) -> dict[str, Any]:
        # Return this call's own snapshot, not seen[-1]: threads share `seen`, and another
        # thread's append between the two statements would hand back its request instead.
        snapshot = copy.deepcopy(kwargs)
        seen.append(snapshot)
        if len(seen) > MAX_REQUESTS:
            # A retry loop that never ends must fail the test quickly instead of wedging it.
            raise RuntimeError(f"runaway retry loop: more than {MAX_REQUESTS} requests")
        return snapshot

    def converse(**kwargs: Any) -> Any:
        request = record(kwargs)
        return responder(request) if responder else (response or OK_RESPONSE)

    def converse_stream(**kwargs: Any) -> Any:
        request = record(kwargs)
        if responder:
            responder(request)
        return {"stream": []}

    client.converse.side_effect = converse
    client.converse_stream.side_effect = converse_stream
    return client, seen


def raising(error: BaseException) -> Responder:
    """A responder that fails every request with *error*."""

    def responder(_request: dict[str, Any]) -> dict[str, Any]:
        raise error

    return responder


def reject_budget_thinking(request: dict[str, Any]) -> dict[str, Any]:
    """Behave like the live Sonnet 5 models: refuse reasoning_config, accept anything else."""
    if "reasoning_config" in request.get("additionalModelRequestFields", {}):
        raise ValidationException("ValidationException: " + LIVE_ENABLED_THINKING_REJECTION)
    return OK_RESPONSE


def model_unavailable_for(model_id: str) -> Responder:
    """A responder for which *model_id* is not available on demand (the condition that
    makes the backend swap to its fallback) and every other model answers."""

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        if request["modelId"] == model_id:
            raise ValidationException(
                "ValidationException: Invocation of model ID with on-demand throughput "
                "isn't supported."
            )
        return OK_RESPONSE

    return responder
