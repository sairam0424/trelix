"""Shared fakes for the AnthropicBackend thinking-mode tests (no network, no SDK needed).

Not a test module (no ``test_`` prefix), imported as ``tests.unit.anthropic_thinking_harness``
like ``bedrock_thinking_harness``. The error text is the one the Bedrock Converse API returned
live on 2026-10-04 for us.anthropic.claude-sonnet-5 and -5-5 when sent the budget shape; the
direct Anthropic API behaviour is from the Claude API reference, not measured.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage

# Fake key for testing -- not a real credential
_TEST_FAKE_KEY = "test-anthropic-fake-key-for-unit-tests"

LIVE_ENABLED_THINKING_REJECTION = (
    '"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" '
    'and "output_config.effort" to control thinking behavior.'
)

# A model the classifier has never seen: budget by default, so the first thinking request
# is the budget shape and only a live rejection moves it to adaptive.
UNCLASSIFIED_MODEL = "claude-nova-7"

MESSAGES = [ChatMessage(role="user", content="hi")]


class FakeBadRequest(Exception):
    """Stands in for ``anthropic.BadRequestError``: what the backend reads from it is the
    ``status_code`` attribute and the message."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch sys.modules so `import anthropic` works without the real package, and drop any
    cached backend module so the next import binds to the fake. Each test module wraps this
    in its own ``mock_anthropic`` fixture."""
    module = MagicMock()
    module.Anthropic = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.delitem(sys.modules, "trelix.llm.providers.anthropic_backend", raising=False)
    return module


def ok_response(blocks: list[tuple[str, str]] | None = None) -> MagicMock:
    """A Messages API response. ``blocks`` is (type, value) pairs; default is one text
    block, which is what an adaptive request on an easy question returned live."""
    response = MagicMock()
    content = []
    for block_type, value in blocks or [("text", "the answer")]:
        block = MagicMock()
        block.type = block_type
        if block_type == "text":
            block.text = value
        else:
            block.thinking = value
            block.signature = "sig"
        content.append(block)
    response.content = content
    response.model = "claude-test"
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 1
    response.usage.output_tokens = 1
    return response


def make_backend(model: str, budget: int = 2048) -> Any:
    from trelix.llm.providers.anthropic_backend import AnthropicBackend

    config = LLMConfig(
        provider="anthropic",
        anthropic_api_key=_TEST_FAKE_KEY,
        model=model,
        thinking_budget_tokens=budget,
        _env_file=None,  # type: ignore[call-arg]
    )
    return AnthropicBackend(config)


def stream_manager(chunks: list[str]) -> MagicMock:
    manager = MagicMock()
    stream = MagicMock()
    stream.text_stream = iter(chunks)
    manager.__enter__.return_value = stream
    return manager


def failing_stream_manager(error: Exception) -> MagicMock:
    """The real SDK makes the HTTP request in ``__enter__``, so that is where a 400 raises."""
    manager = MagicMock()
    manager.__enter__.side_effect = error
    return manager
