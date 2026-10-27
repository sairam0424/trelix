"""AnthropicBackend: thinking request shape per model (mocked, no network).

Claude 5 and newer, plus Opus 4.7 and 4.8, reject ``thinking={"type": "enabled",
"budget_tokens": N}`` with a 400 and take ``thinking={"type": "adaptive"}``. Every other
model, including ids nobody has classified yet, keeps the budget request it always had.
The direct-API behaviour is from the Claude API reference, not measured; the retry-and-
remember path for unclassified ids, with the error text the Bedrock Converse API returned
live on 2026-10-04, is in test_llm_adaptive_thinking_anthropic_retry.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.unit.anthropic_thinking_harness import (
    MESSAGES,
    UNCLASSIFIED_MODEL,
    install_fake_anthropic,
    make_backend,
    ok_response,
    stream_manager,
)

ADAPTIVE_ONLY_IDS = [
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-sonnet-5-5-20261001",
    "claude-opus-5",
    "claude-haiku-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
]

BUDGET_IDS = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-opus-4-6",
    "claude-opus-4-1-20250805",
    "claude-haiku-4-5",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    UNCLASSIFIED_MODEL,
    "gpt-4o",
]


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    return install_fake_anthropic(monkeypatch)


def _thinking_sent_by_complete(model: str, budget: int = 2048) -> dict[str, Any]:
    backend = make_backend(model, budget)
    client = MagicMock()
    client.messages.create.return_value = ok_response()
    backend._client = client
    backend.complete(MESSAGES, thinking=True)
    return client.messages.create.call_args[1]


def _thinking_sent_by_stream(model: str, budget: int = 2048) -> dict[str, Any]:
    backend = make_backend(model, budget)
    client = MagicMock()
    client.messages.stream.return_value = stream_manager(["ok"])
    backend._client = client
    assert list(backend.stream(MESSAGES, thinking=True)) == ["ok"]
    return client.messages.stream.call_args[1]


SENDERS = [
    pytest.param(_thinking_sent_by_complete, id="complete"),
    pytest.param(_thinking_sent_by_stream, id="stream"),
]


class TestRequestShapeByModel:
    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS)
    @pytest.mark.parametrize("send", SENDERS)
    def test_adaptive_only_models_get_adaptive_thinking(
        self, mock_anthropic: MagicMock, model: str, send: Any
    ) -> None:
        sent = send(model)

        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["model"] == model
        assert "temperature" not in sent

    @pytest.mark.parametrize("model", BUDGET_IDS)
    @pytest.mark.parametrize("send", SENDERS)
    def test_every_other_model_keeps_the_budget_request(
        self, mock_anthropic: MagicMock, model: str, send: Any
    ) -> None:
        sent = send(model)

        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert sent["model"] == model

    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS)
    @pytest.mark.parametrize("send", SENDERS)
    def test_the_budget_setting_is_not_sent_to_adaptive_models(
        self, mock_anthropic: MagicMock, model: str, send: Any
    ) -> None:
        sent = send(model, budget=9999)

        assert sent["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in repr(sent)
        assert "9999" not in repr(sent["thinking"])

    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS + BUDGET_IDS)
    def test_thinking_false_sends_no_thinking_parameter_for_any_model(
        self, mock_anthropic: MagicMock, model: str
    ) -> None:
        backend = make_backend(model)
        client = MagicMock()
        client.messages.create.return_value = ok_response()
        client.messages.stream.return_value = stream_manager(["ok"])
        backend._client = client

        backend.complete(MESSAGES, thinking=False)
        list(backend.stream(MESSAGES, thinking=False))

        assert "thinking" not in client.messages.create.call_args[1]
        assert "thinking" not in client.messages.stream.call_args[1]

    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS + BUDGET_IDS)
    def test_thinking_kwargs_is_empty_when_thinking_is_off(
        self, mock_anthropic: MagicMock, model: str
    ) -> None:
        assert make_backend(model)._thinking_kwargs(thinking=False) == {}


class TestAdaptiveResponses:
    def test_an_adaptive_response_without_a_thinking_block_parses(
        self, mock_anthropic: MagicMock
    ) -> None:
        """The model decides whether to think: live, an easy question came back with only a
        text block. That is a normal response, not an error."""
        backend = make_backend("claude-sonnet-5")
        client = MagicMock()
        client.messages.create.return_value = ok_response([("text", "4")])
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "4"
        assert result.thinking is None
        assert result.thinking_blocks == []
        assert result.finish_reason == "stop"

    def test_an_adaptive_response_with_a_thinking_block_still_surfaces_it(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend("claude-sonnet-5-5")
        client = MagicMock()
        client.messages.create.return_value = ok_response(
            [("thinking", "weighing it"), ("text", "done")]
        )
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "done"
        assert result.thinking == "weighing it"
        assert [b.type for b in result.thinking_blocks] == ["thinking"]
