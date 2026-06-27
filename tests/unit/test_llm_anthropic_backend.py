"""Tests for AnthropicBackend (mocked — no real API calls)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse

# Fake key for testing — not a real credential
_FAKE_ANT_KEY = "test-anthropic-api-key-fake"


def _make_mock_anthropic_module() -> ModuleType:
    """Build a minimal fake anthropic module so the backend can be instantiated."""
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
    return mock_anthropic


@pytest.fixture()
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch sys.modules so `import anthropic` works without the real package."""
    module = _make_mock_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", module)
    # Also remove cached import of anthropic_backend so it re-runs with the patch
    monkeypatch.delitem(sys.modules, "trelix.llm.providers.anthropic_backend", raising=False)
    return module


class TestAnthropicBackend:
    def _make_backend(self, mock_anthropic_module: MagicMock):
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        cfg = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_FAKE_ANT_KEY,
            model="claude-3-5-sonnet-20241022",
            _env_file=None,  # type: ignore[call-arg]
        )
        return AnthropicBackend(cfg)

    def test_complete_returns_chat_response(self, mock_anthropic: MagicMock) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"
        assert result.finish_reason == "stop"  # normalized from "end_turn"

    def test_uses_max_tokens_not_max_completion_tokens(self, mock_anthropic: MagicMock) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hi")], max_tokens=100)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs
        assert call_kwargs["max_tokens"] == 100

    def test_system_as_separate_param(self, mock_anthropic: MagicMock) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete(
            [ChatMessage(role="user", content="hi")],
            system="You are a bot.",
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "You are a bot."
        # system must NOT appear in messages list
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    def test_finish_reason_normalization(self, mock_anthropic: MagicMock) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.model = "claude"
        mock_response.stop_reason = "max_tokens"  # Anthropic name
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])
        assert result.finish_reason == "length"  # normalized

    def test_import_error_when_anthropic_not_installed(self) -> None:
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        cfg = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_FAKE_ANT_KEY,
            _env_file=None,  # type: ignore[call-arg]
        )
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install"):
                AnthropicBackend(cfg)
