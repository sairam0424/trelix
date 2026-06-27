"""Tests for LiteLLMBackend (mocked — litellm not required)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse


def _litellm_module():
    """Return a minimal mock for litellm."""
    mock_litellm = MagicMock()
    mock_litellm.drop_params = False
    return mock_litellm


class TestLiteLLMBackend:
    def _make_backend(self):
        mock_litellm = _litellm_module()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            from trelix.llm.providers.litellm_backend import LiteLLMBackend

            cfg = LLMConfig(
                provider="litellm",
                litellm_model="bedrock/claude-3-5-sonnet",
                _env_file=None,  # type: ignore[call-arg]
            )
            backend = LiteLLMBackend(cfg)
        # Replace the internal litellm ref with the same mock for assertions
        backend._litellm = mock_litellm
        return backend, mock_litellm

    def test_complete_calls_litellm_completion(self) -> None:
        backend, mock_litellm = self._make_backend()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "bedrock/claude-3-5-sonnet"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"

    def test_uses_litellm_model_string(self) -> None:
        backend, mock_litellm = self._make_backend()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "bedrock/claude-3-5-sonnet"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_litellm.completion.return_value = mock_response

        backend.complete([ChatMessage(role="user", content="hi")])
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "bedrock/claude-3-5-sonnet"

    def test_import_error_when_litellm_not_installed(self) -> None:
        # Remove cached litellm_backend to force fresh import
        for key in list(sys.modules.keys()):
            if "litellm_backend" in key:
                del sys.modules[key]
        with patch.dict("sys.modules", {"litellm": None}):
            from trelix.llm.providers.litellm_backend import LiteLLMBackend

            cfg = LLMConfig(provider="litellm", _env_file=None)  # type: ignore[call-arg]
            with pytest.raises(ImportError, match="pip install"):
                LiteLLMBackend(cfg)
