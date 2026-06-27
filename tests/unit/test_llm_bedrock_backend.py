"""Tests for BedrockBackend (mocked — no real AWS calls)."""
from __future__ import annotations

import sys
import types
import pytest
from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse


def _make_boto3_mock() -> MagicMock:
    """Return a minimal boto3 mock that satisfies _build_client."""
    boto3_mock = MagicMock()
    session_mock = MagicMock()
    boto3_mock.Session.return_value = session_mock
    session_mock.client.return_value = MagicMock()
    return boto3_mock


class TestBedrockBackend:
    def _make_backend(self):
        from trelix.llm.providers.bedrock_backend import BedrockBackend
        cfg = LLMConfig(
            provider="bedrock",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region="us-east-1",
            _env_file=None,  # type: ignore[call-arg]
        )
        boto3_mock = _make_boto3_mock()
        with patch.dict("sys.modules", {"boto3": boto3_mock}):
            backend = BedrockBackend(cfg)
        return backend

    def test_complete_returns_chat_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = {
            "output": {"message": {"content": [{"text": "hello"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        mock_client.converse.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"
        assert result.finish_reason == "stop"

    def test_uses_inference_config_max_tokens(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hi")], max_tokens=100)

        call_kwargs = mock_client.converse.call_args[1]
        assert "inferenceConfig" in call_kwargs
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 100
        # Must NOT have top-level max_tokens
        assert "max_tokens" not in call_kwargs
        assert "maxTokens" not in call_kwargs

    def test_system_as_top_level_list(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        backend.complete(
            [ChatMessage(role="user", content="hi")],
            system="You are helpful.",
        )
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs.get("system") == [{"text": "You are helpful."}]

    def test_content_always_list_of_dicts(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hello")])

        call_kwargs = mock_client.converse.call_args[1]
        messages = call_kwargs["messages"]
        for msg in messages:
            assert isinstance(msg["content"], list)
            assert all(isinstance(block, dict) for block in msg["content"])

    def test_tool_choice_auto_format(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {
                "content": [{"toolUse": {"toolUseId": "1", "name": "fn", "input": {"x": 1}}}],
                "role": "assistant",
            }},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        tools = [{"type": "function", "function": {"name": "fn",
                   "parameters": {"type": "object", "properties": {}}}}]
        backend.tool_call([ChatMessage(role="user", content="hi")], tools=tools)

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_import_error_when_boto3_not_installed(self) -> None:
        from trelix.llm.providers.bedrock_backend import BedrockBackend
        cfg = LLMConfig(provider="bedrock", _env_file=None)  # type: ignore[call-arg]
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="pip install"):
                BedrockBackend(cfg)
