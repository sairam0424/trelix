"""Tests for OpenAIBackend (mocked — no real API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse
from trelix.llm.providers.openai_backend import OpenAIBackend, _token_limit_param

_FAKE_KEY = "test-k"  # short enough not to trigger secret scanner; never sent to any service


class TestTokenLimitParam:
    def test_gpt4o_uses_max_completion_tokens(self) -> None:
        result = _token_limit_param("gpt-4o", 100)
        assert result == {"max_completion_tokens": 100}

    def test_gpt4_uses_max_tokens(self) -> None:
        result = _token_limit_param("gpt-4", 100)
        assert result == {"max_tokens": 100}

    def test_gpt35_turbo_uses_max_tokens(self) -> None:
        result = _token_limit_param("gpt-3.5-turbo", 100)
        assert result == {"max_tokens": 100}

    def test_o3_uses_max_completion_tokens(self) -> None:
        result = _token_limit_param("o3", 100)
        assert result == {"max_completion_tokens": 100}

    def test_azure_deployment_name_uses_max_completion_tokens(self) -> None:
        # Azure deployment names don't start with legacy prefixes
        result = _token_limit_param("my-gpt4o-deployment", 100)
        assert result == {"max_completion_tokens": 100}


class TestOpenAIBackendComplete:
    def _make_backend(self, provider: str = "openai") -> OpenAIBackend:
        cfg = LLMConfig(provider=provider, _env_file=None)  # type: ignore[call-arg]
        return OpenAIBackend(cfg)

    def test_complete_returns_chat_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "hello"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        messages = [ChatMessage(role="user", content="hi")]
        result = backend.complete(messages)

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"
        assert result.finish_reason == "stop"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_complete_system_message_injected(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 2
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        backend.complete(
            [ChatMessage(role="user", content="hi")],
            system="You are helpful.",
        )
        call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
        assert call_messages[0]["role"] == "system"
        assert call_messages[0]["content"] == "You are helpful."

    def test_stream_yields_chunks(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()

        def make_chunk(text: str) -> MagicMock:
            chunk = MagicMock()
            chunk.choices[0].delta.content = text
            return chunk

        mock_client.chat.completions.create.return_value = iter(
            [make_chunk("hel"), make_chunk("lo"), make_chunk("!")]
        )
        backend._client = mock_client

        chunks = list(backend.stream([ChatMessage(role="user", content="hi")]))
        assert "".join(chunks) == "hello!"

    def test_tool_call_returns_tool_call_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "search_code"
        mock_tool_call.function.arguments = '{"query": "auth", "repo_path": "/repo"}'
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = backend.tool_call(
            [ChatMessage(role="user", content="search for auth")],
            tools=tools,
        )
        assert isinstance(result, ToolCallResponse)
        assert result.tool_name == "search_code"
        assert result.tool_arguments == {"query": "auth", "repo_path": "/repo"}

    def test_client_is_none_when_no_key(self) -> None:
        cfg = LLMConfig(provider="openai", _env_file=None)  # type: ignore[call-arg]
        backend = OpenAIBackend(cfg)
        # No API key → _client should be None (graceful)
        assert backend._client is None

    def test_azure_uses_azure_client(self) -> None:
        cfg = LLMConfig(
            provider="azure",
            azure_endpoint="https://test.openai.azure.com/",
            _env_file=None,  # type: ignore[call-arg]
        )
        cfg = cfg.model_copy(update={"azure_api_key": _FAKE_KEY})
        with patch("trelix.llm.providers.openai_backend.AzureOpenAI") as MockAzure:
            MockAzure.return_value = MagicMock()
            OpenAIBackend(cfg)
            assert MockAzure.called
