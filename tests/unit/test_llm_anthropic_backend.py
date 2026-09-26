"""Tests for AnthropicBackend (mocked — no real API calls)."""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import httpx
import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse, ImageContent


def _retryable_error(status_code: int = 503) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"{status_code} error", request=request, response=response)


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


class TestClientRetryConfiguration:
    """The anthropic SDK's own default retry (max_retries=2) would
    otherwise stack underneath @with_retry's 5-attempt tenacity layer,
    multiplying worst-case wall-clock time on a persistent outage far
    beyond what max_attempts=5 implies. Uses the REAL anthropic SDK client
    (skips if the optional 'anthropic' extra isn't installed) so this
    actually proves what value reaches the SDK, not what trelix's own
    code believes it passed."""

    def test_anthropic_client_has_sdk_retries_disabled(self) -> None:
        pytest.importorskip("anthropic")
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        cfg = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_FAKE_ANT_KEY,
            model="claude-3-5-sonnet-20241022",
            _env_file=None,  # type: ignore[call-arg]
        )
        backend = AnthropicBackend(cfg)
        assert backend._client is not None
        assert backend._client.max_retries == 0


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
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "hello"
        mock_response.content = [mock_content_block]
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

    def test_complete_with_images_builds_correct_content_blocks(
        self, mock_anthropic: MagicMock
    ) -> None:
        """images=[ImageContent(...)] must produce the Anthropic Messages API
        content-block shape: one base64 `{"type": "image", ...}` block per
        ImageContent, followed by a trailing `{"type": "text", ...}` block
        carrying message.content."""
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "a cat"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        image_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes"
        result = backend.complete(
            [
                ChatMessage(
                    role="user",
                    content="what is in this image?",
                    images=[ImageContent(data=image_bytes, media_type="image/png")],
                )
            ]
        )

        assert result.content == "a cat"
        call_kwargs = mock_client.messages.create.call_args[1]
        sent_content = call_kwargs["messages"][0]["content"]
        assert sent_content == [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode(),
                },
            },
            {"type": "text", "text": "what is in this image?"},
        ]

    def test_complete_text_only_message_content_is_plain_string(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Regression: images=None (the default) must still send a bare
        string for `content`, byte-identical to the pre-vision request shape
        — not wrapped in a content-block list."""
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "ok"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hi")])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_uses_max_tokens_not_max_completion_tokens(self, mock_anthropic: MagicMock) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "ok"
        mock_response.content = [mock_content_block]
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
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "ok"
        mock_response.content = [mock_content_block]
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
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "ok"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude"
        mock_response.stop_reason = "max_tokens"  # Anthropic name
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])
        assert result.finish_reason == "length"  # normalized

    def test_complete_does_not_pass_temperature_to_create(self, mock_anthropic: MagicMock) -> None:
        """anthropic-sdk-python v1.0.0 removed temperature from Messages.create —
        confirmed via inspect.signature against the real installed SDK (no
        'temperature' parameter). complete() must never pass it."""
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "ok"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hi")], temperature=0.5)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs

    def test_stream_does_not_pass_temperature(self, mock_anthropic: MagicMock) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["hi"])
        mock_manager = MagicMock()
        mock_manager.__enter__.return_value = mock_stream
        mock_client.messages.stream.return_value = mock_manager
        backend._client = mock_client

        list(backend.stream([ChatMessage(role="user", content="hi")], temperature=0.5))

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert "temperature" not in call_kwargs

    def test_complete_warns_once_on_ignored_temperature(
        self, mock_anthropic: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "ok"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        import logging

        with caplog.at_level(logging.WARNING, logger="trelix.llm.anthropic_backend"):
            backend.complete([ChatMessage(role="user", content="hi")], temperature=0.7)
            backend.complete([ChatMessage(role="user", content="hi")], temperature=0.7)

        temperature_warnings = [r for r in caplog.records if "temperature" in r.message]
        assert len(temperature_warnings) == 1

    def test_split_content_handles_redacted_thinking(self, mock_anthropic: MagicMock) -> None:
        """Confirmed bug: _split_content only branched on 'text'/'thinking' block
        types, silently dropping any 'redacted_thinking' block Anthropic returns."""
        backend = self._make_backend(mock_anthropic)
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "answer"
        redacted_block = MagicMock()
        redacted_block.type = "redacted_thinking"
        redacted_block.data = "opaque123"

        text, thinking_blocks = backend._split_content([text_block, redacted_block])

        assert text == "answer"
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].type == "redacted_thinking"
        assert thinking_blocks[0].data == "opaque123"
        assert thinking_blocks[0].thinking is None

    def test_split_content_handles_thinking_with_signature(self, mock_anthropic: MagicMock) -> None:
        backend = self._make_backend(mock_anthropic)
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "because X"
        thinking_block.signature = "sig456"

        text, thinking_blocks = backend._split_content([thinking_block])

        assert text == ""
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].type == "thinking"
        assert thinking_blocks[0].thinking == "because X"
        assert thinking_blocks[0].signature == "sig456"

    def test_complete_populates_reasoning_content_and_thinking_blocks(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "step 1"
        thinking_block.signature = "sig1"
        redacted_block = MagicMock()
        redacted_block.type = "redacted_thinking"
        redacted_block.data = "blob"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "final answer"
        mock_response.content = [thinking_block, redacted_block, text_block]
        mock_response.model = "claude-sonnet"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")], thinking=True)

        assert result.content == "final answer"
        assert result.thinking == "step 1"  # only the "thinking"-kind block's text
        assert len(result.thinking_blocks) == 2
        assert result.thinking_blocks[0].type == "thinking"
        assert result.thinking_blocks[1].type == "redacted_thinking"

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

    def test_complete_retries_on_503_then_succeeds(self, mock_anthropic: MagicMock) -> None:
        """A transient 5xx must be retried, not surfaced immediately —
        confirms the shared retry contract is wired into complete()."""
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "ok"
        mock_response.content = [mock_content_block]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.side_effect = [_retryable_error(503), mock_response]
        backend._client = mock_client

        with patch("tenacity.nap.time.sleep"):
            result = backend.complete([ChatMessage(role="user", content="hi")])

        assert result.content == "ok"
        assert mock_client.messages.create.call_count == 2

    def test_complete_400_is_not_retried(self, mock_anthropic: MagicMock) -> None:
        """A non-retryable client error must fail on the first attempt."""
        backend = self._make_backend(mock_anthropic)
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _retryable_error(400)
        backend._client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            backend.complete([ChatMessage(role="user", content="hi")])

        assert mock_client.messages.create.call_count == 1
