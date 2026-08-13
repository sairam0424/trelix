"""
Comprehensive test suite for Extended Thinking feature (trelix v3.0).

Tests cover:
- _split_content() with all edge cases (text-only, thinking-then-text, redacted, empty)
- _thinking_kwargs() with disabled, enabled, and per-call overrides
- ChatResponse thinking field and cache token population
- Content[0] crash fix verification (thinking blocks don't break direct access)
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse

# Fake key for testing — not a real credential
_TEST_FAKE_KEY = "test-anthropic-fake-key-for-unit-tests"


def _make_mock_anthropic_module() -> ModuleType:
    """Build a minimal fake anthropic module so the backend can be instantiated."""
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
    return mock_anthropic


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch sys.modules so `import anthropic` works without the real package."""
    module = _make_mock_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", module)
    # Also remove cached import of anthropic_backend so it re-runs with the patch
    monkeypatch.delitem(sys.modules, "trelix.llm.providers.anthropic_backend", raising=False)
    return module


class MockAnthropicContent:
    """Mock Anthropic content block."""

    def __init__(self, block_type: str, text: str | None = None, thinking: str | None = None):
        self.type = block_type
        self.text = text
        self.thinking = thinking


class MockAnthropicUsage:
    """Mock Anthropic usage data with cache fields."""

    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cache_read: int = 10,
        cache_write: int = 5,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


class MockAnthropicResponse:
    """Mock Anthropic API response."""

    def __init__(
        self,
        content_blocks: list,
        model: str = "claude-3-5-sonnet-20241022",
        stop_reason: str = "end_turn",
        usage: MockAnthropicUsage | None = None,
    ):
        self.content = content_blocks
        self.model = model
        self.stop_reason = stop_reason
        self.usage = usage or MockAnthropicUsage()


# =============================================================================
# Test _split_content() — All Edge Cases
# =============================================================================


class TestSplitContent:
    """Test _split_content() helper with all content block variations."""

    def test_text_only(self, mock_anthropic: MagicMock):
        """Text-only response (no thinking blocks)."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("text", text="Hello, world!"),
        ]
        text, thinking = backend._split_content(content_blocks)

        assert text == "Hello, world!"
        assert thinking is None

    def test_thinking_then_text(self, mock_anthropic: MagicMock):
        """Thinking block followed by text block (typical extended thinking flow)."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("thinking", thinking="Let me analyze this carefully..."),
            MockAnthropicContent("text", text="The answer is 42."),
        ]
        text, thinking = backend._split_content(content_blocks)

        assert text == "The answer is 42."
        assert thinking == "Let me analyze this carefully..."

    def test_redacted_thinking(self, mock_anthropic: MagicMock):
        """Thinking block with empty/redacted content."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        # Redacted thinking returns empty string but block still exists
        content_blocks = [
            MockAnthropicContent("thinking", thinking=""),
            MockAnthropicContent("text", text="Here's my answer."),
        ]
        text, thinking = backend._split_content(content_blocks)

        assert text == "Here's my answer."
        # Empty thinking string still returns "" (not None), as there WAS a thinking block
        assert thinking == ""

    def test_empty_list(self, mock_anthropic: MagicMock):
        """Empty content list (edge case from API error or truncation)."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        text, thinking = backend._split_content([])

        assert text == ""
        assert thinking is None

    def test_multiple_text_blocks(self, mock_anthropic: MagicMock):
        """Multiple text blocks concatenate correctly."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("text", text="First part. "),
            MockAnthropicContent("text", text="Second part. "),
            MockAnthropicContent("text", text="Third part."),
        ]
        text, thinking = backend._split_content(content_blocks)

        assert text == "First part. Second part. Third part."
        assert thinking is None

    def test_multiple_thinking_blocks(self, mock_anthropic: MagicMock):
        """Multiple thinking blocks concatenate correctly."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("thinking", thinking="First thought. "),
            MockAnthropicContent("thinking", thinking="Second thought. "),
            MockAnthropicContent("text", text="My conclusion."),
        ]
        text, thinking = backend._split_content(content_blocks)

        assert text == "My conclusion."
        assert thinking == "First thought. Second thought. "

    def test_interleaved_blocks(self, mock_anthropic: MagicMock):
        """Text and thinking blocks can appear in any order."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("text", text="Start: "),
            MockAnthropicContent("thinking", thinking="[internal reasoning] "),
            MockAnthropicContent("text", text="Middle: "),
            MockAnthropicContent("thinking", thinking="[more reasoning] "),
            MockAnthropicContent("text", text="End."),
        ]
        text, thinking = backend._split_content(content_blocks)

        assert text == "Start: Middle: End."
        assert thinking == "[internal reasoning] [more reasoning] "


# =============================================================================
# Test _thinking_kwargs() — Configuration Scenarios
# =============================================================================


class TestThinkingKwargs:
    """Test _thinking_kwargs() with different config states."""

    def test_disabled(self, mock_anthropic: MagicMock):
        """When thinking=False, returns empty dict (no API kwargs)."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        kwargs = backend._thinking_kwargs(thinking=False)

        assert kwargs == {}

    def test_enabled_with_default_budget(self, mock_anthropic: MagicMock):
        """When thinking=True, returns thinking dict with default budget (4096)."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_TEST_FAKE_KEY,
            thinking_budget_tokens=4096,
            _env_file=None,
        )
        backend = AnthropicBackend(config)

        kwargs = backend._thinking_kwargs(thinking=True)

        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 4096

    def test_enabled_with_custom_budget(self, mock_anthropic: MagicMock):
        """Config.thinking_budget_tokens is respected in API call."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_TEST_FAKE_KEY,
            thinking_budget_tokens=8192,
            _env_file=None,
        )
        backend = AnthropicBackend(config)

        kwargs = backend._thinking_kwargs(thinking=True)

        assert kwargs["thinking"]["budget_tokens"] == 8192

    def test_per_call_override_via_parameter(self, mock_anthropic: MagicMock):
        """Thinking can be enabled per-call even if config.thinking_enabled=False."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_TEST_FAKE_KEY,
            thinking_enabled=False,  # disabled globally
            _env_file=None,
        )
        backend = AnthropicBackend(config)

        # Per-call override: thinking=True
        kwargs = backend._thinking_kwargs(thinking=True)

        assert kwargs != {}  # Should NOT be empty
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"


# =============================================================================
# Test ChatResponse Population
# =============================================================================


class TestChatResponsePopulation:
    """Test ChatResponse fields are correctly populated from Anthropic response."""

    def test_thinking_field_populated(self, mock_anthropic: MagicMock):
        """ChatResponse.thinking is populated when thinking block exists."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        mock_response = MockAnthropicResponse(
            content_blocks=[
                MockAnthropicContent("thinking", thinking="Extended reasoning here..."),
                MockAnthropicContent("text", text="Final answer"),
            ],
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert isinstance(result, ChatResponse)
        assert result.content == "Final answer"
        assert result.thinking == "Extended reasoning here..."

    def test_thinking_field_none_when_absent(self, mock_anthropic: MagicMock):
        """ChatResponse.thinking is None when no thinking block in response."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        mock_response = MockAnthropicResponse(
            content_blocks=[
                MockAnthropicContent("text", text="Just regular text"),
            ],
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert result.thinking is None

    def test_cache_read_tokens_populated(self, mock_anthropic: MagicMock):
        """ChatResponse.cache_read_tokens extracts cache_read_input_tokens from usage."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        mock_response = MockAnthropicResponse(
            content_blocks=[MockAnthropicContent("text", text="ok")],
            usage=MockAnthropicUsage(cache_read=250, cache_write=50),
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert result.cache_read_tokens == 250

    def test_cache_write_tokens_populated(self, mock_anthropic: MagicMock):
        """ChatResponse.cache_write_tokens extracts cache_creation_input_tokens from usage."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        mock_response = MockAnthropicResponse(
            content_blocks=[MockAnthropicContent("text", text="ok")],
            usage=MockAnthropicUsage(cache_read=100, cache_write=75),
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert result.cache_write_tokens == 75

    def test_cache_tokens_default_to_zero_when_missing(self, mock_anthropic: MagicMock):
        """Cache token fields default to 0 if not present in API response."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        # Mock usage object without cache fields (older API version)
        # Use spec to limit attributes - only input_tokens and output_tokens exist
        mock_usage = MagicMock(spec=["input_tokens", "output_tokens"])
        mock_usage.input_tokens = 50
        mock_usage.output_tokens = 25
        # cache_read_input_tokens and cache_creation_input_tokens do NOT exist

        mock_response = MockAnthropicResponse(
            content_blocks=[MockAnthropicContent("text", text="ok")],
        )
        mock_response.usage = mock_usage
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert result.cache_read_tokens == 0
        assert result.cache_write_tokens == 0


# =============================================================================
# Test Content[0] Crash Fix
# =============================================================================


class TestContentAccessCrashFix:
    """Verify the fix: thinking blocks don't crash when code tries content[0].text."""

    def test_thinking_block_first_does_not_crash(self, mock_anthropic: MagicMock):
        """
        OLD CODE BUG: response.content[0].text crashes when content[0] is a thinking block.
        NEW CODE FIX: _split_content() safely iterates all blocks, handling any type.

        This test verifies the crash is fixed by ensuring complete() succeeds when
        the first content block is a thinking block (not text).
        """
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        # Thinking block FIRST (this would crash old code doing content[0].text)
        mock_response = MockAnthropicResponse(
            content_blocks=[
                MockAnthropicContent("thinking", thinking="Reasoning..."),
                MockAnthropicContent("text", text="Answer"),
            ],
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        # Should NOT raise AttributeError
        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert result.content == "Answer"
        assert result.thinking == "Reasoning..."

    def test_only_thinking_block_no_text(self, mock_anthropic: MagicMock):
        """Response with only thinking blocks (no text) should not crash."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        # Only thinking, no text blocks (edge case)
        mock_response = MockAnthropicResponse(
            content_blocks=[
                MockAnthropicContent("thinking", thinking="Internal monologue"),
            ],
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        # Should NOT crash
        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert result.content == ""  # No text blocks → empty string
        assert result.thinking == "Internal monologue"

    def test_mixed_unknown_block_types_ignored(self, mock_anthropic: MagicMock):
        """Unknown block types (future API additions) are safely ignored."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key=_TEST_FAKE_KEY, _env_file=None)
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        # Future block type "unknown_type" (API evolution)
        mock_unknown_block = MockAnthropicContent("unknown_type")
        mock_response = MockAnthropicResponse(
            content_blocks=[
                mock_unknown_block,
                MockAnthropicContent("text", text="Known text"),
            ],
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        # Should NOT crash, unknown block is ignored
        result = backend.complete([ChatMessage(role="user", content="Test")])

        assert result.content == "Known text"
        assert result.thinking is None


# =============================================================================
# Integration: Temperature Auto-Correction
# =============================================================================


class TestTemperatureAutoCorrection:
    """Verify thinking=True forces temperature=1.0 (API requirement)."""

    def test_temperature_forced_to_one_when_thinking_enabled(self, mock_anthropic: MagicMock):
        """When thinking=True, temperature must be 1.0 regardless of config."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_TEST_FAKE_KEY,
            temperature=0.5,  # User wants 0.5
            _env_file=None,
        )
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        mock_response = MockAnthropicResponse(
            content_blocks=[MockAnthropicContent("text", text="ok")],
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="Test")], thinking=True)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0  # Forced, not 0.5

    def test_temperature_respected_when_thinking_disabled(self, mock_anthropic: MagicMock):
        """When thinking=False, config.temperature is used normally."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key=_TEST_FAKE_KEY,
            temperature=0.3,
            _env_file=None,
        )
        backend = AnthropicBackend(config)

        mock_client = MagicMock()
        mock_response = MockAnthropicResponse(
            content_blocks=[MockAnthropicContent("text", text="ok")],
        )
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="Test")], thinking=False)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3  # Config value respected
