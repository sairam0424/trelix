"""
Unit tests for extended thinking support (Feature 1, trelix v3.0).

Tests:
- Config defaults (thinking_enabled=False, thinking_budget_tokens=4096)
- ChatResponse new fields (thinking, cache_read_tokens, cache_write_tokens)
- Anthropic backend thinking support and crash fix
- Other backends accept thinking parameter
"""

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatResponse, TrelixChatClient


class TestConfig:
    """Test LLMConfig thinking fields."""

    def test_thinking_fields_exist(self):
        """Config has thinking_enabled and thinking_budget_tokens fields."""
        config = LLMConfig()
        assert hasattr(config, "thinking_enabled")
        assert hasattr(config, "thinking_budget_tokens")

    def test_thinking_defaults(self):
        """thinking_enabled defaults to False, thinking_budget_tokens to 4096."""
        config = LLMConfig()
        assert config.thinking_enabled is False
        assert config.thinking_budget_tokens == 4096

    def test_thinking_env_override(self, monkeypatch):
        """Env vars can override thinking config."""
        monkeypatch.setenv("TRELIX_LLM_THINKING_ENABLED", "true")
        monkeypatch.setenv("TRELIX_LLM_THINKING_BUDGET_TOKENS", "8192")
        config = LLMConfig()
        assert config.thinking_enabled is True
        assert config.thinking_budget_tokens == 8192


class TestChatResponse:
    """Test ChatResponse new fields."""

    def test_new_fields_exist(self):
        """ChatResponse has thinking, cache_read_tokens, cache_write_tokens."""
        response = ChatResponse(
            content="test",
            model="test-model",
            finish_reason="stop",
        )
        assert hasattr(response, "thinking")
        assert hasattr(response, "cache_read_tokens")
        assert hasattr(response, "cache_write_tokens")

    def test_new_fields_defaults(self):
        """New fields default to None/0."""
        response = ChatResponse(
            content="test",
            model="test-model",
            finish_reason="stop",
        )
        assert response.thinking is None
        assert response.cache_read_tokens == 0
        assert response.cache_write_tokens == 0

    def test_new_fields_settable(self):
        """New fields can be set."""
        response = ChatResponse(
            content="test",
            model="test-model",
            finish_reason="stop",
            thinking="extended thinking content",
            cache_read_tokens=100,
            cache_write_tokens=50,
        )
        assert response.thinking == "extended thinking content"
        assert response.cache_read_tokens == 100
        assert response.cache_write_tokens == 50


class TestTrelixChatClientABC:
    """Test abstract base class signature updates."""

    def test_complete_accepts_thinking(self):
        """TrelixChatClient.complete() accepts thinking parameter."""
        # Can't instantiate ABC directly, but we can check the signature
        import inspect

        sig = inspect.signature(TrelixChatClient.complete)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_stream_accepts_thinking(self):
        """TrelixChatClient.stream() accepts thinking parameter."""
        import inspect

        sig = inspect.signature(TrelixChatClient.stream)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False


class MockAnthropicContent:
    """Mock Anthropic content block."""

    def __init__(self, block_type: str, text: str | None = None, thinking: str | None = None):
        self.type = block_type
        self.text = text
        self.thinking = thinking


class MockAnthropicResponse:
    """Mock Anthropic API response."""

    def __init__(
        self, content_blocks: list, model: str = "claude-test", stop_reason: str = "end_turn"
    ):
        self.content = content_blocks
        self.model = model
        self.stop_reason = stop_reason
        self.usage = MockUsage()


class MockUsage:
    """Mock usage data."""

    def __init__(self):
        self.input_tokens = 100
        self.output_tokens = 50
        self.cache_read_input_tokens = 10
        self.cache_creation_input_tokens = 5


class TestAnthropicBackend:
    """Test Anthropic backend thinking support."""

    def test_split_content_text_only(self):
        """_split_content extracts text when no thinking blocks."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key="test")
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("text", text="Hello world"),
        ]
        text, thinking = backend._split_content(content_blocks)
        assert text == "Hello world"
        assert thinking is None

    def test_split_content_with_thinking(self):
        """_split_content extracts both text and thinking."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key="test")
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("thinking", thinking="Let me think..."),
            MockAnthropicContent("text", text="The answer is 42"),
        ]
        text, thinking = backend._split_content(content_blocks)
        assert text == "The answer is 42"
        assert thinking == "Let me think..."

    def test_split_content_multiple_blocks(self):
        """_split_content concatenates multiple blocks of same type."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key="test")
        backend = AnthropicBackend(config)

        content_blocks = [
            MockAnthropicContent("thinking", thinking="First thought"),
            MockAnthropicContent("thinking", thinking="Second thought"),
            MockAnthropicContent("text", text="First part"),
            MockAnthropicContent("text", text="Second part"),
        ]
        text, thinking = backend._split_content(content_blocks)
        assert text == "First partSecond part"
        assert thinking == "First thoughtSecond thought"

    def test_split_content_empty(self):
        """_split_content handles empty content."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key="test")
        backend = AnthropicBackend(config)

        text, thinking = backend._split_content([])
        assert text == ""
        assert thinking is None

    def test_thinking_kwargs_disabled(self):
        """_thinking_kwargs returns empty dict when thinking disabled."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(provider="anthropic", anthropic_api_key="test")
        backend = AnthropicBackend(config)

        kwargs = backend._thinking_kwargs(thinking=False)
        assert kwargs == {}

    def test_thinking_kwargs_enabled(self):
        """_thinking_kwargs returns thinking dict when enabled."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key="test",
            thinking_budget_tokens=8192,
        )
        backend = AnthropicBackend(config)

        kwargs = backend._thinking_kwargs(thinking=True)
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 8192

    def test_complete_signature_accepts_thinking(self):
        """complete() method accepts thinking parameter."""
        import inspect

        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        sig = inspect.signature(AnthropicBackend.complete)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_stream_signature_accepts_thinking(self):
        """stream() method accepts thinking parameter."""
        import inspect

        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        sig = inspect.signature(AnthropicBackend.stream)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False


class TestOtherBackends:
    """Test other backends accept thinking parameter."""

    def test_openai_complete_accepts_thinking(self):
        """OpenAI backend complete() accepts thinking."""
        import inspect

        from trelix.llm.providers.openai_backend import OpenAIBackend

        sig = inspect.signature(OpenAIBackend.complete)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_openai_stream_accepts_thinking(self):
        """OpenAI backend stream() accepts thinking."""
        import inspect

        from trelix.llm.providers.openai_backend import OpenAIBackend

        sig = inspect.signature(OpenAIBackend.stream)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_bedrock_complete_accepts_thinking(self):
        """Bedrock backend complete() accepts thinking."""
        import inspect

        from trelix.llm.providers.bedrock_backend import BedrockBackend

        sig = inspect.signature(BedrockBackend.complete)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_bedrock_stream_accepts_thinking(self):
        """Bedrock backend stream() accepts thinking."""
        import inspect

        from trelix.llm.providers.bedrock_backend import BedrockBackend

        sig = inspect.signature(BedrockBackend.stream)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_vertex_complete_accepts_thinking(self):
        """Vertex backend complete() accepts thinking."""
        import inspect

        from trelix.llm.providers.vertex_backend import VertexBackend

        sig = inspect.signature(VertexBackend.complete)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_vertex_stream_accepts_thinking(self):
        """Vertex backend stream() accepts thinking."""
        import inspect

        from trelix.llm.providers.vertex_backend import VertexBackend

        sig = inspect.signature(VertexBackend.stream)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_litellm_complete_accepts_thinking(self):
        """LiteLLM backend complete() accepts thinking."""
        import inspect

        from trelix.llm.providers.litellm_backend import LiteLLMBackend

        sig = inspect.signature(LiteLLMBackend.complete)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False

    def test_litellm_stream_accepts_thinking(self):
        """LiteLLM backend stream() accepts thinking."""
        import inspect

        from trelix.llm.providers.litellm_backend import LiteLLMBackend

        sig = inspect.signature(LiteLLMBackend.stream)
        assert "thinking" in sig.parameters
        assert sig.parameters["thinking"].default is False


class TestSynthesizer:
    """Test Synthesizer thinking integration."""

    def test_synthesizer_has_llm_config(self):
        """Synthesizer stores llm_config."""
        from trelix.core.config import EmbedderConfig, LLMConfig
        from trelix.retrieval.synthesizer import Synthesizer

        embedder_cfg = EmbedderConfig()
        llm_cfg = LLMConfig(provider="openai", openai_api_key="test")
        synth = Synthesizer(embedder_cfg, llm_config=llm_cfg)

        assert hasattr(synth, "_llm_config")
        assert synth._llm_config is llm_cfg

    def test_synthesizer_llm_config_from_shim(self):
        """Synthesizer creates llm_config from embedder config when not provided."""
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.synthesizer import Synthesizer

        embedder_cfg = EmbedderConfig(provider="openai", openai_api_key="test")
        synth = Synthesizer(embedder_cfg)

        assert hasattr(synth, "_llm_config")
        assert synth._llm_config is not None
