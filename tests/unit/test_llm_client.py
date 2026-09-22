"""Tests for TrelixChatClient ABC and dataclasses."""

from __future__ import annotations

import pytest

from trelix.llm.client import (
    ChatMessage,
    ChatResponse,
    ThinkingBlock,
    ToolCallResponse,
    TrelixChatClient,
)


class TestDataclasses:
    def test_chat_message_fields(self) -> None:
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_chat_response_defaults(self) -> None:
        r = ChatResponse(content="hi", model="gpt-4o", finish_reason="stop")
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.thinking is None
        assert r.thinking_blocks == []

    def test_chat_response_full(self) -> None:
        r = ChatResponse(
            content="hi", model="gpt-4o", finish_reason="stop", input_tokens=10, output_tokens=5
        )
        assert r.input_tokens == 10
        assert r.output_tokens == 5

    def test_tool_call_response(self) -> None:
        t = ToolCallResponse(tool_name="fn", tool_arguments={"x": 1}, raw_response=None)
        assert t.tool_name == "fn"
        assert t.tool_arguments == {"x": 1}


class TestThinkingBlock:
    """Field names mirror litellm's ChatCompletionThinkingBlock/
    ChatCompletionRedactedThinkingBlock TypedDicts (litellm/types/llms/openai.py) for
    interop: trelix normalizes Anthropic's thinking/redacted_thinking blocks and Bedrock
    Converse's reasoningContent block into this one shape.
    """

    def test_thinking_block_carries_text_and_signature(self) -> None:
        block = ThinkingBlock(type="thinking", thinking="because X implies Y", signature="sig123")
        assert block.type == "thinking"
        assert block.thinking == "because X implies Y"
        assert block.signature == "sig123"
        assert block.data is None

    def test_redacted_thinking_block_carries_only_data(self) -> None:
        block = ThinkingBlock(type="redacted_thinking", data="opaque-blob")
        assert block.type == "redacted_thinking"
        assert block.data == "opaque-blob"
        assert block.thinking is None
        assert block.signature is None

    def test_chat_response_thinking_blocks_accepts_a_list(self) -> None:
        blocks = [
            ThinkingBlock(type="thinking", thinking="step 1", signature="sig1"),
            ThinkingBlock(type="redacted_thinking", data="blob"),
        ]
        r = ChatResponse(
            content="answer",
            model="claude-sonnet",
            finish_reason="stop",
            thinking="step 1",
            thinking_blocks=blocks,
        )
        assert r.thinking == "step 1"
        assert r.thinking_blocks == blocks


class TestTrelixChatClientABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            TrelixChatClient()  # type: ignore[abstract]

    def test_abstract_methods_present(self) -> None:
        assert "complete" in TrelixChatClient.__abstractmethods__
        assert "stream" in TrelixChatClient.__abstractmethods__
        assert "tool_call" in TrelixChatClient.__abstractmethods__

    def test_concrete_subclass_must_implement_all(self) -> None:
        class Partial(TrelixChatClient):
            def complete(self, messages, max_tokens=None, temperature=None, system=None):
                return ChatResponse("", "", "stop")

            # missing stream and tool_call

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


class TestLLMConfig:
    @pytest.fixture(autouse=True)
    def _unconfigured_llm_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scrub the two vars whose ambient presence defeats a "code default" claim.

        `LLMConfig(_env_file=None)` silences the ./.env FILE source only. The
        process environment is a separate, higher-precedence pydantic-settings
        source, so the two default assertions below were really asserting
        "nothing exported TRELIX_LLM_PROVIDER/TRELIX_LLM_MODEL in this process".

        That is not hypothetical. Importing `litellm` — tests/unit/test_retry.py
        does, for its error-classification checks — runs load_dotenv() at import
        time and publishes this repo's root .env into os.environ for the whole
        process. This repo's .env sets TRELIX_LLM_PROVIDER=azure, so
        test_default_provider_is_openai failed whenever that import landed first
        (reverse collection order does exactly that); TRELIX_LLM_MODEL=gpt-4o
        happens to equal the code default, so test_default_model_is_gpt4o passed
        by coincidence rather than by testing anything.

        The tests below that DO exercise env precedence set these vars themselves
        via monkeypatch, which runs after this fixture.
        """
        monkeypatch.delenv("TRELIX_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("TRELIX_LLM_MODEL", raising=False)

    def test_default_provider_is_openai(self) -> None:
        from trelix.core.config import LLMConfig

        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.provider == "openai"

    def test_default_model_is_gpt4o(self) -> None:
        from trelix.core.config import LLMConfig

        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.model == "gpt-4o"

    def test_llm_field_on_index_config(self) -> None:
        import tempfile

        from trelix.core.config import LLMConfig

        # Build LLMConfig directly with no env file and no env override
        # to verify the default values — don't instantiate via IndexConfig
        # because its default_factory would read the real .env.
        with tempfile.TemporaryDirectory() as tmp:
            from trelix.core.config import IndexConfig

            cfg = IndexConfig(repo_path=tmp)
            assert hasattr(cfg, "llm")
            assert isinstance(cfg.llm, LLMConfig)

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import LLMConfig

        monkeypatch.setenv("TRELIX_LLM_PROVIDER", "anthropic")
        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.provider == "anthropic"

    def test_aws_region_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """anthropic-sdk-python v1.0.0 made AnthropicBedrock raise if no region is
        configured, instead of silently defaulting to us-east-1 — BedrockBackend
        now matches that posture (see test_llm_bedrock_backend.py) rather than
        silently picking a region the caller never chose."""
        from trelix.core.config import LLMConfig

        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.aws_region is None
