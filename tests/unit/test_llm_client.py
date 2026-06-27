"""Tests for TrelixChatClient ABC and dataclasses."""
from __future__ import annotations

import pytest
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient


class TestDataclasses:
    def test_chat_message_fields(self) -> None:
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_chat_response_defaults(self) -> None:
        r = ChatResponse(content="hi", model="gpt-4o", finish_reason="stop")
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_chat_response_full(self) -> None:
        r = ChatResponse(content="hi", model="gpt-4o", finish_reason="stop",
                         input_tokens=10, output_tokens=5)
        assert r.input_tokens == 10
        assert r.output_tokens == 5

    def test_tool_call_response(self) -> None:
        t = ToolCallResponse(tool_name="fn", tool_arguments={"x": 1}, raw_response=None)
        assert t.tool_name == "fn"
        assert t.tool_arguments == {"x": 1}


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
    def test_default_provider_is_openai(self) -> None:
        from trelix.core.config import LLMConfig
        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.provider == "openai"

    def test_default_model_is_gpt4o(self) -> None:
        from trelix.core.config import LLMConfig
        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.model == "gpt-4o"

    def test_llm_field_on_index_config(self) -> None:
        from trelix.core.config import LLMConfig
        import tempfile
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
