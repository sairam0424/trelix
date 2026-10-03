"""Every backend's stop signal maps to one vocabulary, and nothing unknown reads as a clean stop.

A provider reports a truncated, refused or filtered reply as an ordinary successful response.
The only tell is the stop field. These tables are written out as literals on purpose: they are
the contract, and a backend that quietly starts mapping `refusal` to `stop` has to change a row.

Mutations each table was checked against (every one fails a test below):
- `normalise()` falling back to "stop" instead of "unknown" (the unknown and None rows fail);
- the refusal override in `classify_chat_choice` removed (the refusal rows fail);
- `guardrail_intervened` mapped to "stop" (the Bedrock row fails);
- `_candidate_finish_name` answering "STOP" when there are no candidates (the Vertex row fails).
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse
from trelix.llm.finish_reasons import is_complete, normalise

_FAKE_KEY = "test-k"  # short enough not to trigger secret scanner; never sent to any service
_MESSAGES = [ChatMessage(role="user", content="hi")]


def _anthropic_backend(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setitem(sys.modules, "anthropic", MagicMock())
    monkeypatch.delitem(sys.modules, "trelix.llm.providers.anthropic_backend", raising=False)
    from trelix.llm.providers.anthropic_backend import AnthropicBackend

    cfg = LLMConfig(
        provider="anthropic",
        anthropic_api_key=_FAKE_KEY,
        model="claude-3-5-sonnet-20241022",
        _env_file=None,  # type: ignore[call-arg]
    )
    return AnthropicBackend(cfg)


def _bedrock_backend() -> Any:
    from trelix.llm.providers.bedrock_backend import BedrockBackend

    cfg = LLMConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-4-6",
        aws_region="us-east-1",
        _env_file=None,  # type: ignore[call-arg]
    )
    boto3_mock = MagicMock()
    boto3_mock.Session.return_value.client.return_value = MagicMock()
    botocore_mock = MagicMock()
    modules = {
        "boto3": boto3_mock,
        "botocore": botocore_mock,
        "botocore.config": botocore_mock.config,
    }
    with patch.dict("sys.modules", modules):
        return BedrockBackend(cfg)


def _openai_backend() -> Any:
    from trelix.llm.providers.openai_backend import OpenAIBackend

    backend = OpenAIBackend(LLMConfig(provider="openai", _env_file=None))  # type: ignore[call-arg]
    backend._client = MagicMock()
    return backend


def _litellm_backend() -> Any:
    from trelix.llm.providers.litellm_backend import LiteLLMBackend

    cfg = LLMConfig(
        provider="litellm",
        litellm_model="bedrock/claude-3-5-sonnet",
        _env_file=None,  # type: ignore[call-arg]
    )
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        backend = LiteLLMBackend(cfg)
    backend._litellm = MagicMock()
    return backend


def _vertex_modules() -> dict[str, MagicMock]:
    genai = MagicMock()
    genai.Client = MagicMock(return_value=MagicMock())
    genai.types = MagicMock()
    google = MagicMock()
    google.genai = genai
    return {"google": google, "google.genai": genai, "google.genai.types": genai.types}


def _vertex_backend(modules: dict[str, MagicMock]) -> Any:
    from trelix.llm.providers.vertex_backend import VertexBackend

    cfg = LLMConfig(
        provider="vertex",
        model="gemini-2.0-flash",
        google_api_key=_FAKE_KEY,
        _env_file=None,  # type: ignore[call-arg]
    )
    with patch.dict("sys.modules", modules):
        backend = VertexBackend(cfg)
    backend._client = MagicMock()
    return backend


def _chat_choice(
    finish_reason: object, *, refusal: object = None, model_extra: object = None
) -> SimpleNamespace:
    return SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content="[]", refusal=refusal),
        model_extra=model_extra,
    )


def _chat_response(choice: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(model="m", choices=[choice], usage=None)


def _real_openai_completion(choice: dict[str, Any]) -> Any:
    """A response built by the real openai SDK, so field names cannot drift from a fake."""
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, **choice}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def _real_litellm_completion(choice: dict[str, Any]) -> Any:
    """A response built by litellm's own converter from raw OpenAI-shaped JSON."""
    pytest.importorskip("litellm")
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )
    from litellm.types.utils import ModelResponse

    return convert_to_model_response_object(
        response_object={
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, **choice}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        model_response_object=ModelResponse(),
        response_type="completion",
    )


class TestIsComplete:
    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("stop", True),
            ("tool_calls", True),
            ("length", False),
            ("refusal", False),
            ("content_filter", False),
            ("paused", False),
            ("error", False),
            ("unknown", False),
            ("", False),
            ("end_turn", False),
        ],
    )
    def test_only_stop_and_tool_calls_are_complete(self, reason: str, expected: bool) -> None:
        assert is_complete(reason) is expected

    @pytest.mark.parametrize("raw", [None, 3, b"stop", ["stop"]])
    def test_a_non_string_value_is_unknown(self, raw: object) -> None:
        assert normalise({"stop": "stop"}, raw) == "unknown"


class TestAnthropicStopReasons:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("pause_turn", "paused"),
            ("refusal", "refusal"),
            ("a_value_added_next_year", "unknown"),
            (None, "unknown"),
        ],
    )
    def test_complete_maps_every_documented_value(
        self, monkeypatch: pytest.MonkeyPatch, raw: object, expected: str
    ) -> None:
        backend = _anthropic_backend(monkeypatch)
        backend._client = MagicMock()
        backend._client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="[]")],
            model="claude-x",
            stop_reason=raw,
            usage=None,
        )

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == expected
        assert result.raw_finish_reason == (raw if isinstance(raw, str) else None)


class TestBedrockStopReasons:
    @staticmethod
    def _converse_reply(stop_reason: object) -> dict[str, Any]:
        reply: dict[str, Any] = {
            "output": {"message": {"role": "assistant", "content": [{"text": "[]"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        if stop_reason is not None:
            reply["stopReason"] = stop_reason
        return reply

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("guardrail_intervened", "content_filter"),
            ("content_filtered", "content_filter"),
            ("malformed_model_output", "error"),
            ("malformed_tool_use", "error"),
            ("a_value_added_next_year", "unknown"),
            (None, "unknown"),
        ],
    )
    def test_complete_maps_every_documented_value(self, raw: object, expected: str) -> None:
        backend = _bedrock_backend()
        backend._client = MagicMock()
        backend._client.converse.return_value = self._converse_reply(raw)

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == expected
        assert result.raw_finish_reason == (raw if isinstance(raw, str) else None)


_CHAT_ROWS = [
    ("stop", "stop"),
    ("tool_calls", "tool_calls"),
    ("function_call", "tool_calls"),
    ("length", "length"),
    ("content_filter", "content_filter"),
    ("a_value_added_next_year", "unknown"),
    (None, "unknown"),
]
_OPENAI_KNOWN = {"stop", "tool_calls", "function_call", "length", "content_filter"}


# litellm rewrites a missing or unknown finish_reason to "stop" itself, so those rows cannot be
# reproduced through its real objects.
_LITELLM_SDK_ROWS = [(raw, expected) for raw, expected in _CHAT_ROWS if raw in _OPENAI_KNOWN]


class TestOpenAIShapedChoices:
    @pytest.mark.parametrize(("raw", "expected"), _CHAT_ROWS)
    def test_openai_backend_maps_finish_reason(self, raw: object, expected: str) -> None:
        backend = _openai_backend()
        backend._client.chat.completions.create.return_value = _chat_response(_chat_choice(raw))

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == expected
        assert result.refusal is None
        assert result.signals == []

    @pytest.mark.parametrize(("raw", "expected"), _CHAT_ROWS)
    def test_litellm_backend_maps_finish_reason(self, raw: object, expected: str) -> None:
        backend = _litellm_backend()
        backend._litellm.completion.return_value = _chat_response(_chat_choice(raw))

        assert backend.complete(_MESSAGES).finish_reason == expected

    def test_a_refusal_arriving_with_stop_is_a_refusal(self) -> None:
        backend = _openai_backend()
        backend._client.chat.completions.create.return_value = _chat_response(
            _chat_choice("stop", refusal="I can't help with that.")
        )

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == "refusal"
        assert result.refusal == "I can't help with that."
        assert result.raw_finish_reason == "stop"

    def test_a_truncation_stays_a_truncation_when_a_refusal_is_also_present(self) -> None:
        backend = _openai_backend()
        backend._client.chat.completions.create.return_value = _chat_response(
            _chat_choice("length", refusal="I can't")
        )

        assert backend.complete(_MESSAGES).finish_reason == "length"

    @pytest.mark.parametrize("refusal", ["", "   ", None, 7, MagicMock()])
    def test_an_empty_or_non_text_refusal_field_is_ignored(self, refusal: object) -> None:
        backend = _openai_backend()
        backend._client.chat.completions.create.return_value = _chat_response(
            _chat_choice("stop", refusal=refusal)
        )

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == "stop"
        assert result.refusal is None

    def test_an_azure_filter_outage_keeps_the_reply_complete_but_is_recorded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _openai_backend()
        extra = {"content_filter_results": {"error": {"code": "content_filter_error"}}}
        backend._client.chat.completions.create.return_value = _chat_response(
            _chat_choice("stop", model_extra=extra)
        )

        with caplog.at_level(logging.WARNING, logger="trelix.llm.openai_backend"):
            result = backend.complete(_MESSAGES)

        assert result.finish_reason == "stop"
        assert result.signals == ["content_filter_error"]
        assert "content filter did not run" in caplog.text

    def test_a_healthy_azure_filter_result_records_nothing(self) -> None:
        backend = _openai_backend()
        extra = {"content_filter_results": {"hate": {"filtered": False, "severity": "safe"}}}
        backend._client.chat.completions.create.return_value = _chat_response(
            _chat_choice("stop", model_extra=extra)
        )

        assert backend.complete(_MESSAGES).signals == []


class TestRealSdkObjects:
    """The same rules on objects the real SDKs build, where a fake could hide a renamed field."""

    def test_openai_sdk_refusal_with_stop_is_a_refusal(self) -> None:
        backend = _openai_backend()
        completion = _real_openai_completion(
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": None, "refusal": "I can't do that."},
            }
        )
        backend._client.chat.completions.create.return_value = completion

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == "refusal"
        assert result.refusal == "I can't do that."
        assert result.content == ""

    def test_openai_sdk_azure_filter_outage_is_recorded_and_stays_complete(self) -> None:
        backend = _openai_backend()
        completion = _real_openai_completion(
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "[]"},
                "content_filter_results": {"error": {"code": "content_filter_error"}},
            }
        )
        backend._client.chat.completions.create.return_value = completion

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == "stop"
        assert result.signals == ["content_filter_error"]

    def test_litellm_refusal_with_stop_is_a_refusal(self) -> None:
        # litellm keeps the refusal in provider_specific_fields and drops `message.refusal`.
        completion = _real_litellm_completion(
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": None, "refusal": "No."},
            }
        )
        backend = _litellm_backend()
        backend._litellm.completion.return_value = completion

        result = backend.complete(_MESSAGES)

        assert result.finish_reason == "refusal"
        assert result.refusal == "No."

    @pytest.mark.parametrize(("raw", "expected"), _LITELLM_SDK_ROWS)
    def test_litellm_sdk_objects_map_the_openai_set(self, raw: str, expected: str) -> None:
        completion = _real_litellm_completion(
            {"finish_reason": raw, "message": {"role": "assistant", "content": "[]"}}
        )
        backend = _litellm_backend()
        backend._litellm.completion.return_value = completion

        assert backend.complete(_MESSAGES).finish_reason == expected


class TestClassifyChatChoiceNeverRaises:
    @pytest.mark.parametrize(
        "choice",
        [
            SimpleNamespace(),
            SimpleNamespace(finish_reason="stop", message=None),
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(), model_extra=5),
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(), model_extra=["x"]),
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(),
                model_extra={"content_filter_results": "oops"},
            ),
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(provider_specific_fields="not a dict"),
            ),
            MagicMock(),
        ],
    )
    def test_odd_shapes_classify_without_raising(self, choice: object) -> None:
        from trelix.llm.finish_reasons import classify_chat_choice

        info = classify_chat_choice(choice)

        assert info.refusal is None
        assert info.signals == ()
        assert info.finish_reason in {"stop", "unknown"}

    def test_a_missing_finish_reason_on_a_bare_object_is_unknown(self) -> None:
        from trelix.llm.finish_reasons import classify_chat_choice

        assert classify_chat_choice(SimpleNamespace()).finish_reason == "unknown"

    def test_a_refusal_in_provider_specific_fields_is_read(self) -> None:
        from trelix.llm.finish_reasons import classify_chat_choice

        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(provider_specific_fields={"refusal": "  Not today.  "}),
        )

        info = classify_chat_choice(choice)

        assert info.finish_reason == "refusal"
        assert info.refusal == "Not today."


class TestNotConfiguredPlaceholders:
    """A backend with no credentials answers with a placeholder that still says "stop".

    Callers tell it apart by its model (`UNCONFIGURED_MODEL`, "none"), not by the finish
    reason, so the pair is pinned here as literals: changing either alone would silently break
    the reviewer's guard.
    """

    def test_openai_placeholder_is_identified_by_model(self) -> None:
        backend = _openai_backend()
        backend._client = None

        result = backend.complete(_MESSAGES)

        assert result.model == "none"
        assert result.finish_reason == "stop"

    def test_anthropic_placeholder_is_identified_by_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _anthropic_backend(monkeypatch)
        backend._client = None

        result = backend.complete(_MESSAGES)

        assert result.model == "none"
        assert result.finish_reason == "stop"

    def test_vertex_placeholder_is_identified_by_model(self) -> None:
        backend = _vertex_backend(_vertex_modules())
        backend._client = None

        result = backend.complete(_MESSAGES)

        assert result.model == "none"
        assert result.finish_reason == "stop"


class TestVertexEnumNames:
    def test_the_sdk_enum_names_match_the_table_keys(self) -> None:
        types = pytest.importorskip("google.genai.types")

        assert types.FinishReason.STOP.name == "STOP"
        assert types.FinishReason.MAX_TOKENS.name == "MAX_TOKENS"


class TestVertexFinishReasons:
    @staticmethod
    def _reply(finish_name: object, *, has_candidates: bool = True) -> SimpleNamespace:
        candidates = (
            [SimpleNamespace(finish_reason=SimpleNamespace(name=finish_name))]
            if has_candidates
            else []
        )
        return SimpleNamespace(
            text="[]",
            candidates=candidates,
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        )

    def _complete(self, reply: SimpleNamespace) -> ChatResponse:
        modules = _vertex_modules()
        backend = _vertex_backend(modules)
        backend._client.models.generate_content.return_value = reply
        with patch.dict("sys.modules", modules):
            result: ChatResponse = backend.complete(_MESSAGES)
        return result

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "unknown"),
            ("RECITATION", "unknown"),
            ("PROHIBITED_CONTENT", "unknown"),
            ("MALFORMED_FUNCTION_CALL", "unknown"),
            ("A_VALUE_ADDED_NEXT_YEAR", "unknown"),
            (None, "unknown"),
        ],
    )
    def test_only_stop_and_max_tokens_are_classified(self, name: object, expected: str) -> None:
        result = self._complete(self._reply(name))

        assert result.finish_reason == expected
        assert result.raw_finish_reason == (name if isinstance(name, str) else None)

    def test_a_response_with_no_candidates_is_unknown_not_stop(self) -> None:
        result = self._complete(self._reply("STOP", has_candidates=False))

        assert result.finish_reason == "unknown"
        assert result.raw_finish_reason is None


class TestChatResponseDefaults:
    def test_new_fields_default_to_nothing_reported(self) -> None:
        r = ChatResponse(content="x", model="m", finish_reason="stop")

        assert r.raw_finish_reason is None
        assert r.refusal is None
        assert r.signals == []

    def test_signals_are_not_shared_between_instances(self) -> None:
        first = ChatResponse(content="x", model="m", finish_reason="stop")
        second = ChatResponse(content="x", model="m", finish_reason="stop")

        first.signals.append("content_filter_error")

        assert second.signals == []
