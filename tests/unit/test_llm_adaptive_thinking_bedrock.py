"""BedrockBackend: thinking request shape per model (mocked, no network).

Adaptive-only models (Claude 5 and newer, Opus 4.7 and 4.8) get
``additionalModelRequestFields {"thinking": {"type": "adaptive"}}`` and no forced
temperature; every other model, including ids nobody has classified yet, keeps the
``reasoning_config`` budget request with temperature forced to 1.0. The live evidence is
summarised in tests/unit/bedrock_thinking_harness.py; the retry-and-remember path for
unclassified ids is in test_llm_adaptive_thinking_bedrock_retry.py.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from tests.unit.bedrock_thinking_harness import (
    HAIKU_FALLBACK,
    MESSAGES,
    UNCLASSIFIED_MODEL,
    make_backend,
    recording_client,
)
from trelix.llm.client import ChatMessage

ADAPTIVE_ONLY_IDS = [
    "anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5-5",
    "global.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5-5",
    "us.anthropic.claude-sonnet-5-5-20261001-v1:0",
    "anthropic.claude-sonnet-5-v1:0",
    "eu.anthropic.claude-opus-4-8",
    "anthropic.claude-opus-4-7",
    "global.anthropic.claude-opus-4-7",
    "claude-sonnet-5",
]

BUDGET_IDS = [
    "us.anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-opus-4-6-v1",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    UNCLASSIFIED_MODEL,
    "some-future-model",
]

METHODS = [pytest.param("complete", id="complete"), pytest.param("stream", id="stream")]


def _send(backend: Any, method: str, **kwargs: Any) -> None:
    if method == "complete":
        backend.complete(MESSAGES, **kwargs)
    else:
        list(backend.stream(MESSAGES, **kwargs))


class TestRequestShapeByModel:
    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS)
    @pytest.mark.parametrize("method", METHODS)
    def test_adaptive_only_models_get_adaptive_thinking_and_no_temperature(
        self, model: str, method: str
    ) -> None:
        backend = make_backend(model)
        client, seen = recording_client()
        backend._client = client

        _send(backend, method, temperature=0.2, thinking=True)

        assert len(seen) == 1
        assert seen[0]["modelId"] == model
        assert seen[0]["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}
        assert "temperature" not in seen[0]["inferenceConfig"]
        assert "budget_tokens" not in repr(seen[0])
        assert "reasoning_config" not in repr(seen[0])

    @pytest.mark.parametrize("model", BUDGET_IDS)
    @pytest.mark.parametrize("method", METHODS)
    def test_every_other_model_keeps_reasoning_config_and_forced_temperature(
        self, model: str, method: str
    ) -> None:
        backend = make_backend(model)
        client, seen = recording_client()
        backend._client = client

        _send(backend, method, temperature=0.2, thinking=True)

        assert len(seen) == 1
        assert seen[0]["additionalModelRequestFields"] == {
            "reasoning_config": {"type": "enabled", "budget_tokens": 2048}
        }
        assert seen[0]["inferenceConfig"]["temperature"] == 1.0

    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS)
    def test_the_budget_setting_is_not_sent_to_adaptive_models(self, model: str) -> None:
        backend = make_backend(model, budget=9999)
        client, seen = recording_client()
        backend._client = client

        backend.complete(MESSAGES, thinking=True)

        assert "9999" not in repr(seen[0])

    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS + BUDGET_IDS)
    @pytest.mark.parametrize("method", METHODS)
    def test_thinking_false_is_unchanged_for_every_model(self, model: str, method: str) -> None:
        backend = make_backend(model)
        client, seen = recording_client()
        backend._client = client

        _send(backend, method, temperature=0.3, thinking=False)

        assert "additionalModelRequestFields" not in seen[0]
        assert seen[0]["inferenceConfig"]["temperature"] == 0.3

    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS + BUDGET_IDS)
    def test_tool_calls_never_carry_thinking_fields(self, model: str) -> None:
        backend = make_backend(model)
        client, seen = recording_client(
            response={
                "output": {
                    "message": {
                        "content": [{"toolUse": {"name": "pick", "input": {"a": 1}}}],
                        "role": "assistant",
                    }
                }
            }
        )
        backend._client = client
        tool = {"function": {"name": "pick", "parameters": {"type": "object"}}}

        backend.tool_call(MESSAGES, [tool], force_tool="pick")

        assert "additionalModelRequestFields" not in seen[0]

    @pytest.mark.parametrize("model", ADAPTIVE_ONLY_IDS + BUDGET_IDS)
    @pytest.mark.parametrize("method", METHODS)
    def test_a_known_temperature_rejection_keeps_temperature_out_of_plain_requests(
        self, model: str, method: str
    ) -> None:
        backend = make_backend(model)
        backend._temperature_rejected = True
        client, seen = recording_client()
        backend._client = client

        _send(backend, method, thinking=False)
        _send(backend, method, temperature=0.9, thinking=False)

        assert len(seen) == 2
        assert "temperature" not in seen[0]["inferenceConfig"]
        assert "temperature" not in seen[1]["inferenceConfig"]

    def test_a_known_temperature_rejection_still_leaves_adaptive_without_temperature(
        self,
    ) -> None:
        backend = make_backend("us.anthropic.claude-sonnet-5")
        backend._temperature_rejected = True
        client, seen = recording_client()
        backend._client = client

        backend.complete(MESSAGES, temperature=0.7, thinking=True)

        assert seen[0]["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}
        assert "temperature" not in seen[0]["inferenceConfig"]

    def test_a_known_temperature_rejection_still_drops_temperature_on_budget_models(
        self,
    ) -> None:
        backend = make_backend("us.anthropic.claude-sonnet-4-6")
        backend._temperature_rejected = True
        client, seen = recording_client()
        backend._client = client

        backend.complete(MESSAGES, thinking=True)

        assert seen[0]["additionalModelRequestFields"] == {
            "reasoning_config": {"type": "enabled", "budget_tokens": 2048}
        }
        assert "temperature" not in seen[0]["inferenceConfig"]

    def test_the_thinking_shape_follows_the_model_id_the_request_carries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_build_request reads the shared model attribute once. A second read could see the
        fallback another thread just swapped in and send one model's id with the other
        model's thinking shape: here the first read is an adaptive-only model and every
        later read is the (budget) fallback."""
        backend = make_backend("us.anthropic.claude-sonnet-5")
        reads = itertools.chain(["us.anthropic.claude-sonnet-5"], itertools.repeat(HAIKU_FALLBACK))
        monkeypatch.setattr(
            type(backend),
            "_model",
            property(lambda _self: next(reads), lambda _self, _value: None),
            raising=False,
        )

        request = backend._build_request(MESSAGES, None, None, thinking=True)

        assert request["modelId"] == "us.anthropic.claude-sonnet-5"
        assert request["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}
        assert "temperature" not in request["inferenceConfig"]


class TestAdaptiveResponses:
    def test_a_response_without_a_reasoning_block_parses(self) -> None:
        """The model decides whether to think: live, an easy question came back with only a
        text block. That is a normal response, not an error."""
        backend = make_backend("us.anthropic.claude-sonnet-5-5")
        client, _seen = recording_client(
            response={
                "output": {"message": {"content": [{"text": "4"}], "role": "assistant"}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1},
            }
        )
        backend._client = client

        result = backend.complete([ChatMessage(role="user", content="2+2?")], thinking=True)

        assert result.content == "4"
        assert result.thinking is None
        assert result.thinking_blocks == []
        assert result.model == "us.anthropic.claude-sonnet-5-5"

    def test_a_response_with_a_reasoning_block_still_surfaces_it(self) -> None:
        backend = make_backend("us.anthropic.claude-sonnet-5")
        client, _seen = recording_client(
            response={
                "output": {
                    "message": {
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {"text": "weighing it", "signature": "s1"}
                                }
                            },
                            {"text": "done"},
                        ],
                        "role": "assistant",
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1},
            }
        )
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "done"
        assert result.thinking == "weighing it"
