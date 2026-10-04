"""BedrockBackend: learning that a model needs adaptive thinking, and re-shaping on a model
swap (mocked, no network).

A model the classifier does not know rejects the budget shape with the ValidationException
text Bedrock returned live on 2026-10-04 (LIVE_ENABLED_THINKING_REJECTION): the backend
switches that model to adaptive, re-sends exactly once, and remembers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from tests.unit.bedrock_thinking_harness import (
    HAIKU_FALLBACK,
    LIVE_ENABLED_THINKING_REJECTION,
    MESSAGES,
    OK_RESPONSE,
    UNCLASSIFIED_MODEL,
    ValidationException,
    make_backend,
    model_unavailable_for,
    raising,
    recording_client,
    reject_budget_thinking,
)

BUDGET_REQUEST = {"reasoning_config": {"type": "enabled", "budget_tokens": 2048}}
ADAPTIVE_REQUEST = {"thinking": {"type": "adaptive"}}
LIVE_ERROR = ValidationException("ValidationException: " + LIVE_ENABLED_THINKING_REJECTION)

CONCURRENT_CALLS = 8
# Far longer than eight threads need to reach the barrier, short enough that a broken
# interleaving fails the test instead of waiting for the global pytest timeout.
BARRIER_TIMEOUT_SECONDS = 10


class TestRetryAndRemember:
    def test_complete_retries_once_in_adaptive_mode_without_temperature(self) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client, seen = recording_client(reject_budget_thinking)
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "ok"
        assert len(seen) == 2
        assert seen[0]["additionalModelRequestFields"] == BUDGET_REQUEST
        assert seen[0]["inferenceConfig"]["temperature"] == 1.0
        assert seen[1]["additionalModelRequestFields"] == ADAPTIVE_REQUEST
        assert "temperature" not in seen[1]["inferenceConfig"]

    def test_the_rejection_does_not_demote_the_model_to_the_fallback(self) -> None:
        """The live message contains "not supported", which the model-unavailable check
        also matches. It must be recognised first, or the primary model is abandoned for
        the fallback for the rest of the instance's life."""
        backend = make_backend(UNCLASSIFIED_MODEL)
        client, seen = recording_client(reject_budget_thinking)
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert backend._model == UNCLASSIFIED_MODEL
        assert result.model == UNCLASSIFIED_MODEL
        assert [request["modelId"] for request in seen] == [UNCLASSIFIED_MODEL] * 2

    def test_stream_retries_once_and_remembers(self) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client, seen = recording_client(reject_budget_thinking)
        backend._client = client

        list(backend.stream(MESSAGES, thinking=True))
        assert len(seen) == 2
        assert seen[1]["additionalModelRequestFields"] == ADAPTIVE_REQUEST

        seen.clear()
        list(backend.stream(MESSAGES, thinking=True))
        assert len(seen) == 1
        assert seen[0]["additionalModelRequestFields"] == ADAPTIVE_REQUEST

    def test_a_second_call_goes_straight_to_adaptive_without_another_rejection(self) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client, seen = recording_client(reject_budget_thinking)
        backend._client = client
        backend.complete(MESSAGES, thinking=True)
        seen.clear()

        backend.complete(MESSAGES, thinking=True)

        assert len(seen) == 1
        assert seen[0]["additionalModelRequestFields"] == ADAPTIVE_REQUEST
        assert "temperature" not in seen[0]["inferenceConfig"]

    def test_the_switch_logs_one_warning_naming_the_model_and_mode_without_the_provider_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client, _seen = recording_client(reject_budget_thinking)
        backend._client = client

        with caplog.at_level(logging.WARNING, logger="trelix.llm.bedrock_backend"):
            backend.complete(MESSAGES, thinking=True)
            backend.complete(MESSAGES, thinking=True)
            backend.complete(MESSAGES, thinking=True)

        warnings = [r for r in caplog.records if r.name == "trelix.llm.bedrock_backend"]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert UNCLASSIFIED_MODEL in message
        assert "adaptive" in message
        assert "output_config.effort" not in message
        assert "is not supported for this model" not in message

    def test_a_second_rejection_is_raised_not_retried_again(self) -> None:
        """The primary doubles as its own fallback so that the pre-existing
        model-unavailable branch (which also matches "not supported") cannot answer
        instead and hide an extra retry."""
        backend = make_backend(UNCLASSIFIED_MODEL, fallback=UNCLASSIFIED_MODEL)
        client, seen = recording_client(raising(LIVE_ERROR))
        backend._client = client

        with pytest.raises(ValidationException):
            backend.complete(MESSAGES, thinking=True)

        assert len(seen) == 2

    def test_the_whole_recovery_is_bounded_when_every_model_keeps_refusing(self) -> None:
        """With a distinct fallback the pre-existing model swap gets its turn after the
        adaptive retry fails, and the fallback is given its own one adaptive switch. Four
        requests in all (primary budget, primary adaptive, fallback budget, fallback
        adaptive), then the error propagates: nothing loops."""
        backend = make_backend(UNCLASSIFIED_MODEL, fallback=HAIKU_FALLBACK)
        client, seen = recording_client(raising(LIVE_ERROR))
        backend._client = client

        with pytest.raises(ValidationException):
            backend.complete(MESSAGES, thinking=True)

        assert [
            (request["modelId"], next(iter(request["additionalModelRequestFields"])))
            for request in seen
        ] == [
            (UNCLASSIFIED_MODEL, "reasoning_config"),
            (UNCLASSIFIED_MODEL, "thinking"),
            (HAIKU_FALLBACK, "reasoning_config"),
            (HAIKU_FALLBACK, "thinking"),
        ]

    def test_the_longest_recovery_is_four_retries_and_five_requests(self) -> None:
        """Every adjustment the loop knows, once each, in a single call: the primary
        refuses temperature, then the budget, then is unavailable on demand, and the
        fallback refuses the budget as well. This is the bound _try_with_fallback()'s
        docstring states; the temperature stays dropped on the fallback."""
        backend = make_backend(UNCLASSIFIED_MODEL, fallback=HAIKU_FALLBACK)

        def responder(request: dict[str, Any]) -> dict[str, Any]:
            if request["modelId"] != UNCLASSIFIED_MODEL:
                return reject_budget_thinking(request)
            if "temperature" in request["inferenceConfig"]:
                raise ValidationException(
                    "ValidationException: `temperature` is deprecated for this model."
                )
            if "reasoning_config" in request["additionalModelRequestFields"]:
                raise LIVE_ERROR
            raise ValidationException(
                "ValidationException: Invocation of model ID with on-demand throughput "
                "isn't supported."
            )

        client, seen = recording_client(responder)
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "ok"
        assert backend._model == HAIKU_FALLBACK
        assert [
            (
                request["modelId"],
                next(iter(request["additionalModelRequestFields"])),
                "temperature" in request["inferenceConfig"],
            )
            for request in seen
        ] == [
            (UNCLASSIFIED_MODEL, "reasoning_config", True),
            (UNCLASSIFIED_MODEL, "reasoning_config", False),
            (UNCLASSIFIED_MODEL, "thinking", False),
            (HAIKU_FALLBACK, "reasoning_config", False),
            (HAIKU_FALLBACK, "thinking", False),
        ]

    def test_an_unrelated_validation_exception_is_not_swallowed_or_remembered(self) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client, seen = recording_client(
            raising(ValidationException("ValidationException: messages must alternate roles"))
        )
        backend._client = client

        with pytest.raises(ValidationException, match="alternate"):
            backend.complete(MESSAGES, thinking=True)
        assert len(seen) == 1

        healthy_client, healthy_seen = recording_client()
        backend._client = healthy_client
        backend.complete(MESSAGES, thinking=True)
        assert healthy_seen[0]["additionalModelRequestFields"] == BUDGET_REQUEST

    def test_the_marker_text_without_a_validation_exception_is_not_the_rejection(self) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client, seen = recording_client(raising(RuntimeError(LIVE_ENABLED_THINKING_REJECTION)))
        backend._client = client

        with pytest.raises(RuntimeError):
            backend.complete(MESSAGES, thinking=True)

        assert len(seen) == 1

    def test_an_adaptive_request_that_gets_the_rejection_is_raised_not_looped(self) -> None:
        """Only a request still carrying reasoning_config can match, which is what bounds
        the switch to one. The primary doubles as its own fallback so that the
        model-unavailable branch cannot answer instead and hide a loop."""
        backend = make_backend(
            "us.anthropic.claude-sonnet-5", fallback="us.anthropic.claude-sonnet-5"
        )
        client, seen = recording_client(raising(LIVE_ERROR))
        backend._client = client

        with pytest.raises(ValidationException):
            backend.complete(MESSAGES, thinking=True)

        assert len(seen) == 1

    def test_thinking_false_never_takes_the_thinking_branch(self) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL, fallback=UNCLASSIFIED_MODEL)
        client, seen = recording_client(raising(LIVE_ERROR))
        backend._client = client

        with pytest.raises(ValidationException):
            backend.complete(MESSAGES, thinking=False)

        assert len(seen) == 1
        assert "additionalModelRequestFields" not in seen[0]

    def test_a_real_botocore_client_error_is_recognised(self) -> None:
        botocore_exceptions = pytest.importorskip("botocore.exceptions")
        error = botocore_exceptions.ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": LIVE_ENABLED_THINKING_REJECTION,
                },
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "Converse",
        )
        backend = make_backend(UNCLASSIFIED_MODEL)
        calls: list[dict[str, Any]] = []

        def responder(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            if len(calls) == 1:
                raise error
            return OK_RESPONSE

        client, seen = recording_client(responder)
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "ok"
        assert [r["additionalModelRequestFields"] for r in seen] == [
            BUDGET_REQUEST,
            ADAPTIVE_REQUEST,
        ]

    def test_concurrent_rejections_converge_and_log_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every thread has its budget request in flight before any rejection comes back:
        the fake client holds each budget request on a barrier until all eight have
        arrived. Without that, the threads run one after another, the first one teaches
        the backend, and the rest never send a budget request, so the "only the call that
        first records the switch may log" rule would never be exercised."""
        backend = make_backend(UNCLASSIFIED_MODEL)
        all_budget_requests_in_flight = threading.Barrier(
            CONCURRENT_CALLS, timeout=BARRIER_TIMEOUT_SECONDS
        )

        def responder(request: dict[str, Any]) -> dict[str, Any]:
            if "reasoning_config" in request["additionalModelRequestFields"]:
                all_budget_requests_in_flight.wait()
            return reject_budget_thinking(request)

        client, seen = recording_client(responder)
        backend._client = client
        results: list[str] = []
        errors: list[BaseException] = []

        def call() -> None:
            try:
                results.append(backend.complete(MESSAGES, thinking=True).content)
            except BaseException as exc:  # noqa: BLE001 -- surfaced by the assertion below
                errors.append(exc)

        with caplog.at_level(logging.WARNING, logger="trelix.llm.bedrock_backend"):
            threads = [threading.Thread(target=call) for _ in range(CONCURRENT_CALLS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert errors == []
        assert results == ["ok"] * CONCURRENT_CALLS
        shapes = [next(iter(request["additionalModelRequestFields"])) for request in seen]
        assert (
            sorted(shapes)
            == ["reasoning_config"] * CONCURRENT_CALLS + ["thinking"] * CONCURRENT_CALLS
        )
        assert len([r for r in caplog.records if r.name == "trelix.llm.bedrock_backend"]) == 1

    def test_a_rejection_is_remembered_for_the_model_that_was_sent_not_the_current_one(
        self,
    ) -> None:
        """Another thread can swap the instance to the fallback model while this call is
        still in flight. The rejection is a fact about the model in the request, so that
        is the model to remember as adaptive, and the (budget) fallback keeps its own
        shape. The responder makes the swap the way the other thread would."""
        backend = make_backend(UNCLASSIFIED_MODEL, fallback=HAIKU_FALLBACK)

        def responder(request: dict[str, Any]) -> dict[str, Any]:
            backend._model = HAIKU_FALLBACK
            return reject_budget_thinking(request)

        client, seen = recording_client(responder)
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "ok"
        assert [
            (request["modelId"], next(iter(request["additionalModelRequestFields"])))
            for request in seen
        ] == [
            (UNCLASSIFIED_MODEL, "reasoning_config"),
            (UNCLASSIFIED_MODEL, "thinking"),
        ]
        assert backend._thinking_modes.mode_for(UNCLASSIFIED_MODEL) == "adaptive"
        assert backend._thinking_modes.mode_for(HAIKU_FALLBACK) == "budget"


class TestModelSwapReshapesThinking:
    """The fallback model can take the other thinking shape, so swapping to it must rebuild
    the thinking fields and the temperature instead of resending the primary's."""

    def test_adaptive_primary_falls_back_to_a_budget_fallback_with_a_budget_request(
        self,
    ) -> None:
        backend = make_backend("us.anthropic.claude-sonnet-5", fallback=HAIKU_FALLBACK)
        client, seen = recording_client(model_unavailable_for("us.anthropic.claude-sonnet-5"))
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.model == HAIKU_FALLBACK
        assert len(seen) == 2
        assert seen[0]["additionalModelRequestFields"] == ADAPTIVE_REQUEST
        assert "temperature" not in seen[0]["inferenceConfig"]
        assert seen[1]["modelId"] == HAIKU_FALLBACK
        assert seen[1]["additionalModelRequestFields"] == BUDGET_REQUEST
        assert seen[1]["inferenceConfig"]["temperature"] == 1.0

    def test_budget_primary_falls_back_to_an_adaptive_fallback_with_an_adaptive_request(
        self,
    ) -> None:
        backend = make_backend(
            "us.anthropic.claude-sonnet-4-6", fallback="us.anthropic.claude-sonnet-5"
        )
        client, seen = recording_client(model_unavailable_for("us.anthropic.claude-sonnet-4-6"))
        backend._client = client

        backend.complete(MESSAGES, thinking=True)

        assert seen[0]["additionalModelRequestFields"] == BUDGET_REQUEST
        assert seen[1]["modelId"] == "us.anthropic.claude-sonnet-5"
        assert seen[1]["additionalModelRequestFields"] == ADAPTIVE_REQUEST
        assert "temperature" not in seen[1]["inferenceConfig"]

    def test_a_non_thinking_request_gains_no_thinking_fields_on_a_model_swap(self) -> None:
        backend = make_backend("us.anthropic.claude-sonnet-5", fallback=HAIKU_FALLBACK)
        client, seen = recording_client(model_unavailable_for("us.anthropic.claude-sonnet-5"))
        backend._client = client

        backend.complete(MESSAGES, temperature=0.4, thinking=False)

        assert len(seen) == 2
        assert "additionalModelRequestFields" not in seen[1]
        assert seen[1]["inferenceConfig"]["temperature"] == 0.4

    def test_calls_after_a_swap_are_built_for_the_fallback_model(self) -> None:
        """After the swap the instance's active model is the fallback, so the next call's
        request must be shaped for it, not for the original primary."""
        backend = make_backend("us.anthropic.claude-sonnet-5", fallback=HAIKU_FALLBACK)
        client, seen = recording_client(model_unavailable_for("us.anthropic.claude-sonnet-5"))
        backend._client = client
        backend.complete(MESSAGES, thinking=True)
        seen.clear()

        backend.complete(MESSAGES, thinking=True)

        assert len(seen) == 1
        assert seen[0]["modelId"] == HAIKU_FALLBACK
        assert seen[0]["additionalModelRequestFields"] == BUDGET_REQUEST
        assert seen[0]["inferenceConfig"]["temperature"] == 1.0

    def test_what_one_model_learned_does_not_leak_to_the_fallback(self) -> None:
        """The unclassified primary rejects budget thinking and is remembered as adaptive;
        when it later becomes unavailable, the (budget) fallback must still get budget."""
        backend = make_backend(UNCLASSIFIED_MODEL, fallback=HAIKU_FALLBACK)
        primary_down = model_unavailable_for(UNCLASSIFIED_MODEL)
        state = {"primary_down": False}

        def responder(request: dict[str, Any]) -> dict[str, Any]:
            if state["primary_down"]:
                return primary_down(request)
            return reject_budget_thinking(request)

        client, seen = recording_client(responder)
        backend._client = client
        backend.complete(MESSAGES, thinking=True)
        seen.clear()
        state["primary_down"] = True

        backend.complete(MESSAGES, thinking=True)

        assert seen[0]["modelId"] == UNCLASSIFIED_MODEL
        assert seen[0]["additionalModelRequestFields"] == ADAPTIVE_REQUEST
        assert seen[1]["modelId"] == HAIKU_FALLBACK
        assert seen[1]["additionalModelRequestFields"] == BUDGET_REQUEST
