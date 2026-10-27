"""AnthropicBackend: learning that a model needs adaptive thinking (mocked, no network).

A model the classifier does not know rejects the budget shape with a 400 carrying the text
the Bedrock Converse API returned live on 2026-10-04 (LIVE_ENABLED_THINKING_REJECTION): the
backend switches that model to adaptive, re-sends exactly once, and remembers. The request
shape per known model id is in test_llm_adaptive_thinking_anthropic.py.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.unit.anthropic_thinking_harness import (
    LIVE_ENABLED_THINKING_REJECTION,
    MESSAGES,
    UNCLASSIFIED_MODEL,
    FakeBadRequest,
    failing_stream_manager,
    install_fake_anthropic,
    make_backend,
    ok_response,
    stream_manager,
)

CONCURRENT_CALLS = 8
# Far longer than eight threads need to reach the barrier, short enough that a broken
# interleaving fails the test instead of waiting for the global pytest timeout.
BARRIER_TIMEOUT_SECONDS = 10


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    return install_fake_anthropic(monkeypatch)


class TestRetryAndRemember:
    """A model the classifier does not know rejects the budget shape: switch that model to
    adaptive, re-send exactly once, remember."""

    def _backend_rejecting_budget(self) -> tuple[Any, MagicMock]:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        seen: list[dict[str, Any]] = []

        def create(**kwargs: Any) -> MagicMock:
            seen.append(dict(kwargs))
            if kwargs.get("thinking", {}).get("type") == "enabled":
                raise FakeBadRequest(LIVE_ENABLED_THINKING_REJECTION)
            return ok_response()

        client.messages.create.side_effect = create
        client.seen = seen
        backend._client = client
        return backend, client

    def test_complete_retries_once_in_adaptive_mode_and_returns_the_answer(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend, client = self._backend_rejecting_budget()

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "the answer"
        assert [c["thinking"] for c in client.seen] == [
            {"type": "enabled", "budget_tokens": 2048},
            {"type": "adaptive"},
        ]

    def test_the_retry_keeps_everything_else_in_the_request(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend, client = self._backend_rejecting_budget()

        backend.complete(MESSAGES, max_tokens=321, system="be brief", thinking=True)

        first, second = client.seen
        assert {k: v for k, v in first.items() if k != "thinking"} == {
            k: v for k, v in second.items() if k != "thinking"
        }
        assert second["max_tokens"] == 321
        assert second["system"] == "be brief"
        assert second["model"] == UNCLASSIFIED_MODEL

    def test_a_second_call_goes_straight_to_adaptive_without_another_rejection(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend, client = self._backend_rejecting_budget()
        backend.complete(MESSAGES, thinking=True)
        client.seen.clear()

        backend.complete(MESSAGES, thinking=True)

        assert [c["thinking"] for c in client.seen] == [{"type": "adaptive"}]

    def test_stream_retries_once_and_remembers(self, mock_anthropic: MagicMock) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        seen: list[dict[str, Any]] = []

        def stream(**kwargs: Any) -> MagicMock:
            seen.append(dict(kwargs))
            if kwargs.get("thinking", {}).get("type") == "enabled":
                return failing_stream_manager(FakeBadRequest(LIVE_ENABLED_THINKING_REJECTION))
            return stream_manager(["a", "b"])

        client.messages.stream.side_effect = stream
        backend._client = client

        assert list(backend.stream(MESSAGES, thinking=True)) == ["a", "b"]
        assert [c["thinking"] for c in seen] == [
            {"type": "enabled", "budget_tokens": 2048},
            {"type": "adaptive"},
        ]

        seen.clear()
        assert list(backend.stream(MESSAGES, thinking=True)) == ["a", "b"]
        assert [c["thinking"] for c in seen] == [{"type": "adaptive"}]

    def test_the_switch_logs_one_warning_naming_the_model_and_mode_without_the_provider_text(
        self, mock_anthropic: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend, _client = self._backend_rejecting_budget()

        with caplog.at_level(logging.WARNING, logger="trelix.llm.anthropic_backend"):
            backend.complete(MESSAGES, thinking=True)
            backend.complete(MESSAGES, thinking=True)
            backend.complete(MESSAGES, thinking=True)

        warnings = [r for r in caplog.records if r.name == "trelix.llm.anthropic_backend"]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert UNCLASSIFIED_MODEL in message
        assert "adaptive" in message
        assert "output_config.effort" not in message
        assert "is not supported for this model" not in message

    def test_a_second_rejection_is_raised_not_retried_again(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        client.messages.create.side_effect = FakeBadRequest(LIVE_ENABLED_THINKING_REJECTION)
        backend._client = client

        with pytest.raises(FakeBadRequest):
            backend.complete(MESSAGES, thinking=True)

        assert client.messages.create.call_count == 2

    def test_a_different_error_on_the_adaptive_retry_propagates(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        client.messages.create.side_effect = [
            FakeBadRequest(LIVE_ENABLED_THINKING_REJECTION),
            FakeBadRequest("max_tokens: must be greater than 0"),
        ]
        backend._client = client

        with pytest.raises(FakeBadRequest, match="max_tokens"):
            backend.complete(MESSAGES, thinking=True)

        assert client.messages.create.call_count == 2

    def test_an_unrelated_bad_request_is_not_swallowed_or_remembered(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        client.messages.create.side_effect = FakeBadRequest(
            "messages: text content blocks must be non-empty"
        )
        backend._client = client

        with pytest.raises(FakeBadRequest, match="non-empty"):
            backend.complete(MESSAGES, thinking=True)
        assert client.messages.create.call_count == 1

        client.messages.create.reset_mock()
        client.messages.create.side_effect = None
        client.messages.create.return_value = ok_response()
        backend.complete(MESSAGES, thinking=True)
        assert client.messages.create.call_args[1]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }

    def test_the_marker_text_on_a_non_400_status_is_not_the_rejection(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        client.messages.create.side_effect = FakeBadRequest(
            LIVE_ENABLED_THINKING_REJECTION, status_code=422
        )
        backend._client = client

        with pytest.raises(FakeBadRequest):
            backend.complete(MESSAGES, thinking=True)

        assert client.messages.create.call_count == 1

    def test_the_marker_text_on_an_exception_without_a_status_is_not_the_rejection(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError(LIVE_ENABLED_THINKING_REJECTION)
        backend._client = client

        with pytest.raises(RuntimeError):
            backend.complete(MESSAGES, thinking=True)

        assert client.messages.create.call_count == 1

    def test_a_400_about_thinking_is_not_retried_when_thinking_was_not_requested(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend(UNCLASSIFIED_MODEL)
        client = MagicMock()
        client.messages.create.side_effect = FakeBadRequest(LIVE_ENABLED_THINKING_REJECTION)
        backend._client = client

        with pytest.raises(FakeBadRequest):
            backend.complete(MESSAGES, thinking=False)

        assert client.messages.create.call_count == 1

    def test_an_adaptive_request_that_gets_the_rejection_is_raised_not_looped(
        self, mock_anthropic: MagicMock
    ) -> None:
        backend = make_backend("claude-sonnet-5")
        client = MagicMock()
        client.messages.create.side_effect = FakeBadRequest(LIVE_ENABLED_THINKING_REJECTION)
        backend._client = client

        with pytest.raises(FakeBadRequest):
            backend.complete(MESSAGES, thinking=True)

        assert client.messages.create.call_count == 1

    def test_the_memory_is_per_backend_instance(self, mock_anthropic: MagicMock) -> None:
        learned, _client = self._backend_rejecting_budget()
        learned.complete(MESSAGES, thinking=True)

        fresh = make_backend(UNCLASSIFIED_MODEL)

        assert fresh._thinking_kwargs(thinking=True) == {
            "thinking": {"type": "enabled", "budget_tokens": 2048}
        }
        assert learned._thinking_kwargs(thinking=True) == {"thinking": {"type": "adaptive"}}

    def test_concurrent_rejections_converge_and_log_one_warning(
        self, mock_anthropic: MagicMock, caplog: pytest.LogCaptureFixture
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
        thinking_types: list[str] = []

        def create(**kwargs: Any) -> MagicMock:
            thinking_type = kwargs["thinking"]["type"]
            thinking_types.append(thinking_type)
            if thinking_type == "enabled":
                all_budget_requests_in_flight.wait()
                raise FakeBadRequest(LIVE_ENABLED_THINKING_REJECTION)
            return ok_response()

        client = MagicMock()
        client.messages.create.side_effect = create
        backend._client = client
        results: list[str] = []
        errors: list[BaseException] = []

        def call() -> None:
            try:
                results.append(backend.complete(MESSAGES, thinking=True).content)
            except BaseException as exc:  # noqa: BLE001 -- surfaced by the assertion below
                errors.append(exc)

        with caplog.at_level(logging.WARNING, logger="trelix.llm.anthropic_backend"):
            threads = [threading.Thread(target=call) for _ in range(CONCURRENT_CALLS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert errors == []
        assert results == ["the answer"] * CONCURRENT_CALLS
        assert (
            sorted(thinking_types)
            == ["adaptive"] * CONCURRENT_CALLS + ["enabled"] * CONCURRENT_CALLS
        )
        assert len([r for r in caplog.records if r.name == "trelix.llm.anthropic_backend"]) == 1


class TestAgainstTheRealSdkException:
    def test_the_sdks_own_bad_request_error_is_recognised(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        backend = make_backend(UNCLASSIFIED_MODEL)
        response = MagicMock()
        response.status_code = 400
        response.headers = {}
        error = anthropic.BadRequestError(
            "Error code: 400 - " + LIVE_ENABLED_THINKING_REJECTION, response=response, body=None
        )
        client = MagicMock()
        client.messages.create.side_effect = [error, ok_response()]
        backend._client = client

        result = backend.complete(MESSAGES, thinking=True)

        assert result.content == "the answer"
        assert [c[1]["thinking"] for c in client.messages.create.call_args_list] == [
            {"type": "enabled", "budget_tokens": 2048},
            {"type": "adaptive"},
        ]
