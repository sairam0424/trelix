"""Shared fakes for the C-8 chat-span tests (test_otel_llm_spans.py, test_otel_llm_wrapper.py,
test_otel_llm_forwarding.py).

Not a test module. A TrelixChatClient stand-in whose replies the test chooses and which records
what it RECEIVED, a recording stand-in for util-genai's TelemetryHandler (so a test can see what
trelix ASSIGNED, not what the library later chose to emit), the LLMConfig/ChatResponse builders,
the canaries and the 1.2b0 floor check. Every expected value stays a literal in the test files;
nothing here is imported from the module under test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib.metadata import version
from typing import Any

import pytest

from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient

CANARIES = ("CANARY-IN-7f3a", "CANARY-SYS-9c1e", "CANARY-OUT-5b2d")
UPSTREAM_MODE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def require_util_genai_at_the_floor(*, sdk: bool) -> None:
    """Skip unless opentelemetry-util-genai >= 1.2b0 (the `otel` extra's floor: `suspend()` and
    the `gen_ai.usage.cache_write.input_tokens` name are 1.2b0-only) and, with *sdk*, the SDK."""
    if sdk:
        pytest.importorskip("opentelemetry.sdk", reason="requires pip install trelix[otel]")
    pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
    from packaging.version import Version

    installed = version("opentelemetry-util-genai")
    if Version(installed) < Version("1.2b0"):
        pytest.skip(
            f"requires opentelemetry-util-genai>=1.2b0 (installed {installed}); "
            "pip install trelix[otel]"
        )


def cfg(provider: str = "anthropic", model: str = "claude-sonnet-4-6", **fields: Any) -> Any:
    from trelix.core.config import LLMConfig

    return LLMConfig(
        _env_file=None,  # type: ignore[call-arg]
        provider=provider,
        model=model,
        max_tokens=fields.pop("max_tokens", 2048),
        **fields,
    )


def reply(**overrides: Any) -> ChatResponse:
    """The R2 fixture reply: Anthropic-shaped usage with both cache counts."""
    fields: dict[str, Any] = {
        "content": "CANARY-OUT-5b2d",
        "model": "claude-sonnet-4-6-20260101",
        "finish_reason": "length",
        "raw_finish_reason": "max_tokens",
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 7,
        "cache_write_tokens": 2,
    }
    return ChatResponse(**{**fields, **overrides})


class FakeBackend(TrelixChatClient):
    """Returns what it is given and records what it RECEIVED: `calls` holds one
    `(method, {parameter: value})` entry per call, by parameter name, so a test can assert
    that the wrapper forwarded every argument. An exception in `chunks` is raised mid-stream."""

    _model = "fake-model-1"

    def __init__(
        self,
        response: ChatResponse | None = None,
        error: BaseException | None = None,
        chunks: tuple[Any, ...] = ("a", "b"),
    ) -> None:
        self.response = response
        self.error = error
        self.chunks = chunks
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, method: str, **received: Any) -> None:
        self.calls.append((method, received))

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> ChatResponse:
        self._record(
            "complete",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            thinking=thinking,
        )
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> Iterator[str]:
        self._record(
            "stream",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            thinking=thinking,
        )
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolCallResponse:
        self._record(
            "tool_call",
            messages=messages,
            tools=tools,
            force_tool=force_tool,
            max_tokens=max_tokens,
        )
        if self.error is not None:
            raise self.error
        return ToolCallResponse(tool_name="t", tool_arguments={"k": 1})


class RecordingInvocation:
    """What `handler.inference()` returns: records every assignment and lifecycle call."""

    def __init__(self, provider: str, request_model: str | None) -> None:
        self.provider = provider
        self.request_model = request_model
        self.attributes: dict[str, Any] = {}
        self.lifecycle: list[Any] = []

    def stop(self) -> None:
        self.lifecycle.append("stop")

    def fail(self, error: Any) -> None:
        self.lifecycle.append(error)

    def suspend(self) -> None:
        self.lifecycle.append("suspend")


class RecordingHandler:
    """Stands in for TelemetryHandler; `captures` is what should_capture_content() answers."""

    def __init__(self, captures: bool = True) -> None:
        self.invocations: list[RecordingInvocation] = []
        self._captures = captures

    def inference(self, provider: str, *, request_model: str | None = None) -> RecordingInvocation:
        invocation = RecordingInvocation(provider, request_model)
        self.invocations.append(invocation)
        return invocation

    def should_capture_content(self) -> bool:
        return self._captures


def span_text(span: Any) -> str:
    """Everything a finished span can carry text in, as one JSON string."""
    events = [{"name": e.name, "attributes": dict(e.attributes or {})} for e in span.events]
    return json.dumps(
        {
            "attributes": dict(span.attributes or {}),
            "events": events,
            "status": span.status.description,
        },
        default=str,
    )


def canaries_absent(spans: Any) -> bool:
    return all(canary not in span_text(s) for canary in CANARIES for s in spans)
