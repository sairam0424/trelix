"""
GenAI `chat` spans around TrelixChatClient calls (roadmap C-8 PR 3), on REAL spans.

Covers R2-R5 of the design with an InMemorySpanExporter attached to the process's global
TracerProvider (the fixture is tests/unit/test_otel_tracing.py's), plus D4 (the unconfigured
placeholder is recorded) and the factory's on path. The wrapper's SDK-free contracts -- the off
path, the `_client` property, `request_model`, `seed_kwargs`, the content gate against a
recording handler and the one-time WARNING without the library -- are in
test_otel_llm_wrapper.py; the fakes both files share are in tests/unit/otel_llm_fakes.py.

Every test here needs opentelemetry-sdk AND opentelemetry-util-genai >= 1.2b0, the `otel`
extra's floor since this PR (`suspend()` and the `gen_ai.usage.cache_write.input_tokens` name
are 1.2b0-only); an older install skips with the floor in the reason.

Listed in scripts/mutation.py::DESELECTED_FILES for the reason test_otel_tracing.py is: the
global TracerProvider is a one-shot process resource.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.unit.otel_llm_fakes import (
    UPSTREAM_MODE,
    FakeBackend,
    canaries_absent,
    cfg,
    reply,
    require_util_genai_at_the_floor,
    span_text,
)
from trelix.llm.client import ChatMessage, TrelixChatClient

# What every span carries; a `complete()` adds the response and usage keys on top.
_REQUEST_ATTRIBUTES = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "fake-model-1",
    "gen_ai.request.max_tokens": 2048,
}


def _traced(backend: TrelixChatClient, config: Any) -> Any:
    """The wrapper the factory would build, through the real `traced()` and the real handler."""
    from trelix.llm.otel import traced

    client = traced(backend, config)
    assert type(client).__name__ == "TracedChatClient", "traced() returned the bare backend"
    return client


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.fixture(scope="module")
def _test_tracer_provider():
    """An InMemorySpanExporter fed by the process's global TracerProvider.

    Attaches to the incumbent provider rather than installing its own, exactly as
    tests/unit/test_otel_tracing.py::_test_tracer_provider does and for the reason written
    there: the global TracerProvider is a one-shot process resource.
    """
    require_util_genai_at_the_floor(sdk=True)

    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "trelix-test"}))
        trace.set_tracer_provider(provider)
        installed = trace.get_tracer_provider()
        if installed is not provider:
            pytest.fail(
                "the global TracerProvider is held by a non-SDK "
                f"{type(installed).__name__}, which cannot export spans"
            )

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    try:
        yield exporter
    finally:
        processor.shutdown()


@pytest.fixture()
def otel_test_exporter(_test_tracer_provider, monkeypatch: pytest.MonkeyPatch):
    """The exporter, cleared; OTel on from the environment; trelix's memoised handler, settings
    and one-time flags reset; the two upstream content variables removed BEFORE the handler is
    rebuilt (it reads them once, when built)."""
    import trelix.llm.otel as llm_otel
    import trelix.retrieval.otel_tracing as otel_tracing

    monkeypatch.delenv(UPSTREAM_MODE, raising=False)
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", raising=False)
    monkeypatch.setenv("TRELIX_OTEL_ENABLED", "true")
    _test_tracer_provider.clear()
    prev_handler = otel_tracing._handler
    prev_service_name = otel_tracing._handler_service_name
    otel_tracing._handler = None
    otel_tracing._handler_service_name = None
    monkeypatch.setattr(otel_tracing, "_env_otel_settings", None)
    monkeypatch.setattr(otel_tracing, "_capture_inert_warned", False)
    monkeypatch.setattr(llm_otel, "_unavailable_warned", False)
    try:
        yield _test_tracer_provider
    finally:
        otel_tracing._handler = prev_handler
        otel_tracing._handler_service_name = prev_service_name


class TestCompleteSpan:
    def test_complete_emits_one_chat_span_with_exactly_these_attributes(
        self, otel_test_exporter
    ) -> None:
        """R2. MUTATIONS that must make this fail: `input_tokens` without the cache sum (19 ->
        10); `finish_reasons` from `finish_reason` (("length",)); `request.max_tokens` from the
        call argument only (key missing); provider row anthropic -> aws.bedrock;
        `response_model_name` from cfg.model."""
        from opentelemetry.trace import SpanKind, StatusCode

        response = reply()
        client = _traced(FakeBackend(response=response), cfg())
        result = client.complete([ChatMessage("user", "CANARY-IN-7f3a")], system="CANARY-SYS-9c1e")

        assert result is response
        (span,) = otel_test_exporter.get_finished_spans()
        assert span.name == "chat fake-model-1"
        assert span.kind is SpanKind.CLIENT
        assert span.status.status_code is StatusCode.UNSET
        assert dict(span.attributes) == {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": "fake-model-1",
            "gen_ai.request.max_tokens": 2048,
            "gen_ai.response.model": "claude-sonnet-4-6-20260101",
            "gen_ai.response.finish_reasons": ("max_tokens",),
            "gen_ai.usage.input_tokens": 19,
            "gen_ai.usage.output_tokens": 5,
            "gen_ai.usage.cache_read.input_tokens": 7,
            "gen_ai.usage.cache_write.input_tokens": 2,
            "trelix.finish_reason": "length",
        }
        assert canaries_absent([span])

    def test_the_span_keeps_the_librarys_instrumentation_scope(self, otel_test_exporter) -> None:
        """Critique C7: no scope kwargs to TelemetryHandler(). MUTATION: pass
        instrumentation_scope_name="trelix"."""
        _traced(FakeBackend(response=reply()), cfg()).complete([ChatMessage("user", "q")])

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.instrumentation_scope.name == "opentelemetry.util.genai.handler"

    def test_the_call_argument_is_the_cap_when_given(self, otel_test_exporter) -> None:
        """R2(a)."""
        client = _traced(FakeBackend(response=reply()), cfg())
        client.complete([ChatMessage("user", "q")], max_tokens=512)

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.attributes["gen_ai.request.max_tokens"] == 512

    def test_no_raw_finish_reason_means_no_finish_reasons_key(self, otel_test_exporter) -> None:
        """R2(b): the provider reported none (Vertex with no candidates)."""
        response = reply(raw_finish_reason=None, finish_reason="unknown")
        _traced(FakeBackend(response=response), cfg()).complete([ChatMessage("user", "q")])

        (span,) = otel_test_exporter.get_finished_spans()
        assert "gen_ai.response.finish_reasons" not in span.attributes
        assert span.attributes["trelix.finish_reason"] == "unknown"

    def test_zero_cache_counts_leave_no_cache_keys(self, otel_test_exporter) -> None:
        """R2(c): the non-Anthropic backends."""
        response = reply(cache_read_tokens=0, cache_write_tokens=0)
        _traced(FakeBackend(response=response), cfg()).complete([ChatMessage("user", "q")])

        (span,) = otel_test_exporter.get_finished_spans()
        assert "gen_ai.usage.cache_read.input_tokens" not in span.attributes
        assert "gen_ai.usage.cache_write.input_tokens" not in span.attributes
        assert span.attributes["gen_ai.usage.input_tokens"] == 10

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"provider": "openai", "model": "gpt-4o"}, "openai"),
            ({"provider": "azure", "model": "gpt-4o"}, "azure.ai.openai"),
            ({"provider": "anthropic"}, "anthropic"),
            ({"provider": "bedrock", "aws_region": "us-east-1"}, "aws.bedrock"),
            ({"provider": "litellm"}, "litellm"),
            ({"provider": "vertex", "google_api_key": "test-k"}, "gcp.gemini"),
            ({"provider": "vertex", "google_project_id": "p"}, "gcp.vertex_ai"),
        ],
    )
    def test_provider_name_on_the_span(
        self, otel_test_exporter, fields: dict[str, Any], expected: str
    ) -> None:
        """R2(d). MUTATION: swap any row of the provider table."""
        _traced(FakeBackend(response=reply()), cfg(**fields)).complete([ChatMessage("user", "q")])

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.attributes["gen_ai.provider.name"] == expected

    def test_the_unconfigured_placeholder_is_recorded_like_any_reply(
        self, otel_test_exporter
    ) -> None:
        """D4: `model == "none"` with zero usage, through the real OpenAI backend (no key), and
        the factory's on path. MUTATION: never wrap -> type name OpenAIBackend."""
        from trelix.llm.factory import build_chat_client

        client = build_chat_client(cfg(provider="openai", model="gpt-4o"))
        assert type(client).__name__ == "TracedChatClient"
        assert type(client._backend).__name__ == "OpenAIBackend"
        response = client.complete([ChatMessage("user", "q")])

        assert response.model == "none"
        (span,) = otel_test_exporter.get_finished_spans()
        assert span.name == "chat gpt-4o"
        assert span.attributes["gen_ai.provider.name"] == "openai"
        assert span.attributes["gen_ai.response.model"] == "none"
        assert span.attributes["gen_ai.usage.input_tokens"] == 0
        assert span.attributes["gen_ai.usage.output_tokens"] == 0


class _ProviderError(Exception):
    pass


class TestErrorPath:
    def test_backend_exception_is_recorded_as_error_and_re_raised(self, otel_test_exporter) -> None:
        """R3. MUTATIONS: swallow the exception; stop() instead of fail() (status UNSET);
        fail(exc) instead of the class name (description "boom")."""
        from opentelemetry.trace import StatusCode

        client = _traced(FakeBackend(error=RuntimeError("boom")), cfg())
        with pytest.raises(RuntimeError, match="boom"):
            client.complete([ChatMessage("user", "CANARY-IN-7f3a")])

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description == "RuntimeError"
        assert span.attributes["error.type"] == "RuntimeError"
        assert not any(key.startswith("gen_ai.usage.") for key in span.attributes)
        for key, value in _REQUEST_ATTRIBUTES.items():
            assert span.attributes[key] == value
        assert "boom" not in span_text(span)
        assert canaries_absent([span])

    def test_a_non_builtin_exception_is_module_qualified(self, otel_test_exporter) -> None:
        client = _traced(FakeBackend(error=_ProviderError("body")), cfg())
        with pytest.raises(_ProviderError):
            client.complete([ChatMessage("user", "q")])

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.attributes["error.type"].endswith("._ProviderError")
        assert span.status.description == span.attributes["error.type"]
        assert "body" not in span_text(span)


class TestStreamAndToolCallSpans:
    def test_stream_yields_the_chunks_and_emits_a_request_only_span(
        self, otel_test_exporter
    ) -> None:
        """R4(a). MUTATION: assign usage 0 on stream spans (`gen_ai.usage.input_tokens: 0`)."""
        from opentelemetry.trace import StatusCode

        client = _traced(FakeBackend(), cfg())
        assert list(client.stream([ChatMessage("user", "q")], max_tokens=64)) == ["a", "b"]

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.name == "chat fake-model-1"
        assert span.status.status_code is StatusCode.UNSET
        assert dict(span.attributes) == {**_REQUEST_ATTRIBUTES, "gen_ai.request.max_tokens": 64}

    def test_a_suspended_stream_does_not_parent_the_consumers_own_spans(
        self, otel_test_exporter
    ) -> None:
        """R4(b). MUTATION: remove suspend() -> unrelated.parent.span_id == chat.span_id."""
        from opentelemetry import trace

        client = _traced(FakeBackend(), cfg())
        it = client.stream([ChatMessage("user", "q")])
        assert next(it) == "a"
        with trace.get_tracer("trelix.test").start_as_current_span("unrelated"):
            pass
        assert list(it) == ["b"]

        spans = {s.name: s for s in otel_test_exporter.get_finished_spans()}
        assert set(spans) == {"unrelated", "chat fake-model-1"}
        assert spans["unrelated"].parent is None
        assert spans["chat fake-model-1"].end_time > spans["unrelated"].end_time

    def test_closing_the_stream_early_ends_the_span_unset_with_request_attributes(
        self, otel_test_exporter
    ) -> None:
        from opentelemetry.trace import StatusCode

        client = _traced(FakeBackend(chunks=("a", "b", "c")), cfg())
        it = client.stream([ChatMessage("user", "q")])
        assert next(it) == "a"
        it.close()

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.status.status_code is StatusCode.UNSET
        assert dict(span.attributes) == _REQUEST_ATTRIBUTES

    def test_a_stream_that_is_never_iterated_produces_no_span(self, otel_test_exporter) -> None:
        client = _traced(FakeBackend(), cfg())
        client.stream([ChatMessage("user", "q")])

        assert otel_test_exporter.get_finished_spans() == ()

    def test_a_backend_error_mid_stream_ends_the_span_with_error(self, otel_test_exporter) -> None:
        from opentelemetry.trace import StatusCode

        client = _traced(FakeBackend(chunks=("a", RuntimeError("boom"))), cfg())
        it = client.stream([ChatMessage("user", "q")])
        assert next(it) == "a"
        with pytest.raises(RuntimeError, match="boom"):
            next(it)

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.status.status_code is StatusCode.ERROR
        assert span.attributes["error.type"] == "RuntimeError"

    def test_tool_call_emits_a_request_only_span(self, otel_test_exporter) -> None:
        """R4(c), D5. MUTATION: tool_call() passing None for its own max_tokens -> the span
        reports the config cap (1024) instead of the call's (16)."""
        client = _traced(FakeBackend(), cfg(provider="openai", model="gpt-4o", max_tokens=1024))
        result = client.tool_call(
            [ChatMessage("user", "q")], tools=[{"type": "function"}], force_tool="t", max_tokens=16
        )

        assert result.tool_name == "t"
        (span,) = otel_test_exporter.get_finished_spans()
        assert dict(span.attributes) == {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "fake-model-1",
            "gen_ai.request.max_tokens": 16,
        }

    def test_tool_call_error_is_recorded_and_re_raised(self, otel_test_exporter) -> None:
        from opentelemetry.trace import StatusCode

        client = _traced(FakeBackend(error=RuntimeError("boom")), cfg())
        with pytest.raises(RuntimeError, match="boom"):
            client.tool_call([ChatMessage("user", "q")], tools=[])

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.status.status_code is StatusCode.ERROR
        assert span.attributes["error.type"] == "RuntimeError"


class TestContentGate:
    def test_upstream_span_only_alone_puts_no_content_on_the_span(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R5(b): the host opted another library in; trelix's gate is off. MUTATION: hand the
        messages to the invocation unconditionally -> canaries present."""
        monkeypatch.setenv(UPSTREAM_MODE, "SPAN_ONLY")
        client = _traced(FakeBackend(response=reply()), cfg())
        client.complete([ChatMessage("user", "CANARY-IN-7f3a")], system="CANARY-SYS-9c1e")

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 1
        assert "gen_ai.input.messages" not in spans[0].attributes
        assert canaries_absent(spans)

    def test_both_switches_on_put_the_prompt_instruction_and_reply_on_the_span(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R5(c): the literal JSON the library writes (separators (",", ":"))."""
        monkeypatch.setenv(UPSTREAM_MODE, "SPAN_ONLY")
        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "true")
        client = _traced(FakeBackend(response=reply()), cfg())
        client.complete([ChatMessage("user", "CANARY-IN-7f3a")], system="CANARY-SYS-9c1e")

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.attributes["gen_ai.input.messages"] == (
            '[{"role":"user","parts":[{"content":"CANARY-IN-7f3a","type":"text"}],"name":null}]'
        )
        assert span.attributes["gen_ai.system_instructions"] == (
            '[{"content":"CANARY-SYS-9c1e","type":"text"}]'
        )
        assert span.attributes["gen_ai.output.messages"] == (
            '[{"role":"assistant","parts":[{"content":"CANARY-OUT-5b2d","type":"text"}],'
            '"finish_reason":null,"name":null}]'
        )

    def test_the_first_system_role_message_is_the_instruction(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critique C12: exactly what the backends send when `system` is not passed."""
        monkeypatch.setenv(UPSTREAM_MODE, "SPAN_ONLY")
        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "true")
        client = _traced(FakeBackend(response=reply()), cfg())
        client.complete(
            [ChatMessage("system", "CANARY-SYS-9c1e"), ChatMessage("user", "CANARY-IN-7f3a")]
        )

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.attributes["gen_ai.system_instructions"] == (
            '[{"content":"CANARY-SYS-9c1e","type":"text"}]'
        )
        assert span.attributes["gen_ai.input.messages"] == (
            '[{"role":"user","parts":[{"content":"CANARY-IN-7f3a","type":"text"}],"name":null}]'
        )

    def test_trelix_flag_alone_records_nothing_and_warns_exactly_once(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R5(d). MUTATIONS: remove the warning (0 records); warn per call (2 records)."""
        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "true")
        client = _traced(FakeBackend(response=reply()), cfg())
        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.otel"):
            for _ in range(2):
                client.complete([ChatMessage("user", "CANARY-IN-7f3a")], system="CANARY-SYS-9c1e")

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 2
        assert canaries_absent(spans)
        records = _warnings(caplog)
        assert len(records) == 1
        assert "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT" in records[0].getMessage()

    def test_stream_replies_are_never_captured_even_with_both_switches_on(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(UPSTREAM_MODE, "SPAN_ONLY")
        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "true")
        client = _traced(FakeBackend(chunks=("CANARY-OUT-5b2d",)), cfg())
        assert list(client.stream([ChatMessage("user", "CANARY-IN-7f3a")])) == ["CANARY-OUT-5b2d"]

        (span,) = otel_test_exporter.get_finished_spans()
        assert "gen_ai.input.messages" in span.attributes
        assert "gen_ai.output.messages" not in span.attributes
        assert "CANARY-OUT-5b2d" not in span_text(span)
