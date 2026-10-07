"""
Unit tests for trelix.retrieval.otel_tracing.

Covers:
- Zero behavior change / zero opentelemetry import when TRELIX_OTEL_ENABLED=false
  (the single most important test — proves the feature is truly opt-in)
- Correct gen_ai.* span attributes when enabled
- Thread-context propagation across ThreadPoolExecutor via with_current_context()
- The content gate: query text reaches a leg span only with TRELIX_OTEL_CAPTURE_CONTENT=true
  (trelix's AND-gate in front of OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT), and a
  one-time WARNING when the trelix flag is on but the upstream mode discards the text
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

_CANARY = "CANARY-QUERY-1a2b"
_UPSTREAM_MODE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_INERT_WARNING = (
    "TRELIX_OTEL_CAPTURE_CONTENT is true but OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    " is NO_CONTENT (unset or invalid) — no prompt, reply or query text will be recorded."
    " Set it to SPAN_ONLY, EVENT_ONLY or SPAN_AND_EVENT."
)


def _cfg(
    otel_enabled: bool,
    service_name: str = "trelix-test",
    otel_exporter_endpoint: str | None = None,
    otel_capture_content: bool | None = None,
) -> SimpleNamespace:
    """A RetrievalConfig stand-in. `otel_capture_content=None` leaves the attribute OFF the
    object, the shape of every config built before the flag existed."""
    fields: dict[str, Any] = {
        "otel_enabled": otel_enabled,
        "otel_service_name": service_name,
        "otel_exporter_endpoint": otel_exporter_endpoint,
    }
    if otel_capture_content is not None:
        fields = {**fields, "otel_capture_content": otel_capture_content}
    return SimpleNamespace(**fields)


def _span_text(span: Any) -> str:
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


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """trelix's WARNING records only: another library's (an SDK span processor another module left
    attached) must not count against the warn-once and no-warning pins here."""
    return [
        r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("trelix")
    ]


# ---------------------------------------------------------------------------
# Disabled path — zero cost, zero import
# ---------------------------------------------------------------------------


class TestDisabledIsNoOp:
    def test_retrieval_leg_span_is_noop_when_disabled(self) -> None:
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        cfg = _cfg(otel_enabled=False)
        with retrieval_leg_span(cfg, "vector", query_text="foo", top_k=10) as span:
            span.set_result_count(5)  # must not raise even though nothing is tracing

    def test_pipeline_stage_span_is_noop_when_disabled(self) -> None:
        from trelix.retrieval.otel_tracing import pipeline_stage_span

        cfg = _cfg(otel_enabled=False)
        with pipeline_stage_span(cfg, "fusion", {"rrf_k": 60}):
            pass

    def test_with_current_context_is_passthrough_when_disabled(self) -> None:
        """Without a real TracerProvider/opentelemetry.context installed, the
        wrapped function still behaves identically — no exception, same result."""
        from trelix.retrieval.otel_tracing import with_current_context

        def add(a: int, b: int) -> int:
            return a + b

        wrapped = with_current_context(add)
        assert wrapped(2, 3) == 5

    def test_is_enabled_reads_cfg_without_importing_opentelemetry(self) -> None:
        """is_enabled() must never trigger an opentelemetry import — it's called
        on every single retrieval leg regardless of the flag's value."""
        from trelix.retrieval.otel_tracing import is_enabled

        # Purge opentelemetry.* from sys.modules to catch an accidental import.
        purged = {k: v for k, v in sys.modules.items() if k.startswith("opentelemetry")}
        for k in purged:
            del sys.modules[k]
        try:
            assert is_enabled(_cfg(otel_enabled=False)) is False
            assert is_enabled(_cfg(otel_enabled=True)) is True
            assert not any(k.startswith("opentelemetry") for k in sys.modules)
        finally:
            sys.modules.update(purged)

    def test_retriever_import_does_not_import_opentelemetry(self) -> None:
        """Importing the retriever module itself must not eagerly import OTel —
        only actually enabling the flag at call time should."""
        purged = {k: v for k, v in sys.modules.items() if k.startswith("opentelemetry")}
        retriever_mod = sys.modules.pop("trelix.retrieval.retriever", None)
        otel_tracing_mod = sys.modules.pop("trelix.retrieval.otel_tracing", None)
        for k in purged:
            del sys.modules[k]
        try:
            import trelix.retrieval.retriever  # noqa: F401

            assert not any(k.startswith("opentelemetry") for k in sys.modules)
        finally:
            sys.modules.update(purged)
            # Restoring sys.modules is NOT enough, and the difference is a real bug this
            # test used to cause. The fresh `import trelix.retrieval.retriever` above pulls
            # in a NEW otel_tracing module object and binds it as the PACKAGE ATTRIBUTE
            # `trelix.retrieval.otel_tracing`. Putting the original back in sys.modules
            # leaves that attribute pointing at the orphan, so any later
            # `from trelix.retrieval import otel_tracing` — or a helper that resets
            # module-level metric state — operates on a different object than the code under
            # test. Measured: `pytest test_otel_metrics.py` alone passed 12/12, while
            # `pytest test_otel_tracing.py test_otel_metrics.py` failed 5, purely on
            # alphabetical ordering.
            #
            # Same defect class as the `watchfiles` reload earlier in this release, where a
            # module left half-restored silently disabled the deletion path for every
            # subsequent test in the session.
            package = sys.modules.get("trelix.retrieval")
            for name, module in (
                ("retriever", retriever_mod),
                ("otel_tracing", otel_tracing_mod),
            ):
                if module is None:
                    continue
                sys.modules[f"trelix.retrieval.{name}"] = module
                if package is not None:
                    setattr(package, name, module)


# ---------------------------------------------------------------------------
# Enabled path — real spans via InMemorySpanExporter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _test_tracer_provider():
    """
    Get an InMemorySpanExporter fed by the process's global TracerProvider.

    ATTACHES to whatever provider is installed rather than assuming it can
    install its own, because the global TracerProvider is a ONE-SHOT process
    resource: `set_tracer_provider()` is guarded by a `Once`, so the second
    caller anywhere in the session gets a silent no-op plus a logged
    "Overriding of current TracerProvider is not allowed".

    This module used to be that second caller. tests/unit/test_structured_logging.py
    installs a bare `TracerProvider()` (no span processors) and never restores it, so
    under reverse collection order — where test_structured_logging sorts after
    test_otel_tracing and therefore runs first — this fixture's provider was
    discarded, every span went to a processor-less provider, and all five
    "enabled path" tests below failed on empty span lists (three on
    `len(spans) == 1` against `()`, two on a KeyError/`in spans` miss). Measured:
    `pytest tests/unit/test_otel_tracing.py tests/unit/test_otel_metrics.py
    tests/unit/test_structured_logging.py` was 38 passed forward, 6 failed
    reversed — these 5 plus one unrelated failure inside test_otel_metrics.py.

    Adding a SpanProcessor to the incumbent SDK provider is enough — the
    exporter sees every sampled span regardless of who owns the provider — and
    is order-independent in both directions. A non-SDK incumbent (e.g. a
    NoOpTracerProvider) has no `add_span_processor`, so that case fails loudly
    instead of silently collecting zero spans.

    Skips (rather than errors) when `opentelemetry-sdk` isn't installed —
    CI's default `pip install -e ".[local,dev]"` deliberately does NOT include
    the optional `otel` extra, since otel_enabled=False is the documented,
    tested-elsewhere default; these "enabled path" tests only run when a
    developer/CI job explicitly installs `trelix[otel]`.
    """
    pytest.importorskip("opentelemetry.sdk", reason="requires pip install trelix[otel]")

    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        # Nothing real installed yet (ProxyTracerProvider), so claim the slot.
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "trelix-test"}))
        trace.set_tracer_provider(provider)
        installed = trace.get_tracer_provider()
        if installed is not provider:
            pytest.fail(
                "the global TracerProvider is held by a non-SDK "
                f"{type(installed).__name__}, which cannot export spans — some earlier "
                "test set it and did not restore it"
            )

    from tests.unit.otel_llm_fakes import exporting_spans

    # The provider is shared with the rest of the session: the processor is detached
    # again on teardown (shutting it down alone leaves it attached, see exporting_spans).
    with exporting_spans(provider) as exporter:
        yield exporter


@pytest.fixture()
def otel_test_exporter(_test_tracer_provider, monkeypatch: pytest.MonkeyPatch):
    """
    Yield the module-wide InMemorySpanExporter, cleared before each test.
    Also resets trelix's memoized TelemetryHandler so _get_handler() rebuilds
    it against the real (already-installed) TracerProvider on next use,
    since a freshly-imported handler observed the ProxyTracerProvider only
    once, at first import.

    The two upstream content variables are removed BEFORE the handler is reset:
    the handler reads them once, when it is built, so a developer shell with
    SPAN_ONLY exported would otherwise leak into every rebuilt handler. The
    content-gate state (`_env_otel_settings`, `_capture_inert_warned`) is reset
    alongside the handler so each test starts from "never warned".
    """
    import trelix.retrieval.otel_tracing as otel_tracing

    monkeypatch.delenv(_UPSTREAM_MODE, raising=False)
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", raising=False)
    _test_tracer_provider.clear()
    prev_handler = otel_tracing._handler
    prev_service_name = otel_tracing._handler_service_name
    otel_tracing._handler = None
    otel_tracing._handler_service_name = None
    monkeypatch.setattr(otel_tracing, "_env_otel_settings", None)
    monkeypatch.setattr(otel_tracing, "_capture_inert_warned", False)
    try:
        yield _test_tracer_provider
    finally:
        otel_tracing._handler = prev_handler
        otel_tracing._handler_service_name = prev_service_name


class TestEnabledEmitsSpans:
    def test_retrieval_leg_span_emits_retrieval_span_with_gen_ai_attributes(
        self, otel_test_exporter
    ) -> None:
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        cfg = _cfg(otel_enabled=True)
        with retrieval_leg_span(cfg, "vector", query_text="auth handler", top_k=20) as span:
            span.set_result_count(7)

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 1
        (s,) = spans
        assert s.attributes.get("gen_ai.operation.name") == "retrieval"
        assert s.attributes.get("gen_ai.data_source.id") == "vector"

    def test_pipeline_stage_span_emits_trelix_namespaced_span(self, otel_test_exporter) -> None:
        from trelix.retrieval.otel_tracing import pipeline_stage_span

        cfg = _cfg(otel_enabled=True)
        with pipeline_stage_span(cfg, "fusion", {"rrf_k": 60}):
            pass

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 1
        (s,) = spans
        assert s.name == "trelix.fusion"
        assert s.attributes.get("trelix.fusion.rrf_k") == 60

    def test_leg_span_records_exception_via_fail_not_stop(self, otel_test_exporter) -> None:
        """An exception inside the `with` block must not be swallowed, and the
        span should still be finalized (via .fail(), not .stop())."""
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        cfg = _cfg(otel_enabled=True)
        with pytest.raises(ValueError):
            with retrieval_leg_span(cfg, "bm25", query_text="q", top_k=10):
                raise ValueError("boom")

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 1


class TestOtlpExporterWiring:
    def test_no_endpoint_configures_provider_with_no_span_processors(self) -> None:
        pytest.importorskip("opentelemetry.sdk", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import _build_tracer_provider

        provider = _build_tracer_provider("trelix-test", None)
        # No public API to introspect processor count; constructing without
        # raising and without importing the OTLP exporter module is the
        # observable contract here.
        assert provider.resource.attributes["service.name"] == "trelix-test"

    def test_endpoint_configured_adds_otlp_processor_without_raising(self) -> None:
        """Constructing an OTLPSpanExporter must never raise just because the
        collector endpoint is unreachable — export failures happen later,
        asynchronously, inside BatchSpanProcessor's background thread."""
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            reason="requires pip install trelix[otel]",
        )
        from trelix.retrieval.otel_tracing import _build_tracer_provider

        provider = _build_tracer_provider("trelix-test", "http://localhost:4318/v1/traces")
        assert provider.resource.attributes["service.name"] == "trelix-test"
        provider.shutdown()


# ---------------------------------------------------------------------------
# Thread-context propagation — the genuinely tricky part
# ---------------------------------------------------------------------------


class TestThreadContextPropagation:
    def test_with_current_context_nests_child_span_under_parent_across_thread(
        self, otel_test_exporter
    ) -> None:
        """A span started inside a ThreadPoolExecutor worker must be parented
        under the span active in the submitting thread at pool.submit() time —
        this is exactly the retriever.py _run_subquery_legs() scenario."""
        from opentelemetry import trace

        from trelix.retrieval.otel_tracing import with_current_context

        tracer = trace.get_tracer("trelix.test")

        def leg_work() -> None:
            with tracer.start_as_current_span("child_leg"):
                pass

        with tracer.start_as_current_span("root_query") as root_span:
            root_span_id = root_span.get_span_context().span_id
            traced_leg_work = with_current_context(leg_work)
            with ThreadPoolExecutor() as pool:
                pool.submit(traced_leg_work).result()

        spans = {s.name: s for s in otel_test_exporter.get_finished_spans()}
        assert "root_query" in spans
        assert "child_leg" in spans
        child = spans["child_leg"]
        assert child.parent is not None
        assert child.parent.span_id == root_span_id

    def test_without_with_current_context_child_span_is_unparented(
        self, otel_test_exporter
    ) -> None:
        """Control case: submitting the bare function (no with_current_context
        wrapping) produces a span that does NOT nest under the root — this is
        the exact bug with_current_context exists to prevent."""
        from opentelemetry import trace

        tracer = trace.get_tracer("trelix.test")

        def leg_work() -> None:
            with tracer.start_as_current_span("child_leg_unwrapped"):
                pass

        with tracer.start_as_current_span("root_query_2") as root_span:
            root_span_id = root_span.get_span_context().span_id
            with ThreadPoolExecutor() as pool:
                pool.submit(leg_work).result()

        spans = {s.name: s for s in otel_test_exporter.get_finished_spans()}
        child = spans["child_leg_unwrapped"]
        assert child.parent is None or child.parent.span_id != root_span_id


# ---------------------------------------------------------------------------
# Content gate — TRELIX_OTEL_CAPTURE_CONTENT in front of the upstream opt-in
# ---------------------------------------------------------------------------


class _RecordingHandler:
    """Stands in for TelemetryHandler: keeps every invocation trelix asked for, so a test can
    see what trelix ASSIGNED (not what the library later chose to emit)."""

    def __init__(self, captures: bool) -> None:
        self.invocations: list[SimpleNamespace] = []
        self._captures = captures

    def retrieval(self, *, data_source_id: str) -> SimpleNamespace:
        invocation = SimpleNamespace(
            data_source_id=data_source_id, stop=lambda: None, fail=lambda exc: None
        )
        self.invocations.append(invocation)
        return invocation

    def should_capture_content(self) -> bool:
        return self._captures


class TestContentGateOnTheInvocation:
    """The gate as trelix applies it, against a recording fake handler: no opentelemetry
    needed, so these run in the default CI job without the `otel` extra."""

    @pytest.fixture()
    def recording_handler(self, monkeypatch: pytest.MonkeyPatch) -> _RecordingHandler:
        import trelix.retrieval.otel_tracing as otel_tracing

        handler = _RecordingHandler(captures=True)
        monkeypatch.setattr(otel_tracing, "_handler_for", lambda cfg: handler)
        monkeypatch.setattr(otel_tracing, "_capture_inert_warned", False)
        return handler

    def test_query_text_is_never_assigned_when_the_flag_is_absent(
        self, recording_handler: _RecordingHandler
    ) -> None:
        """A config built before the flag existed (no attribute at all) means OFF.

        MUTATION that must make this fail: assign `query_text` unconditionally.
        """
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        with retrieval_leg_span(_cfg(otel_enabled=True), "vector", query_text=_CANARY, top_k=10):
            pass

        (invocation,) = recording_handler.invocations
        assert not hasattr(invocation, "query_text")
        assert invocation.top_k == 10.0

    def test_query_text_is_never_assigned_when_the_flag_is_false(
        self, recording_handler: _RecordingHandler
    ) -> None:
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        cfg = _cfg(otel_enabled=True, otel_capture_content=False)
        with retrieval_leg_span(cfg, "bm25", query_text=_CANARY, top_k=5):
            pass

        (invocation,) = recording_handler.invocations
        assert not hasattr(invocation, "query_text")

    def test_query_text_is_assigned_when_the_flag_is_true(
        self, recording_handler: _RecordingHandler
    ) -> None:
        """MUTATION that must make this fail: invert the gate (`not capture_content_enabled`)."""
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        cfg = _cfg(otel_enabled=True, otel_capture_content=True)
        with retrieval_leg_span(cfg, "vector", query_text=_CANARY, top_k=10):
            pass

        (invocation,) = recording_handler.invocations
        assert invocation.query_text == "CANARY-QUERY-1a2b"

    def test_inert_capture_warns_once_across_spans_with_a_fake_handler(
        self, recording_handler: _RecordingHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The fake says it will not capture; two spans with the flag on -> ONE warning."""
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        recording_handler._captures = False
        cfg = _cfg(otel_enabled=True, otel_capture_content=True)
        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.otel"):
            for leg in ("vector", "bm25"):
                with retrieval_leg_span(cfg, leg, query_text="q", top_k=1):
                    pass

        assert [r.getMessage() for r in _warnings(caplog)] == [_INERT_WARNING]


class TestContentGateOnRealSpans:
    """R5(e): what a finished span carries, with the upstream opt-in set to SPAN_ONLY."""

    def test_upstream_span_only_alone_puts_no_query_text_on_the_span(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The host opted another library into content capture; trelix's text stays home.

        MUTATION that must make this fail: hand `query_text` to the invocation unconditionally.
        """
        pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        monkeypatch.setenv(_UPSTREAM_MODE, "SPAN_ONLY")
        with retrieval_leg_span(_cfg(otel_enabled=True), "vector", query_text=_CANARY, top_k=10):
            pass

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 1
        assert "gen_ai.retrieval.query.text" not in spans[0].attributes
        assert all("CANARY-QUERY-1a2b" not in _span_text(s) for s in spans)

    def test_both_switches_on_put_the_query_text_on_the_span(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        monkeypatch.setenv(_UPSTREAM_MODE, "SPAN_ONLY")
        cfg = _cfg(otel_enabled=True, otel_capture_content=True)
        with retrieval_leg_span(cfg, "vector", query_text=_CANARY, top_k=10):
            pass

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.retrieval.query.text"] == "CANARY-QUERY-1a2b"

    def test_trelix_flag_alone_puts_no_query_text_on_the_span(self, otel_test_exporter) -> None:
        """Upstream unset (NO_CONTENT): the library drops the text trelix handed over."""
        pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        cfg = _cfg(otel_enabled=True, otel_capture_content=True)
        with retrieval_leg_span(cfg, "vector", query_text=_CANARY, top_k=10):
            pass

        spans = otel_test_exporter.get_finished_spans()
        assert len(spans) == 1
        assert all("CANARY-QUERY-1a2b" not in _span_text(s) for s in spans)


class TestInertCaptureWarning:
    """R5(g): the one-time WARNING when the trelix flag is on and upstream is NO_CONTENT."""

    def test_warns_exactly_once_for_two_spans(
        self, otel_test_exporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATIONS that must make this fail: remove the call (0 records); warn on every
        span instead of once (2 records)."""
        pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        cfg = _cfg(otel_enabled=True, otel_capture_content=True)
        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.otel"):
            for leg in ("vector", "bm25"):
                with retrieval_leg_span(cfg, leg, query_text="q", top_k=1):
                    pass

        records = _warnings(caplog)
        assert len(records) == 1
        assert "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT" in records[0].getMessage()
        assert records[0].getMessage() == _INERT_WARNING

    def test_no_warning_when_the_trelix_flag_is_off(
        self, otel_test_exporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nothing was asked for, so nothing is inert — the default must stay silent."""
        pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.otel"):
            with retrieval_leg_span(_cfg(otel_enabled=True), "vector", query_text="q", top_k=1):
                pass

        assert _warnings(caplog) == []

    def test_no_warning_when_upstream_captures(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        monkeypatch.setenv(_UPSTREAM_MODE, "SPAN_ONLY")
        cfg = _cfg(otel_enabled=True, otel_capture_content=True)
        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.otel"):
            with retrieval_leg_span(cfg, "vector", query_text="q", top_k=1):
                pass

        assert _warnings(caplog) == []

    def test_the_memoised_handler_decides_not_the_live_environment(
        self, otel_test_exporter, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The handler reads the upstream variable once, when it is built. A value exported
        afterwards changes nothing about what the spans carry, so the warning must follow the
        handler, not the environment.

        MUTATION that must make this fail: decide from
        `opentelemetry.util.genai.utils.get_content_capturing_mode()` (a re-read of the
        environment) instead of `handler.should_capture_content()` -> 0 records.
        """
        pytest.importorskip("opentelemetry.util.genai", reason="requires pip install trelix[otel]")
        from trelix.retrieval.otel_tracing import retrieval_leg_span

        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.otel"):
            # Build (and memoise) the handler while upstream is unset: it decides NO_CONTENT.
            with retrieval_leg_span(_cfg(otel_enabled=True), "vector", query_text="q", top_k=1):
                pass
            assert _warnings(caplog) == []
            monkeypatch.setenv(_UPSTREAM_MODE, "SPAN_ONLY")
            cfg = _cfg(otel_enabled=True, otel_capture_content=True)
            with retrieval_leg_span(cfg, "bm25", query_text="q", top_k=1):
                pass

        assert len(_warnings(caplog)) == 1


class TestCaptureContentConfig:
    """R5(f): the RetrievalConfig field and what a malformed value does."""

    def test_default_is_false(self) -> None:
        """MUTATION that must make this fail: `default=True` in config.py."""
        from trelix.core.config import RetrievalConfig

        assert RetrievalConfig.model_fields["otel_capture_content"].default is False
        assert RetrievalConfig(_env_file=None).otel_capture_content is False

    def test_env_true_turns_it_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "true")
        assert RetrievalConfig(_env_file=None).otel_capture_content is True

    @pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
    def test_env_value_reaches_the_environment_resolution_path(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
    ) -> None:
        """`capture_content_enabled()` with no cfg (the embedder's path today, the chat spans'
        in PR 3) resolves TRELIX_OTEL_CAPTURE_CONTENT through RetrievalConfig.

        MUTATION that must make this fail: replace `bool(resolved.otel_capture_content)` in
        `_otel_settings` with a constant (`True` fails "false", `False` fails "true") or with
        `bool(resolved.otel_enabled)` (fails "true").
        """
        import trelix.retrieval.otel_tracing as otel_tracing

        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", value)
        monkeypatch.setattr(otel_tracing, "_env_otel_settings", None)
        assert otel_tracing._otel_settings(None) == (False, "trelix", None, expected)
        assert otel_tracing.capture_content_enabled() is expected

    @pytest.mark.parametrize("value", ["maybe", ""])
    def test_malformed_value_raises_on_construction_and_resolves_to_disabled(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """pydantic's bool parser rejects it (every IndexConfig construction raises); the
        environment-resolution path swallows that into "telemetry off, content off".

        MUTATION that must make this fail: change the swallow-path fallback in
        `_otel_settings` to `_OtelSettings(False, "trelix", None, True)`. (A `default=True`
        on the field does NOT reach this path — `test_default_is_false` pins that.)
        """
        from pydantic import ValidationError

        import trelix.retrieval.otel_tracing as otel_tracing
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", value)
        with pytest.raises(ValidationError):
            RetrievalConfig(_env_file=None)

        monkeypatch.setattr(otel_tracing, "_env_otel_settings", None)
        assert otel_tracing._otel_settings(None) == (False, "trelix", None, False)


class TestExportingSpans:
    """The real-span fixture plumbing both OTel test modules share (otel_llm_fakes.py): the
    processor must be DETACHED on exit, not only shut down, or opentelemetry-sdk >= 1.45 logs a
    WARNING for every later span in the process and the next module's warn-count pins fail."""

    def test_the_processor_is_detached_on_exit(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION that must make this fail: `processor.shutdown()` alone in the `finally`
        (the spy then records "after" too; on sdk >= 1.45 the WARNING is logged as well)."""
        pytest.importorskip("opentelemetry.sdk", reason="requires pip install trelix[otel]")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from tests.unit.otel_llm_fakes import exporting_spans

        seen: list[str] = []
        real_on_end = SimpleSpanProcessor.on_end

        def spy(processor: Any, span: Any) -> None:
            seen.append(span.name)
            real_on_end(processor, span)

        monkeypatch.setattr(SimpleSpanProcessor, "on_end", spy)
        provider = TracerProvider()  # this test's own; never installed as the global one
        with exporting_spans(provider) as exporter:
            provider.get_tracer("probe").start_span("inside").end()
        with caplog.at_level(logging.WARNING):
            provider.get_tracer("probe").start_span("after").end()

        assert [s.name for s in exporter.get_finished_spans()] == ["inside"]
        assert seen == ["inside"]
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
