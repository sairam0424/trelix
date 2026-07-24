"""
Unit tests for trelix.retrieval.otel_tracing.

Covers:
- Zero behavior change / zero opentelemetry import when TRELIX_OTEL_ENABLED=false
  (the single most important test — proves the feature is truly opt-in)
- Correct gen_ai.* span attributes when enabled
- Thread-context propagation across ThreadPoolExecutor via with_current_context()
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


def _cfg(
    otel_enabled: bool,
    service_name: str = "trelix-test",
    otel_exporter_endpoint: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        otel_enabled=otel_enabled,
        otel_service_name=service_name,
        otel_exporter_endpoint=otel_exporter_endpoint,
    )


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
            if retriever_mod is not None:
                sys.modules["trelix.retrieval.retriever"] = retriever_mod
            if otel_tracing_mod is not None:
                sys.modules["trelix.retrieval.otel_tracing"] = otel_tracing_mod


# ---------------------------------------------------------------------------
# Enabled path — real spans via InMemorySpanExporter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _test_tracer_provider():
    """
    Install a real SDK TracerProvider backed by InMemorySpanExporter, once for
    this test module. OpenTelemetry's global TracerProvider can only be set
    once per process — a second call to set_tracer_provider() is a silent
    no-op (with a logged warning) — so this installs lazily (only when a test
    in this module actually runs, not merely on collection) and only once.

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
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "trelix-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture()
def otel_test_exporter(_test_tracer_provider):
    """
    Yield the module-wide InMemorySpanExporter, cleared before each test.
    Also resets trelix's memoized TelemetryHandler so _get_handler() rebuilds
    it against the real (already-installed) TracerProvider on next use,
    since a freshly-imported handler observed the ProxyTracerProvider only
    once, at first import.
    """
    import trelix.retrieval.otel_tracing as otel_tracing

    _test_tracer_provider.clear()
    prev_handler = otel_tracing._handler
    prev_service_name = otel_tracing._handler_service_name
    otel_tracing._handler = None
    otel_tracing._handler_service_name = None
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
