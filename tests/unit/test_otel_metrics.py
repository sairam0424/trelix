"""
Unit tests for the OpenTelemetry *metrics* seam — counters, not spans.

Spans answer "did a request happen"; they cannot answer "what did it cost".
Embedding is the only operation in trelix that is billed per call, so the
tests that matter here are:

- the counters actually record under a real SDK MeterProvider
  (InMemoryMetricReader — the OTel API itself is never mocked)
- the seam is a true no-op when opentelemetry is not installed, which is the
  *default* install: `otel` is an optional extra
- provider-reported token counts are recorded; missing ones are left missing
  rather than estimated (a fabricated cost series is worse than no series)
- when the flag is on but the metrics API is unavailable, that is logged at
  WARNING — an absent cost counter otherwise reads as "we spent nothing"

Embedder stubs are built with object.__new__ + the attributes the embed()
path actually reads, so no provider SDK, credential or network is involved;
the real embed()/_embed_batch()/_embed_one() bodies still run.
"""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

_ATTR_PROVIDER = "trelix.embedder.provider"
_ATTR_MODEL = "gen_ai.request.model"


def _reset_metrics_state() -> None:
    """Clear otel_tracing's memoized meter / instruments / resolved settings."""
    import trelix.retrieval.otel_tracing as otel_tracing

    otel_tracing._env_otel_settings = None
    otel_tracing._meter = None
    otel_tracing._meter_service_name = None
    otel_tracing._embedding_counters = None
    otel_tracing._metrics_unavailable_logged = False


def _absent_opentelemetry() -> dict[str, None]:
    """sys.modules patch that makes every `import opentelemetry...` fail."""
    return dict.fromkeys(
        [k for k in sys.modules if k.startswith("opentelemetry")]
        + ["opentelemetry", "opentelemetry.metrics", "opentelemetry.sdk.metrics"]
    )


# ---------------------------------------------------------------------------
# Embedder stubs — real embed() bodies, canned provider responses
# ---------------------------------------------------------------------------


def _stub_openai_embedder(*, tokens: int | None, batch_size: int = 1) -> Any:
    """OpenAIEmbedder whose SDK client returns a canned response.

    batch_size=1 makes the per-HTTP-request counting observable: two texts
    become two calls through the embed() loop.
    """
    from trelix.embedder.base import OpenAIEmbedder

    embedder = object.__new__(OpenAIEmbedder)
    embedder._model = "text-embedding-3-large"
    embedder._dimensions = 3
    embedder._batch_size = batch_size

    usage = SimpleNamespace(total_tokens=tokens) if tokens is not None else None
    response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])], usage=usage)

    class _StubEmbeddings:
        def create(self, **_kwargs: Any) -> Any:
            return response

    embedder._client = SimpleNamespace(embeddings=_StubEmbeddings())
    return embedder


def _stub_bedrock_client(payload: dict[str, Any]) -> Any:
    """boto3 bedrock-runtime stand-in returning *payload* as an invoke_model body."""
    body = json.dumps(payload).encode()

    class _Body:
        def read(self) -> bytes:
            return body

    return SimpleNamespace(invoke_model=lambda **_kwargs: {"body": _Body()})


# ---------------------------------------------------------------------------
# Disabled / opentelemetry-absent path — must be a true no-op
# ---------------------------------------------------------------------------


class TestNoOpWithoutOtel:
    def test_metrics_enabled_is_false_and_recording_is_a_noop_when_flag_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_OTEL_ENABLED", "false")
        _reset_metrics_state()
        from trelix.retrieval.otel_tracing import metrics_enabled, record_embedding_call

        assert metrics_enabled() is False
        # Must not raise even though nothing is metering.
        record_embedding_call(
            provider="openai", model="text-embedding-3-large", texts=2, characters=9, tokens=7
        )
        _reset_metrics_state()

    def test_metrics_enabled_does_not_import_opentelemetry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """metrics_enabled() is consulted on every embed call — it must never
        pull opentelemetry in, exactly like is_enabled() for spans."""
        monkeypatch.setenv("TRELIX_OTEL_ENABLED", "false")
        _reset_metrics_state()
        from trelix.retrieval.otel_tracing import metrics_enabled

        purged = {k: v for k, v in sys.modules.items() if k.startswith("opentelemetry")}
        for k in purged:
            del sys.modules[k]
        try:
            assert metrics_enabled() is False
            assert not any(k.startswith("opentelemetry") for k in sys.modules)
        finally:
            sys.modules.update(purged)
            _reset_metrics_state()

    def test_recording_is_a_noop_when_opentelemetry_is_not_installed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Flag ON with `opentelemetry` unimportable (the default install plus
        TRELIX_OTEL_ENABLED=true) — recording must not raise, and must say so
        once at WARNING instead of silently dropping cost data."""
        monkeypatch.setenv("TRELIX_OTEL_ENABLED", "true")
        _reset_metrics_state()
        from trelix.retrieval.otel_tracing import metrics_enabled, record_embedding_call

        assert metrics_enabled() is True
        with patch.dict(sys.modules, _absent_opentelemetry()), caplog.at_level(logging.WARNING):
            record_embedding_call(
                provider="openai", model="text-embedding-3-large", texts=1, characters=3
            )
            record_embedding_call(
                provider="openai", model="text-embedding-3-large", texts=1, characters=3
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, "should warn exactly once, not once per call"
        assert "metric" in warnings[0].getMessage().lower()
        _reset_metrics_state()

    def test_embedder_still_embeds_when_opentelemetry_is_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real embed() path with the flag on and no opentelemetry present
        must return embeddings unchanged — instrumentation cannot break work."""
        monkeypatch.setenv("TRELIX_OTEL_ENABLED", "true")
        _reset_metrics_state()
        embedder = _stub_openai_embedder(tokens=5)

        with patch.dict(sys.modules, _absent_opentelemetry()):
            vectors = embedder.embed(["alpha", "beta"])

        assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        _reset_metrics_state()


# ---------------------------------------------------------------------------
# Enabled path — real SDK MeterProvider + InMemoryMetricReader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _test_meter_provider():
    """
    Install a real SDK MeterProvider backed by InMemoryMetricReader, once for
    this module. Like the TracerProvider, OTel's global MeterProvider can only
    be set once per process (a second set_meter_provider() is a logged no-op),
    so this installs lazily and only once.

    Skips when opentelemetry-sdk isn't installed — CI's default
    `pip install -e ".[local,dev]"` deliberately omits the optional `otel`
    extra, mirroring tests/unit/test_otel_tracing.py.
    """
    pytest.importorskip("opentelemetry.sdk.metrics", reason="requires pip install trelix[otel]")

    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    reader = InMemoryMetricReader()
    metrics.set_meter_provider(
        MeterProvider(
            resource=Resource.create({SERVICE_NAME: "trelix-test"}),
            metric_readers=[reader],
        )
    )
    return reader


@pytest.fixture()
def metric_reader(_test_meter_provider, monkeypatch: pytest.MonkeyPatch):
    """Flag on + trelix's memoized meter reset so it binds to the provider
    installed above (a fresh import saw only the _ProxyMeterProvider)."""
    monkeypatch.setenv("TRELIX_OTEL_ENABLED", "true")
    _reset_metrics_state()
    try:
        yield _test_meter_provider
    finally:
        _reset_metrics_state()


def _counter_total(reader: Any, metric_name: str, provider: str) -> float | None:
    """Summed value of *metric_name* for *provider*, or None if never recorded.

    Counters are cumulative for the life of the process, so each test below
    uses a different provider label to keep its series disjoint.
    """
    total: float | None = None
    for resource_metric in reader.get_metrics_data().resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name != metric_name:
                    continue
                for point in metric.data.data_points:
                    if point.attributes.get(_ATTR_PROVIDER) != provider:
                        continue
                    total = (total or 0) + point.value
    return total


class TestEnabledRecordsCounters:
    def test_openai_embed_counts_one_request_per_api_call(self, metric_reader) -> None:
        """batch_size=1 over two texts = 2 HTTP requests, 2 texts, 9 characters,
        and the provider-reported token total summed across both calls."""
        embedder = _stub_openai_embedder(tokens=7)

        vectors = embedder.embed(["alpha", "beta"])  # 5 + 4 characters
        assert len(vectors) == 2

        assert _counter_total(metric_reader, "trelix.embedder.requests", "openai") == 2
        assert _counter_total(metric_reader, "trelix.embedder.texts", "openai") == 2
        assert _counter_total(metric_reader, "trelix.embedder.characters", "openai") == 9
        assert _counter_total(metric_reader, "trelix.embedder.tokens", "openai") == 14

    def test_counter_carries_provider_and_model_attributes(self, metric_reader) -> None:
        embedder = _stub_openai_embedder(tokens=3, batch_size=64)
        embedder.embed(["x"])

        models = {
            point.attributes.get(_ATTR_MODEL)
            for rm in metric_reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for metric in sm.metrics
            if metric.name == "trelix.embedder.requests"
            for point in metric.data.data_points
            if point.attributes.get(_ATTR_PROVIDER) == "openai"
        }
        assert models == {"text-embedding-3-large"}

    def test_no_token_series_when_provider_reports_no_usage(self, metric_reader) -> None:
        """Cohere on Bedrock returns no token count. The requests counter must
        still move; the tokens counter must stay absent rather than carry an
        invented estimate."""
        from trelix.embedder.base import BedrockCohereEmbedder

        embedder = object.__new__(BedrockCohereEmbedder)
        embedder._model = "cohere.embed-english-v3"
        embedder._client = _stub_bedrock_client({"embeddings": [[0.1, 0.2]]})

        assert embedder.embed(["hello"]) == [[0.1, 0.2]]
        assert _counter_total(metric_reader, "trelix.embedder.requests", "bedrock-cohere") == 1
        assert _counter_total(metric_reader, "trelix.embedder.characters", "bedrock-cohere") == 5
        assert _counter_total(metric_reader, "trelix.embedder.tokens", "bedrock-cohere") is None

    def test_bedrock_titan_records_provider_reported_tokens(self, metric_reader) -> None:
        """Titan reports inputTextTokenCount, one text per invoke_model call."""
        from trelix.embedder.base import BedrockTitanEmbedder

        embedder = object.__new__(BedrockTitanEmbedder)
        embedder._model = "amazon.titan-embed-text-v2:0"
        embedder._dims = 2
        embedder._normalize = True
        embedder._client = _stub_bedrock_client({"embedding": [0.1, 0.2], "inputTextTokenCount": 4})

        assert embedder.embed(["ab", "cd"]) == [[0.1, 0.2], [0.1, 0.2]]
        assert _counter_total(metric_reader, "trelix.embedder.requests", "bedrock-titan") == 2
        assert _counter_total(metric_reader, "trelix.embedder.tokens", "bedrock-titan") == 8


# ---------------------------------------------------------------------------
# OTLP metrics endpoint derivation
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_traces_path_is_swapped_for_metrics_path(self) -> None:
        """OBSERVABILITY.md documents OTEL_EXPORTER_OTLP_ENDPOINT with the
        /v1/traces suffix, and an explicit endpoint= is used verbatim by the
        OTLP exporter. Reusing it would POST metrics to /v1/traces, which
        collectors reject."""
        from trelix.retrieval.otel_tracing import _metrics_endpoint

        assert (
            _metrics_endpoint("http://localhost:4318/v1/traces")
            == "http://localhost:4318/v1/metrics"
        )

    def test_bare_base_url_gets_the_metrics_path_appended(self) -> None:
        from trelix.retrieval.otel_tracing import _metrics_endpoint

        assert _metrics_endpoint("http://localhost:4318") == "http://localhost:4318/v1/metrics"

    def test_no_endpoint_stays_none(self) -> None:
        from trelix.retrieval.otel_tracing import _metrics_endpoint

        assert _metrics_endpoint(None) is None

    def test_build_meter_provider_with_unreachable_endpoint_does_not_raise(self) -> None:
        """Constructing an OTLPMetricExporter must never raise just because the
        collector is unreachable — export failures happen later, in the
        PeriodicExportingMetricReader's background thread. (MeterProvider
        exposes no public `.resource`, unlike TracerProvider, so the observable
        contract is that construction and get_meter() both work.)"""
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.metric_exporter",
            reason="requires pip install trelix[otel]",
        )
        from trelix.retrieval.otel_tracing import _build_meter_provider

        provider = _build_meter_provider("trelix-test", "http://localhost:4318/v1/traces")
        assert provider.get_meter("trelix.test") is not None
        provider.shutdown()
