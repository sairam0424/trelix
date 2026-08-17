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
- _get_meter()'s *install* branch — the one a real `pip install trelix[otel]`
  deployment takes, where no host has configured a MeterProvider and trelix
  installs its own. The fixture below configures a provider first, so every
  in-process test here takes the other branch; the install branch is covered
  in a fresh subprocess (see TestProviderInstallation)

Embedder stubs are built with object.__new__ + the attributes the embed()
path actually reads, so no provider SDK, credential or network is involved;
the real embed()/_embed_batch()/_embed_one() bodies still run.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
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


def _counter_deltas(
    reader: Any, provider: str, names: tuple[str, ...]
) -> tuple[dict[str, float], Any]:
    """Snapshot the current totals for *provider*, so a test can assert an INCREASE.

    Necessary because OTel's global MeterProvider can be set only once per process, so the
    reader below is shared and every counter is cumulative for the whole session.
    `_counter_total`'s docstring says each test keeps its series disjoint by using a distinct
    provider label — but two tests here legitimately need the real `"openai"` label, so they
    share a series and their absolute totals depend on collection order.

    Measured: `tests/unit` in reverse collection order failed
    `test_openai_embed_counts_one_request_per_api_call` with `assert 3 == 2`, purely because
    another `"openai"` test had already incremented the counter. A delta is correct under any
    ordering and does not need the labels to stay disjoint.
    """
    return ({n: (_counter_total(reader, n, provider) or 0.0) for n in names}, reader)


def _counter_total(reader: Any, metric_name: str, provider: str) -> float | None:
    """Summed value of *metric_name* for *provider*, or None if never recorded.

    Counters are cumulative for the life of the process, so each test below
    uses a different provider label to keep its series disjoint.
    """
    total: float | None = None
    # `get_metrics_data()` returns None, not an empty container, when the reader has
    # collected nothing yet — so a caller taking a BASELINE before any recording (see
    # _counter_deltas) hits AttributeError on a fresh reader rather than reading zero.
    data = reader.get_metrics_data()
    if data is None:
        return None
    for resource_metric in data.resource_metrics:
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

        names = (
            "trelix.embedder.requests",
            "trelix.embedder.texts",
            "trelix.embedder.characters",
            "trelix.embedder.tokens",
        )
        before, _ = _counter_deltas(metric_reader, "openai", names)

        vectors = embedder.embed(["alpha", "beta"])  # 5 + 4 characters
        assert len(vectors) == 2

        after, _ = _counter_deltas(metric_reader, "openai", names)
        # Deltas, not absolutes: this test shares the "openai" series with
        # test_counter_carries_provider_and_model_attributes, so an absolute total depends
        # on which ran first.
        assert after["trelix.embedder.requests"] - before["trelix.embedder.requests"] == 2
        assert after["trelix.embedder.texts"] - before["trelix.embedder.texts"] == 2
        assert after["trelix.embedder.characters"] - before["trelix.embedder.characters"] == 9
        assert after["trelix.embedder.tokens"] - before["trelix.embedder.tokens"] == 14

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


# ---------------------------------------------------------------------------
# _get_meter()'s provider-install branch — what production actually runs
# ---------------------------------------------------------------------------
#
# The tests above all install a MeterProvider (the module fixture) *before*
# trelix looks, so _get_meter() sees a configured provider and skips its
# set_meter_provider() call. A user who runs `pip install trelix[otel]` and sets
# TRELIX_OTEL_ENABLED has no MeterProvider of their own, so the branch they take
# is the one that installs trelix's — and OTel's global MeterProvider is
# one-shot per process (opentelemetry-api 1.44.0 `_METER_PROVIDER_SET_ONCE`;
# a second set_meter_provider() keeps the first provider and only logs
# "Overriding of current MeterProvider is not allowed", verified in source).
#
# So the install branch cannot be reached in this interpreter at all: this
# module's fixture claims the global, and tests/unit/test_otel_tracing.py
# manipulates sys.modules and the global TracerProvider besides. A fresh
# subprocess is the only honest way to exercise it, and it makes the two
# probe-backed tests below immune to test ordering: they read and write no
# parent-process state, so neither file's globals can reach them.


def _run_probe(script: str, cwd: Path, **env_overrides: str) -> str:
    """Run *script* in a fresh interpreter with a clean OTel env; return stdout.

    All OTEL_* vars are dropped from the inherited environment (notably
    OTEL_PYTHON_METER_PROVIDER, which would make get_meter_provider() install an
    entry-point provider instead of returning the unset placeholder) before
    *env_overrides* is applied. *cwd* is a tmp dir so RetrievalConfig's
    ``env_file=".env"`` cannot pick up the repo's own .env.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env["TRELIX_OTEL_ENABLED"] = "true"
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout


# Prepended to every probe. The probes exit via os._exit() (see below), which
# skips atexit — and atexit is how coverage's subprocess bootstrap
# (a1_coverage.pth, which only calls coverage.process_startup() when
# COVERAGE_PROCESS_START or COVERAGE_PROCESS_CONFIG is set) writes its data
# file, so without this hand-flush the probe's lines are lost even when that
# bootstrap did run. Verified: with COVERAGE_PROCESS_START + parallel=True,
# `coverage combine` picks the probe up and _get_meter()'s install line stops
# being reported missing. _run_probe() strips only OTEL_*, so those two vars
# reach the child untouched.
#
# Note that pytest-cov 7.1.0, as this repo invokes it, does NOT arm that
# bootstrap (it sets no COV_CORE_*/COVERAGE_PROCESS_* at all — checked), so a
# plain `pytest --cov` run still lists the install line as uncovered. It is
# exercised regardless: inverting _get_meter()'s provider guard in either
# direction turns these tests red.
_PROBE_PRELUDE = """
import sys


def _save_coverage():
    coverage = sys.modules.get("coverage")
    active = coverage.Coverage.current() if coverage is not None else None
    if active is not None:
        active.stop()
        active.save()

"""


# record_embedding_call() is the entry point, not _get_meter(): the point is
# that the whole documented setup (install the extra, set the two env vars, embed
# something) ends with a provider wired to the operator's collector.
_PROBE_INSTALL = (
    _PROBE_PRELUDE
    + """
import os, sys

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from trelix.retrieval.otel_tracing import metrics_enabled, record_embedding_call

# The production precondition: nothing has configured OTel metrics, so the API
# returns its unset placeholder and trelix's guard opens.
assert type(metrics.get_meter_provider()).__name__ == "_ProxyMeterProvider"
assert metrics_enabled() is True

record_embedding_call(
    provider="openai", model="text-embedding-3-large", texts=1, characters=3, tokens=2
)

provider = metrics.get_meter_provider()
assert isinstance(provider, MeterProvider), type(provider).__name__
print("SERVICE_NAME", provider._sdk_config.resource.attributes["service.name"])

readers = provider._metric_readers
assert len(readers) == 1, readers
(reader,) = readers
assert isinstance(reader, PeriodicExportingMetricReader), type(reader).__name__
assert isinstance(reader._exporter, OTLPMetricExporter), type(reader._exporter).__name__
print("ENDPOINT", reader._exporter._endpoint)

sys.stdout.flush()
_save_coverage()
# os._exit skips atexit: MeterProvider registers a shutdown that flushes the
# reader, and that export retries against the collector nothing is listening on
# at :4318 for ~6s before giving up (measured).
os._exit(0)
"""
)


_PROBE_SINGLE_INSTALL = (
    _PROBE_PRELUDE
    + """
import logging, os, sys

from opentelemetry import metrics

import trelix.retrieval.otel_tracing as otel

assert type(metrics.get_meter_provider()).__name__ == "_ProxyMeterProvider"

# A second set_meter_provider() is swallowed by opentelemetry-api: the first
# provider is kept and the only trace of the attempt is this WARNING. Capture it
# — trelix must be skipping the second install outright, not relying on the
# one-shot to absorb it (which would silently pin the export endpoint to
# whichever config happened to be resolved first).
overrides = []


class _Capture(logging.Handler):
    def emit(self, record):
        overrides.append(record.getMessage())


logging.getLogger("opentelemetry.metrics._internal").addHandler(_Capture())

otel._get_meter("probe-one", "http://localhost:4318/v1/traces")
installed = metrics.get_meter_provider()
assert type(installed).__name__ == "MeterProvider", type(installed).__name__

# Re-enter past the memo the way a second, differently-named config would.
otel._meter = None
otel._meter_service_name = None
otel._get_meter("probe-two", "http://localhost:9999/v1/traces")

assert metrics.get_meter_provider() is installed, "global MeterProvider was replaced"
assert overrides == [], overrides
print("SINGLE_INSTALL")

# Memoized fast path: a sentinel proves the early return fired rather than the
# meter being re-derived from the provider (MeterProvider caches meters by name,
# so plain identity of the returned meter would pass either way).
sentinel = object()
otel._meter = sentinel
assert otel._get_meter("probe-two", None) is sentinel, "memoized fast path did not fire"
print("MEMOIZED")

sys.stdout.flush()
_save_coverage()
os._exit(0)
"""
)


class TestProviderInstallation:
    def test_install_branch_wires_a_provider_to_the_configured_otlp_endpoint(
        self, tmp_path: Path
    ) -> None:
        """`pip install trelix[otel]` + the two documented env vars, in a fresh
        process: trelix installs its own MeterProvider carrying the configured
        service.name and exporting to the metrics signal path."""
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.metric_exporter",
            reason="requires pip install trelix[otel]",
        )
        out = _run_probe(
            _PROBE_INSTALL,
            tmp_path,
            OTEL_SERVICE_NAME="trelix-probe",
            OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces",
        )
        assert "SERVICE_NAME trelix-probe" in out
        # /v1/traces swapped for /v1/metrics — see TestMetricsEndpoint.
        assert "ENDPOINT http://localhost:4318/v1/metrics" in out

    def test_reentering_the_install_branch_does_not_install_a_second_provider(
        self, tmp_path: Path
    ) -> None:
        """Second entry must be skipped by trelix's own guard, leaving no
        "Overriding of current MeterProvider" warning behind."""
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.metric_exporter",
            reason="requires pip install trelix[otel]",
        )
        out = _run_probe(_PROBE_SINGLE_INSTALL, tmp_path)
        assert "SINGLE_INSTALL" in out
        assert "MEMOIZED" in out

    def test_get_meter_raises_rather_than_returning_none_without_opentelemetry(self) -> None:
        """_get_meter() promises it *raises* on import failure — that is what
        makes _embedding_counters_for() emit its one WARNING instead of quietly
        yielding no counters. Pinned because the tests above are the only ones
        that call _get_meter() directly, so nothing else guards the contract."""
        import trelix.retrieval.otel_tracing as otel_tracing

        _reset_metrics_state()
        try:
            with patch.dict(sys.modules, _absent_opentelemetry()), pytest.raises(ImportError):
                otel_tracing._get_meter("trelix-absent", "http://localhost:4318/v1/traces")
        finally:
            _reset_metrics_state()

    def test_a_host_configured_provider_is_never_clobbered(
        self, metric_reader: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The other branch, named: with a provider already configured (as the
        module fixture does, and as a host application embedding trelix would),
        _get_meter() must leave it in place. If it called set_meter_provider()
        anyway the global would be unchanged but the attempt would be logged —
        so an empty log is the assertion that the guard, not OTel's one-shot,
        is what protects the host's provider."""
        from opentelemetry import metrics

        import trelix.retrieval.otel_tracing as otel_tracing

        before = metrics.get_meter_provider()
        with caplog.at_level(logging.WARNING):
            meter = otel_tracing._get_meter("trelix-would-be-clobbered", None)

        assert meter is not None
        assert metrics.get_meter_provider() is before
        assert [r.getMessage() for r in caplog.records if "Overriding" in r.getMessage()] == []
