"""
OTLP metrics wiring and the embedding counter instruments — the stateless part of
otel_tracing's metrics half.

otel_tracing keeps the process-wide memo (`_meter`, `_embedding_counters`, the one-time
warning flag), the settings resolver and the recording entry points: `embedder/base.py`
calls `otel_tracing.record_embedding_call()`, and tests/unit/test_otel_metrics*.py rebind
and probe those names on `otel_tracing` (`otel_tracing._meter = None`,
`otel_tracing._get_meter(...)`). Everything here is a pure function of its arguments, so
it can live apart from that state; otel_tracing re-exports `_metrics_endpoint` and
`_build_meter_provider` so every existing import keeps working.

Imports nothing from opentelemetry at module level, so the off path
(`TRELIX_OTEL_ENABLED=false`) still never loads the SDK.
"""

from __future__ import annotations

from typing import Any

# Counter names are trelix's own (`trelix.*`): the GenAI metric conventions
# cover chat token usage, not embedding volume, so there is nothing to borrow.
# Attributes deliberately mix namespaces: `gen_ai.request.model` is the
# conventional free-form model attribute (joins these counters to the
# gen_ai.* spans), while the provider is trelix's own selector value
# ("bedrock-titan", "local-code", ...) and NOT a `gen_ai.provider.name` enum
# member, so it keeps a trelix.* name rather than pretending to conform.
_ATTR_PROVIDER = "trelix.embedder.provider"
_ATTR_MODEL = "gen_ai.request.model"


def _metrics_endpoint(traces_endpoint: str | None) -> str | None:
    """Map the configured OTLP endpoint onto the metrics signal path.

    docs/OBSERVABILITY.md documents OTEL_EXPORTER_OTLP_ENDPOINT with a
    ``/v1/traces`` suffix, and a value passed as ``endpoint=`` is used verbatim
    by the OTLP exporter (unlike the env-var form, no signal path is appended).
    Reusing it for metrics would POST metric payloads to the traces route,
    which collectors reject — so the suffix is swapped, not shared.
    """
    if not traces_endpoint:
        return None
    base = traces_endpoint.rstrip("/")
    if base.endswith("/v1/traces"):
        base = base[: -len("/v1/traces")]
    return f"{base}/v1/metrics"


def _build_meter_provider(service_name: str, otlp_endpoint: str | None) -> Any:
    """Build a MeterProvider for *service_name*, exporting to *otlp_endpoint* if set.

    Mirrors otel_tracing._build_tracer_provider(), including being separated out
    so the exporter wiring is testable without installing a real (one-shot)
    global provider.
    """
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    readers = []
    endpoint = _metrics_endpoint(otlp_endpoint)
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint)))
    return MeterProvider(
        resource=Resource.create({SERVICE_NAME: service_name}),
        metric_readers=readers,
    )


def create_embedding_counters(meter: Any) -> dict[str, Any]:
    """The four embedding counters on *meter*, keyed requests/texts/characters/tokens."""
    return {
        "requests": meter.create_counter(
            "trelix.embedder.requests",
            unit="{request}",
            description="Embedding provider calls (one per API request/model invocation)",
        ),
        "texts": meter.create_counter(
            "trelix.embedder.texts",
            unit="{text}",
            description="Texts (chunks/queries) submitted for embedding",
        ),
        "characters": meter.create_counter(
            "trelix.embedder.characters",
            unit="{character}",
            description="Characters submitted for embedding — the volume proxy for "
            "providers that report no token usage",
        ),
        "tokens": meter.create_counter(
            "trelix.embedder.tokens",
            unit="{token}",
            description="Provider-reported tokens embedded — the billed quantity; "
            "absent for providers that report none",
        ),
    }
