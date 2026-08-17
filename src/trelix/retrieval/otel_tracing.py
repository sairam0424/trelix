"""
OpenTelemetry tracing + cost metrics — off by default.

Emits one span per retrieval leg (vector, BM25, grep, sparse, sub-chunk,
file-summary) plus root/planner/fusion/expansion/rerank/assembly spans, using
the official `gen_ai.*` semantic conventions (status: Development, not yet
Stable — attribute names may still shift upstream; see
docs/OBSERVABILITY.md).

Also emits counters (see "Metrics" below): spans record that a request
happened, never what it cost, so token/request volume needs instruments of its
own. `record_embedding_call()` is the one that matters most — embedding is the
only operation in trelix billed per call.

Requires `pip install trelix[otel]`. When `TRELIX_OTEL_ENABLED=false`
(default), every function here is a cheap no-op and the `opentelemetry.*`
packages are never imported — zero cost on the hot path.

Cross-thread propagation: `_run_subquery_legs()` runs inside a
ThreadPoolExecutor (see retriever.py), and OTel's context is contextvars-based
— it does not automatically cross a `pool.submit()` boundary. Callers must
capture the current context in the submitting thread and pass it through
`with_current_context()` so leg spans nest correctly under the root span
instead of starting as new, unparented traces.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from opentelemetry.util.genai.handler import TelemetryHandler

logger = logging.getLogger("trelix.retrieval.otel")

_T = TypeVar("_T")

_handler: TelemetryHandler | None = None
_handler_service_name: str | None = None

_meter: Any = None
_meter_service_name: str | None = None
_embedding_counters: dict[str, Any] | None = None
_env_otel_settings: tuple[bool, str, str | None] | None = None
_metrics_unavailable_logged = False


def _build_tracer_provider(service_name: str, otlp_endpoint: str | None) -> Any:
    """Build a TracerProvider for *service_name*, exporting to *otlp_endpoint* if set.

    Separated from _get_handler() so the OTLP-exporter wiring is directly
    testable without needing to install it as the process's real (one-shot)
    global TracerProvider.
    """
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    if otlp_endpoint:
        # opentelemetry-exporter-otlp-proto-http is only imported (and only
        # needs to be installed) when an endpoint is actually configured.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _get_handler(service_name: str, otlp_endpoint: str | None) -> TelemetryHandler | None:
    """Lazily build (and memoize) the TelemetryHandler. None on any import/init failure."""
    global _handler, _handler_service_name
    if _handler is not None and _handler_service_name == service_name:
        return _handler
    try:
        from opentelemetry import trace
        from opentelemetry.util.genai.handler import TelemetryHandler

        # Only install a real SDK TracerProvider if one isn't already
        # configured (e.g. by a host application embedding trelix) — never
        # clobber an existing provider.
        current = trace.get_tracer_provider()
        if type(current).__name__ == "ProxyTracerProvider":
            trace.set_tracer_provider(_build_tracer_provider(service_name, otlp_endpoint))

        _handler = TelemetryHandler()
        _handler_service_name = service_name
        return _handler
    except Exception as exc:
        logger.debug("OpenTelemetry init failed (tracing disabled): %s", exc)
        return None


def is_enabled(cfg: Any) -> bool:
    """True if OTel tracing should run for this config, without importing opentelemetry."""
    return bool(getattr(cfg, "otel_enabled", False))


def _handler_for(cfg: Any) -> TelemetryHandler | None:
    """Build/reuse the memoized TelemetryHandler for *cfg*'s service name + OTLP endpoint."""
    return _get_handler(
        getattr(cfg, "otel_service_name", "trelix"),
        getattr(cfg, "otel_exporter_endpoint", None),
    )


def with_current_context(fn: Callable[..., _T]) -> Callable[..., _T]:
    """
    Wrap *fn* so it runs under the OTel context captured at wrap time.

    Use at a ThreadPoolExecutor submission site so a worker thread's spans
    nest under the submitting thread's active span:

        ctx_fn = with_current_context(self._run_subquery_legs)
        pool.submit(ctx_fn, sq, strategy)

    No-op passthrough (returns *fn* unchanged) when tracing isn't active, so
    this never imports opentelemetry when the feature flag is off.
    """
    try:
        from opentelemetry import context as otel_context
    except ImportError:
        return fn

    captured = otel_context.get_current()

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> _T:
        token = otel_context.attach(captured)
        try:
            return fn(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return _wrapped


class retrieval_leg_span:
    """
    Context manager wrapping one retrieval leg (vector/bm25/grep/sparse/
    sub-chunk/file-summary) in a `gen_ai.*` retrieval span via
    `TelemetryHandler.retrieval()`. No-op (never raises, never imports
    opentelemetry) when *cfg* has otel_enabled=False or init fails.
    """

    def __init__(
        self,
        cfg: Any,
        leg: str,
        *,
        query_text: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self._leg = leg
        self._query_text = query_text
        self._top_k = top_k
        self._invocation: Any = None
        if is_enabled(cfg):
            handler = _handler_for(cfg)
            if handler is not None:
                try:
                    self._invocation = handler.retrieval(data_source_id=leg)
                    if query_text is not None:
                        self._invocation.query_text = query_text
                    if top_k is not None:
                        self._invocation.top_k = float(top_k)
                except Exception as exc:
                    logger.debug("Failed to start '%s' retrieval span: %s", leg, exc)
                    self._invocation = None

    def set_result_count(self, count: int) -> None:
        """Best-effort — record how many results this leg returned."""
        if self._invocation is None:
            return
        try:
            self._invocation.attributes = {
                **(self._invocation.attributes or {}),
                "trelix.leg.result_count": count,
            }
        except Exception as exc:
            logger.debug("Failed to set result_count on '%s' span: %s", self._leg, exc)

    def __enter__(self) -> retrieval_leg_span:
        return self

    def __exit__(
        self, _exc_type: type[BaseException] | None, exc: BaseException | None, _tb: Any
    ) -> None:
        if self._invocation is None:
            return
        try:
            if exc is not None:
                self._invocation.fail(exc)
            else:
                self._invocation.stop()
        except Exception as inner_exc:
            logger.debug("Failed to finalize '%s' span: %s", self._leg, inner_exc)


class pipeline_stage_span:
    """
    Context manager for a non-leg pipeline stage (planner, fusion, expansion,
    rerank, assembly) using a plain OTel span with `trelix.*`-namespaced
    attributes (these are trelix-specific pipeline concepts, not `gen_ai.*`
    operations, so they get trelix's own namespace rather than borrowing the
    GenAI conventions). No-op under the same conditions as retrieval_leg_span.
    """

    def __init__(self, cfg: Any, stage: str, attributes: Mapping[str, Any] | None = None) -> None:
        self._stage = stage
        self._span_cm: Any = None
        self._span: Any = None
        if is_enabled(cfg):
            handler = _handler_for(cfg)
            if handler is not None:
                try:
                    from opentelemetry import trace

                    tracer = trace.get_tracer("trelix.retrieval")
                    attrs = {f"trelix.{stage}.{k}": v for k, v in (attributes or {}).items()}
                    self._span_cm = tracer.start_as_current_span(
                        f"trelix.{stage}", attributes=attrs
                    )
                except Exception as exc:
                    logger.debug("Failed to start '%s' pipeline span: %s", stage, exc)
                    self._span_cm = None

    def __enter__(self) -> pipeline_stage_span:
        if self._span_cm is not None:
            try:
                self._span = self._span_cm.__enter__()
            except Exception as exc:
                logger.debug("Failed to enter '%s' pipeline span: %s", self._stage, exc)
                self._span_cm = None
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        if self._span_cm is not None:
            try:
                self._span_cm.__exit__(exc_type, exc, tb)
            except Exception as inner_exc:
                logger.debug("Failed to exit '%s' pipeline span: %s", self._stage, inner_exc)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics — counters, gated on the same flag as the spans above
# ═══════════════════════════════════════════════════════════════════════════


def _otel_settings(cfg: Any) -> tuple[bool, str, str | None]:
    """(enabled, service_name, otlp_endpoint) — from *cfg*, or from the
    environment when *cfg* is None.

    Every span helper above is handed a RetrievalConfig by its caller. The
    embedder cannot be: EmbedderConfig carries no otel_* fields, and all four
    construction sites (indexing/indexer.py, indexing/artifact_linker.py,
    retrieval/retriever.py, embedder/cache.py) pass only `config.embedder`.
    Reading os.environ directly there would silently disagree with the span
    path for anyone who sets TRELIX_OTEL_ENABLED in .env rather than the
    process env, so cfg=None resolves through RetrievalConfig itself — same
    field, same precedence — once per process (~15 ms, measured).
    """
    if cfg is not None:
        return (
            bool(getattr(cfg, "otel_enabled", False)),
            getattr(cfg, "otel_service_name", "trelix"),
            getattr(cfg, "otel_exporter_endpoint", None),
        )
    global _env_otel_settings
    if _env_otel_settings is None:
        try:
            from trelix.core.config import RetrievalConfig

            resolved = RetrievalConfig()
            _env_otel_settings = (
                bool(resolved.otel_enabled),
                resolved.otel_service_name,
                resolved.otel_exporter_endpoint,
            )
        except Exception as exc:
            logger.debug("Could not resolve OTel settings from the environment: %s", exc)
            _env_otel_settings = (False, "trelix", None)
    return _env_otel_settings


def metrics_enabled(cfg: Any = None) -> bool:
    """True if metrics should be recorded, without importing opentelemetry.

    Counterpart to is_enabled() for call sites that hold no config object.
    """
    return _otel_settings(cfg)[0]


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

    Mirrors _build_tracer_provider(), including being separated out so the
    exporter wiring is testable without installing a real (one-shot) global
    provider.
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


def _get_meter(service_name: str, otlp_endpoint: str | None) -> Any:
    """Lazily build (and memoize) the meter. Raises on import/init failure —
    callers report that once, loudly (see _embedding_counters_for)."""
    global _meter, _meter_service_name
    if _meter is not None and _meter_service_name == service_name:
        return _meter

    from opentelemetry import metrics

    # Only install our own MeterProvider when nothing has configured OTel
    # metrics yet, same rule as _get_handler() applies to tracing. The API's
    # unset placeholder is named `_ProxyMeterProvider` (opentelemetry-api
    # 1.44.0, verified); the underscore-free spelling is accepted in case that
    # private name changes. An explicit NoOpMeterProvider is a deliberate host
    # choice and is left alone.
    current = metrics.get_meter_provider()
    if type(current).__name__.lstrip("_") == "ProxyMeterProvider":
        metrics.set_meter_provider(_build_meter_provider(service_name, otlp_endpoint))

    _meter = metrics.get_meter("trelix.embedder")
    _meter_service_name = service_name
    return _meter


# Counter names are trelix's own (`trelix.*`): the GenAI metric conventions
# cover chat token usage, not embedding volume, so there is nothing to borrow.
# Attributes deliberately mix namespaces: `gen_ai.request.model` is the
# conventional free-form model attribute (joins these counters to the
# gen_ai.* spans), while the provider is trelix's own selector value
# ("bedrock-titan", "local-code", ...) and NOT a `gen_ai.provider.name` enum
# member, so it keeps a trelix.* name rather than pretending to conform.
_ATTR_PROVIDER = "trelix.embedder.provider"
_ATTR_MODEL = "gen_ai.request.model"


def _embedding_counters_for(cfg: Any) -> dict[str, Any] | None:
    """Build/reuse the four embedding counters. None when metrics can't be recorded."""
    global _embedding_counters, _metrics_unavailable_logged
    if _embedding_counters is not None:
        return _embedding_counters
    _, service_name, otlp_endpoint = _otel_settings(cfg)
    try:
        meter = _get_meter(service_name, otlp_endpoint)
        _embedding_counters = {
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
        return _embedding_counters
    except Exception as exc:
        # WARNING, not debug (unlike the span helpers): the operator asked for
        # observability and would otherwise read an empty cost dashboard as
        # "we spent nothing". Logged once per process — this sits on the embed
        # path. The failed import itself is retried on every later call (54 µs
        # measured for a missing package), which is noise next to any embedding
        # call and keeps this to one piece of state instead of two.
        if not _metrics_unavailable_logged:
            _metrics_unavailable_logged = True
            logger.warning(
                "TRELIX_OTEL_ENABLED is set but OpenTelemetry metrics are unavailable (%s) — "
                "embedding cost counters will NOT be recorded. Install: pip install 'trelix[otel]'",
                exc,
            )
        return None


def record_embedding_call(
    *,
    provider: str,
    model: str,
    texts: int,
    characters: int,
    tokens: int | None = None,
    cfg: Any = None,
) -> None:
    """
    Count one embedding provider call, keyed by provider and model.

    Call once per *provider call*, not per batch of texts: OpenAI/Azure/Cohere
    send one request per batch chunk, Titan one per text.

    *tokens* is the provider-reported total, or None when the provider reports
    none (local models, Cohere on Bedrock). None leaves the tokens series
    untouched — a series that is silently a chars/4 guess is worse than a
    series that is visibly absent.

    No-op (never raises) when metrics are disabled or opentelemetry is absent.
    """
    if not metrics_enabled(cfg):
        return
    counters = _embedding_counters_for(cfg)
    if counters is None:
        return
    attrs = {_ATTR_PROVIDER: provider, _ATTR_MODEL: model}
    try:
        counters["requests"].add(1, attrs)
        counters["texts"].add(texts, attrs)
        counters["characters"].add(characters, attrs)
        if tokens is not None:
            counters["tokens"].add(tokens, attrs)
    except Exception as exc:
        logger.debug("Failed to record embedding counters for '%s': %s", provider, exc)
