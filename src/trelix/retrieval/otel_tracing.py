"""
OpenTelemetry tracing for the retrieval pipeline — off by default.

Emits one span per retrieval leg (vector, BM25, grep, sparse, sub-chunk,
file-summary) plus root/planner/fusion/expansion/rerank/assembly spans, using
the official `gen_ai.*` semantic conventions (status: Development, not yet
Stable — attribute names may still shift upstream; see
docs/OBSERVABILITY.md).

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
