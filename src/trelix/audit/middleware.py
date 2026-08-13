"""
Request-level audit middleware.

A Starlette ``BaseHTTPMiddleware`` that records exactly ONE
:class:`~trelix.audit.events.AuditEvent` per HTTP request into an
:class:`~trelix.audit.store.AuditStore`. It is registered as the outermost
middleware in ``create_app()`` so it observes the *final* status code —
including 401s produced by the auth dependency and 500s produced by an
unhandled route error — after every inner layer has run.

Design notes:
  - **Never stores the API key/token.** The event carries only the request
    path (``resource``), a coarse action, the outcome, the status code, timing,
    and request/trace correlation ids — never a header value or query text.
  - **Identity seam.** ``principal`` is read from ``request.state.principal``
    (set by ``authenticate`` to the local ``"static-token"`` default, and the
    exact attribute the OIDC path overwrites with a real ``sub@iss``).
    Falls back to ``"static-token"`` when unset (e.g. the unauthenticated
    ``/health`` route), so a missing principal never crashes the recorder.
  - **Resilience.** Writes go through ``AuditStore.append`` which swallows and
    logs failures by default; only ``AuditConfig.fail_closed=True`` makes a
    write failure propagate. A broken audit sink therefore never takes down a
    request unless the operator explicitly opted into fail-closed.
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from trelix.audit.events import (
    ACTION_ADMIN,
    ACTION_ASK,
    ACTION_INDEX,
    ACTION_SEARCH,
    OUTCOME_DENIED,
    OUTCOME_ERROR,
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    AuditEvent,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

    from trelix.audit.store import AuditStore
    from trelix.core.config import AuditConfig

# Default identity for local (API-key / open) mode. The seam an OIDC layer
# overwrites via request.state.principal — kept as a module constant so the
# middleware fallback and authenticate agree on the exact same string.
DEFAULT_PRINCIPAL = "static-token"

# Coarse path -> action mapping. Prefix match keeps sub-paths
# (e.g. /graph/communities) grouped; anything unrecognized is ACTION_ADMIN.
_PATH_ACTIONS: tuple[tuple[str, str], ...] = (
    ("/search", ACTION_SEARCH),
    ("/ask", ACTION_ASK),
    ("/index", ACTION_INDEX),
    ("/parse", ACTION_INDEX),
)


def _action_for_path(path: str) -> str:
    for prefix, action in _PATH_ACTIONS:
        if path == prefix or path.startswith(prefix + "/"):
            return action
    return ACTION_ADMIN


def _outcome_for_status(status_code: int) -> str:
    """Map an HTTP status to an audit outcome.

    401/403 -> denied, 5xx -> error, other 4xx -> failure, else success.
    """
    if status_code in (401, 403):
        return OUTCOME_DENIED
    if status_code >= 500:
        return OUTCOME_ERROR
    if status_code >= 400:
        return OUTCOME_FAILURE
    return OUTCOME_SUCCESS


def _current_trace_id() -> str | None:
    """Return the active OTel trace id (32-hex) or None.

    Mirrors logging_setup._inject_trace_context: never imports opentelemetry
    unless it is already loaded in the process, so this is a true no-op when
    tracing is disabled — and never makes a network call.
    """
    if "opentelemetry" not in sys.modules:
        return None
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            return format(span_context.trace_id, "032x")
    except Exception:  # noqa: BLE001 — audit trace lookup must never crash a request
        return None
    return None


class AuditMiddleware(BaseHTTPMiddleware):
    """Record one AuditEvent per request into an AuditStore."""

    def __init__(self, app: object, *, store: AuditStore, config: AuditConfig) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._store = store
        self._config = config

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled route error -> a 500 will be produced by the outer
            # ServerErrorMiddleware. Record it as an error, then re-raise so
            # the normal error-response path is unchanged.
            self._emit(
                request,
                status_code=500,
                outcome=OUTCOME_ERROR,
                duration_ms=self._elapsed_ms(start),
            )
            raise

        self._emit(
            request,
            status_code=response.status_code,
            outcome=_outcome_for_status(response.status_code),
            duration_ms=self._elapsed_ms(start),
        )
        return response

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    def _emit(self, request: Request, *, status_code: int, outcome: str, duration_ms: int) -> None:
        principal = getattr(request.state, "principal", DEFAULT_PRINCIPAL)
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        client_ip = request.client.host if request.client else None
        event = AuditEvent(
            ts=datetime.now(tz=UTC).isoformat(),
            principal=principal,
            action=_action_for_path(request.url.path),
            resource=request.url.path,
            outcome=outcome,
            status_code=status_code,
            client_ip=client_ip,
            request_id=request_id,
            trace_id=_current_trace_id(),
            duration_ms=duration_ms,
            detail=None,  # never persist raw query text / headers from the request line
        )
        self._store.append(
            event,
            log_queries=self._config.log_queries,
            fail_closed=self._config.fail_closed,
        )
