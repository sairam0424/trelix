"""
Unified retry/backoff contract for every outbound network call in trelix.

Before this module, retry logic was either absent (all 5 LLM backends, all
remote embedder providers, GitHubPRClient, the agentic ReAct loop) or
duplicated ad hoc (JiraConnector/TestRailConnector each hand-rolled a
near-identical backoff loop — see git_linker.py's sibling module docstrings
for the git-log precedent this mirrors for HTTP instead). A single transient
429/timeout could abort an entire indexing run, ReAct session, or PR review.

Built on tenacity (github.com/jd/tenacity) rather than hand-rolled: it
auto-dispatches sync vs. async based on whether the wrapped callable is a
coroutine (one decorator covers both trelix's fully-sync LLM/connector code
and the embedder layer's real async paths), and `wait_random_exponential` is
the exact pattern OpenAI's own cookbook recommends for rate-limited API
calls — full-jitter backoff that reduces collision when multiple concurrent
requests (e.g. the indexer's asyncio.Semaphore(4) batch-embedding loop) hit
the same rate-limited resource simultaneously.

Deliberately NOT a circuit breaker: tenacity has no failure-threshold/
open-circuit semantics, only per-call retry/wait/stop logic. "Stop hammering
a persistently-down connector across many calls" is a distinct future item
(e.g. pybreaker), not something this module attempts.
"""

from __future__ import annotations

import email.utils
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

if TYPE_CHECKING:
    from tenacity import RetryCallState

logger = logging.getLogger("trelix.core.retry")

# 429 (rate limited) and 5xx (transient server-side failure) are retryable.
# 4xx other than 429 (bad request, unauthorized, not found, ...) indicates a
# real problem with the request itself — retrying won't fix it, and doing so
# just delays a failure the caller needs to see immediately.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def is_retryable_http_error(exc: BaseException) -> bool:
    """
    True if *exc* represents a transient failure worth retrying: a
    connection-level error (always retryable — the request never reached
    the server) or an HTTP response whose status code is in
    RETRYABLE_STATUS_CODES.

    Recognizes httpx, requests, and boto3/botocore exception shapes without
    hard-importing any of them — trelix's optional extras mean not every
    installation has all three HTTP libraries available.
    """
    status_code = _extract_status_code(exc)
    if status_code is not None:
        return status_code in RETRYABLE_STATUS_CODES

    # No status code could be extracted — this is a connection-level
    # failure (DNS, timeout, connection refused, ...) rather than a real
    # HTTP response. Those are always worth retrying.
    return _is_connection_level_error(exc)


def _extract_status_code(exc: BaseException) -> int | None:
    """Pull an HTTP status code out of exc, if it has one. Returns None for
    connection-level errors that never got a response at all.

    Covers both raw HTTP-client exceptions (httpx, requests, botocore) and
    LLM-SDK exceptions (openai, anthropic, google-genai) — the latter wrap an
    httpx.Response internally but expose their own status attribute rather
    than being httpx.HTTPStatusError instances themselves. litellm's
    exceptions subclass openai's, so the openai check covers both.

    Each block catches Exception, not just ImportError: this function's
    entire contract is "never raise, always degrade to None on anything
    unexpected" (is_retryable_http_error() falls back to treating an
    unclassifiable exception as non-retryable). A malformed or mocked
    exception object — e.g. a test harness's stand-in whose .response is a
    plain dict instead of the real SDK's response type, or an SDK version
    bump that changes an attribute's shape — could raise AttributeError/
    TypeError from the attribute access itself, not just from a missing
    import; letting that propagate would crash tenacity's retry predicate
    mid-retry-loop instead of degrading gracefully.
    """
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
    except Exception:  # noqa: BLE001
        pass

    try:
        import requests

        if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
            return exc.response.status_code
    except Exception:  # noqa: BLE001
        pass

    try:
        from botocore.exceptions import ClientError

        if isinstance(exc, ClientError) and exc.response is not None:
            code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            return int(code) or None
    except Exception:  # noqa: BLE001
        pass

    try:
        import openai

        if isinstance(exc, openai.APIStatusError):
            return exc.status_code
    except Exception:  # noqa: BLE001
        pass

    try:
        import anthropic

        if isinstance(exc, anthropic.APIStatusError):
            return int(exc.status_code)
    except Exception:  # noqa: BLE001
        pass

    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError):
            return int(exc.code)
    except Exception:  # noqa: BLE001
        pass

    try:
        import voyageai.error as voyage_errors

        if isinstance(exc, voyage_errors.VoyageError) and exc.http_status is not None:
            return int(exc.http_status)
    except Exception:  # noqa: BLE001
        pass

    return None


def _is_connection_level_error(exc: BaseException) -> bool:
    """True for errors that mean the request never got a response at all
    (DNS failure, connection refused, timeout) — as opposed to an error
    response the server actually sent back.

    Each block catches Exception, not just ImportError — see
    _extract_status_code()'s docstring for why: this function must never
    raise, only ever degrade to False."""
    try:
        import httpx

        if isinstance(exc, httpx.HTTPError) and not isinstance(exc, httpx.HTTPStatusError):
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        import requests

        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        import openai

        if isinstance(exc, openai.APIConnectionError):
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        import anthropic

        if isinstance(exc, anthropic.APIConnectionError):
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        import voyageai.error as voyage_errors

        if isinstance(exc, (voyage_errors.APIConnectionError, voyage_errors.Timeout)):
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        from botocore.exceptions import BotoCoreError

        # BotoCoreError (EndpointConnectionError, ConnectTimeoutError,
        # ReadTimeoutError, ...) is boto3/botocore's connection-level
        # failure hierarchy — disjoint from ClientError (already handled
        # in _extract_status_code(), which carries a real HTTP status via
        # ResponseMetadata). Without this branch, a DNS failure or connect/
        # read timeout talking to Bedrock was never retried, unlike every
        # other backend/connector wired into this same contract.
        if isinstance(exc, BotoCoreError):
            return True
    except Exception:  # noqa: BLE001
        pass

    return False


def _extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Pull a Retry-After delay (in seconds) out of exc's response headers,
    if it has one. Returns None when there's no response, no header, or the
    header is present but unparseable — callers fall back to exponential
    backoff in every one of those cases.

    Retry-After is valid as either a delay-in-seconds integer or an HTTP
    date (RFC 9110 §10.2.3); both forms are handled. httpx/requests/openai/
    anthropic exceptions all expose the same `.response.headers` shape
    (openai/anthropic's APIStatusError wraps a real httpx.Response
    internally), so one extraction path covers all four without needing
    per-SDK branches the way _extract_status_code() does.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    delay = parsed.timestamp() - time.time()
    return max(0.0, delay)


class _wait_retry_after_or_exponential:
    """Honor a server-supplied Retry-After header when present, falling
    back to wait_random_exponential's full-jitter backoff otherwise.

    Before this, migrating Jira/TestRail's hand-rolled backoff (which read
    Retry-After explicitly) onto the shared tenacity contract silently
    dropped that behavior — every retry used blind exponential backoff
    regardless of what the server explicitly requested, risking either
    retrying sooner than a rate limiter wants (escalating the throttle) or
    waiting longer than necessary when the header requested a short delay.
    """

    def __init__(self, min_wait_seconds: float, max_wait_seconds: float) -> None:
        self._exponential = wait_random_exponential(min=min_wait_seconds, max=max_wait_seconds)

    def __call__(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        if outcome is not None and outcome.failed:
            exc = outcome.exception()
            if exc is not None:
                retry_after = _extract_retry_after_seconds(exc)
                if retry_after is not None:
                    return retry_after
        return self._exponential(retry_state)


_F = TypeVar("_F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 5,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
) -> Callable[[_F], _F]:
    """
    Return a tenacity @retry decorator implementing trelix's unified retry
    contract: honors a server-supplied Retry-After header when present,
    otherwise full-jitter exponential backoff; retries only on
    is_retryable_http_error()'s classification, giving up after
    *max_attempts*. Works on both sync and async functions — tenacity
    auto-detects which at decoration time.

    Usage::

        @with_retry()
        def call_api() -> Response:
            return httpx.get(url)

        @with_retry(max_attempts=3)
        async def call_api_async() -> Response:
            return await client.get(url)
    """
    return retry(
        retry=retry_if_exception(is_retryable_http_error),
        wait=_wait_retry_after_or_exponential(min_wait_seconds, max_wait_seconds),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
