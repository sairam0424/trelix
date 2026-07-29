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

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

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
    exceptions subclass openai's, so the openai check covers both."""
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
    except ImportError:
        pass

    try:
        import requests

        if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
            return exc.response.status_code
    except ImportError:
        pass

    try:
        from botocore.exceptions import ClientError

        if isinstance(exc, ClientError) and exc.response is not None:
            code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            return int(code) or None
    except ImportError:
        pass

    try:
        import openai

        if isinstance(exc, openai.APIStatusError):
            return exc.status_code
    except ImportError:
        pass

    try:
        import anthropic

        if isinstance(exc, anthropic.APIStatusError):
            return int(exc.status_code)
    except ImportError:
        pass

    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError):
            return int(exc.code)
    except ImportError:
        pass

    try:
        import voyageai.error as voyage_errors

        if isinstance(exc, voyage_errors.VoyageError) and exc.http_status is not None:
            return int(exc.http_status)
    except ImportError:
        pass

    return None


def _is_connection_level_error(exc: BaseException) -> bool:
    """True for errors that mean the request never got a response at all
    (DNS failure, connection refused, timeout) — as opposed to an error
    response the server actually sent back."""
    try:
        import httpx

        if isinstance(exc, httpx.HTTPError) and not isinstance(exc, httpx.HTTPStatusError):
            return True
    except ImportError:
        pass

    try:
        import requests

        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
    except ImportError:
        pass

    try:
        import openai

        if isinstance(exc, openai.APIConnectionError):
            return True
    except ImportError:
        pass

    try:
        import anthropic

        if isinstance(exc, anthropic.APIConnectionError):
            return True
    except ImportError:
        pass

    try:
        import voyageai.error as voyage_errors

        if isinstance(exc, (voyage_errors.APIConnectionError, voyage_errors.Timeout)):
            return True
    except ImportError:
        pass

    return False


_F = TypeVar("_F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 5,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
) -> Callable[[_F], _F]:
    """
    Return a tenacity @retry decorator implementing trelix's unified retry
    contract: full-jitter exponential backoff, retrying only on
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
        wait=wait_random_exponential(min=min_wait_seconds, max=max_wait_seconds),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
