"""The two 403 texts of `RequestGuardMiddleware`, and the parameters that replace them.

`create_app` never passes a text, so the REST API's bodies must stay what they were: the
literals below are copied in, not imported (test_api_request_guard.py checks the same texts
through the real app against the module's constants). trelix-mcp passes its own texts, because
it reads TRELIX_MCP_ALLOWED_HOSTS first and the REST texts name the other variable
(packages/trelix-mcp/tests pin those texts on the wire). What is pinned here is the mechanism:
each parameter replaces the text of exactly its own refusals, and nothing else changes.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from trelix.api.request_guard import RequestGuardMiddleware, bounded_repr

REST_REJECTION = (
    "Request rejected: the Host or Origin header is not allowed for this trelix server. "
    "If the hostname is legitimate, add it to TRELIX_API_ALLOWED_HOSTS (comma-separated). "
    "Set TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)
REST_CROSS_SITE = (
    "Request rejected: this trelix server refuses cross-site browser requests. Call it "
    "from a page served from the same host as the API, or set "
    "TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)
CUSTOM_REJECTION = "custom host text"
CUSTOM_CROSS_SITE = "custom cross-site text"

GOOD_HOST = (b"host", b"localhost:8765")
# Every way the guard refuses, and whether it is the cross-site text that answers.
REFUSALS = [
    pytest.param([], False, id="host-missing"),
    pytest.param([(b"host", b"evil.example")], False, id="host-foreign"),
    pytest.param(
        [GOOD_HOST, (b"origin", b"http://localhost"), (b"origin", b"http://localhost")],
        False,
        id="origin-repeated",
    ),
    pytest.param([GOOD_HOST, (b"origin", b"null")], False, id="origin-foreign"),
    pytest.param([GOOD_HOST, (b"sec-fetch-site", b"cross-site")], True, id="cross-site"),
]


def _refuse(headers: list[tuple[bytes, bytes]], **details: str) -> tuple[int, dict[str, str]]:
    """Run one refused request through a guard built with `details`; return (status, JSON body)."""
    sent: list[dict[str, Any]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        raise AssertionError("a refused request must not reach the app")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    guard = RequestGuardMiddleware(
        inner, allowed_hosts=frozenset({"localhost", "127.0.0.1", "::1"}), **details
    )
    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}
    asyncio.run(guard(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], json.loads(body)


@pytest.mark.parametrize(("headers", "is_cross_site"), REFUSALS)
def test_without_overrides_every_refusal_keeps_the_rest_text(
    headers: list[tuple[bytes, bytes]], is_cross_site: bool
) -> None:
    expected = REST_CROSS_SITE if is_cross_site else REST_REJECTION
    assert _refuse(headers) == (403, {"detail": expected})


@pytest.mark.parametrize(("headers", "is_cross_site"), REFUSALS)
def test_each_override_replaces_only_its_own_text(
    headers: list[tuple[bytes, bytes]], is_cross_site: bool
) -> None:
    both = _refuse(headers, rejection_detail=CUSTOM_REJECTION, cross_site_detail=CUSTOM_CROSS_SITE)
    assert both == (403, {"detail": CUSTOM_CROSS_SITE if is_cross_site else CUSTOM_REJECTION})

    only_host = _refuse(headers, rejection_detail=CUSTOM_REJECTION)
    assert only_host == (
        403,
        {"detail": REST_CROSS_SITE if is_cross_site else CUSTOM_REJECTION},
    )

    only_cross_site = _refuse(headers, cross_site_detail=CUSTOM_CROSS_SITE)
    assert only_cross_site == (
        403,
        {"detail": CUSTOM_CROSS_SITE if is_cross_site else REST_REJECTION},
    )


def test_bounded_repr_keeps_hostile_text_on_one_short_line() -> None:
    assert bounded_repr("/mcp\r\nWARNING x") == "'/mcp\\r\\nWARNING x'"
    assert bounded_repr(None) == "None"
    assert len(bounded_repr("a" * 5000)) == 100
