"""`BearerTokenMiddleware` and `register_health_route` from `trelix_mcp.http`.

The middleware is driven as raw ASGI around a two-line echo app, and then inside a throwaway
`FastMCP("throwaway")` whose `http_app(**http_app_kwargs(settings))` is built directly, so the
Host/Origin guard, the bearer check and the MCP endpoint are seen in their real order. Nothing
here touches `trelix_mcp.server`, which does not wire this module yet. Every expected value is a
literal written here (the 401 and 403 bodies are copied in, never imported); the one import used
as an expected value is `trelix_mcp.__version__`, the package's identity. This package's tests
are governed by packages/trelix-mcp/pyproject.toml (no socket ban, no timeout), so the markers
below supply both.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import trelix_mcp
from fastmcp import FastMCP
from http_support import INIT_BODY, MCP_HEADERS, asgi_factory, raw, serving
from trelix_mcp.http import (
    BearerTokenMiddleware,
    http_app_kwargs,
    register_health_route,
    resolve_http_settings,
)

pytestmark = [pytest.mark.disable_socket, pytest.mark.timeout(60)]

LOGGER = "trelix_mcp.http"
UNAUTHORIZED = (
    "Invalid or missing bearer token. Send Authorization: Bearer <token> or "
    "X-Trelix-Api-Key: <token> matching the server's TRELIX_MCP_AUTH_TOKEN "
    "(or TRELIX_API_AUTH_TOKEN)."
)
REJECTION = (
    "Request rejected: the Host or Origin header is not allowed for this trelix server. "
    "If the hostname is legitimate, add it to TRELIX_API_ALLOWED_HOSTS (comma-separated). "
    "Set TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)
CROSS_SITE = (
    "Request rejected: this trelix server refuses cross-site browser requests. Call it "
    "from a page served from the same host as the API, or set "
    "TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)
HOST = ("host", "127.0.0.1:8766")
EVIL_HOST = ("host", "evil.example")
BEARER = ("authorization", "Bearer t0k")


async def _echo(scope: Any, receive: Any, send: Any) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


_guarded = BearerTokenMiddleware(_echo, token="t0k")


@pytest.mark.parametrize(
    "credentials",
    [
        [("authorization", "Bearer t0k")],
        [("authorization", "bearer t0k")],
        [("x-trelix-api-key", "t0k")],
        # Any presented credential may match (B3; REST's rule).
        [("authorization", "Bearer nope"), ("x-trelix-api-key", "t0k")],
    ],
)
async def test_bearer_accepts_one_matching_credential(credentials: list[tuple[str, str]]) -> None:
    status, _, text = await raw(_guarded, "POST", "/mcp", headers=[HOST, *credentials])
    assert (status, text) == (200, "ok")


@pytest.mark.parametrize(
    "credentials",
    [
        [],
        [("authorization", "Bearer nope")],
        [("authorization", "Bearer t0k"), ("authorization", "Bearer t0k")],
        [("authorization", "Bearer  t0k")],
        [("authorization", "Bearer")],
        [("x-trelix-api-key", "Bearer t0k")],
        [("x-trelix-api-key", "t0k"), ("x-trelix-api-key", "t0k")],
        [("authorization", "Basic dDBr")],
    ],
)
async def test_bearer_refuses_missing_wrong_or_repeated(
    credentials: list[tuple[str, str]],
) -> None:
    status, headers, text = await raw(_guarded, "POST", "/mcp", headers=[HOST, *credentials])
    assert status == 401
    assert headers["www-authenticate"] == "Bearer"
    assert headers["content-type"] == "application/json"
    assert headers["content-length"] == str(len(text.encode("utf-8")))
    assert json.loads(text) == {"detail": UNAUTHORIZED}


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("GET", "/health", 200),
        ("HEAD", "/health", 200),
        ("GET", "/health/", 401),
        ("GET", "/HEALTH", 401),
        ("POST", "/health", 401),
    ],
)
async def test_health_is_exempt_from_the_bearer_check(method: str, path: str, status: int) -> None:
    got, _, _ = await raw(_guarded, method, path, headers=[HOST])
    assert got == status


async def test_lifespan_passes_and_websocket_is_closed() -> None:
    reached: list[str] = []
    sent: list[dict[str, Any]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["type"])

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    app = BearerTokenMiddleware(inner, token="t0k")
    await app({"type": "lifespan"}, None, send)
    anonymous = {"type": "websocket", "path": "/mcp", "headers": [(b"host", b"127.0.0.1:8766")]}
    await app(anonymous, None, send)
    assert (reached, sent) == (["lifespan"], [{"type": "websocket.close", "code": 1008}])
    bearer = {**anonymous, "headers": [*anonymous["headers"], (b"authorization", b"Bearer t0k")]}
    await app(bearer, None, send)
    assert reached == ["lifespan", "websocket"]


async def test_401_logs_no_header_value(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await raw(_guarded, "POST", "/mcp", headers=[HOST, ("authorization", "Bearer nope")])
    records = [r for r in caplog.records if r.name == LOGGER]
    assert [(r.levelname, r.getMessage()) for r in records] == [
        ("WARNING", "Rejected POST /mcp: missing or invalid bearer token")
    ]
    assert "nope" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await raw(_guarded, "POST", "/" + "a" * 2000, headers=[HOST])
    (record,) = [r for r in caplog.records if r.name == LOGGER]
    assert len(record.getMessage()) < 300


def _throwaway_app(tmp_path: Path, *, stateful: bool = False) -> Any:
    mcp = FastMCP("throwaway")
    register_health_route(mcp)
    settings = resolve_http_settings(
        host="127.0.0.1",
        port=None,
        path=None,
        allowed_hosts=(),
        stateful=stateful,
        allow_unauthenticated=False,
        roots=(tmp_path,),
        environ={"TRELIX_MCP_AUTH_TOKEN": "t0k"},
    )
    return mcp.http_app(**http_app_kwargs(settings))


async def test_health_route_on_a_fastmcp_app(tmp_path: Path) -> None:
    body = {"status": "ok", "version": trelix_mcp.__version__}
    app = _throwaway_app(tmp_path)
    async with serving(app):
        status, _, text = await raw(app, "GET", "/health", headers=[EVIL_HOST])
        assert (status, json.loads(text)) == (200, body)
        # Starlette writes the JSON body for HEAD too; stripping it is the ASGI server's job.
        status, _, text = await raw(app, "HEAD", "/health", headers=[EVIL_HOST])
        assert (status, json.loads(text)) == (200, body)
        status, _, _ = await raw(app, "GET", "/healthz", headers=[HOST, BEARER])
        assert status == 404

        client = asgi_factory(app)(headers={"host": "127.0.0.1:8766"}, follow_redirects=True)
        async with client:
            response = await client.get("/health")
            assert (response.status_code, response.json()) == (200, body)
            response = await client.head("/health")
            assert (response.status_code, response.text) == (200, "")


@pytest.mark.parametrize(
    ("headers", "status", "detail", "www_authenticate"),
    [
        # No credential and a foreign Host: 403, not 401, so the guard runs before the bearer check.
        ([EVIL_HOST], 403, REJECTION, None),
        ([EVIL_HOST, BEARER], 403, REJECTION, None),
        ([HOST, ("origin", "null"), BEARER], 403, REJECTION, None),
        ([HOST, ("sec-fetch-site", "cross-site"), BEARER], 403, CROSS_SITE, None),
        ([HOST], 401, UNAUTHORIZED, "Bearer"),
    ],
)
async def test_guard_runs_before_the_bearer_check(
    tmp_path: Path,
    headers: list[tuple[str, str]],
    status: int,
    detail: str,
    www_authenticate: str | None,
) -> None:
    app = _throwaway_app(tmp_path)
    async with serving(app):
        got, out, text = await raw(app, "POST", "/mcp", headers=headers, body=INIT_BODY.encode())
    assert (got, json.loads(text)) == (status, {"detail": detail})
    assert "evil.example" not in text
    assert out.get("www-authenticate") == www_authenticate


async def test_initialize_streams_and_get_is_405_when_stateless(tmp_path: Path) -> None:
    app = _throwaway_app(tmp_path)
    async with serving(app):
        headers = [*MCP_HEADERS, BEARER]
        status, out, text = await raw(app, "POST", "/mcp", headers=headers, body=INIT_BODY.encode())
        assert status == 200
        assert out["content-type"].startswith("text/event-stream")
        assert '"protocolVersion"' in text
        # Stateless: the MCP route takes POST and DELETE only, whatever the Accept header says.
        status, _, _ = await raw(app, "GET", "/mcp", headers=headers)
        assert status == 405
        status, _, _ = await raw(app, "GET", "/mcp", headers=[HOST, BEARER])
        assert status == 405


async def test_stateful_app_wants_a_session(tmp_path: Path) -> None:
    app = _throwaway_app(tmp_path, stateful=True)
    async with serving(app):
        # MCP_HEADERS' `accept: text/event-stream` is what turns 406 into this 400.
        status, _, text = await raw(app, "GET", "/mcp", headers=[*MCP_HEADERS, BEARER])
    assert status == 400
    assert "Missing session ID" in text
