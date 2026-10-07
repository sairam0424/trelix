"""Helpers for the trelix-mcp HTTP tests: a raw ASGI driver, the lifespan and an httpx2 factory.

There is deliberately no `build_app` here: a test builds its own two-line echo app or a throwaway
`FastMCP("throwaway")` whose `http_app(**http_app_kwargs(settings))` it calls directly, so what is
pinned is the module under test and not a helper's idea of the wiring.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx2

INIT_BODY = (
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25",'
    '"capabilities":{},"clientInfo":{"name":"t","version":"0"}}}'
)
MCP_HEADERS: list[tuple[str, str]] = [
    ("host", "127.0.0.1:8766"),
    ("content-type", "application/json"),
    ("accept", "application/json, text/event-stream"),
]
# Shorter than the 60 s test timeout so a wrong implementation fails with TimeoutError here.
_RAW_TIMEOUT_SECONDS = 20


def _scope(method: str, path: str, headers: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """tests/unit/test_api_request_guard.py::_scope; headers keep their order, so repeats work."""
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(n.lower().encode("latin-1"), v.encode("latin-1")) for n, v in headers],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8766),
    }


async def raw(
    app: Any,
    method: str,
    path: str,
    *,
    headers: Sequence[tuple[str, str]],
    body: bytes = b"",
) -> tuple[int, dict[str, str], str]:
    """Run one ASGI http request; return (status, lower-cased headers, body as utf-8 text)."""
    sent: list[dict[str, Any]] = []
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            # mcp answers a POST through sse_starlette, whose disconnect listener loops on
            # receive() until http.disconnect. A repeated http.request would spin forever and an
            # http.disconnect would cut the stream before the result is written; a client that is
            # still reading sends nothing, so wait until the response ends and cancels us.
            await asyncio.Event().wait()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await asyncio.wait_for(app(_scope(method, path, headers), receive, send), _RAW_TIMEOUT_SECONDS)
    start = next(m for m in sent if m["type"] == "http.response.start")
    out_headers = {
        k.decode("latin-1").lower(): v.decode("latin-1") for k, v in start.get("headers", [])
    }
    text = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], out_headers, text.decode("utf-8")


@asynccontextmanager
async def serving(app: Any) -> AsyncIterator[Any]:
    """FastMCP's session manager exists only inside the lifespan; every app test runs in it."""
    async with app.router.lifespan_context(app):
        yield app


def asgi_factory(app: Any) -> Any:
    """An `httpx_client_factory` for `StreamableHttpTransport` (mcp 2.3.0 passes `headers`,
    `timeout`, `auth` and `follow_redirects`, hence `**extra`), and a plain client for tests."""

    def factory(
        headers: Any = None, timeout: Any = None, auth: Any = None, **extra: Any
    ) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://127.0.0.1:8766",
            headers=headers,
            timeout=timeout,
            auth=auth,
            **extra,
        )

    return factory
