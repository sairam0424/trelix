"""The HTTP pieces of trelix-mcp: settings, a bearer-token check and a ``/health`` route.

Nothing wires them yet. A later change gives ``server.main()`` a ``--transport http`` flag
that calls ``resolve_http_settings`` once, ``register_health_route(mcp)`` at import and
``mcp.run(**run_kwargs(settings))``; until then ``trelix-mcp`` serves stdio only and this
module is importable but idle.

Every setting is read from the ``environ`` mapping the caller passes, never a ``.env`` or
``OPERATOR_ENV_FILE``: ``TRELIX_MCP_AUTH_TOKEN`` (fallback ``TRELIX_API_AUTH_TOKEN``) is the
bearer token, ``TRELIX_MCP_ALLOWED_HOSTS`` (fallback ``TRELIX_API_ALLOWED_HOSTS``) the Host
allow-list. ``trelix-mcp`` reads ``TRELIX_MCP_ALLOWED_HOSTS`` first and falls back to
``TRELIX_API_ALLOWED_HOSTS`` only when the MCP variable is unset or blank; when both are set,
only the MCP one counts, including ``*``.

The Host/Origin guard is the REST API's ``RequestGuardMiddleware``, so the 403 bodies are the
REST guard's and name ``TRELIX_API_ALLOWED_HOSTS``. The bearer check is this module's own
raw-ASGI ``BearerTokenMiddleware``: a 401 with ``WWW-Authenticate: Bearer`` and a fixed body,
``GET``/``HEAD`` ``/health`` exempt. The guard is always the outer of the two.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from trelix.api.request_guard import (
    HEALTH_PATH,
    LOOPBACK_HOSTS,
    RequestGuardMiddleware,
    is_health_probe,
    loopback_bind_allowed_hosts,
    normalise_host,
    resolve_allowed_hosts,
)
from trelix_mcp import __version__

_log = logging.getLogger("trelix_mcp.http")

AUTH_TOKEN_ENV = "TRELIX_MCP_AUTH_TOKEN"  # noqa: S105  # the variable's NAME, not a secret
API_AUTH_TOKEN_ENV = "TRELIX_API_AUTH_TOKEN"  # noqa: S105  # the variable's NAME, not a secret
ALLOWED_HOSTS_ENV = "TRELIX_MCP_ALLOWED_HOSTS"
API_ALLOWED_HOSTS_ENV = "TRELIX_API_ALLOWED_HOSTS"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_PATH = "/mcp"
_MAX_LOGGED_CHARS = 100
_WS_POLICY_VIOLATION = 1008

# Fixed on purpose: it names the fix and never echoes what the client sent.
UNAUTHORIZED_DETAIL = (
    "Invalid or missing bearer token. Send Authorization: Bearer <token> or "
    "X-Trelix-Api-Key: <token> matching the server's TRELIX_MCP_AUTH_TOKEN "
    "(or TRELIX_API_AUTH_TOKEN)."
)

_Scope = MutableMapping[str, Any]
_Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
_Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
_ASGIApp = Callable[[_Scope, _Receive, _Send], Awaitable[None]]


class HttpConfigError(ValueError):
    """A start-up configuration error; the CLI turns it into exit 2."""


@dataclass(frozen=True)
class HttpSettings:
    host: str
    port: int
    path: str
    stateless: bool
    guard_hosts: frozenset[str] | None  # None = guard off because the env said `*`
    token: str | None  # None = open (loopback or --allow-unauthenticated)


def first_env(environ: Mapping[str, str], *names: str) -> tuple[str, str] | None:
    """The first ``(name, value)`` whose value is set and not blank; the value is unmodified."""
    for name in names:
        value = environ.get(name)
        if value is not None and value.strip():
            return name, value
    return None


def _is_specific(name: str) -> bool:
    """True for a hostname or a specific address; False for ``0.0.0.0`` / ``::``."""
    try:
        return not ipaddress.ip_address(name).is_unspecified
    except ValueError:
        return True


def _check_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise HttpConfigError(f"--port must be between 1 and 65535, got {port}")


def _check_path(path: str) -> None:
    if not path.startswith("/") or path == HEALTH_PATH:
        raise HttpConfigError(f"--path must start with / and cannot be /health, got {path!r}")


def _bind_name(host: str) -> str:
    name = normalise_host(host)
    if name is None:
        raise HttpConfigError(
            f"--host must be a hostname or IP address, optionally in brackets, got {host!r}"
        )
    return name


def _token_from(environ: Mapping[str, str]) -> str | None:
    found = first_env(environ, AUTH_TOKEN_ENV, API_AUTH_TOKEN_ENV)
    if found is None:
        return None
    name, value = found
    # Headers are decoded as latin-1 and the token encoded as utf-8: a non-ASCII token could
    # never match, so refuse to start rather than serve a server nobody can log in to.
    if not (value.isascii() and value.isprintable()):
        raise HttpConfigError(f"the token in {name} must be printable ASCII")
    return value


def _guard_hosts(
    name: str, allowed_hosts: Sequence[str], environ: Mapping[str, str]
) -> frozenset[str] | None:
    explicit = [*LOOPBACK_HOSTS, *([name] if _is_specific(name) else []), *allowed_hosts]
    env_hosts = first_env(environ, ALLOWED_HOSTS_ENV, API_ALLOWED_HOSTS_ENV)
    guard_hosts = resolve_allowed_hosts(explicit, env_hosts[1] if env_hosts else None)
    if guard_hosts is None:
        _log.warning("Host/Origin guard is OFF (allowed hosts set to *)")
    else:
        _log.info("Host/Origin guard is ON (allowed hosts: %s)", ", ".join(sorted(guard_hosts)))
    return guard_hosts


def resolve_http_settings(
    *,
    host: str | None,
    port: int | None,
    path: str | None,
    allowed_hosts: Sequence[str],
    stateful: bool,
    allow_unauthenticated: bool,
    roots: Sequence[Path],
    environ: Mapping[str, str],
) -> HttpSettings:
    """Validate the HTTP flags and environment in the order port, path, roots, host, token and
    the loopback rule; raise ``HttpConfigError`` with a fixed text."""
    host = host or DEFAULT_HOST
    port = DEFAULT_PORT if port is None else port
    path = path or DEFAULT_PATH
    _check_port(port)
    _check_path(path)
    if not roots:
        raise HttpConfigError(
            "--transport http needs at least one --root or TRELIX_ALLOWED_REPO_ROOTS; every "
            "tool call names a repository path and paths outside these roots are refused"
        )
    name = _bind_name(host)
    token = _token_from(environ)
    loopback = loopback_bind_allowed_hosts(host) is not None
    if token is None and not loopback and not allow_unauthenticated:
        raise HttpConfigError(
            f"--host {host} is not a loopback address and no token is set; set "
            "TRELIX_MCP_AUTH_TOKEN (or TRELIX_API_AUTH_TOKEN), or pass --allow-unauthenticated "
            "to serve without one"
        )
    if token is None and not loopback:
        _log.warning(
            "Serving MCP over HTTP on %s:%s with no token: every tool is open to anyone who "
            "can reach this port (--allow-unauthenticated)",
            host,
            port,
        )
    return HttpSettings(
        host=host,
        port=port,
        path=path,
        stateless=not stateful,
        guard_hosts=_guard_hosts(name, allowed_hosts, environ),
        token=token,
    )


def _header_values(scope: _Scope, name: bytes) -> list[str]:
    return [
        value.decode("latin-1") for key, value in scope.get("headers", ()) if key.lower() == name
    ]


def _credentials(scope: _Scope) -> list[str]:
    """Every credential the request presents; a header sent zero or several times adds none."""
    candidates: list[str] = []
    authorization = _header_values(scope, b"authorization")
    if len(authorization) == 1:
        scheme, _, rest = authorization[0].partition(" ")
        if scheme.lower() == "bearer":
            candidates.append(rest)
    api_keys = _header_values(scope, b"x-trelix-api-key")
    if len(api_keys) == 1:
        candidates.append(api_keys[0])
    return candidates


def _authorised(scope: _Scope, token: bytes) -> bool:
    ok = False
    # Every candidate is compared, no early return: timing must not reveal which header was wrong.
    for candidate in _credentials(scope):
        ok |= hmac.compare_digest(candidate.encode("latin-1"), token)
    return ok


async def _reject(scope: _Scope, send: _Send) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
        return
    body = json.dumps({"detail": UNAUTHORIZED_DETAIL}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BearerTokenMiddleware:
    """Refuse every request but ``GET``/``HEAD`` ``/health`` unless it carries the token."""

    def __init__(self, app: _ASGIApp, *, token: str) -> None:
        self.app = app
        self._token = token.encode("utf-8")

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        exempt = scope["type"] not in ("http", "websocket") or is_health_probe(scope)
        if exempt or _authorised(scope, self._token):
            await self.app(scope, receive, send)
            return
        _log.warning(
            "Rejected %s %s: missing or invalid bearer token",
            scope.get("method", scope["type"]),
            str(scope.get("path") or "")[:_MAX_LOGGED_CHARS],
        )
        await _reject(scope, send)


def asgi_middleware(settings: HttpSettings) -> list[Middleware]:
    """The guard first (outermost), then the bearer check; each only when it applies."""
    layers: list[Middleware] = []
    if settings.guard_hosts is not None:
        layers.append(Middleware(RequestGuardMiddleware, allowed_hosts=settings.guard_hosts))
    if settings.token is not None:
        layers.append(Middleware(BearerTokenMiddleware, token=settings.token))
    return layers


def http_app_kwargs(settings: HttpSettings) -> dict[str, Any]:
    """Keyword arguments for ``FastMCP.http_app``. ``path``, ``stateless_http``, ``json_response``
    and ``host_origin_protection`` are passed explicitly, so the ``FASTMCP_*`` variables behind
    them cannot change them. FastMCP's own Host/Origin guard and the SDK's DNS-rebinding check
    stay off: the REST guard is the one allow-list."""
    return {
        "path": settings.path,
        "stateless_http": settings.stateless,
        "json_response": False,
        "host_origin_protection": False,
        "middleware": asgi_middleware(settings),
    }


def run_kwargs(settings: HttpSettings) -> dict[str, Any]:
    """Keyword arguments for ``FastMCP.run``."""
    return {
        "transport": "http",
        "host": settings.host,
        "port": settings.port,
        **http_app_kwargs(settings),
    }


def register_health_route(mcp: FastMCP[Any]) -> None:
    """Serve ``GET /health`` (Starlette adds ``HEAD``) with the package version; custom routes
    are read only by ``http_app``, so a stdio server is unaffected."""

    @mcp.custom_route(HEALTH_PATH, methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})
