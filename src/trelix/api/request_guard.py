"""Host / Origin / fetch-metadata guard for the REST API.

The threat: ``trelix serve`` is open by design when no credential is configured, and it
binds to loopback by default. Nothing used to look at the ``Host`` or ``Origin`` header,
so a page on an attacker's domain that is re-resolved to 127.0.0.1 (DNS rebinding) became
same-origin with the server and could READ ``/search``, ``/ask`` and ``/parse``; and even
without rebinding any web page can fire ``no-cors`` GETs with side effects (``/ask``
spends LLM money). Loopback is a network boundary, not a browser boundary.

What this module checks, when it is switched on:

* ``Host`` must be present exactly once and its hostname (port ignored) must be in the
  allow-list;
* an ``Origin`` header, when present (once), must be an ``http``/``https`` origin whose
  hostname is in the allow-list (any port); the literal ``null`` is refused;
* ``Sec-Fetch-Site: cross-site`` is refused.

A request with neither ``Origin`` nor ``Sec-Fetch-Site`` (curl, Python clients, the trelix
clients) is unaffected. There is deliberately NO CORS support here. ``GET``/``HEAD``
``/health`` is exempt from every check: kubelet probes send ``Host: <pod-ip>:<port>``.

When it is switched on is decided elsewhere (``create_app`` / ``trelix serve``); this
module holds the pure matching helpers, the settings class for
``TRELIX_API_ALLOWED_HOSTS`` and the ASGI middleware. It is raw ASGI on purpose: no
starlette import, so the module stays importable without ``trelix[serve]`` like
``api/app.py``.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
from collections.abc import Awaitable, Callable, Collection, Iterable, MutableMapping, Sequence
from typing import Any, NamedTuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from trelix.core.config import OPERATOR_ENV_FILE

logger = logging.getLogger("trelix.api")

# Names a browser uses for the loopback interface. Hostnames only — ports never count.
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")

# The single value of TRELIX_API_ALLOWED_HOSTS that switches the guard off.
ALLOW_ALL = "*"

# Fixed on purpose: it names the fix and never echoes the offending header value.
REJECTION_DETAIL = (
    "Request rejected: the Host or Origin header is not allowed for this trelix server. "
    "If the hostname is legitimate, add it to TRELIX_API_ALLOWED_HOSTS (comma-separated). "
    "Set TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)

CROSS_SITE_DETAIL = (
    "Request rejected: this trelix server refuses cross-site browser requests. Call it "
    "from a page served from the same host as the API, or set "
    "TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)

HEALTH_PATH = "/health"
_HEALTH_METHODS = frozenset({"GET", "HEAD"})
_ORIGIN_SCHEMES = frozenset({"http", "https"})
_CROSS_SITE = "cross-site"
_FORBIDDEN_HOST_CHARS = frozenset("@/\\?#%")
_HOST_LABEL = re.compile(r"[a-z0-9_-]+")
_MAX_PORT_DIGITS = 5
_MAX_LOGGED_CHARS = 100
_WS_POLICY_VIOLATION = 1008

_Scope = MutableMapping[str, Any]
_Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
_Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
_ASGIApp = Callable[[_Scope, _Receive, _Send], Awaitable[None]]


class RequestGuardSettings(BaseSettings):
    """``TRELIX_API_ALLOWED_HOSTS`` — kept apart from ``_ApiAuthSettings``.

    A plain ``str``: pydantic-settings would try to JSON-decode a ``list[str]`` field
    from the environment, and a comma-separated list is not JSON.
    """

    model_config = SettingsConfigDict(
        env_file=OPERATOR_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    allowed_hosts: str = Field(default="", alias="TRELIX_API_ALLOWED_HOSTS")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _is_plain_ascii(value: str) -> bool:
    """Printable, non-space ASCII only: rejects whitespace, controls, NUL and non-ASCII."""
    return all(0x20 < ord(char) < 0x7F for char in value)


def _is_port(text: str) -> bool:
    return text.isdigit() and len(text) <= _MAX_PORT_DIGITS


def _normalise_ipv6(text: str) -> str | None:
    try:
        return str(ipaddress.IPv6Address(text))
    except ValueError:
        return None


def _normalise_bracketed(value: str) -> str | None:
    end = value.find("]")
    if end == -1:
        return None
    rest = value[end + 1 :]
    if rest and not (rest.startswith(":") and _is_port(rest[1:])):
        return None
    return _normalise_ipv6(value[1:end])


def _normalise_name(name: str) -> str | None:
    lowered = name.lower()
    # ONE trailing dot ("localhost.") is the absolute form of the same name.
    trimmed = lowered[:-1] if lowered.endswith(".") else lowered
    if not trimmed:
        return None
    if not all(_HOST_LABEL.fullmatch(label) for label in trimmed.split(".")):
        return None
    return trimmed


def normalise_host(value: str | None) -> str | None:
    """Reduce a ``Host`` value (or allow-list entry) to a bare lowercase hostname.

    Ports are dropped, IPv6 brackets removed (``[::1]:8765`` -> ``::1``), case folded and
    one trailing dot ignored. Anything that is not a plain ``host[:port]`` returns
    ``None``: whitespace, control characters, NUL, userinfo (``@``), a path, query or
    fragment, a malformed bracket, an empty label.
    """
    if not value or not _is_plain_ascii(value) or _FORBIDDEN_HOST_CHARS & set(value):
        return None
    if value.startswith("["):
        return _normalise_bracketed(value)
    if value.count(":") > 1:
        # An unbracketed IPv6 literal: tolerated in config entries, never carries a port.
        return _normalise_ipv6(value)
    name, separator, port = value.partition(":")
    if separator and not _is_port(port):
        return None
    return _normalise_name(name)


def is_allowed_host(value: str | None, allowed: Collection[str]) -> bool:
    """True when ``value`` normalises to a hostname in ``allowed`` (already normalised)."""
    hostname = normalise_host(value)
    return hostname is not None and hostname in allowed


def is_allowed_origin(origin: str | None, allowed: Collection[str]) -> bool:
    """True for an ``http``/``https`` origin whose hostname (any port) is allowed.

    ``null`` (sandboxed iframes, ``file://``, redirects), other schemes, and anything with
    a path, query, fragment or userinfo are refused.
    """
    if not origin:
        return False
    scheme, separator, authority = origin.partition("://")
    if not separator or scheme.lower() not in _ORIGIN_SCHEMES:
        return False
    return is_allowed_host(authority, allowed)


def _bounded_repr(value: object) -> str:
    """``repr()`` cut to a fixed length: hostile header bytes can neither span lines nor flood."""
    return repr(value)[:_MAX_LOGGED_CHARS]


def _normalise_entries(entries: Iterable[str]) -> frozenset[str]:
    kept: set[str] = set()
    for entry in entries:
        hostname = normalise_host(entry)
        if hostname is None:
            logger.warning(
                "Ignoring invalid TRELIX_API_ALLOWED_HOSTS entry %s: give a bare hostname "
                "such as dev.example.com or [::1] (optional :port); URLs with a scheme and "
                "wildcards like *.example.com are not supported; * alone disables the guard",
                _bounded_repr(entry),
            )
            continue
        kept.add(hostname)
    return frozenset(kept)


def resolve_allowed_hosts(
    explicit: Sequence[str] | None, env_value: str | None
) -> frozenset[str] | None:
    """Decide whether the guard is on and with which hostnames; ``None`` means off.

    * ``env_value`` is exactly ``*``: off, whatever ``explicit`` says (explicit opt-out).
    * ``explicit`` given: on, with those hosts plus the env hosts.
    * ``explicit`` is None and the env lists at least one host: on, with the loopback
      names plus the env hosts (so an ASGI-factory deployment can opt in).
    * neither: off, which is the behavior before this guard existed.

    Entries are normalised like header values (ports, brackets, case, whitespace and
    empty entries tolerated); invalid ones are dropped with a warning rather than
    disabling the guard.
    """
    raw = (env_value or "").strip()
    if raw == ALLOW_ALL:
        return None
    env_entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if explicit is None and not env_entries:
        return None
    base: Sequence[str] = LOOPBACK_HOSTS if explicit is None else explicit
    return _normalise_entries([*base, *env_entries])


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def loopback_bind_allowed_hosts(bind_host: str) -> tuple[str, ...] | None:
    """Allowed hostnames for a loopback bind, or ``None`` when the bind is not loopback.

    Loopback is ``localhost``, ``::1`` or anything in ``127.0.0.0/8``. The result is the
    three loopback names plus the bind host when it is a more specific loopback address.
    """
    hostname = normalise_host(bind_host)
    if hostname is None or not _is_loopback(hostname):
        return None
    return LOOPBACK_HOSTS if hostname in LOOPBACK_HOSTS else (*LOOPBACK_HOSTS, hostname)


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------


def _header_values(scope: _Scope, name: bytes) -> list[str]:
    return [
        value.decode("latin-1") for key, value in scope.get("headers", ()) if key.lower() == name
    ]


def _route_path(scope: _Scope) -> str:
    """The path the router matches on.

    uvicorn (``--root-path``) and other servers put ``root_path`` in front of ``path``;
    Starlette strips it again before routing, so the health check must too. The rule is
    Starlette's: strip only on a whole-segment prefix, otherwise leave the path alone.
    """
    path = scope.get("path") or ""
    root = scope.get("root_path") or ""
    if not root or not path.startswith(root):
        return str(path)
    rest = path[len(root) :]
    return str(rest) if not rest or rest.startswith("/") else str(path)


def _is_health_probe(scope: _Scope) -> bool:
    """Exact equality on the route path: ``/health/``, ``/HEALTH``, ``//health`` are not it."""
    return (
        scope["type"] == "http"
        and scope.get("method") in _HEALTH_METHODS
        and _route_path(scope) == HEALTH_PATH
    )


class _Violation(NamedTuple):
    reason: str
    offending: object
    detail: str


class RequestGuardMiddleware:
    """Refuse foreign ``Host`` / ``Origin`` / cross-site fetches with a fixed 403."""

    def __init__(self, app: _ASGIApp, *, allowed_hosts: Collection[str]) -> None:
        self.app = app
        self._allowed_hosts = frozenset(allowed_hosts)

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope["type"] not in ("http", "websocket") or _is_health_probe(scope):
            await self.app(scope, receive, send)
            return
        violation = self._violation(scope)
        if violation is None:
            await self.app(scope, receive, send)
            return
        logger.warning(
            "Rejected %s %s: %s: %s",
            scope["type"],
            _bounded_repr(scope.get("path")),
            violation.reason,
            _bounded_repr(violation.offending),
        )
        await self._reject(scope, send, violation.detail)

    def _violation(self, scope: _Scope) -> _Violation | None:
        hosts = _header_values(scope, b"host")
        if len(hosts) != 1:
            return _Violation("Host header missing or repeated", hosts, REJECTION_DETAIL)
        if not is_allowed_host(hosts[0], self._allowed_hosts):
            return _Violation("Host header not allowed", hosts[0], REJECTION_DETAIL)

        origins = _header_values(scope, b"origin")
        if len(origins) > 1:
            return _Violation("Origin header repeated", origins, REJECTION_DETAIL)
        if origins and not is_allowed_origin(origins[0], self._allowed_hosts):
            return _Violation("Origin header not allowed", origins[0], REJECTION_DETAIL)

        for site in _header_values(scope, b"sec-fetch-site"):
            if site.strip().lower() == _CROSS_SITE:
                return _Violation("cross-site fetch", site, CROSS_SITE_DETAIL)
        return None

    @staticmethod
    async def _reject(scope: _Scope, send: _Send, detail: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
            return
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
