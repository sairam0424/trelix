"""`trelix_mcp.http` settings: what `resolve_http_settings` returns, raises and logs.

The module is not wired into `server.py` yet (a later change adds `--transport http`), so these
tests drive it directly and none touches `trelix_mcp.server`. Every expected value is a literal
written here: the error and log texts are copied in, never imported. This package's tests are
governed by packages/trelix-mcp/pyproject.toml (no socket ban, no timeout), so the markers below
supply both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from trelix_mcp.http import (
    BearerTokenMiddleware,
    HttpConfigError,
    HttpSettings,
    asgi_middleware,
    first_env,
    http_app_kwargs,
    resolve_http_settings,
    run_kwargs,
)

from trelix.api.request_guard import RequestGuardMiddleware

pytestmark = [pytest.mark.disable_socket, pytest.mark.timeout(60)]

LOGGER = "trelix_mcp.http"
MCP_AUTH = "TRELIX_MCP_AUTH_TOKEN"
API_AUTH = "TRELIX_API_AUTH_TOKEN"
MCP_HOSTS = "TRELIX_MCP_ALLOWED_HOSTS"
API_HOSTS = "TRELIX_API_ALLOWED_HOSTS"
TOKEN_SET = {MCP_AUTH: "t"}
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
FULL = LOOPBACK | {"10.0.0.7", "mcp.internal", "a.example"}
E1 = (
    "--transport http needs at least one --root or TRELIX_ALLOWED_REPO_ROOTS; every tool call "
    "names a repository path and paths outside these roots are refused"
)
E2 = (
    "--host {} is not a loopback address and no token is set; set TRELIX_MCP_AUTH_TOKEN "
    "(or TRELIX_API_AUTH_TOKEN), or pass --allow-unauthenticated to serve without one"
)
E3_HIGH = "--port must be between 1 and 65535, got 70000"
E7_MCP = "the token in TRELIX_MCP_AUTH_TOKEN must be printable ASCII"
E7_API = "the token in TRELIX_API_AUTH_TOKEN must be printable ASCII"
E8 = "--host must be a hostname or IP address, optionally in brackets, got 'not a host!'"
W1 = (
    "Serving MCP over HTTP on 0.0.0.0:8766 with no token: every tool is open to anyone who can "
    "reach this port (--allow-unauthenticated)"
)
W2 = "Host/Origin guard is OFF (allowed hosts set to *)"
W3_DEFAULT = "Host/Origin guard is ON (allowed hosts: 127.0.0.1, ::1, localhost)"


def _settings(tmp_path: Path, **overrides: Any) -> HttpSettings:
    """`resolve_http_settings` with the shipped defaults and `(tmp_path,)` as the roots."""
    defaults: dict[str, Any] = {
        "host": None,
        "port": None,
        "path": None,
        "allowed_hosts": (),
        "stateful": False,
        "allow_unauthenticated": False,
        "roots": (tmp_path,),
        "environ": {},
    }
    return resolve_http_settings(**{**defaults, **overrides})


def _logged(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    """(level, message) of this module's records only: caplog holds every logger's."""
    return [(r.levelname, r.getMessage()) for r in caplog.records if r.name == LOGGER]


def test_first_env_returns_the_first_non_blank_value_unmodified() -> None:
    assert first_env({"A": "  ", "B": " api "}, "A", "B") == ("B", " api ")
    assert first_env({"A": ""}, "A", "B") is None
    assert first_env({}, "A") is None


def test_defaults_and_loopback_open(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        settings = _settings(tmp_path)
    assert settings == HttpSettings(
        host="127.0.0.1", port=8766, path="/mcp", stateless=True, guard_hosts=LOOPBACK, token=None
    )
    assert len(asgi_middleware(settings)) == 1
    assert _logged(caplog) == [("INFO", W3_DEFAULT)]


@pytest.mark.parametrize(
    ("host", "allowed_hosts", "environ", "expected"),
    [
        ("10.0.0.7", ["mcp.internal"], {MCP_HOSTS: "a.example", **TOKEN_SET}, FULL),
        # An unspecified bind is never a Host a client legitimately sends (D-2).
        ("0.0.0.0", [], TOKEN_SET, LOOPBACK),  # noqa: S104
        ("::", [], TOKEN_SET, LOOPBACK),
        ("[::]", [], TOKEN_SET, LOOPBACK),
        ("mcp.internal", [], {API_AUTH: "k"}, LOOPBACK | {"mcp.internal"}),
        ("127.0.0.5", [], {}, LOOPBACK | {"127.0.0.5"}),
        ("LOCALHOST", [], {}, LOOPBACK),
        # Python's ipaddress rejects the shorthand, so 127.1 is a hostname to this check.
        ("127.1", [], TOKEN_SET, LOOPBACK | {"127.1"}),
        # TRELIX_API_ALLOWED_HOSTS is read only when the MCP variable is unset or blank (B12).
        (None, [], {MCP_HOSTS: "", API_HOSTS: "b.example"}, LOOPBACK | {"b.example"}),
        (None, [], {MCP_HOSTS: "b.example", API_HOSTS: "*"}, LOOPBACK | {"b.example"}),
        (None, [], {MCP_HOSTS: "a.example", API_HOSTS: "b.example"}, LOOPBACK | {"a.example"}),
        (None, [], {MCP_HOSTS: "  ", API_HOSTS: "*"}, None),
    ],
)
def test_allowed_host_resolution(
    tmp_path: Path,
    host: str | None,
    allowed_hosts: list[str],
    environ: dict[str, str],
    expected: frozenset[str] | None,
) -> None:
    settings = _settings(tmp_path, host=host, allowed_hosts=allowed_hosts, environ=environ)
    assert settings.guard_hosts == expected


def test_guard_off_when_the_env_says_star(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        settings = _settings(tmp_path, host="127.0.0.1", environ={MCP_HOSTS: "*", **TOKEN_SET})
    assert settings.guard_hosts is None
    assert _logged(caplog) == [("WARNING", W2)]
    assert [m.cls for m in asgi_middleware(settings)] == [BearerTokenMiddleware]


def test_bind_host_and_token_shape_the_middleware_list(tmp_path: Path) -> None:
    settings = _settings(tmp_path, host="mcp.internal", environ={API_AUTH: "k"})
    assert settings.token == "k"
    assert [m.cls for m in asgi_middleware(settings)] == [
        RequestGuardMiddleware,
        BearerTokenMiddleware,
    ]
    settings = _settings(tmp_path, host="127.0.0.5")
    assert settings.token is None
    assert [m.cls for m in asgi_middleware(settings)] == [RequestGuardMiddleware]


@pytest.mark.parametrize(
    ("environ", "token"),
    [
        ({MCP_AUTH: "  "}, None),
        ({MCP_AUTH: "", API_AUTH: "api"}, "api"),
        # The value is kept as given, spaces included (B3; REST's rule).
        ({MCP_AUTH: "  ", API_AUTH: " api "}, " api "),
        ({MCP_AUTH: "mcp", API_AUTH: "api"}, "mcp"),
    ],
)
def test_token_resolution(tmp_path: Path, environ: dict[str, str], token: str | None) -> None:
    settings = _settings(tmp_path, environ=environ)
    assert settings.token == token
    assert len(asgi_middleware(settings)) == (1 if token is None else 2)


@pytest.mark.parametrize(
    ("environ", "message"),
    [({MCP_AUTH: "tök"}, E7_MCP), ({MCP_AUTH: "", API_AUTH: "a\tb"}, E7_API)],
)
def test_non_ascii_token_is_a_startup_error(
    tmp_path: Path, environ: dict[str, str], message: str
) -> None:
    with pytest.raises(HttpConfigError) as info:
        _settings(tmp_path, environ=environ)
    assert str(info.value) == message


def test_nonloopback_requires_token(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(HttpConfigError) as info:
        _settings(tmp_path, host="0.0.0.0")  # noqa: S104
    assert str(info.value) == E2.format("0.0.0.0")  # noqa: S104
    with pytest.raises(HttpConfigError) as info:
        _settings(tmp_path, host="127.1")
    assert str(info.value) == E2.format("127.1")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        settings = _settings(tmp_path, host="0.0.0.0", allow_unauthenticated=True)  # noqa: S104
    assert settings.token is None
    assert [m.cls for m in asgi_middleware(settings)] == [RequestGuardMiddleware]
    assert [msg for level, msg in _logged(caplog) if level == "WARNING"] == [W1]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        settings = _settings(tmp_path, host="::1")
    assert settings.token is None
    assert [msg for level, msg in _logged(caplog) if level == "WARNING"] == []
    # A loopback bind never warns, flag or not: W1 is about exposure and nothing is exposed.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        settings = _settings(tmp_path, host="127.0.0.1", allow_unauthenticated=True)
    assert settings.token is None
    assert [msg for level, msg in _logged(caplog) if level == "WARNING"] == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"port": 70000}, E3_HIGH),
        ({"port": 0}, "--port must be between 1 and 65535, got 0"),
        ({"path": "health"}, "--path must start with / and cannot be /health, got 'health'"),
        ({"path": "/health"}, "--path must start with / and cannot be /health, got '/health'"),
        ({"roots": ()}, E1),
        ({"host": "not a host!", "environ": TOKEN_SET}, E8),
        # The port is checked before the roots.
        ({"port": 70000, "roots": ()}, E3_HIGH),
    ],
)
def test_port_path_roots_and_host_are_validated(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(HttpConfigError) as info:
        _settings(tmp_path, **overrides)
    assert str(info.value) == message


def test_port_bounds_are_inclusive(tmp_path: Path) -> None:
    assert _settings(tmp_path, port=1).port == 1
    assert _settings(tmp_path, port=65535).port == 65535


def test_run_kwargs_are_explicit(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path, host="127.0.0.1", port=8766, path="/mcp", environ={MCP_AUTH: "t0k"}
    )
    kwargs = run_kwargs(settings)
    assert {k: v for k, v in kwargs.items() if k != "middleware"} == {
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8766,
        "path": "/mcp",
        "stateless_http": True,
        "json_response": False,
        "host_origin_protection": False,
    }
    assert [(m.cls, m.kwargs) for m in kwargs["middleware"]] == [
        (RequestGuardMiddleware, {"allowed_hosts": frozenset({"localhost", "127.0.0.1", "::1"})}),
        (BearerTokenMiddleware, {"token": "t0k"}),
    ]
    assert http_app_kwargs(_settings(tmp_path, stateful=True))["stateless_http"] is False
