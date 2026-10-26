"""`trelix-mcp --transport http` at the command line, through `trelix_mcp.server.main()`.

These tests pin what reaches `mcp.run`, which flags and environments are exit 2 with which text,
and the start-up log lines; requests over the built app are test_http_transport.py's and the stdio
argv contract (`--help`, `--version`, an unknown flag, a blank `--root`) is test_server_cli.py's.
Same style as that file: patch `sys.argv` and `mcp.run`, so whichever branch runs, the test sees
what the server would have served.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import trelix_mcp.server as server_module
from trelix_mcp.http import BearerTokenMiddleware

from trelix.api.request_guard import RequestGuardMiddleware

LOGGER = "trelix_mcp"
TOKEN_ENV = "TRELIX_MCP_AUTH_TOKEN"
E1 = (
    "--transport http needs at least one --root or TRELIX_ALLOWED_REPO_ROOTS; every tool call "
    "names a repository path and paths outside these roots are refused"
)
E2 = (
    "--host 0.0.0.0 is not a loopback address "  # noqa: S104
    "and no token is set; set TRELIX_MCP_AUTH_TOKEN (or TRELIX_API_AUTH_TOKEN), or pass "
    "--allow-unauthenticated to serve without one"
)
E5 = (
    "--host, --port, --path, --allowed-host, --stateful and --allow-unauthenticated apply to "
    "--transport http only"
)
W1 = (
    "Serving MCP over HTTP on 0.0.0.0:8766 "  # noqa: S104
    "with no token: every tool is open to anyone who can reach this port (--allow-unauthenticated)"
)
W4_STATELESS = "trelix-mcp starting (transport=http, 127.0.0.1:8766/mcp, stateless=True, roots=1)"
W4_STATEFUL = "trelix-mcp starting (transport=http, 127.0.0.1:8766/mcp, stateless=False, roots=1)"
W4_IPV6 = "trelix-mcp starting (transport=http, [::1]:8766/mcp, stateless=True, roots=1)"


def _plain(text: str) -> str:
    """argparse may colour its usage text on a terminal; the assertions read the words."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _http_argv(repo: Path, *extra: str) -> list[str]:
    return ["trelix-mcp", "--transport", "http", "--root", str(repo), *extra]


@pytest.mark.parametrize(
    ("extra", "stateless", "log"),
    [([], True, W4_STATELESS), (["--stateful"], False, W4_STATEFUL)],
    ids=["stateless", "stateful"],
)
def test_http_flags_reach_run(
    extra: list[str],
    stateless: bool,
    log: str,
    tmp_path: Path,
    mark_indexed: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The kwargs `mcp.run` receives are `run_kwargs(settings)`, nothing dropped or renamed, and
    the error mask is already set when `run` is called (the real `run` blocks until shutdown).

    The SIGTERM handler is installed before `run` too, so `docker stop` ends the HTTP server by the
    handler's clean `sys.exit(0)`, not the default signal action. A stand-in `signal` module keeps
    the test off the process's real handlers.

    MUTATIONS: `**http_app_kwargs(settings)` instead of `**run_kwargs(settings)` (host and port
    missing); `stateful=not args.stateful` (the stateful row fails); the mask assignment moved
    below the `run(...)` call; the `signal.signal(...)` call moved below the HTTP branch.
    """
    repo = mark_indexed(tmp_path / "repo")
    argv = ["--host", "127.0.0.1", "--port", "8766", "--path", "/mcp", *extra]
    monkeypatch.setattr("sys.argv", _http_argv(repo, *argv))
    monkeypatch.setenv(TOKEN_ENV, "t0k")
    mask_when_run: list[bool] = []
    events: list[str] = []

    def _record_mask(**_: object) -> None:
        events.append("run")
        mask_when_run.append(server_module.mcp._mask_error_details)

    mock_run = MagicMock(side_effect=_record_mask)
    monkeypatch.setattr(server_module.mcp, "run", mock_run)
    fake_signal = MagicMock()
    fake_signal.signal.side_effect = lambda *_: events.append("sigterm")
    monkeypatch.setattr("trelix_mcp.cli.signal", fake_signal)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        server_module.main()

    mock_run.assert_called_once()
    assert events == ["sigterm", "run"]
    assert fake_signal.signal.call_args.args[0] is fake_signal.SIGTERM
    assert fake_signal.signal.call_args.args[1].__name__ == "_handle_sigterm"
    assert mask_when_run == [True]
    kwargs = mock_run.call_args.kwargs
    assert set(kwargs) == {
        "transport",
        "host",
        "port",
        "path",
        "stateless_http",
        "json_response",
        "host_origin_protection",
        "middleware",
    }
    assert kwargs["transport"] == "http"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8766
    assert kwargs["path"] == "/mcp"
    assert kwargs["stateless_http"] is stateless
    assert kwargs["host_origin_protection"] is False
    assert [(m.cls, m.kwargs) for m in kwargs["middleware"]] == [
        (RequestGuardMiddleware, {"allowed_hosts": frozenset({"localhost", "127.0.0.1", "::1"})}),
        (BearerTokenMiddleware, {"token": "t0k"}),
    ]
    assert log in caplog.text


def test_ipv6_bind_is_bracketed_in_the_start_line(
    tmp_path: Path,
    mark_indexed: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`--host [::1]` binds the bare `::1` (what `mcp.run` receives) and logs the authority form
    `[::1]:8766/mcp`, not `::1:8766/mcp`. MUTATION: log `settings.host` instead of
    `url_host(settings.host)`."""
    repo = mark_indexed(tmp_path / "repo")
    monkeypatch.setattr("sys.argv", _http_argv(repo, "--host", "[::1]"))
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        server_module.main()

    assert mock_run.call_args.kwargs["host"] == "::1"
    assert W4_IPV6 in caplog.text


@pytest.mark.parametrize(
    "flag",
    [
        ["--port", "9000"],
        ["--port", "0"],
        ["--host", "127.0.0.1"],
        ["--path", "/x"],
        ["--allowed-host", "a.example"],
        ["--stateful"],
        ["--allow-unauthenticated"],
    ],
    ids=["port", "port-zero", "host", "path", "allowed-host", "stateful", "allow-unauthenticated"],
)
def test_http_only_flags_are_rejected_on_stdio(
    flag: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """E5: an HTTP-only flag without `--transport http` is a usage error, not silently ignored.

    MUTATIONS: delete the `_reject_http_only_flags` call in `main()`; `any(value for ...)` instead
    of `value is not None` in `_reject_http_only_flags` (the falsy `--port 0` row passes through).
    """
    monkeypatch.setattr("sys.argv", ["trelix-mcp", *flag])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    assert E5 in _plain(capsys.readouterr().err)


def test_http_only_flag_error_wins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """E5 is reported before an env limit or the port range: it runs right after `parse_args`.

    MUTATION: move the `_reject_http_only_flags` call after `limits_from_env()`.
    """
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--port", "70000"])
    monkeypatch.setenv("TRELIX_MCP_MAX_K", "many")
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    err = _plain(capsys.readouterr().err)
    assert E5 in err
    assert "TRELIX_MCP_MAX_K" not in err
    assert "--port must be between" not in err


def test_an_unknown_transport_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--transport` takes `stdio` or `http` only; anything else is argparse's exit 2, not a
    silent stdio start (the HTTP branch is `args.transport == "http"`, so a stray value would fall
    through to stdio).

    MUTATIONS: add "sse" to the `--transport` `choices`; delete the `choices` argument.
    """
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--transport", "sse"])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    assert "invalid choice: 'sse'" in _plain(capsys.readouterr().err)


@pytest.mark.parametrize(
    ("extra_argv", "env", "expected"),
    [
        (["--port", "70000"], {}, "--port must be between 1 and 65535, got 70000"),
        (["--path", "health"], {}, "--path must start with / and cannot be /health, got 'health'"),
        (
            ["--path", "/health"],
            {},
            "--path must start with / and cannot be /health, got '/health'",
        ),
        ([], {TOKEN_ENV: "tök"}, "the token in TRELIX_MCP_AUTH_TOKEN must be printable ASCII"),
        (
            ["--host", "not a host!"],
            {},
            "--host must be a hostname or IP address, optionally in brackets, got 'not a host!'",
        ),
        (
            ["--host", "example.com:9000"],
            {},
            "--host must be a hostname or IP address, optionally in brackets, got "
            "'example.com:9000'",
        ),
        (
            ["--allowed-host", "mcp.internal,mcp.example"],
            {},
            "--allowed-host must be one hostname or IP address, optionally in brackets (repeat "
            "the flag for more), got 'mcp.internal,mcp.example'",
        ),
        # `main()` reads the TRELIX_MCP_* limits before the transport branch, so HTTP refuses an
        # unusable one like stdio does. MUTATION: wrap `limits_from_env()` in `if args.transport
        # == "stdio":` (the server would start with a limit nobody can use).
        (
            [],
            {"TRELIX_MCP_MAX_K": "many"},
            "TRELIX_MCP_MAX_K must be an integer of at least 1, got 'many'",
        ),
    ],
    ids=[
        "port",
        "path-relative",
        "path-health",
        "token-non-ascii",
        "host",
        "host-with-port",
        "allowed-host-comma-list",
        "env-limit",
    ],
)
def test_http_config_errors_exit_2(
    extra_argv: list[str],
    env: dict[str, str],
    expected: str,
    tmp_path: Path,
    mark_indexed: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every `HttpConfigError` is `parser.error`: exit 2 with the resolver's text, `run` not called;
    so is an unusable `TRELIX_MCP_*` limit, which `main()` checks first.

    The texts themselves are pinned in test_http_settings.py. MUTATION: drop the
    `except HttpConfigError` clause in `_start_http` (the exception propagates instead).
    """
    repo = mark_indexed(tmp_path / "repo")
    monkeypatch.setattr("sys.argv", _http_argv(repo, *extra_argv))
    for name, value in (env or {TOKEN_ENV: "t0k"}).items():
        monkeypatch.setenv(name, value)
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    assert expected in _plain(capsys.readouterr().err)


def test_http_without_roots_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-1: HTTP with no `--root` and no TRELIX_ALLOWED_REPO_ROOTS is E1, not a server that
    refuses every call.

    MUTATION: pass `roots=()` to `resolve_http_settings` (the resolver sees no roots either way,
    so the real roots must be what is passed).
    """
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--transport", "http"])
    monkeypatch.setenv(TOKEN_ENV, "t0k")
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    assert E1 in _plain(capsys.readouterr().err)


def test_http_roots_may_come_from_the_environment_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The container route: `--transport http` with no `--root` serves when
    TRELIX_ALLOWED_REPO_ROOTS names the root, and the start line counts it.

    MUTATIONS: pass `roots=tuple(Path(r) for r in args.root)` to `resolve_http_settings` in
    `_start_http` (E1 exits 2 although the environment supplies the root); log
    `len(args.root)` instead of `len(roots)` (the start line says `roots=0`).
    """
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--transport", "http"])
    monkeypatch.setenv(TOKEN_ENV, "t0k")
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(tmp_path))
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        server_module.main()

    mock_run.assert_called_once()
    assert W4_STATELESS in caplog.text


def test_nonloopback_requires_token(
    tmp_path: Path,
    mark_indexed: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`--host 0.0.0.0` with no token is E2; `--allow-unauthenticated` serves with the guard only
    and W1 in the log (the loopback case is test_http_settings.py's).

    MUTATION: pass `allow_unauthenticated=True` to `resolve_http_settings` unconditionally.
    """
    repo = mark_indexed(tmp_path / "repo")
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)
    monkeypatch.setattr("sys.argv", _http_argv(repo, "--host", "0.0.0.0"))  # noqa: S104

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    assert E2 in _plain(capsys.readouterr().err)

    argv = _http_argv(repo, "--host", "0.0.0.0", "--allow-unauthenticated")  # noqa: S104
    monkeypatch.setattr("sys.argv", argv)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        server_module.main()
    mock_run.assert_called_once()
    middleware = mock_run.call_args.kwargs["middleware"]
    assert [m.cls for m in middleware] == [RequestGuardMiddleware]
    assert W1 in caplog.text


def test_stdio_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-4: the error mask is HTTP-only; a stdio start leaves FastMCP's default in place.

    MUTATION: set `server.mcp._mask_error_details = True` in `main()` instead of `_start_http`.
    """
    monkeypatch.setattr("sys.argv", ["trelix-mcp"])
    monkeypatch.setattr(server_module.mcp, "run", MagicMock())

    server_module.main()

    assert server_module.mcp._mask_error_details is False
