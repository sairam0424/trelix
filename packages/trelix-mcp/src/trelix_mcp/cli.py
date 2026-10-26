"""The `trelix-mcp` command line: `_build_parser` and `main`.

`trelix_mcp.server.main` — the `[project.scripts]` target — calls `main()` here, so the argv
parsing and start-up live outside the server module while the console script keeps its entry
point. Every server-owned name is reached through the module (`server.mcp`,
`server._install_confinement`, `server.mcp.run(...)`), never imported by name: a test that
patches `trelix_mcp.server.mcp.run` and calls `server.main()` must see its patch, and a by-name
import would freeze the unpatched object here. `server.py`'s `logging.basicConfig` runs on
import, before this module's first log line.

`--transport http` serves Streamable HTTP through `trelix_mcp.http` (loopback by default, the REST
Host/Origin guard always on, a bearer token when set, unexpected tool errors masked, a root
required); stdio, the default, is unchanged. Every configuration error is `parser.error` (exit 2).
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

import trelix_mcp.server as server
from trelix.core.confinement import resolve_allowed_roots
from trelix_mcp import __version__
from trelix_mcp.budget import BudgetConfigError, limits_from_env
from trelix_mcp.http import HttpConfigError, resolve_http_settings, run_kwargs, url_host
from trelix_mcp.tool_metadata import TOOL_PROFILES, apply_tool_profile

_log = logging.getLogger("trelix_mcp")

# E5: an HTTP-only flag on the stdio transport; checked before the limits and the roots.
HTTP_ONLY_FLAGS_ERROR = (
    "--host, --port, --path, --allowed-host, --stateful and --allow-unauthenticated apply to "
    "--transport http only"
)


def _build_parser() -> argparse.ArgumentParser:
    """The `trelix-mcp` argument parser: `--version`, `--tools`, `--root` and the HTTP flags."""
    parser = argparse.ArgumentParser(
        prog="trelix-mcp",
        description="MCP server for trelix — semantic code search over stdio or Streamable HTTP.",
    )
    parser.add_argument("--version", action="version", version=f"trelix-mcp {__version__}")
    parser.add_argument(
        "--tools",
        choices=TOOL_PROFILES,
        default="full",
        help="tool profile: 'full' (default) lists every tool, 'core' lists only the everyday "
        "search and indexing tools and hides the rest",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="PATH",
        help="a repository root every repo_path, federation path and trelix://repo/... URI must "
        "lie inside; repeatable, and TRELIX_ALLOWED_REPO_ROOTS (os.pathsep-separated) adds more. "
        "With neither, nothing is confined",
    )
    _add_http_bind_flags(parser)
    _add_http_policy_flags(parser)
    return parser


def _add_http_bind_flags(parser: argparse.ArgumentParser) -> None:
    """`--transport` and the three bind flags; their `None` default is how `_reject_http_only_flags`
    tells "not given" from a value, and `resolve_http_settings` fills in the real defaults."""
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="transport to serve: stdio (default; the MCP client spawns this process) or http "
        "(Streamable HTTP; needs --root)",
    )
    parser.add_argument(
        "--host",
        default=None,
        metavar="HOST",
        help="address to bind, default 127.0.0.1; a non-loopback address needs a token or "
        "--allow-unauthenticated, and a port here is refused: pass it with --port "
        "(--transport http only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="port to bind, 1-65535, default 8766 (--transport http only)",
    )
    parser.add_argument(
        "--path",
        default=None,
        metavar="PATH",
        help="URL path of the MCP endpoint, default /mcp; must start with / and cannot be /health "
        "(--transport http only)",
    )


def _add_http_policy_flags(parser: argparse.ArgumentParser) -> None:
    """`--allowed-host`, `--stateful` and `--allow-unauthenticated`."""
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="a hostname the Host/Origin guard accepts besides localhost, 127.0.0.1, ::1 and the "
        "bind address; repeatable, and TRELIX_MCP_ALLOWED_HOSTS (comma-separated; fallback "
        "TRELIX_API_ALLOWED_HOSTS) adds more; only the variable's * turns the guard off, here it "
        "is refused (--transport http only)",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        help="keep per-client sessions (needed by clients that open a server-initiated GET "
        "stream); default stateless (--transport http only)",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="serve a non-loopback --host with no token in TRELIX_MCP_AUTH_TOKEN (fallback "
        "TRELIX_API_AUTH_TOKEN); without it that is a start-up error (--transport http only)",
    )


def _reject_http_only_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """E5: an HTTP-only flag on stdio is an error before the limits or the roots are read."""
    given = any(value is not None for value in (args.host, args.port, args.path))
    switched = bool(args.allowed_host) or args.stateful or args.allow_unauthenticated
    if args.transport == "stdio" and (given or switched):
        parser.error(HTTP_ONLY_FLAGS_ERROR)


def _start_http(
    parser: argparse.ArgumentParser, args: argparse.Namespace, roots: tuple[Path, ...]
) -> None:
    """Resolve the HTTP settings (every error is exit 2), mask unexpected tool errors and serve."""
    try:
        settings = resolve_http_settings(
            host=args.host,
            port=args.port,
            path=args.path,
            allowed_hosts=args.allowed_host,
            stateful=args.stateful,
            allow_unauthenticated=args.allow_unauthenticated,
            roots=roots,
            environ=os.environ,
        )
    except HttpConfigError as exc:
        parser.error(str(exc))
    # D-4: over HTTP an exception a tool did not expect answers `Error calling tool '<name>'` (no
    # path, no traceback) while a ToolError text passes unchanged; stdio keeps its unmasked texts.
    # FastMCP's private attribute: test_fastmcp_mask_attribute_canary watches the name.
    server.mcp._mask_error_details = True
    # An IPv6 bind reads as a URL authority: `[::1]:8766/mcp`, not `::1:8766/mcp`.
    _log.info(
        "trelix-mcp starting (transport=http, %s:%d%s, stateless=%s, roots=%d)",
        url_host(settings.host),
        settings.port,
        settings.path,
        settings.stateless,
        len(roots),
    )
    server.mcp.run(**run_kwargs(settings))


def main() -> None:
    """Entry point for the trelix-mcp server.

    Parses argv and rejects unknown flags — the normal path (no args, launched by an MCP client's
    server config) falls straight through to running the stdio server with every tool and no
    confinement, unchanged from before this parser existed. The checks run in this order so that
    the first error reported is the one the operator can act on: an HTTP-only flag on stdio (E5),
    a blank `--root`, an unusable `TRELIX_MCP_*` limit, then the HTTP settings.
    """
    parser = _build_parser()
    args = parser.parse_args()
    _reject_http_only_flags(parser, args)
    if any(not root.strip() for root in args.root):
        parser.error("--root must not be blank")
    try:
        limits_from_env()
    except BudgetConfigError as exc:
        parser.error(str(exc))
    apply_tool_profile(server.mcp, args.tools)
    roots = resolve_allowed_roots(*args.root)
    server._install_confinement(roots)

    def _handle_sigterm(signum: int, frame: Any) -> None:
        _log.info("Received SIGTERM — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    if args.transport == "http":
        _start_http(parser, args, roots)
        return
    _log.info("trelix-mcp starting (transport=stdio)")
    server.mcp.run(transport="stdio")
