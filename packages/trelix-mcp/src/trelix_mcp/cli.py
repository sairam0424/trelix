"""The `trelix-mcp` command line: `_build_parser` and `main`.

`trelix_mcp.server.main` — the `[project.scripts]` target — calls `main()` here, so the argv
parsing and start-up live outside the server module while the console script keeps its entry
point. Every server-owned name is reached through the module (`server.mcp`,
`server._install_confinement`, `server.mcp.run(...)`), never imported by name: a test that
patches `trelix_mcp.server.mcp.run` and calls `server.main()` must see its patch, and a by-name
import would freeze the unpatched object here. `server.py`'s `logging.basicConfig` runs on
import, before this module's first log line.
"""

import argparse
import logging
import signal
import sys
from typing import Any

import trelix_mcp.server as server
from trelix.core.confinement import resolve_allowed_roots
from trelix_mcp import __version__
from trelix_mcp.budget import BudgetConfigError, limits_from_env
from trelix_mcp.tool_metadata import TOOL_PROFILES, apply_tool_profile

_log = logging.getLogger("trelix_mcp")


def _build_parser() -> argparse.ArgumentParser:
    """The `trelix-mcp` argument parser: `--version`, `--tools` and `--root`."""
    parser = argparse.ArgumentParser(
        prog="trelix-mcp",
        description="MCP server for trelix — semantic code search over stdio.",
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
    return parser


def main() -> None:
    """Entry point for the trelix-mcp server (stdio transport).

    Parses argv for --help/--version/--tools/--root and rejects unknown flags — the normal path
    (no args, launched by an MCP client's server config) falls straight through to running
    the server with every tool and no confinement, unchanged from before this parser existed.
    """
    parser = _build_parser()
    args = parser.parse_args()
    if any(not root.strip() for root in args.root):
        parser.error("--root must not be blank")
    try:
        limits_from_env()
    except BudgetConfigError as exc:
        parser.error(str(exc))
    apply_tool_profile(server.mcp, args.tools)
    server._install_confinement(resolve_allowed_roots(*args.root))

    def _handle_sigterm(signum: int, frame: Any) -> None:
        _log.info("Received SIGTERM — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    _log.info("trelix-mcp starting (transport=stdio)")
    server.mcp.run(transport="stdio")
