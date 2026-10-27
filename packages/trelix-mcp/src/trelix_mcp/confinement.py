"""Repository-root confinement for trelix-mcp: which paths a caller may name at all.

`server.py` installs `RepoConfinementMiddleware` when `trelix-mcp --root PATH` or
`TRELIX_ALLOWED_REPO_ROOTS` names at least one root (see `server._install_confinement`). From then
on every `tools/call` whose arguments carry a string `repo_path`, and `federation_add_repo`'s
`path`, must resolve inside one of the roots or the call is refused before the tool runs. The rule
is `trelix.core.confinement.is_within_allowed_roots`, the one the REST API applies: both sides
resolved, `is_relative_to`, so `..`, a symlink inside a root that points outside, a sibling
`<root>-evil`, a blank value (which `Path("")` turns into the server's working directory) and a
value the filesystem cannot resolve at all (a NUL byte, a symlink loop) are all refused. The
refusal names the field and nothing else; the value and the roots go to the operator log only.

`entries_inside_roots` is the same rule for the federation registry, which `federation_search_all`
reads from a file rather than from the caller. A non-string `repo_path` is left alone here:
FastMCP's argument validation answers it, as it did before any confinement existed.

The `trelix://repo/...` resources are confined in `server.py`, where FastMCP has already parsed the
URI into `repo_path`; a second URI parser here could only disagree with FastMCP's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp.types import CallToolRequestParams

from trelix.core.confinement import is_within_allowed_roots
from trelix.federation.registry import RepoEntry

# The whole refusal: the field name and nothing the caller sent (a path outside the roots is
# exactly what must not be echoed, and the roots themselves are the operator's business).
REFUSAL = "{field} is not inside an allowed repository root"
# A refused value is logged for the operator, cut short so a pasted file cannot flood the log.
LOGGED_VALUE_CHARS = 100

_log = logging.getLogger("trelix_mcp.confinement")


def _confined_fields(name: str, arguments: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The (field, value) pairs of `arguments` the roots apply to.

    `repo_path` on any tool, plus `path` on `federation_add_repo` (the one tool that names a
    repository under another name). A value that is not a string is not listed: FastMCP's own
    validation rejects it with the message it always had.
    """
    candidates = [("repo_path", arguments.get("repo_path"))]
    if name == "federation_add_repo":
        candidates.append(("path", arguments.get("path")))
    return [(field, value) for field, value in candidates if isinstance(value, str)]


def inside_roots(value: str, roots: Sequence[Path]) -> bool:
    """`is_within_allowed_roots`, answering False for a value that cannot be resolved at all.

    `Path.resolve()` raises instead of answering for a NUL byte (ValueError), for a symlink loop on
    Python 3.12 (RuntimeError) and for a relative value when the working directory no longer exists
    (OSError). None of those is inside a root, and the exception escaping would hand the client
    `Internal server error` with a traceback in the log instead of the refusal.
    """
    try:
        return is_within_allowed_roots(value, roots)
    except (ValueError, OSError, RuntimeError):
        return False


class RepoConfinementMiddleware(Middleware):
    """Refuse a tool call that names a repository outside the allowed roots."""

    def __init__(self, roots: Sequence[Path]) -> None:
        self._roots = tuple(roots)

    def _refuse_if_outside(self, field: str, value: str) -> None:
        """A `ToolError` unless `value` is non-blank and resolves inside a root."""
        if value.strip() and inside_roots(value, self._roots):
            return
        _log.warning(
            "Refused %s=%r: outside the allowed repository roots %s",
            field,
            value[:LOGGED_VALUE_CHARS],
            [str(root) for root in self._roots],
        )
        raise ToolError(REFUSAL.format(field=field))

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        arguments = context.message.arguments or {}
        for field, value in _confined_fields(context.message.name, arguments):
            self._refuse_if_outside(field, value)
        return await call_next(context)


def entries_inside_roots(
    entries: Sequence[RepoEntry], roots: Sequence[Path]
) -> tuple[list[RepoEntry], int]:
    """`(kept, outside)`: the registry entries inside `roots`, and how many were not.

    With no roots every entry is kept and none is read: that is the stdio default, and it keeps
    `federation_search_all` byte-identical to its unconfined self.
    """
    if not roots:
        return list(entries), 0
    kept = [entry for entry in entries if inside_roots(entry.path, roots)]
    return kept, len(entries) - len(kept)
