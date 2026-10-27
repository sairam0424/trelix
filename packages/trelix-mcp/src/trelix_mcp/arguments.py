"""Checks on the arguments a tool receives, each answered as a `ToolError`.

A `ToolError` is this server's normal error result: `isError: true` with the message as the only
text block, and the session carries on. Every message here names the argument, shows what was given
(cut short), and says what would be valid, so an agent can correct the call without a traceback to
read. FastMCP already rejects an argument of the wrong type, or outside a `Literal`'s values, before
a tool runs; these checks cover what a JSON schema cannot say: blank text, a path that is not a
directory, a number that must be positive. A repository with no index is `server._require_index`.

Kept out of budget.py, which is about the size of a result, and out of server.py, which is already
very large and only calls in here.
"""

from __future__ import annotations

import math
from pathlib import Path

from fastmcp.exceptions import ToolError

# A value is echoed back so the caller recognises the call, not so it can be read in full; a
# repository path fits whole, a pasted file does not.
SHOWN_MAX_CHARS = 200
REPO_ROOT_HINT = "the absolute path of the repository root"


def shown(value: object) -> str:
    """`value` quoted for an error message, cut to SHOWN_MAX_CHARS characters."""
    text = str(value)
    if len(text) > SHOWN_MAX_CHARS:
        text = text[:SHOWN_MAX_CHARS] + "..."
    return f"'{text}'"


def check_text(value: str, name: str, valid: str) -> None:
    """A `ToolError` when `value` is empty or only whitespace.

    `valid` completes "pass ...": what the argument should hold, e.g. "the text to search for".
    """
    if value.strip():
        return
    raise ToolError(f"{name} must not be empty or whitespace (got {shown(value)}); pass {valid}.")


def check_repo_dir(repo_path: str, name: str = "repo_path") -> None:
    """A `ToolError` unless `repo_path` names an existing directory.

    The three cases are told apart because each has a different remedy: a blank value needs a
    path, a missing one needs the right path, and a file needs the directory above it.
    """
    check_text(repo_path, name, REPO_ROOT_HINT)
    _check_existing_dir(repo_path, name)


def _check_existing_dir(repo_path: str, name: str) -> None:
    """The exists/is-a-directory half of `check_repo_dir`, after the blank check."""
    path = Path(repo_path)
    try:
        exists, is_dir = path.exists(), path.is_dir()
    except OSError:
        # Python 3.12 and 3.13 raise here for a name the filesystem rejects (a component over
        # 255 bytes) or a parent that cannot be read; 3.14's pathlib answers False. Answering
        # False too keeps one message per version and the raw OSError, which carries the whole
        # path and names no argument, out of the client's hands.
        exists = is_dir = False
    if not exists:
        raise ToolError(f"{name} does not exist: {shown(repo_path)}; pass {REPO_ROOT_HINT}.")
    if not is_dir:
        raise ToolError(
            f"{name} is not a directory: {shown(repo_path)}; pass the repository root, not a "
            "file in it."
        )


def check_absolute_repo_dir(path: str, name: str) -> None:
    """`check_repo_dir`, and the path must also be absolute (a registry entry outlives the cwd).

    The blank check comes first so a blank path is reported as blank, not as relative.
    """
    check_text(path, name, REPO_ROOT_HINT)
    if not Path(path).is_absolute():
        raise ToolError(
            f"{name} must be an absolute path (got {shown(path)}); pass {REPO_ROOT_HINT}."
        )
    _check_existing_dir(path, name)


def check_positive_weight(weight: float) -> None:
    """A `ToolError` unless `weight` is a positive, finite number (RRF multiplies ranks by it)."""
    if weight > 0 and math.isfinite(weight):
        return
    raise ToolError(
        f"weight must be a positive number (got {weight}); 1.0 is the default, and a higher "
        "value ranks that repo's results higher."
    )


def check_session_id(session_id: str, hint: str) -> None:
    """A `ToolError` when a given session id is blank; `hint` says where a real one comes from."""
    check_text(session_id, "session_id", hint)
