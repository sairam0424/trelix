"""Repository-root confinement shared by the REST API and trelix-mcp.

Authentication answers *who*; it says nothing about *where*. These two helpers answer
``where``: every caller-supplied repository path must resolve inside a root the operator
configured out-of-band (the path ``trelix serve`` was pointed at, or the
``TRELIX_ALLOWED_REPO_ROOTS`` list). With no root configured, nothing is reachable.

Moved verbatim from ``trelix.api.app`` so that ``trelix-mcp`` can apply the same rule to its
``repo_path`` arguments without importing the REST module; ``trelix.api.app`` re-exports
them under its old private names.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

# Env var holding extra allow-listed repository roots, os.pathsep-separated
# (":" on POSIX, ";" on Windows) — the same convention as PATH, so operators
# do not have to learn a trelix-specific separator.
#
# Read straight from os.environ rather than through a BaseSettings with
# env_file=".env" (as _ApiAuthSettings does) ON PURPOSE. A repo-local `.env` is
# already a live configuration source for this process, and the repositories
# this API serves are exactly the untrusted content an attacker can plant one
# in. A `.env` that could widen the containment allow-list would let the
# indexed material grant itself access to the rest of the host, which is the
# one place that amplification must not reach.
ALLOWED_ROOTS_ENV = "TRELIX_ALLOWED_REPO_ROOTS"


def resolve_allowed_roots(*explicit: str | Path | None) -> tuple[Path, ...]:
    """Canonicalize the allow-list once, at app construction.

    Each non-``None`` ``explicit`` root comes first, in order, then the entries of
    ``TRELIX_ALLOWED_REPO_ROOTS``. Resolving here rather than per-request is what
    makes the roots untrusted input's opposite: nothing a caller sends can extend
    this tuple. Both sides of the later comparison are resolved, which matters on
    macOS where ``/tmp`` is a symlink to ``/private/tmp`` — an unresolved root would
    reject every legitimate request under it.
    """
    candidates: list[Path] = [Path(root) for root in explicit if root is not None]
    candidates.extend(
        Path(entry)
        for entry in os.environ.get(ALLOWED_ROOTS_ENV, "").split(os.pathsep)
        if entry.strip()
    )
    # dict.fromkeys de-duplicates while preserving order; a new tuple is built
    # rather than mutating anything the caller handed in.
    return tuple(dict.fromkeys(p.expanduser().resolve() for p in candidates))


def is_within_allowed_roots(candidate: str | Path, allowed_roots: Sequence[Path]) -> bool:
    """True when ``candidate`` resolves inside one of ``allowed_roots``.

    ``is_relative_to`` on resolved paths, never ``str.startswith`` — the same
    property the per-route checks in the REST API already had, now applied to
    the root itself. A prefix match would accept a sibling ``<root>-evil`` that
    merely begins with the same characters. Equality is covered:
    ``Path("/a").is_relative_to(Path("/a"))`` is True.

    An empty allow-list returns False for everything. That is the whole point:
    the previous behavior — no root configured, therefore every absolute path
    on the host accepted — is the defect, not the compatible default.
    """
    resolved = Path(candidate).expanduser().resolve()
    return any(resolved.is_relative_to(root) for root in allowed_roots)
