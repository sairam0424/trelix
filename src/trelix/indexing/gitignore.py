"""
Shared nested-`.gitignore` matching, extracted from `FileWalker`.

`FileWalker._spec_for_dir` / `_gitignore_chain` / `_is_gitignored` only ever
touched three pieces of instance state: `self.repo_root`, `self.config.walker
.respect_gitignore`, and `self._spec_cache` (a plain dict). Lifting their
bodies into free functions here -- taking that state as explicit parameters
instead -- lets any other local-file connector (e.g. ImageConnector) apply
the exact same nested-`.gitignore` semantics FileWalker uses, without pulling
in FileWalker's Language/extension allow-list machinery. FileWalker's three
methods are now one-line wrappers around these; see their docstrings there
for the two behaviors this preserves byte for byte (per-anchor-relative
pattern matching, and "proximity wins" for nested overrides).

Callers that do not want a persistent cache (e.g. a one-shot discovery walk
that will not be re-run) should pass a fresh `{}` per call; FileWalker
instead keeps its own `_spec_cache` alive across the whole walk.
"""

from __future__ import annotations

from pathlib import Path

import pathspec


def spec_for_dir(
    directory: Path,
    cache: dict[Path, pathspec.PathSpec | None],  # type: ignore[type-arg]
) -> pathspec.PathSpec | None:  # type: ignore[type-arg]
    """Parse and cache the `.gitignore` sitting directly inside `directory`."""
    if directory in cache:
        return cache[directory]

    spec: pathspec.PathSpec | None = None  # type: ignore[type-arg]
    gitignore_path = directory / ".gitignore"
    try:
        if gitignore_path.is_file():
            patterns = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            spec = pathspec.PathSpec.from_lines("gitignore", patterns)
    except OSError:
        # Unreadable .gitignore (permissions, races during a watch). Treating it as
        # absent keeps the walk going; the alternative is aborting the whole
        # discovery pass over one file we were not able to read.
        spec = None

    cache[directory] = spec
    return spec


def gitignore_chain(
    repo_root: Path,
    path: Path,
    cache: dict[Path, pathspec.PathSpec | None],  # type: ignore[type-arg]
) -> list[tuple[Path, pathspec.PathSpec]]:  # type: ignore[type-arg]
    """`(anchor_dir, spec)` pairs governing `path`, ordered shallowest -> deepest.

    Every `.gitignore` from `repo_root` down to `path`'s own directory,
    skipping directories that do not contain one.
    """
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        # Outside the repo entirely — no .gitignore of ours has authority over it.
        return []

    # repo_root first, then each intermediate directory down to path's parent.
    directories = [repo_root]
    current = repo_root
    for part in rel.parts[:-1]:
        current = current / part
        directories.append(current)

    chain: list[tuple[Path, pathspec.PathSpec]] = []  # type: ignore[type-arg]
    for directory in directories:
        spec = spec_for_dir(directory, cache)
        if spec is not None:
            chain.append((directory, spec))
    return chain


def is_path_gitignored(
    repo_root: Path,
    path: Path,
    *,
    is_dir: bool,
    respect_gitignore: bool,
    cache: dict[Path, pathspec.PathSpec | None] | None = None,  # type: ignore[type-arg]
) -> bool:
    """Apply the full nested-`.gitignore` chain to `path`.

    Two details make this match git rather than merely approximate it:

    1. Each `.gitignore`'s patterns are matched against the path *relative to that
       file's own directory*. Anchored patterns (`/rooted.py`) and directory patterns
       (`harness/`) are meaningless otherwise — a repo-root-relative path would make
       `/rooted.py` in `sub/.gitignore` silently match nothing.
    2. Proximity wins. Walking shallowest → deepest and letting each *explicit*
       verdict overwrite the previous one means a deeper `!keep.log` re-includes a
       file its parent excluded, while an unmentioned path (`include is None`) leaves
       the inherited verdict untouched.

    `cache` defaults to a fresh, call-scoped dict when omitted — pass one
    explicitly (and reuse it across calls) to avoid re-parsing the same
    `.gitignore` for every path in a discovery loop.
    """
    if not respect_gitignore:
        return False
    if cache is None:
        cache = {}

    ignored = False
    for anchor, spec in gitignore_chain(repo_root, path, cache):
        rel = path.relative_to(anchor).as_posix()
        if is_dir:
            rel += "/"
        verdict = spec.check_file(rel).include
        if verdict is not None:
            ignored = verdict
    return ignored
