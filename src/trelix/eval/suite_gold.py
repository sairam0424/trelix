"""Gold paths of a suite: in the repository at the pin, and in the set the run will index.

`trelix eval-validate` proves that a path exists in git. A query scores 0 in every arm when its
gold file exists in git and is still not indexed, and that looks exactly like a retrieval miss.
The default walker ignores a directory called `packages`, for one: the shipped golden file has
two queries whose gold files all live there. So after the clone every gold path must also be in
the set `FileWalker` yields with the configuration the run will use.

That configuration confines the walk to the clone (`walker.follow_symlinks = False`). The
default follows symlinks and does not resolve them, so a tracked symlink to a file outside the
clone would be read, hashed and indexed under a name that looks like it is inside. The walker
also opens `.gitignore` and `package.json` by name and reads them through a symlink wherever it
points, so a tracked symlink with either name, spelled in any case, is refused before the walk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.eval._loading import clip
from trelix.eval.golden_validate import RepoTree, validate_golden
from trelix.eval.harness import _parse_golden
from trelix.eval.suite import SuiteError, SuiteSpec
from trelix.eval.suite_git import tracked_files
from trelix.indexing.walker import FileWalker

_LISTED = 5

# The two files the walker opens by name, which `follow_symlinks = False` does not cover: a
# `.gitignore` in every directory it enters, and a `package.json` beside a `packages` or `bin`
# directory. Both are read through a symlink wherever it points. A tracked name is lower-cased
# before it is compared: the walker asks for `.gitignore` by its fixed name, and a
# case-insensitive filesystem (macOS, Windows) answers with a tracked `.GITIGNORE`.
_OPENED_BY_NAME = (".gitignore", "package.json")

# What the walker raises on a repository's own files, which it does not catch. A `.gitignore`
# line pathspec rejects is a ValueError (GitIgnorePatternError); one it compiles into an invalid
# regex such as `[z-a]` is a re.error, which is not; a `package.json` nested deeper than the
# JSON decoder can follow is a RecursionError, which is not a ValueError either.
_WALK_ERRORS = (ValueError, RecursionError, re.error)


@dataclass(frozen=True)
class GoldCounts:
    """How many golden queries, and how many distinct gold files, were checked."""

    queries: int
    files: int


def indexed_files(clone: Path) -> frozenset[str]:
    """The `rel_path` of every file a run would index in `clone`, walked with the run's config.

    Reads and hashes the files (the walk does) and embeds nothing. Raises SuiteError when the
    configuration cannot be built, when the clone's own `.gitignore` or `package.json` makes the
    walk raise, or when the walk could not read everything: a set that is missing files it could
    not read says nothing about the files it does not list.
    """
    try:
        base = IndexConfig(repo_path=str(clone))
    except ValueError as exc:  # pydantic's ValidationError is one
        raise SuiteError([f"cannot build the index configuration: {clip(str(exc), 200)}"]) from exc
    config = base.model_copy(
        update={"walker": base.walker.model_copy(update={"follow_symlinks": False})}
    )
    walker = FileWalker(config)
    try:
        files = frozenset(indexed.rel_path for indexed in walker.walk())
    except _WALK_ERRORS as exc:
        raise SuiteError(
            [f"the walker failed on the clone: {type(exc).__name__}: {clip(str(exc), 200)}"]
        ) from exc
    if not walker.walk_was_complete:
        unread = walker.incomplete_paths
        raise SuiteError(
            [
                f"the walker could not read {len(unread)} path(s) of the clone, so the set of "
                f"indexed files is incomplete: {', '.join(clip(p) for p in unread[:_LISTED])}"
            ]
        )
    return files


def check_gold(spec: SuiteSpec, clone: Path) -> GoldCounts:
    """Refuse unless every gold path exists at the pin and would be indexed; else count them.

    Raises SuiteError with every reason. Paths that do not exist at the pin are reported by
    `validate_golden` with their line; what is left over is a path the walker does not yield,
    reported for the first five lines with a count of the rest. A tracked symlink named like a file
    the walker opens by name is refused before the walk starts.
    """
    tracked = tracked_files(clone, spec.repo_sha)
    tree = RepoTree(spec.repo_sha, tracked)
    report = validate_golden(spec.golden_file, tree=tree, min_per_stratum=0, min_validated=0.0)
    if report.violations:
        raise SuiteError([f"golden: {violation}" for violation in report.violations])
    entries = _parse_golden(spec.golden_file)
    _refuse_links_opened_by_name(clone, tracked)
    indexed = indexed_files(clone)
    unindexed = [
        (entry.line_no, path)
        for entry in entries
        for path in sorted(entry.relevant_files)
        if path not in indexed
    ]
    if unindexed:
        raise SuiteError(_unindexed_problems(unindexed))
    return GoldCounts(
        queries=len(entries),
        files=len({path for entry in entries for path in entry.relevant_files}),
    )


def _refuse_links_opened_by_name(clone: Path, tracked: frozenset[str]) -> None:
    """Raise SuiteError for a tracked symlink named `.gitignore` or `package.json`, at any depth.

    The walker reads these two wherever a symlink points, outside the clone included. A
    `.gitignore` whose target is a regular file is read whole and its lines shape the walk (a
    line of an operator-side file can reach a refusal line); a `package.json` is read with no
    check of what it is, so a link to `/dev/zero` or a FIFO is never read to its end. The name is
    compared in lower case: a case-insensitive filesystem resolves the fixed name `.gitignore`
    to a tracked `.GITIGNORE`. So this runs before the walk, from the tracked paths, and names
    the paths only: one line, up to five of them, and the count.
    """
    links = sorted(
        path
        for path in tracked
        if path.rpartition("/")[2].lower() in _OPENED_BY_NAME and (clone / path).is_symlink()
    )
    if not links:
        return
    listed = ", ".join(clip(path) for path in links[:_LISTED])
    raise SuiteError(
        [
            f"the clone tracks {len(links)} symlink(s) named .gitignore or package.json, which "
            f"the walker opens by name and reads wherever they point: {listed}"
        ]
    )


def _unindexed_problems(unindexed: list[tuple[int, str]]) -> list[str]:
    problems = [
        f"golden: line {line_no}: {clip(repr(path), 120)} is in the repository but the walker "
        "does not index it (an ignored directory, extension or language, or too large)"
        for line_no, path in unindexed[:_LISTED]
    ]
    if len(unindexed) > _LISTED:
        problems.append(f"golden: and {len(unindexed) - _LISTED} more gold paths are not indexed")
    return problems
