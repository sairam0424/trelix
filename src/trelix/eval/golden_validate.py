"""
`trelix eval-validate`: check a golden file without running a single query.

What is checked, and why:

* every line is schema-valid: the v1 `query` / `relevant_files` rules of the loader
  (`harness._parse_golden`, whose path rule is reused here), and the v2 fields of
  `trelix.eval.golden` when present;
* no two queries are the same once stripped and case-folded, and no two `id`s are equal: a
  repeated query is counted twice in every mean;
* with a repository, every `relevant_files` path exists in it at a revision: a stale path
  scores 0 and looks exactly like a retrieval miss, which `trelix eval` cannot tell apart;
* in a file where some entry has a `kind` or a `gold_status` (a v2 file), every kind present
  has enough queries for a per-kind mean to mean something, and enough entries have a
  reviewed `gold_status`.

A file where no entry has a `kind` or a `gold_status` is checked as v1: the first three only.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any

from trelix.eval.golden import KINDS, REVIEWED_GOLD_STATUSES, validate_entry
from trelix.eval.harness import _path_problem
from trelix.review.diff_parser import _run_git


class RepoTreeError(Exception):
    """git could not list the files of the repository at the revision."""


@dataclass(frozen=True)
class RepoTree:
    """The files of a repository at one revision, as paths relative to the repository directory."""

    rev: str
    files: frozenset[str]


@dataclass(frozen=True)
class _Entry:
    """One golden line that parsed as a JSON object. `line_no` is 1-based, like an editor."""

    line_no: int
    item: dict[str, Any]


# A violation tied to a line of the file: (1-based line number, the text to print).
_LineViolation = tuple[int, str]


@dataclass(frozen=True)
class ValidationReport:
    """Everything `trelix eval-validate` found, ready to print."""

    entries: int
    violations: tuple[str, ...]
    notes: tuple[str, ...]

    def lines(self) -> list[str]:
        """Violations, then notes, then a one-line summary, each already a line of output."""
        verdict = "invalid" if self.violations else "valid"
        summary = f"{verdict}: entries {self.entries}, violations {len(self.violations)}"
        return [*self.violations, *(f"note: {n}" for n in self.notes), summary]


def read_repo_tree(repo: str, rev: str) -> RepoTree:
    """List the files of `repo` at `rev` with `git ls-tree`; raise `RepoTreeError` when git cannot.

    Paths are relative to `repo`, the way `FileWalker` builds `rel_path`, so a `repo` that is a
    subdirectory of a larger checkout lists only its own subtree. `-z` keeps unusual file names
    unquoted. A missing directory, a directory that is no repository and an unknown `rev` each
    end as a `RepoTreeError` carrying git's own first line of explanation. A `rev` that starts
    with `-` is refused before git runs: git would read it as an option, and no revision name
    can start with `-`.
    """
    if not Path(repo).is_dir():
        raise RepoTreeError(f"repository {repo!r} is not a directory")
    if rev.startswith("-"):
        raise RepoTreeError(f"revision {rev!r} starts with '-', which git would read as an option")
    try:
        result = _run_git(repo, "ls-tree", "-r", "--name-only", "-z", rev)
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
        raise RepoTreeError(f"git ls-tree failed: {type(exc).__name__}: {exc}") from exc
    if result.returncode != 0:
        reason = next(iter(result.stderr.strip().splitlines()), "no error output")
        raise RepoTreeError(f"cannot list the files of {repo!r} at {rev!r}: {reason}")
    return RepoTree(rev, frozenset(name for name in result.stdout.split("\0") if name))


def validate_golden(
    path: Path,
    *,
    tree: RepoTree | None,
    min_per_stratum: int,
    min_validated: float,
) -> ValidationReport:
    """Check the golden file at `path`; `tree=None` skips the path-exists check."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ValidationReport(
            0, (f"file: not valid UTF-8 ({exc.reason} at byte {exc.start})",), ()
        )

    entries, line_violations = _parse_lines(text)
    line_violations.extend(_line_problems(entries, tree))
    violations = _in_line_order(line_violations)
    notes: list[str] = []
    if not entries:
        violations.append("file: no golden entries")
        return ValidationReport(0, tuple(violations), ())

    if tree is None:
        notes.append("--repo was not given, so relevant_files paths were not checked")
    if _is_v2(entries):
        violations.extend(_stratum_problems(entries, min_per_stratum))
        violations.extend(_validated_share_problems(entries, min_validated))
    else:
        notes.append(
            "no entry has a kind or a gold_status, so this file is checked as v1: only the "
            "schema, duplicate and (with --repo) path checks ran"
        )
    return ValidationReport(len(entries), tuple(violations), tuple(notes))


def _is_v2(entries: list[_Entry]) -> bool:
    """Whether some entry has a `kind` or a `gold_status`, the two fields the v2 checks read.

    A key counts when present, whatever its value: a `kind` of `null` is a wrong-typed `kind`.
    `id`, `lang`, `source` and `split` are not read by a v2 check and do not make a v2 file.
    """
    return any("kind" in e.item or "gold_status" in e.item for e in entries)


def _parse_lines(text: str) -> tuple[list[_Entry], list[_LineViolation]]:
    """The lines that are JSON objects, and a violation for each line that is not."""
    entries: list[_Entry] = []
    violations: list[_LineViolation] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"not valid JSON ({exc.msg} at column {exc.colno})"
            violations.append((line_no, f"line {line_no}: {message}"))
            continue
        if not isinstance(item, dict):
            message = f"expected a JSON object, got {type(item).__name__}"
            violations.append((line_no, f"line {line_no}: {message}"))
            continue
        entries.append(_Entry(line_no, item))
    return entries, violations


def _line_problems(entries: list[_Entry], tree: RepoTree | None) -> list[_LineViolation]:
    """Schema, duplicate and missing-path problems, each prefixed with its line, in file order."""
    first_query = _first_lines(entries, _query_key)
    first_id = _first_lines(entries, _id_key)
    problems: list[_LineViolation] = []
    for entry in entries:
        found = _schema_problems(entry.item)
        found.extend(_duplicate_problems(entry, first_query, first_id))
        if tree is not None:
            found.extend(_missing_path_problems(entry.item, tree))
        problems.extend((entry.line_no, f"line {entry.line_no}: {p}") for p in found)
    return problems


def _in_line_order(found: list[_LineViolation]) -> list[str]:
    """The texts of `found` by line number; the problems of one line keep their order."""
    return [text for _, text in sorted(found, key=itemgetter(0))]


def _query_key(item: dict[str, Any]) -> str | None:
    query = item.get("query")
    return query.strip().casefold() if isinstance(query, str) and query.strip() else None


def _id_key(item: dict[str, Any]) -> str | None:
    entry_id = item.get("id")
    return entry_id if isinstance(entry_id, str) and entry_id.strip() else None


def _first_lines(
    entries: list[_Entry], key_of: Callable[[dict[str, Any]], str | None]
) -> dict[str, int]:
    """Map each key `key_of` yields to the first line that has it."""
    first: dict[str, int] = {}
    for entry in entries:
        key = key_of(entry.item)
        if key is not None:
            first.setdefault(key, entry.line_no)
    return first


def _schema_problems(item: dict[str, Any]) -> list[str]:
    """What the loader would refuse (`_parse_golden`), plus the v2 fields, all at once."""
    problems: list[str] = []
    query = item.get("query")
    if not isinstance(query, str) or not query.strip():
        problems.append('"query" must be a non-empty string')
    relevant = item.get("relevant_files")
    if not isinstance(relevant, list) or not relevant:
        problems.append('"relevant_files" must be a non-empty list of repo-relative POSIX paths')
    else:
        problems.extend(f'"relevant_files" {p}' for p in map(_path_problem, relevant) if p)
    problems.extend(validate_entry(item))
    return problems


def _duplicate_problems(
    entry: _Entry, first_query: dict[str, int], first_id: dict[str, int]
) -> list[str]:
    """A repeat of an earlier line's query or id; the earlier line is not reported."""
    problems: list[str] = []
    query_key = _query_key(entry.item)
    if query_key is not None and first_query[query_key] != entry.line_no:
        problems.append(
            f"duplicate query (same as line {first_query[query_key]} once stripped and case-folded)"
        )
    id_key = _id_key(entry.item)
    if id_key is not None and first_id[id_key] != entry.line_no:
        problems.append(f"duplicate id {id_key!r} (first used on line {first_id[id_key]})")
    return problems


def _missing_path_problems(item: dict[str, Any], tree: RepoTree) -> list[str]:
    """Each well-formed `relevant_files` path that `tree` does not contain."""
    relevant = item.get("relevant_files")
    if not isinstance(relevant, list):
        return []
    return [
        f'"relevant_files" path {p!r} does not exist at {tree.rev}'
        for p in relevant
        if isinstance(p, str) and _path_problem(p) is None and p not in tree.files
    ]


def _stratum_problems(entries: list[_Entry], min_per_stratum: int) -> list[str]:
    """Each kind that is present but has fewer than `min_per_stratum` queries.

    Only the kinds of `KINDS` form strata; an entry with any other `kind` is already a schema
    violation and is in no stratum.
    """
    counts = Counter(e.item["kind"] for e in entries if isinstance(e.item.get("kind"), str))
    return [
        f"file: kind {kind!r} has {counts[kind]} {'query' if counts[kind] == 1 else 'queries'}, "
        f"fewer than the {min_per_stratum} required per kind (--min-per-stratum)"
        for kind in KINDS
        if 0 < counts[kind] < min_per_stratum
    ]


def _validated_share_problems(entries: list[_Entry], min_validated: float) -> list[str]:
    """The share of reviewed entries when it is below `min_validated`; no status is unreviewed."""
    reviewed = sum(1 for e in entries if e.item.get("gold_status") in REVIEWED_GOLD_STATUSES)
    share = reviewed / len(entries)
    if share >= min_validated:
        return []
    return [
        f"file: {reviewed} of {len(entries)} entries have a gold_status of "
        f"{' or '.join(REVIEWED_GOLD_STATUSES)} ({share:.4f}); --min-validated requires "
        f"{min_validated}"
    ]
