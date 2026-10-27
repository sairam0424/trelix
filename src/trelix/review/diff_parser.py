"""
Unified diff parser for PR/diff review.

Parses `git diff` output (unified diff format) into structured DiffHunk objects.
Each hunk captures the file path, line numbers, added/removed lines, and context.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger("trelix.review.diff_parser")

_GIT_TIMEOUT_SECONDS = 30

_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class DiffHunk:
    """A single changed block in a unified diff."""

    file_path: str
    old_start: int
    new_start: int
    old_lines: int
    new_lines: int
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    def to_search_query(self) -> str:
        """Build a retrieval query from this hunk's changed lines."""
        lines = self.added + self.removed
        # Use identifiers extracted from changed lines as the query
        identifiers = []
        for line in lines[:10]:
            words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", line)
            identifiers.extend(words[:5])
        unique = list(dict.fromkeys(identifiers))[:15]
        base = f"changes in {self.file_path}"
        if unique:
            base += " — " + " ".join(unique)
        return base


class GitDiffError(RuntimeError):
    """`git diff` produced no diff: it failed, timed out or printed text that is not UTF-8.

    An empty diff is not this error; that is a diff git produced and found to have no changes.
    """


def _run_git(repo_path: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run `git *args` in `repo_path` with captured text output and the module time limit."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=repo_path,
        timeout=_GIT_TIMEOUT_SECONDS,
    )


class DiffParser:
    """Parse unified diff text into DiffHunk objects."""

    def parse(self, diff_text: str) -> list[DiffHunk]:
        """Parse unified diff text and return list of DiffHunk objects."""
        if not diff_text.strip():
            return []

        hunks: list[DiffHunk] = []
        current_file = ""
        current_hunk: DiffHunk | None = None

        for line in diff_text.splitlines():
            # New file
            m = _FILE_RE.match(line)
            if m:
                current_file = m.group(1)
                continue

            # New hunk header
            m = _HUNK_RE.match(line)
            if m:
                if current_hunk is not None:
                    hunks.append(current_hunk)
                current_hunk = DiffHunk(
                    file_path=current_file,
                    old_start=int(m.group(1)),
                    new_start=int(m.group(3)),
                    old_lines=int(m.group(2)) if m.group(2) else 1,
                    new_lines=int(m.group(4)) if m.group(4) else 1,
                )
                continue

            if current_hunk is None:
                continue

            if line.startswith("+") and not line.startswith("+++"):
                current_hunk.added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk.removed.append(line[1:])
            elif line.startswith(" "):
                current_hunk.context.append(line[1:])

        if current_hunk is not None and (current_hunk.added or current_hunk.removed):
            hunks.append(current_hunk)

        return hunks

    def from_git(
        self,
        repo_path: str,
        base: str = "HEAD~1",
        head: str = "HEAD",
    ) -> list[DiffHunk]:
        """Run git diff and parse the output. Returns [] on any failure."""
        try:
            return self.parse(self.git_diff(repo_path, base, head))
        except Exception as exc:
            logger.debug("DiffParser.from_git failed: %s", exc)
            return []

    def git_diff(self, repo_path: str, base: str = "HEAD~1", head: str = "HEAD") -> str:
        """Return the text of `git diff base head`; raise GitDiffError when git cannot produce it.

        Unlike `from_git`, a failure is not confused with an empty diff. The trailing `--` ends
        the revisions: without it git refuses a ref that is also the name of a file or directory
        ("ambiguous argument 'docs': both revision and filename"), for example a branch `docs`
        in a repository that has a `docs/` directory.
        """
        try:
            result = _run_git(repo_path, "diff", base, head, "--unified=3", "--")
        except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
            raise GitDiffError(f"{type(exc).__name__}: {exc}") from exc
        if result.returncode != 0:
            reason = next(iter(result.stderr.strip().splitlines()), "no error output")
            raise GitDiffError(f"exit {result.returncode}: {reason}")
        return result.stdout

    def ref_resolves(self, repo_path: str, ref: str) -> bool:
        """True when `ref` names an object that exists in the repository; False on any failure.

        `from_git` cannot tell a missing ref from an empty diff (both come back as `[]`), so the
        caller asks here first. Two steps: `rev-parse --verify` understands every revision syntax
        (`:/text`, `rev:path`, `HEAD~1`), but it passes a well-formed 40-digit id whether or not
        the object exists; so the id it prints is then looked up with `cat-file -e`, which fails
        for a missing object (a base SHA that was never fetched). The type is not checked, because
        `git diff` compares commits, tags and trees as well as a pair of blobs
        (`HEAD~1:f HEAD:f`). A range never gets past `--verify`.
        """
        try:
            found = _run_git(repo_path, "rev-parse", "--verify", ref)
            if found.returncode != 0:
                return False
            exists = _run_git(repo_path, "cat-file", "-e", found.stdout.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("DiffParser.ref_resolves failed: %s", exc)
            return False
        return exists.returncode == 0
