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
            result = subprocess.run(
                ["git", "diff", base, head, "--unified=3"],
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=30,
            )
            if result.returncode != 0:
                logger.debug("git diff failed: %s", result.stderr[:200])
                return []
            return self.parse(result.stdout)
        except Exception as exc:
            logger.debug("DiffParser.from_git failed: %s", exc)
            return []
