"""
Git-log ticket linker — walks commit history to link code symbols to
external ticket references (e.g. Jira "PROJ-123", GitHub "#456") found in
commit messages, feeding trelix's generic_edges cross-source graph.

This is entirely new, git-aware code — trelix has no other persisted
git/commit ingestion anywhere. src/trelix/review/diff_parser.py's
DiffParser.from_git() does transient `git diff` parsing for the `trelix
review` CLI command only; nothing there is stored, and no commit
messages/SHAs are ever kept. This module is the first thing in trelix that
actually reads and persists commit history.

Shells out to the `git` CLI (mirroring diff_parser.py's subprocess pattern)
rather than adding a GitPython dependency — keeps trelix's zero-git-dependency
pyproject.toml status quo intact.

Granularity: file-level, not line-range-precise. Every symbol in a file
touched by a ticket-referencing commit is credited with that reference —
a coarse graph signal, not a precise diff-blame. This is an accepted
precision/cost tradeoff, not an oversight.
"""

from __future__ import annotations

import logging
import re
import subprocess

from trelix.core.config import GitLinkerConfig
from trelix.core.models import GenericEdge
from trelix.store.db import Database

logger = logging.getLogger("trelix.indexing.git_linker")

# Separates commits in `git log`'s custom --pretty format below. Chosen to be
# extremely unlikely to appear in a real commit subject/body.
_COMMIT_SEP = "\x1e"
_FIELD_SEP = "\x1f"


class GitLinker:
    """
    Walks `git log` for *repo_path*, regex-matches ticket IDs in commit
    messages, maps each commit's changed files to their indexed symbols, and
    inserts one GenericEdge per (symbol, ticket) pair found.

    Never raises — any git failure (not a git repo, git not installed, a
    shallow clone with no history, a timeout) degrades to "0 edges linked",
    matching DiffParser.from_git()'s existing failure posture exactly.
    """

    def __init__(self, db: Database, config: GitLinkerConfig | None = None) -> None:
        self._db = db
        self._config = config or GitLinkerConfig()
        self._ticket_re = re.compile(self._config.ticket_pattern)

    def link(self, repo_path: str) -> int:
        """
        Run the full walk-log -> match-tickets -> map-to-symbols -> insert
        pipeline. Returns the number of GenericEdges inserted (0 on any
        failure or when no ticket references are found).
        """
        commits = self._walk_log(repo_path)
        if not commits:
            return 0

        edges: list[GenericEdge] = []
        seen: set[tuple[int, str]] = set()  # de-dupe (symbol_id, ticket) pairs
        for ticket_ids, changed_files in commits:
            if not ticket_ids:
                continue
            for rel_path in changed_files:
                symbol_ids = self._db.get_symbol_ids_for_file(rel_path)
                for symbol_id in symbol_ids:
                    for ticket_id in ticket_ids:
                        key = (symbol_id, ticket_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        edges.append(
                            GenericEdge(
                                from_symbol_id=symbol_id,
                                source_ref=f"ticket:{ticket_id}",
                                edge_kind="references_ticket",
                            )
                        )

        if edges:
            self._db.insert_generic_edges(edges)
        logger.info(
            "GitLinker: linked %d symbol-ticket edges from %d commits", len(edges), len(commits)
        )
        return len(edges)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk_log(self, repo_path: str) -> list[tuple[list[str], list[str]]]:
        """
        Return a list of (ticket_ids, changed_files) per commit.

        Never raises: a non-git directory, missing `git` binary, a timeout,
        or any other failure all degrade to an empty list — matching
        DiffParser.from_git()'s existing style exactly.
        """
        cmd = [
            "git",
            "log",
            f"--max-count={self._config.max_commits}",
            "--name-only",
            f"--pretty=format:{_COMMIT_SEP}%s{_FIELD_SEP}%b{_FIELD_SEP}",
        ]
        if self._config.since:
            cmd.append(f"--since={self._config.since}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=60,
            )
            if result.returncode != 0:
                logger.debug("git log failed: %s", result.stderr[:200])
                return []
        except Exception as exc:
            logger.debug("GitLinker._walk_log failed: %s", exc)
            return []

        return self._parse_log_output(result.stdout)

    def _parse_log_output(self, raw: str) -> list[tuple[list[str], list[str]]]:
        """Parse `git log`'s custom-delimited output into per-commit records."""
        commits: list[tuple[list[str], list[str]]] = []
        # First chunk before the first _COMMIT_SEP is empty (format starts
        # each commit with the separator) — skip it.
        for block in raw.split(_COMMIT_SEP)[1:]:
            parts = block.split(_FIELD_SEP)
            if len(parts) < 2:
                continue
            subject, body = parts[0], parts[1]
            # Everything after the second _FIELD_SEP is the --name-only file
            # list, one path per line (with a leading blank line from the
            # format string's trailing separator).
            files_blob = _FIELD_SEP.join(parts[2:]) if len(parts) > 2 else ""
            changed_files = [line.strip() for line in files_blob.splitlines() if line.strip()]

            ticket_ids = self._extract_ticket_ids(f"{subject}\n{body}")
            commits.append((ticket_ids, changed_files))
        return commits

    def _extract_ticket_ids(self, text: str) -> list[str]:
        """De-duplicated, order-preserving list of ticket IDs matched in *text*."""
        return list(dict.fromkeys(self._ticket_re.findall(text)))
