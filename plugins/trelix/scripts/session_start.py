# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""One line of facts about this repository's trelix index, for Claude at session start.

Claude Code runs this script as the plugin's SessionStart hook (`hooks/hooks.json`) and
hands its stdout to the model as context. The pinned trelix-mcp release sends no server
instructions and answers a search on an unindexed repository with an empty result, so this
line is the only place the model learns whether `search_code` can answer at all, how old the
index is, and which absolute `repo_path` to pass.

Rules this file keeps, each pinned by tests/unit/test_claude_plugin_session_start.py:

- Standard library only, nothing from `trelix`: `uv run --no-project --script` runs it in a
  bare interpreter, before the server has installed anything.
- Read-only. The index is opened through a `mode=ro` URI (the form `trelix.store.db` uses),
  so a missing file is not created and a stray write would fail; nothing is written under the
  project. (SQLite may leave `index.db-wal`/`-shm` beside an existing WAL index, as every
  read-only open does; when `.trelix/` is not writable and they are absent it cannot open the
  index at all, and the hook prints nothing.) `trelix.core.config.IndexConfig` is deliberately
  not used: its `db_path_absolute` creates `.trelix/`.
- Never blocks a session: one `git rev-list` child with a 5 s timeout, no network, exit
  status 0 on every path. An error (a stdin the interpreter cannot decode included) prints
  nothing on stdout and one `trelix session_start: skipped (<reason>)` line on stderr, which
  Claude Code keeps in its debug log only.
- At most 1,000 characters on stdout (Claude Code's own cap is 10,000).

Inputs: `CLAUDE_PROJECT_DIR` (set by Claude Code for hook processes) or, when that is unset
or does not name a directory, the hook's stdin JSON `cwd`. No flags, no settings of its own.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.request import pathname2url

MAX_OUTPUT_CHARS = 1000
GIT_TIMEOUT_SECONDS = 5.0
# StoreConfig.db_path's default (src/trelix/core/config.py). An operator who relocated the
# index with TRELIX_STORE_DB_PATH gets the "no index" line; the hook reads no trelix settings.
INDEX_RELATIVE_PATH = Path(".trelix") / "index.db"
PROVENANCE_PREFIX = "provenance."  # src/trelix/store/provenance.py `_PREFIX`
STDIN_LIMIT_BYTES = 65_536
_HEX_DIGITS = frozenset("0123456789abcdef")
# What a stdin that is not a JSON object with a string `cwd` raises on the way to `cwd`.
_NOT_A_HOOK_PAYLOAD = (json.JSONDecodeError, TypeError, AttributeError)


@dataclass(frozen=True)
class IndexFacts:
    files: int
    symbols: int
    git_commit: str | None
    indexed_at: str | None
    embedder: str | None


def _existing_directory(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


def project_root(stdin_text: str, env: Mapping[str, str]) -> Path | None:
    """`CLAUDE_PROJECT_DIR` when it names an existing directory, else stdin's `cwd`, else None.

    The env var wins because it stays put while `cwd` follows Claude into worktrees; a set but
    unusable value falls through to `cwd`. The path is returned as given (no `resolve()`, no
    symlink following) so that the printed `repo_path` is the one Claude Code itself uses.
    """
    from_env = _existing_directory(env.get("CLAUDE_PROJECT_DIR"))
    if from_env is not None:
        return from_env
    try:
        cwd = json.loads(stdin_text).get("cwd")
    except _NOT_A_HOOK_PAYLOAD:
        return None
    return _existing_directory(cwd)


def read_only_uri(db_path: Path) -> str:
    """`file:<path>?mode=ro`, percent-encoded so a `#` or `?` in the path cannot end the URI early
    and drop `mode=ro` (the same reason `trelix.store.db.read_only_uri` gives)."""
    return f"file:{pathname2url(str(db_path))}?mode=ro"


def _provenance(conn: sqlite3.Connection) -> dict[str, str]:
    """The `provenance.*` rows with the prefix stripped; `{}` when the table does not exist.

    Only that one error is tolerated (an index written before provenance existed); any other
    `sqlite3.Error` propagates, and the entry point prints nothing.
    """
    try:
        rows = conn.execute(
            "SELECT key, value FROM index_metadata WHERE key LIKE 'provenance.%'"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if str(exc).startswith("no such table: index_metadata"):
            return {}
        raise
    return {str(key)[len(PROVENANCE_PREFIX) :]: str(value) for key, value in rows}


def read_index_facts(db_path: Path) -> IndexFacts:
    """Row counts of `files` and `symbols`, and the provenance an index run recorded.

    `embedder` is the provider name only (`provenance.embedder_provider`); the model name is
    never printed.
    """
    conn = sqlite3.connect(read_only_uri(db_path), uri=True)
    try:
        files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        symbols = int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])
        provenance = _provenance(conn)
    finally:
        conn.close()
    return IndexFacts(
        files=files,
        symbols=symbols,
        git_commit=provenance.get("git_commit"),
        indexed_at=provenance.get("indexed_at"),
        embedder=provenance.get("embedder_provider"),
    )


def _looks_like_commit(commit: str) -> bool:
    """`^[0-9a-f]{7,64}$`: the only shape that may enter git's argv."""
    return 7 <= len(commit) <= 64 and set(commit) <= _HEX_DIGITS


def commits_ahead(repo: Path, commit: str | None) -> int | None:
    """How many commits HEAD is ahead of `commit`; None whenever git cannot say.

    `git rev-list --count <commit>..HEAD` is 0 when HEAD is an ancestor of the indexed commit
    too (an older checkout), so 0 means "nothing newer than the index", not "same commit".
    """
    if commit is None or not _looks_like_commit(commit):
        return None
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-list", "--count", f"{commit}..HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _distance(ahead: int | None) -> str:
    if ahead is None:
        return "distance from HEAD unknown"
    if ahead == 0:
        return "= HEAD"
    if ahead == 1:
        return "HEAD is 1 commit ahead"
    return f"HEAD is {ahead} commits ahead"


def cap(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def describe(repo: Path, facts: IndexFacts | None, ahead: int | None) -> str:
    """One of exactly three lines: no index, an empty index, or the facts of an index."""
    db = repo / INDEX_RELATIVE_PATH
    index_first = (
        f'Call index_codebase(repo_path="{repo}") before search_code; until then, use grep.'
    )
    if facts is None:
        return cap(f"trelix: no index at {db}. {index_first}")
    if facts.files == 0:
        return cap(f"trelix: the index at {db} is empty (0 files). {index_first}")
    when = facts.indexed_at or "at an unknown time (re-index to record provenance)"
    commit_part = (
        f" from commit {facts.git_commit[:12]} ({_distance(ahead)})" if facts.git_commit else ""
    )
    embedder_part = f"; embedder {facts.embedder}" if facts.embedder else ""
    return cap(
        f"trelix: {repo} is indexed: {facts.files} files, {facts.symbols} symbols; "
        f"built {when}{commit_part}{embedder_part}. "
        f'Pass repo_path="{repo}" to every trelix tool.'
    )


def main(stdin_text: str, env: Mapping[str, str]) -> str:
    """The line to print, or '' when there is nothing to say (reason on stderr)."""
    repo = project_root(stdin_text, env)
    if repo is None:
        sys.stderr.write("trelix session_start: skipped (no project directory)\n")
        return ""
    db_path = repo / INDEX_RELATIVE_PATH
    if not db_path.is_file():
        return describe(repo, None, None)
    facts = read_index_facts(db_path)
    return describe(repo, facts, commits_ahead(repo, facts.git_commit))


if __name__ == "__main__":
    try:
        line = main("" if sys.stdin.isatty() else sys.stdin.read(STDIN_LIMIT_BYTES), os.environ)
        if line:
            sys.stdout.write(line + "\n")
    except Exception as exc:  # the hook must never block a session
        sys.stderr.write(f"trelix session_start: skipped ({type(exc).__name__})\n")
    raise SystemExit(0)
