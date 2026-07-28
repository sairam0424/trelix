"""
Unit tests for GitLinker — walks real git repos (temp fixtures, not mocks,
since the custom-delimited git log format parsing needs proving against real
`git log` output) and asserts symbol-ticket GenericEdges land correctly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from trelix.core.config import GitLinkerConfig
from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.indexing.git_linker import GitLinker
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_git_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)


def _commit_file(repo_dir: Path, rel_path: str, content: str, message: str) -> None:
    file_path = repo_dir / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    subprocess.run(["git", "add", rel_path], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo_dir, check=True)


def _make_file(db: Database, rel_path: str) -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="deadbeef",
        size_bytes=512,
    )
    return db.upsert_file(f)


def _insert_symbol(db: Database, file_id: int, name: str) -> int:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def {name}():",
        body=f"def {name}(): pass",
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    return sym_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGitLinkerRealRepo:
    def test_links_symbol_to_ticket_from_commit_message(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "auth.py", "def login(): pass\n", "PROJ-123: fix login bug")

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        linker = GitLinker(db)
        count = linker.link(str(repo_dir))

        assert count == 1
        targets = db.get_generic_edge_targets(sym_id)
        assert targets == ["ticket:PROJ-123"]

    def test_no_ticket_reference_links_nothing(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "auth.py", "def login(): pass\n", "fix login bug, no ticket here")

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        _insert_symbol(db, file_id, "login")

        linker = GitLinker(db)
        count = linker.link(str(repo_dir))
        assert count == 0

    def test_multiple_tickets_in_one_commit_message(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(
            repo_dir, "auth.py", "def login(): pass\n", "PROJ-1: fix login, also fixes PROJ-2"
        )

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        linker = GitLinker(db)
        count = linker.link(str(repo_dir))

        assert count == 2
        targets = set(db.get_generic_edge_targets(sym_id))
        assert targets == {"ticket:PROJ-1", "ticket:PROJ-2"}

    def test_every_symbol_in_touched_file_credited_file_level_granularity(
        self, tmp_path: Path
    ) -> None:
        """File-level granularity: a commit touching one function still
        credits EVERY symbol in that file, not just the changed lines —
        an accepted precision/cost tradeoff, not a bug."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(
            repo_dir,
            "auth.py",
            "def login(): pass\ndef logout(): pass\n",
            "PROJ-9: fix login only",
        )

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        login_id = _insert_symbol(db, file_id, "login")
        logout_id = _insert_symbol(db, file_id, "logout")

        linker = GitLinker(db)
        count = linker.link(str(repo_dir))

        assert count == 2  # both symbols credited, not just "login"
        assert db.get_generic_edge_targets(login_id) == ["ticket:PROJ-9"]
        assert db.get_generic_edge_targets(logout_id) == ["ticket:PROJ-9"]

    def test_duplicate_symbol_ticket_pairs_are_not_inserted_twice(self, tmp_path: Path) -> None:
        """Two separate commits both referencing PROJ-1 and touching the same
        file must not create two edges for the same (symbol, ticket) pair."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "auth.py", "def login(): pass\n", "PROJ-1: first pass")
        _commit_file(repo_dir, "auth.py", "def login(): pass  # v2\n", "PROJ-1: follow-up fix")

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        linker = GitLinker(db)
        count = linker.link(str(repo_dir))

        assert count == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-1"]

    def test_custom_ticket_pattern_matches_github_issue_style(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "auth.py", "def login(): pass\n", "fixes #456")

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        linker = GitLinker(db, GitLinkerConfig(ticket_pattern=r"#\d+"))
        count = linker.link(str(repo_dir))

        assert count == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:#456"]

    def test_max_commits_bounds_history_walked(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "auth.py", "v1\n", "PROJ-1: first")
        _commit_file(repo_dir, "auth.py", "v2\n", "PROJ-2: second")
        _commit_file(repo_dir, "auth.py", "v3\n", "PROJ-3: third — most recent")

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        # max_commits=1 → only the most recent commit (PROJ-3) is walked.
        linker = GitLinker(db, GitLinkerConfig(max_commits=1))
        count = linker.link(str(repo_dir))

        assert count == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-3"]


class TestGitLinkerFailureModes:
    """Never raises — degrades to 0 edges on any git failure, matching
    DiffParser.from_git()'s existing posture."""

    def test_non_git_directory_returns_zero(self, tmp_path: Path) -> None:
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()

        db = Database(tmp_path / "index.db")
        linker = GitLinker(db)
        count = linker.link(str(not_a_repo))
        assert count == 0

    def test_nonexistent_directory_returns_zero(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        linker = GitLinker(db)
        count = linker.link(str(tmp_path / "does" / "not" / "exist"))
        assert count == 0

    def test_empty_repo_with_no_commits_returns_zero(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        db = Database(tmp_path / "index.db")
        linker = GitLinker(db)
        count = linker.link(str(repo_dir))
        assert count == 0

    def test_touched_file_not_yet_indexed_is_skipped_gracefully(self, tmp_path: Path) -> None:
        """A commit referencing a ticket but touching a file that was never
        indexed (get_symbol_ids_for_file returns []) must not raise."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "unindexed.py", "x = 1\n", "PROJ-1: touches unindexed file")

        db = Database(tmp_path / "index.db")
        linker = GitLinker(db)
        count = linker.link(str(repo_dir))
        assert count == 0
