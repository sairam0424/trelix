"""
Unit tests for GitLinker — walks real git repos (temp fixtures, not mocks,
since the custom-delimited git log format parsing needs proving against real
`git log` output) and asserts symbol-ticket GenericEdges land correctly.
"""

from __future__ import annotations

import subprocess

import pytest
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

    def test_rerunning_link_on_same_repo_does_not_duplicate_edges(self, tmp_path: Path) -> None:
        """Re-running `trelix link-tickets` on a repo it's already linked
        (e.g. a cron re-sync, or a user running the command twice) must not
        duplicate generic_edges rows — the within-run `seen` set in link()
        only dedupes inside a single call, so this exercises the DB-level
        guard (idx_generic_edges_dedup + INSERT OR IGNORE) instead."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "auth.py", "def login(): pass\n", "PROJ-1: add login")

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        first_count = GitLinker(db).link(str(repo_dir))
        second_count = GitLinker(db).link(str(repo_dir))

        assert first_count == 1
        assert second_count == 1  # link() itself still reports what it tried to insert
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-1"]  # but no duplicate row
        row_count = db._conn.execute("SELECT COUNT(*) FROM generic_edges").fetchone()[0]
        assert row_count == 1

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


class TestMergeCommitsAreNotSkipped:
    """A ticket reference in a merge commit must still link to the merged files.

    `git log --name-only` prints NO file list for a merge commit — git shows a diff for
    a merge only when told which parent to diff against. GitLinker passed neither
    `-m` nor `--diff-merges`, so every merge contributed a subject with an empty file
    list and linked to nothing.

    That is the common case on any repository that integrates through pull requests,
    because "Merge pull request #123 from feature/PROJ-456-thing" is exactly where the
    ticket reference lives. Verified on trelix's own history: commit 3dea90a is a merge,
    and `git log -1 --name-only` prints its subject and zero files, while
    `--diff-merges=first-parent` prints the subject and four files.

    `--diff-merges=first-parent` is used rather than `-m`: `-m` emits one diff section
    per parent, so every file in a two-parent merge would be counted twice, and
    `--first-parent` is a different thing again — it changes which commits are traversed
    at all, hiding the individual commits on the merged branch.
    """

    @staticmethod
    def _repo_with_a_merge(repo_dir: Path) -> None:
        """main and a feature branch, merged with --no-ff so a real merge commit exists."""
        _init_git_repo(repo_dir)
        _commit_file(repo_dir, "base.py", "def base(): pass\n", "PROJ-1 initial commit")

        subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=repo_dir, check=True)
        _commit_file(repo_dir, "feature.py", "def shipped(): pass\n", "work in progress")

        subprocess.run(["git", "checkout", "-q", "-"], cwd=repo_dir, check=True)
        # --no-ff forces a merge commit even though the branch could fast-forward.
        subprocess.run(
            ["git", "merge", "-q", "--no-ff", "-m", "PROJ-742 merge the feature", "feature"],
            cwd=repo_dir,
            check=True,
        )

    def test_merge_commit_reports_its_changed_files(self, tmp_path: Path) -> None:
        """The headline regression: the merge's file list must not be empty."""
        repo = tmp_path / "repo"
        repo.mkdir()
        self._repo_with_a_merge(repo)

        linker = GitLinker(GitLinkerConfig(enabled=True))
        records = linker._walk_log(str(repo))

        merges = [(tickets, files) for tickets, files in records if "PROJ-742" in tickets]
        assert merges, f"the merge commit was not found at all in {records}"
        _, merge_files = merges[0]
        assert "feature.py" in merge_files, (
            "the merge commit yielded no file list, so its ticket reference links to "
            f"nothing; got {merge_files}"
        )

    def test_files_are_not_double_counted(self, tmp_path: Path) -> None:
        """Rules out `-m`, which emits one diff section per parent.

        With `-m` a two-parent merge lists every changed file twice, inflating every
        symbol-ticket edge weight derived from it.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        self._repo_with_a_merge(repo)

        linker = GitLinker(GitLinkerConfig(enabled=True))
        records = linker._walk_log(str(repo))

        for tickets, files in records:
            if "PROJ-742" in tickets:
                assert files.count("feature.py") == 1, (
                    f"feature.py appears {files.count('feature.py')} times in one "
                    "merge record"
                )

    def test_ordinary_commits_are_unaffected(self, tmp_path: Path) -> None:
        """Non-merge commits must behave exactly as before."""
        repo = tmp_path / "repo"
        repo.mkdir()
        self._repo_with_a_merge(repo)

        linker = GitLinker(GitLinkerConfig(enabled=True))
        records = linker._walk_log(str(repo))

        initial = [(t, f) for t, f in records if "PROJ-1" in t]
        assert initial, "the initial non-merge commit went missing"
        assert "base.py" in initial[0][1]

    def test_commits_on_the_merged_branch_are_still_traversed(self, tmp_path: Path) -> None:
        """Rules out `--first-parent`, which would hide the branch's own commits.

        The feature branch's commit carries no ticket, but it must still appear as a
        record — dropping it would silently shrink the corpus the linker walks.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        self._repo_with_a_merge(repo)

        linker = GitLinker(GitLinkerConfig(enabled=True))
        records = linker._walk_log(str(repo))

        all_files = [f for _, files in records for f in files]
        assert all_files.count("feature.py") >= 1
        assert len(records) >= 3, (
            f"expected the initial, branch and merge commits; got {len(records)} records"
        )


class TestDefaultTicketPattern:
    """The default pattern must not treat technical constants as ticket ids.

    The previous default, r"[A-Z]+-\\d+", matched any run of capitals followed by a
    hyphen and digits. Measured across 830 commits of this repository it found 12
    strings and every single one was a false positive: UTF-8 (x4), SHA-256, HTTP-400,
    B-1, and two ticket-shaped strings appearing in prose.

    A ticket key and a technical constant are structurally identical — UTF-8 against
    ENG-8 — so anchoring alone cannot separate them. A vocabulary of noise prefixes can,
    and that is what the default now carries.

    The trailing guard allows a hyphen on purpose. Branch names are where ticket keys
    appear in merge subjects, and a guard that rejected "PROJ-456-thing" silently
    discarded exactly the references the merge-commit fix exists to capture.
    """

    @staticmethod
    def _re():  # type: ignore[no-untyped-def]
        import re

        from trelix.core.config import GitLinkerConfig

        return re.compile(GitLinkerConfig().ticket_pattern)

    @pytest.mark.parametrize(
        "text",
        ["PROJ-123", "ENG-45", "SCRUM-1", "AB-9", "TRELIX-9999",
         "Merge pull request #12 from feature/PROJ-456-thing",
         "fix/NS-3-cleanup"],
    )
    def test_real_ticket_shapes_are_matched(self, text: str) -> None:
        assert self._re().search(text), f"{text!r} contains a ticket key and was missed"

    @pytest.mark.parametrize(
        "text",
        ["encoded as UTF-8", "SHA-256 digest", "returns HTTP-400", "BASE-64 payload",
         "ISO-8601 timestamp", "see RFC-2616", "MD-5 hash", "AES-256-GCM"],
    )
    def test_technical_constants_are_not_tickets(self, text: str) -> None:
        match = self._re().search(text)
        assert match is None, f"{text!r} was read as ticket {match.group(0)!r}"

    @pytest.mark.parametrize("text", ["xPROJ-123", "PROJ-123x", "proj-123", "P-1"])
    def test_malformed_keys_are_rejected(self, text: str) -> None:
        assert self._re().search(f" {text} ") is None, f"{text!r} should not be a ticket"

    def test_an_over_long_number_does_not_match_a_prefix_of_itself(self) -> None:
        """PROJ-1234567 must not silently become PROJ-123456."""
        assert self._re().search(" PROJ-1234567 ") is None

    def test_the_pattern_compiles_and_is_not_the_old_one(self) -> None:
        from trelix.core.config import GitLinkerConfig

        pattern = GitLinkerConfig().ticket_pattern
        assert pattern != r"[A-Z]+-\d+", "the permissive default is still in place"

    def test_noise_prefix_list_is_reachable_for_extension(self) -> None:
        """The vocabulary is a named list so adding a prefix is a one-word edit."""
        from trelix.core.config import _TICKET_NOISE_PREFIXES, TICKET_PATTERN_DEFAULT

        assert "UTF" in _TICKET_NOISE_PREFIXES
        assert "UTF" in TICKET_PATTERN_DEFAULT
