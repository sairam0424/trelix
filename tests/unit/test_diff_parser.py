"""Tests for DiffParser — unified diff parsing."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from trelix.review.diff_parser import DiffParser, GitDiffError

_SAMPLE_DIFF = textwrap.dedent("""\
    diff --git a/src/auth.py b/src/auth.py
    index abc123..def456 100644
    --- a/src/auth.py
    +++ b/src/auth.py
    @@ -10,7 +10,9 @@ class AuthService:
         def login(self, user: str, password: str) -> bool:
    -        return self._check(user, password)
    +        if not user or not password:
    +            raise ValueError("credentials required")
    +        return self._check(user, password)

         def logout(self):
    diff --git a/src/db.py b/src/db.py
    index 111111..222222 100644
    --- a/src/db.py
    +++ b/src/db.py
    @@ -5,3 +5,4 @@ class Database:
         def connect(self):
             self._conn = sqlite3.connect(self._path)
    +        self._conn.execute("PRAGMA journal_mode=WAL")
    """)


class TestDiffParser:
    def test_parse_returns_list_of_hunks(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        assert len(hunks) == 2

    def test_hunk_has_correct_file_path(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        assert hunks[0].file_path == "src/auth.py"
        assert hunks[1].file_path == "src/db.py"

    def test_hunk_captures_added_lines(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        auth_hunk = hunks[0]
        assert any("raise ValueError" in line for line in auth_hunk.added)

    def test_hunk_captures_removed_lines(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        auth_hunk = hunks[0]
        assert any("_check" in line for line in auth_hunk.removed)

    def test_hunk_line_numbers(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        assert hunks[0].new_start == 10

    def test_empty_diff_returns_empty_list(self) -> None:
        parser = DiffParser()
        assert parser.parse("") == []

    def test_parse_diff_no_changes_returns_empty(self) -> None:
        parser = DiffParser()
        diff = "diff --git a/unchanged.py b/unchanged.py\n"
        hunks = parser.parse(diff)
        assert isinstance(hunks, list)

    def test_hunk_search_query_is_nonempty(self) -> None:
        parser = DiffParser()
        hunks = parser.parse(_SAMPLE_DIFF)
        for hunk in hunks:
            assert hunk.to_search_query()
            assert isinstance(hunk.to_search_query(), str)

    def test_from_git_calls_subprocess(self) -> None:
        mock_result = type("R", (), {"stdout": _SAMPLE_DIFF, "returncode": 0})()
        with patch("subprocess.run", return_value=mock_result):
            parser = DiffParser()
            hunks = parser.from_git("/fake/repo")
        assert len(hunks) == 2

    def test_from_git_diffs_base_against_head_in_the_repository(self) -> None:
        done = subprocess.CompletedProcess([], 0, stdout=_SAMPLE_DIFF, stderr="")
        with patch("subprocess.run", return_value=done) as run:
            hunks = DiffParser().from_git("/fake/repo", base="main", head="topic")

        assert len(hunks) == 2
        (call,) = run.call_args_list
        assert call.args == (["git", "diff", "main", "topic", "--unified=3", "--"],)
        assert call.kwargs["cwd"] == "/fake/repo"
        assert call.kwargs["timeout"] == 30


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        [
            "git",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


class TestRefResolves:
    """`from_git` returns [] for a missing ref and an empty diff alike; this tells them apart."""

    def _one_commit_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "a.txt").write_text("x\n")
        (repo / "sub").mkdir()
        (repo / "sub" / "b.txt").write_text("y\n")
        _git(repo, "add", "a.txt", "sub/b.txt")
        _git(repo, "commit", "-q", "-m", "only")
        return repo

    def test_head_resolves_and_the_parent_of_the_first_commit_does_not(
        self, tmp_path: Path
    ) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        assert DiffParser().ref_resolves(repo, "HEAD") is True
        assert DiffParser().ref_resolves(repo, "HEAD~1") is False

    def test_a_name_that_is_no_ref_does_not_resolve(self, tmp_path: Path) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        assert DiffParser().ref_resolves(repo, "NOTAREF") is False
        assert DiffParser().ref_resolves(repo, "") is False

    def test_a_well_formed_id_of_an_object_that_does_not_exist_does_not_resolve(
        self, tmp_path: Path
    ) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        assert DiffParser().ref_resolves(repo, "0123456789012345678901234567890123456789") is False

    def test_a_lightweight_and_an_annotated_tag_resolve(self, tmp_path: Path) -> None:
        repo = self._one_commit_repo(tmp_path)
        _git(repo, "tag", "light")
        _git(repo, "tag", "-a", "annotated", "-m", "a tag object, not a commit")

        assert DiffParser().ref_resolves(str(repo), "light") is True
        assert DiffParser().ref_resolves(str(repo), "annotated") is True

    def test_a_tree_resolves_because_git_diff_compares_it(self, tmp_path: Path) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        assert DiffParser().ref_resolves(repo, "HEAD^{tree}") is True
        # The empty tree: `git diff <it> HEAD` is how the first commit of a repository is read.
        assert DiffParser().ref_resolves(repo, "4b825dc642cb6eb9a060e54bf8d69288fbee4904") is True

    def test_a_tree_named_by_revision_and_path_resolves(self, tmp_path: Path) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        # `git diff HEAD~1:src HEAD:src` is valid. A `^{tree}` suffix made the path `sub^{tree}`.
        assert DiffParser().ref_resolves(repo, "HEAD:sub") is True

    def test_a_commit_message_search_resolves(self, tmp_path: Path) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        # `:/text` names the youngest commit whose message matches, and `git diff` takes it.
        assert DiffParser().ref_resolves(repo, ":/only") is True
        assert DiffParser().ref_resolves(repo, ":/nomatch") is False

    def test_a_blob_resolves_because_git_diff_compares_a_pair_of_them(self, tmp_path: Path) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        # `git diff HEAD~1:f HEAD:f` is valid, so the type of the object is not checked.
        assert DiffParser().ref_resolves(repo, "HEAD:a.txt") is True
        assert DiffParser().ref_resolves(repo, "HEAD:sub/b.txt") is True
        assert DiffParser().ref_resolves(repo, "HEAD:absent.txt") is False

    def test_a_range_is_not_something_to_compare(self, tmp_path: Path) -> None:
        repo = str(self._one_commit_repo(tmp_path))

        assert DiffParser().ref_resolves(repo, "HEAD..HEAD") is False

    def test_a_directory_that_is_not_a_repository_resolves_nothing(self, tmp_path: Path) -> None:
        not_a_repo = tmp_path / "plain"
        not_a_repo.mkdir()

        assert DiffParser().ref_resolves(str(not_a_repo), "HEAD") is False

    def test_a_missing_directory_is_false_not_an_exception(self, tmp_path: Path) -> None:
        assert DiffParser().ref_resolves(str(tmp_path / "absent"), "HEAD") is False

    def test_git_resolves_the_ref_then_looks_up_the_id_it_printed(self) -> None:
        found = subprocess.CompletedProcess([], 0, stdout="0123abcd\n", stderr="")
        looked_up = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("subprocess.run", side_effect=[found, looked_up]) as run:
            assert DiffParser().ref_resolves("/fake/repo", "main") is True

        resolve_call, lookup_call = run.call_args_list
        assert resolve_call.args == (["git", "rev-parse", "--verify", "main"],)
        assert lookup_call.args == (["git", "cat-file", "-e", "0123abcd"],)
        for call in (resolve_call, lookup_call):
            assert call.kwargs["cwd"] == "/fake/repo"
            assert call.kwargs["timeout"] == 30

    def test_a_ref_that_git_cannot_resolve_is_not_typed(self) -> None:
        missing = subprocess.CompletedProcess([], 128, stdout="0123abcd\n", stderr="fatal")
        with patch("subprocess.run", side_effect=[missing]) as run:
            assert DiffParser().ref_resolves("/fake/repo", "main") is False

        assert run.call_count == 1

    def test_a_lookup_that_fails_is_false_whatever_it_printed(self) -> None:
        found = subprocess.CompletedProcess([], 0, stdout="0123abcd\n", stderr="")
        failed = subprocess.CompletedProcess([], 1, stdout="commit\n", stderr="")
        with patch("subprocess.run", side_effect=[found, failed]):
            assert DiffParser().ref_resolves("/fake/repo", "main") is False

    @pytest.mark.parametrize(
        "runs",
        [
            pytest.param([FileNotFoundError("git")], id="git-missing"),
            pytest.param([subprocess.TimeoutExpired("git", 30)], id="resolve-timed-out"),
            pytest.param(
                [
                    subprocess.CompletedProcess([], 0, stdout="0123abcd\n", stderr=""),
                    subprocess.TimeoutExpired("git", 30),
                ],
                id="type-timed-out",
            ),
        ],
    )
    def test_a_git_that_cannot_finish_is_false_not_an_exception(self, runs: list[object]) -> None:
        with patch("subprocess.run", side_effect=runs):
            assert DiffParser().ref_resolves("/fake/repo", "HEAD") is False


class TestGitDiff:
    """`git_diff` raises where `from_git` returns []: a diff git never made is not an empty one."""

    def _two_commit_repo(self, tmp_path: Path) -> Path:
        """HEAD~1..HEAD adds a line to docs/a.md; a branch `docs` (and a directory) is HEAD~1."""
        repo = tmp_path / "repo"
        (repo / "docs").mkdir(parents=True)
        _git(repo, "init", "-q")
        (repo / "docs" / "a.md").write_text("one\n")
        _git(repo, "add", "docs/a.md")
        _git(repo, "commit", "-q", "-m", "first")
        _git(repo, "branch", "docs")
        (repo / "docs" / "a.md").write_text("one\ntwo\n")
        _git(repo, "commit", "-q", "-a", "-m", "second")
        return repo

    def test_a_ref_that_is_also_a_directory_name_is_read_as_a_revision(
        self, tmp_path: Path
    ) -> None:
        repo = str(self._two_commit_repo(tmp_path))

        # Without the trailing `--` git dies: ambiguous argument 'docs': both revision and filename.
        assert "+two" in DiffParser().git_diff(repo, "docs", "HEAD")

    def test_no_changes_is_empty_text_not_an_error(self, tmp_path: Path) -> None:
        repo = str(self._two_commit_repo(tmp_path))

        assert DiffParser().git_diff(repo, "HEAD", "HEAD") == ""

    def test_a_pair_that_git_diff_rejects_raises(self, tmp_path: Path) -> None:
        repo = str(self._two_commit_repo(tmp_path))

        # A blob against a commit: git prints its usage text and exits non-zero.
        with pytest.raises(GitDiffError, match=r"^exit [1-9][0-9]*: "):
            DiffParser().git_diff(repo, "HEAD:docs/a.md", "HEAD")

    @pytest.mark.parametrize(
        ("stderr", "message"),
        [
            pytest.param(
                "fatal: bad object\nhint: a second line\n",
                "exit 128: fatal: bad object",
                id="first-line-of-stderr",
            ),
            pytest.param("", "exit 128: no error output", id="silent-failure"),
        ],
    )
    def test_the_message_is_the_exit_code_and_the_first_line_of_stderr(
        self, stderr: str, message: str
    ) -> None:
        failed = subprocess.CompletedProcess([], 128, stdout="partial\n", stderr=stderr)
        with patch("subprocess.run", return_value=failed), pytest.raises(GitDiffError) as caught:
            DiffParser().git_diff("/fake/repo", "a", "b")

        assert str(caught.value) == message

    @pytest.mark.parametrize(
        ("failure", "kind"),
        [
            pytest.param(FileNotFoundError("git"), "FileNotFoundError", id="git-missing"),
            pytest.param(subprocess.TimeoutExpired("git", 30), "TimeoutExpired", id="timed-out"),
            pytest.param(
                UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid continuation byte"),
                "UnicodeDecodeError",
                id="not-utf-8",
            ),
        ],
    )
    def test_a_git_that_cannot_finish_or_be_read_raises(
        self, failure: Exception, kind: str
    ) -> None:
        with patch("subprocess.run", side_effect=failure), pytest.raises(GitDiffError, match=kind):
            DiffParser().git_diff("/fake/repo", "a", "b")

    @pytest.mark.parametrize(
        "outcome",
        [
            pytest.param(
                subprocess.CompletedProcess([], 128, stdout="", stderr="fatal: bad"), id="exit-128"
            ),
            pytest.param(subprocess.TimeoutExpired("git", 30), id="timed-out"),
        ],
    )
    def test_from_git_still_returns_an_empty_list_on_a_failure(self, outcome: object) -> None:
        with patch("subprocess.run", side_effect=[outcome]):
            assert DiffParser().from_git("/fake/repo") == []
