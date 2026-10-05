"""`trelix review --json` on a local diff: one JSON document on stdout, and refs must resolve.

The `--pr` path already did this: every status line goes to stderr and stdout carries only the
JSON array (`[]` when there is nothing to report). The local path printed "Reviewing N hunks
across M files..." to stdout first, and prose ("No issues found.", "No changes found in
diff.", "No findings in the hunks that were reviewed.") instead of an empty array, so a caller
that parsed stdout got a JSON error. A `--base` that git could not resolve made `git diff`
fail, which the command reported as "No changes found in diff." with exit 0.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.llm.client import ChatResponse
from trelix.review.hunk_status import HunkResult, HunkStatus
from trelix.review.reviewer import DiffReviewer, ReviewComment, ReviewOutcome

runner = CliRunner()

_PROGRESS = "Reviewing 1 hunks across 1 files..."
_COMMENT = ReviewComment("src/foo.py", 1, 1, "WARN", "smell here")
_FINDING_JSON = [
    {"file": "src/foo.py", "lines": "1-1", "severity": "WARN", "comment": "smell here"}
]
_DIFF = (
    "diff --git a/src/foo.py b/src/foo.py\n"
    "--- a/src/foo.py\n"
    "+++ b/src/foo.py\n"
    "@@ -1,1 +1,2 @@\n"
    " old\n"
    "+added line\n"
)
_TWO_FILE_DIFF = _DIFF + (
    "diff --git a/src/bar.py b/src/bar.py\n"
    "--- a/src/bar.py\n"
    "+++ b/src/bar.py\n"
    "@@ -5,1 +5,2 @@\n"
    " old\n"
    "+other line\n"
)
_CLEAN = ReviewOutcome(llm_available=True, hunks_total=1, hunks_failed=0)
_PARTIAL = ReviewOutcome(
    llm_available=True,
    hunks_total=4,
    hunks_failed=1,
    hunk_results=(HunkResult("src/foo.py", 1, HunkStatus.PARSE_FAILED, "no_review_array", 0),),
)
_TOLERATE_PARTIAL = {"TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": "1"}


def _fake_review(outcome: ReviewOutcome, comments: list[ReviewComment]) -> Any:
    def _review(self: DiffReviewer, *args: object, **kwargs: object) -> list[ReviewComment]:
        self.last_outcome = outcome
        return comments

    return _review


def _local(
    tmp_path: Path,
    outcome: ReviewOutcome,
    comments: list[ReviewComment],
    *extra: str,
    diff_text: str = _DIFF,
    env: dict[str, str] | None = None,
) -> Any:
    diff_file = tmp_path / "changes.diff"
    diff_file.write_text(diff_text)
    with patch.object(DiffReviewer, "review", _fake_review(outcome, comments)):
        return runner.invoke(
            app, ["review", str(tmp_path), "--diff", str(diff_file), *extra], env=env or {}
        )


class TestJsonStdoutIsOneDocument:
    @pytest.mark.parametrize(
        ("outcome", "comments", "diff_text", "env", "exit_code", "expected", "progress"),
        [
            pytest.param(_CLEAN, [_COMMENT], _DIFF, None, 0, _FINDING_JSON, True, id="findings"),
            pytest.param(_CLEAN, [], _DIFF, None, 0, [], True, id="no-findings"),
            pytest.param(_CLEAN, [], "", None, 0, [], False, id="empty-diff"),
            pytest.param(
                ReviewOutcome(llm_available=False, hunks_total=1, hunks_failed=1),
                [],
                _DIFF,
                None,
                3,
                [],
                True,
                id="exit-3-no-llm",
            ),
            pytest.param(
                ReviewOutcome(llm_available=True, hunks_total=2, hunks_failed=2),
                [],
                _DIFF,
                None,
                3,
                [],
                True,
                id="exit-3-every-call-failed",
            ),
            pytest.param(
                _PARTIAL, [_COMMENT], _DIFF, None, 4, _FINDING_JSON, True, id="exit-4-findings"
            ),
            pytest.param(_PARTIAL, [], _DIFF, None, 4, [], True, id="exit-4-no-findings"),
            pytest.param(
                _PARTIAL, [], _DIFF, _TOLERATE_PARTIAL, 0, [], True, id="tolerated-no-findings"
            ),
        ],
    )
    def test_stdout_parses_and_the_status_lines_are_on_stderr(
        self,
        tmp_path: Path,
        outcome: ReviewOutcome,
        comments: list[ReviewComment],
        diff_text: str,
        env: dict[str, str] | None,
        exit_code: int,
        expected: list[dict[str, str]],
        progress: bool,
    ) -> None:
        result = _local(tmp_path, outcome, comments, "--json", diff_text=diff_text, env=env)

        assert result.exit_code == exit_code, result.stderr
        assert json.loads(result.stdout) == expected
        assert "Reviewing" not in result.stdout
        assert (_PROGRESS in result.stderr) is progress

    def test_the_empty_diff_says_nothing_in_prose(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _CLEAN, [], "--json", diff_text="")

        assert "No changes found" not in result.stdout
        assert "No changes found" not in result.stderr

    def test_a_reply_that_is_prose_leaves_an_empty_array_and_exit_4(self, tmp_path: Path) -> None:
        # No stubbed review(): the real DiffReviewer reads one prose reply and one `[]`.
        prose = ChatResponse(
            content="I could not find any problems in this change.",
            model="gpt-test",
            finish_reason="stop",
            raw_finish_reason="stop",
            output_tokens=0,
        )
        empty = ChatResponse(
            content="[]",
            model="gpt-test",
            finish_reason="stop",
            raw_finish_reason="stop",
            output_tokens=0,
        )
        client = MagicMock()
        client.complete.side_effect = [prose, empty]
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(context_text="ctx", results=[])
        diff_file = tmp_path / "changes.diff"
        diff_file.write_text(_TWO_FILE_DIFF)

        with (
            patch.object(DiffReviewer, "_get_client", return_value=client),
            patch.object(DiffReviewer, "_get_retriever", return_value=retriever),
        ):
            result = runner.invoke(
                app, ["review", str(tmp_path), "--diff", str(diff_file), "--json"]
            )

        assert result.exit_code == 4, result.stderr
        assert json.loads(result.stdout) == []
        assert "Reviewing 2 hunks across 2 files..." in result.stderr
        assert "1 of 2 hunks could not be fully reviewed" in " ".join(result.stderr.split())


class TestHumanOutputIsUnchanged:
    """Regression control: without --json the text, the streams and the exit codes stay."""

    def test_findings_table_and_progress_line_are_on_stdout(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _CLEAN, [_COMMENT])

        assert result.exit_code == 0, result.stderr
        assert _PROGRESS in result.stdout
        assert "Review Results (1 comments)" in result.stdout
        assert "smell here" in result.stdout
        assert "Reviewing" not in result.stderr

    def test_no_findings_says_so_on_stdout(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _CLEAN, [])

        assert result.exit_code == 0, result.stderr
        assert _PROGRESS in result.stdout
        assert "No issues found." in result.stdout
        assert "[]" not in result.stdout

    def test_an_empty_diff_says_so_on_stdout(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _CLEAN, [], diff_text="")

        assert result.exit_code == 0, result.stderr
        assert "No changes found in diff." in result.stdout
        assert "[]" not in result.stdout

    def test_a_partial_review_without_findings_does_not_say_no_issues(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _PARTIAL, [])

        assert result.exit_code == 4
        assert "No findings in the hunks that were reviewed." in result.stdout
        assert "No issues found" not in result.stdout
        assert "[]" not in result.stdout

    def test_exit_3_still_prints_the_reason_on_stderr(self, tmp_path: Path) -> None:
        outcome = ReviewOutcome(llm_available=False, hunks_total=1, hunks_failed=1)

        result = _local(tmp_path, outcome, [])

        assert result.exit_code == 3
        assert "No issues found" not in result.stdout
        assert "not configured" in result.stderr.lower()


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
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A repository whose HEAD~1..HEAD changes src/foo.py and whose HEAD..HEAD changes nothing."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "src" / "foo.py").write_text("old\n")
    _git(repo, "add", "src/foo.py")
    _git(repo, "commit", "-q", "-m", "first")
    (repo / "src" / "foo.py").write_text("old\nadded line\n")
    _git(repo, "commit", "-q", "-a", "-m", "second")
    return repo


def _review_git(
    repo: Path, *extra: str, comments: list[ReviewComment] | None = None
) -> tuple[Any, MagicMock]:
    """Run `trelix review REPO *extra` against `repo`; the stub records whether it was reached."""
    reached = MagicMock()

    def _review(self: DiffReviewer, *args: object, **kwargs: object) -> list[ReviewComment]:
        reached()
        self.last_outcome = _CLEAN
        return comments or []

    with patch.object(DiffReviewer, "review", _review):
        return runner.invoke(app, ["review", str(repo), *extra]), reached


class TestUnknownRefIsAnError:
    def test_an_unknown_base_exits_1_and_names_the_ref(self, git_repo: Path) -> None:
        result, reached = _review_git(git_repo, "--base", "NOTAREF")

        assert result.exit_code == 1
        assert "cannot resolve --base 'NOTAREF' to a git object" in " ".join(result.stderr.split())
        assert result.stdout == ""
        reached.assert_not_called()

    def test_an_unknown_base_with_json_prints_the_error_and_no_stdout(self, git_repo: Path) -> None:
        result, reached = _review_git(git_repo, "--base", "NOTAREF", "--json")

        assert result.exit_code == 1
        assert "cannot resolve --base 'NOTAREF' to a git object" in " ".join(result.stderr.split())
        assert result.stdout == ""
        reached.assert_not_called()

    def test_an_unknown_head_is_the_same_error(self, git_repo: Path) -> None:
        result, reached = _review_git(git_repo, "--head", "NOPE")

        assert result.exit_code == 1
        assert "cannot resolve --head 'NOPE' to a git object" in " ".join(result.stderr.split())
        reached.assert_not_called()

    def test_the_default_base_fails_the_same_way_in_a_one_commit_repository(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "one"
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "a.txt").write_text("x\n")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "only")

        result, _ = _review_git(repo)

        assert result.exit_code == 1
        assert "cannot resolve --base 'HEAD~1' to a git object" in " ".join(result.stderr.split())

    def test_markup_in_the_ref_is_printed_as_typed(self, git_repo: Path) -> None:
        result, _ = _review_git(git_repo, "--base", "[/red]x")

        assert result.exit_code == 1
        assert "cannot resolve --base '[/red]x' to a git object" in " ".join(result.stderr.split())

    def test_a_diff_file_never_looks_at_the_refs(self, git_repo: Path, tmp_path: Path) -> None:
        diff_file = tmp_path / "changes.diff"
        diff_file.write_text(_DIFF)

        with patch.object(DiffReviewer, "review", _fake_review(_CLEAN, [])):
            result = runner.invoke(
                app,
                ["review", str(git_repo), "--diff", str(diff_file), "--base", "NOTAREF", "--json"],
            )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == []


class TestGitDiffFailureIsAnError:
    """Refs that each resolve can still fail together; that is an error, never "no changes"."""

    @pytest.mark.parametrize(
        "extra", [pytest.param((), id="text"), pytest.param(("--json",), id="json")]
    )
    def test_a_blob_against_a_commit_exits_1_and_says_so(
        self, git_repo: Path, extra: tuple[str, ...]
    ) -> None:
        # Both refs resolve, and `git diff` rejects the pair with its usage text (exit 129).
        result, reached = _review_git(git_repo, "--base", "HEAD:src/foo.py", *extra)

        assert result.exit_code == 1
        assert "git diff 'HEAD:src/foo.py' 'HEAD' failed: exit 129" in " ".join(
            result.stderr.split()
        )
        assert result.stdout == ""
        reached.assert_not_called()

    def test_a_diff_that_is_not_utf8_exits_1_and_says_so(self, git_repo: Path) -> None:
        (git_repo / "latin.txt").write_bytes(b"caf\xe9\n")
        _git(git_repo, "add", "latin.txt")
        _git(git_repo, "commit", "-q", "-m", "third")

        result, reached = _review_git(git_repo, "--json")

        assert result.exit_code == 1
        assert "git diff 'HEAD~1' 'HEAD' failed: UnicodeDecodeError" in " ".join(
            result.stderr.split()
        )
        assert result.stdout == ""
        reached.assert_not_called()


class TestResolvedRefsStillWork:
    def test_a_branch_named_like_a_directory_is_still_a_revision(self, git_repo: Path) -> None:
        # `src` is a directory in the work tree and now also a branch, at HEAD~1.
        _git(git_repo, "branch", "src", "HEAD~1")

        result, reached = _review_git(git_repo, "--base", "src", "--json", comments=[_COMMENT])

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == _FINDING_JSON
        reached.assert_called_once_with()

    def test_the_empty_tree_as_base_reviews_the_first_commit(self, tmp_path: Path) -> None:
        repo = tmp_path / "one"
        (repo / "src").mkdir(parents=True)
        _git(repo, "init", "-q")
        (repo / "src" / "foo.py").write_text("only line\n")
        _git(repo, "add", "src/foo.py")
        _git(repo, "commit", "-q", "-m", "only")

        result, reached = _review_git(
            repo, "--base", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "--json"
        )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == []
        assert "Reviewing 1 hunks across 1 files..." in result.stderr
        reached.assert_called_once_with()

    @pytest.mark.parametrize(
        ("base", "head"),
        [
            pytest.param(":/first", "HEAD", id="commit-message-search"),
            pytest.param("HEAD~1:src", "HEAD:src", id="trees-by-revision-and-path"),
            pytest.param("HEAD~1:src/foo.py", "HEAD:src/foo.py", id="blobs-by-revision-and-path"),
        ],
    )
    def test_ref_forms_that_git_diff_accepts_are_reviewed(
        self, git_repo: Path, base: str, head: str
    ) -> None:
        result, reached = _review_git(
            git_repo, "--base", base, "--head", head, "--json", comments=[_COMMENT]
        )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == _FINDING_JSON
        reached.assert_called_once_with()

    def test_a_valid_base_with_no_changes_says_so_and_exits_0(self, git_repo: Path) -> None:
        result, reached = _review_git(git_repo, "--base", "HEAD")

        assert result.exit_code == 0, result.stderr
        assert "No changes found in diff." in result.stdout
        reached.assert_not_called()

    def test_a_valid_base_with_no_changes_under_json_is_an_empty_array(
        self, git_repo: Path
    ) -> None:
        result, reached = _review_git(git_repo, "--base", "HEAD", "--json")

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == []
        reached.assert_not_called()

    def test_a_valid_base_with_changes_is_reviewed_and_the_json_parses(
        self, git_repo: Path
    ) -> None:
        result, reached = _review_git(
            git_repo, "--base", "HEAD~1", "--head", "HEAD", "--json", comments=[_COMMENT]
        )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == _FINDING_JSON
        assert _PROGRESS in result.stderr
        reached.assert_called_once_with()

    def test_the_reviewer_gets_the_change_from_base_to_head_not_the_reverse(
        self, git_repo: Path
    ) -> None:
        seen: list[Any] = []

        def _review(
            self: DiffReviewer, hunks: list[Any], *args: object, **kwargs: object
        ) -> list[ReviewComment]:
            seen.extend(hunks)
            self.last_outcome = _CLEAN
            return []

        with patch.object(DiffReviewer, "review", _review):
            result = runner.invoke(
                app, ["review", str(git_repo), "--base", "HEAD~1", "--head", "HEAD", "--json"]
            )

        assert result.exit_code == 0, result.stderr
        assert [(h.file_path, h.added, h.removed) for h in seen] == [
            ("src/foo.py", ["added line"], [])
        ]
