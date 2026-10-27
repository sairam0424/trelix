"""`trelix review` exits 4 when it reviewed part of the diff but left more unreviewed than allowed.

Exit 3 says "nothing came of the review"; exit 4 says "there are findings, but the review did
not cover everything". The findings are printed first, so a caller reading stdout has them.
`TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION` moves the line (default 0.0, any unreviewed hunk;
1.0 restores the old exit 0), and `TRELIX_REVIEW_OUTCOME_FILE` gets a JSON record of what was
and was not reviewed.

Mutations each group was checked against (every one fails a test below): `>` compared as `>=`,
the default fraction changed, the exit raised before the findings are printed, the outcome file
written only on success, the unreviewed count dropped from the posted review body.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.config import IndexConfig
from trelix.review.github import PRFile
from trelix.review.hunk_status import HunkResult, HunkStatus
from trelix.review.reviewer import DiffReviewer, ReviewComment, ReviewOutcome

runner = CliRunner()

_ENV = {"GITHUB_TOKEN": "test-k"}
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


def _outcome(total: int, unreviewed: int, *, kept: int = 1) -> ReviewOutcome:
    """An outcome with `unreviewed` truncated hunks, the first of which kept `kept` findings."""
    results = [
        HunkResult(
            f"f{i}.py", i + 1, HunkStatus.TRUNCATED, "length_after_retry", kept if i == 0 else 0
        )
        for i in range(unreviewed)
    ] + [HunkResult(f"g{i}.py", i + 1, HunkStatus.REVIEWED) for i in range(total - unreviewed)]
    return ReviewOutcome(
        llm_available=True, hunks_total=total, hunks_failed=unreviewed, hunk_results=tuple(results)
    )


def _fake_review(outcome: ReviewOutcome, comments: list[ReviewComment]) -> Any:
    def _review(self: DiffReviewer, *args: object, **kwargs: object) -> list[ReviewComment]:
        self.last_outcome = outcome
        return comments

    return _review


def _pr_files() -> list[PRFile]:
    return [PRFile("src/foo.py", "modified", 1, 0, "@@ -1,1 +1,2 @@\n+added line")]


def _pr(
    outcome: ReviewOutcome,
    comments: list[ReviewComment],
    *extra: str,
    env: dict[str, str] | None = None,
) -> Any:
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch.object(DiffReviewer, "review", _fake_review(outcome, comments)),
    ):
        return runner.invoke(
            app, ["review", "--pr", "owner/repo#1", *extra], env={**_ENV, **(env or {})}
        )


def _local(
    tmp_path: Path,
    outcome: ReviewOutcome,
    comments: list[ReviewComment],
    *extra: str,
    env: dict[str, str] | None = None,
) -> Any:
    diff_file = tmp_path / "changes.diff"
    diff_file.write_text(_DIFF)
    with patch.object(DiffReviewer, "review", _fake_review(outcome, comments)):
        return runner.invoke(
            app, ["review", str(tmp_path), "--diff", str(diff_file), *extra], env=env or {}
        )


class TestTheThreshold:
    @pytest.mark.parametrize(
        ("total", "unreviewed", "fraction", "expected"),
        [
            (4, 0, None, 0),
            (4, 1, None, 4),
            (4, 1, "0.24", 4),
            (4, 1, "0.25", 0),  # exactly at the line is allowed: only MORE than it is incomplete
            (4, 2, "0.5", 0),
            (4, 3, "0.5", 4),
            (4, 4, "1", 0),  # every hunk cut off but a finding kept, and tolerated
            (4, 4, None, 4),
            (1, 1, "0", 4),
        ],
    )
    def test_exit_code_by_share_of_unreviewed_hunks(
        self, total: int, unreviewed: int, fraction: str | None, expected: int
    ) -> None:
        env = {"TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": fraction} if fraction else None

        result = _pr(_outcome(total, unreviewed), [_COMMENT], "--json", env=env)

        assert result.exit_code == expected, result.stderr

    def test_the_findings_are_on_stdout_whatever_the_exit_code(self) -> None:
        result = _pr(_outcome(5, 2), [_COMMENT], "--json")

        assert result.exit_code == 4
        assert json.loads(result.stdout) == _FINDING_JSON

    def test_the_warning_names_the_counts_and_the_escape_hatch(self) -> None:
        result = _pr(_outcome(5, 2), [_COMMENT], "--json")

        assert "2 of 5 hunks could not be fully reviewed" in result.stderr
        assert "TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION=1" in result.stderr

    def test_a_tolerated_partial_review_warns_without_the_hint(self) -> None:
        result = _pr(
            _outcome(5, 2), [_COMMENT], "--json", env={"TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": "1"}
        )

        assert result.exit_code == 0
        assert "2 of 5 hunks could not be fully reviewed" in result.stderr
        assert "Exiting with code 4" not in result.stderr

    def test_a_complete_review_is_silent_and_exits_zero(self) -> None:
        result = _pr(_outcome(3, 0), [_COMMENT], "--json")

        assert result.exit_code == 0
        assert "could not be fully reviewed" not in result.stderr
        assert "Exiting with code 4" not in result.stderr

    def test_the_table_is_printed_before_the_exit(self) -> None:
        result = _pr(_outcome(5, 2), [_COMMENT])

        assert result.exit_code == 4
        assert "smell here" in result.stdout
        assert "Review Results" in result.stdout

    @pytest.mark.parametrize("value", ["-0.1", "1.1", "abc", "nan", "inf", "50"])
    def test_an_out_of_range_fraction_is_rejected_at_config_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION", value)

        with pytest.raises(ValidationError):
            IndexConfig(repo_path=str(tmp_path))

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_fraction_means_the_default_not_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blank: str
    ) -> None:
        # An undefined CI variable arrives as an empty string.
        monkeypatch.setenv("TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION", blank)

        assert IndexConfig(repo_path=str(tmp_path)).review_max_unreviewed_fraction == 0.0

    def test_the_code_default_fraction_is_zero(self) -> None:
        # The test environment pins the variable to its default (tests/_env_isolation.py), so
        # an instance would always say 0.0; read the field's own default instead.
        assert IndexConfig.model_fields["review_max_unreviewed_fraction"].default == 0.0

    def test_the_outcome_file_is_off_by_default(self) -> None:
        assert IndexConfig.model_fields["review_outcome_file"].default is None


class TestNoFindings:
    def test_no_findings_in_a_partial_review_does_not_claim_no_issues(self) -> None:
        # Nothing kept, some hunks reviewed: the reviewed ones were clean, the rest are unknown.
        outcome = _outcome(4, 1, kept=0)

        result = _pr(outcome, [])

        assert result.exit_code == 4
        assert "No issues found" not in result.stdout
        assert "No findings in the hunks that were reviewed" in result.stdout

    def test_a_clean_review_still_says_no_issues(self) -> None:
        result = _pr(_outcome(4, 0), [])

        assert result.exit_code == 0
        assert "No issues found" in result.stdout

    def test_json_mode_prints_an_empty_array_and_exits_four(self) -> None:
        result = _pr(_outcome(4, 1, kept=0), [], "--json")

        assert result.exit_code == 4
        assert json.loads(result.stdout) == []


class TestPostedReview:
    def _post(self, outcome: ReviewOutcome, comments: list[ReviewComment]) -> tuple[Any, MagicMock]:
        poster = MagicMock()
        with (
            patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
            patch("trelix.review.github.GitHubPRClient.get_pr_head_sha", return_value="abc123"),
            patch("trelix.review.github.GitHubPRClient.post_review", poster),
            patch.object(DiffReviewer, "review", _fake_review(outcome, comments)),
        ):
            result = runner.invoke(
                app, ["review", "--pr", "owner/repo#1", "--post-comments"], env=_ENV
            )
        return result, poster

    def test_the_body_says_how_much_was_not_reviewed(self) -> None:
        result, poster = self._post(_outcome(5, 2), [_COMMENT])

        assert result.exit_code == 4
        body = poster.call_args.kwargs["body"]
        assert body == (
            "trelix review: 1 inline comment(s) found. "
            "2 of 5 hunks could not be fully reviewed, so this review may be incomplete."
        )

    def test_a_complete_review_posts_the_old_body(self) -> None:
        result, poster = self._post(_outcome(5, 0), [_COMMENT])

        assert result.exit_code == 0
        assert poster.call_args.kwargs["body"] == "trelix review: 1 inline comment(s) found."


class TestLocalDiffPath:
    def test_json_mode_exits_four_with_the_findings(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _outcome(3, 1), [_COMMENT], "--json")

        assert result.exit_code == 4, result.stderr
        assert json.loads(result.stdout) == _FINDING_JSON

    def test_table_mode_exits_four_with_the_findings(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _outcome(3, 1), [_COMMENT])

        assert result.exit_code == 4, result.stderr
        assert "smell here" in result.stdout

    def test_no_findings_exits_four_for_a_partial_review(self, tmp_path: Path) -> None:
        result = _local(tmp_path, _outcome(3, 1, kept=0), [])

        assert result.exit_code == 4
        assert "No issues found" not in result.stdout

    def test_a_complete_review_exits_zero(self, tmp_path: Path) -> None:
        assert _local(tmp_path, _outcome(3, 0), [_COMMENT]).exit_code == 0

    def test_the_escape_hatch_applies_here_too(self, tmp_path: Path) -> None:
        result = _local(
            tmp_path,
            _outcome(3, 1),
            [_COMMENT],
            env={"TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": "1"},
        )

        assert result.exit_code == 0


class TestTheOutcomeFile:
    def _env(self, path: Path) -> dict[str, str]:
        return {"TRELIX_REVIEW_OUTCOME_FILE": str(path)}

    def test_an_incomplete_review_records_the_unreviewed_hunks(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"

        result = _pr(_outcome(5, 2), [_COMMENT], "--json", env=self._env(target))

        assert result.exit_code == 4
        assert json.loads(target.read_text(encoding="utf-8")) == {
            "schema_version": 1,
            "hunks_total": 5,
            "hunks_reviewed": 3,
            "hunks_unreviewed": 2,
            "exit_code": 4,
            "hunks": [
                {"file": "f0.py", "line": 1, "status": "truncated", "detail": "length_after_retry"},
                {"file": "f1.py", "line": 2, "status": "truncated", "detail": "length_after_retry"},
            ],
            "hunks_omitted": 0,
        }

    def test_a_review_that_did_not_run_records_exit_three(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"
        results = (HunkResult("a.py", 1, HunkStatus.ERROR, "not_configured"),)
        outcome = ReviewOutcome(
            llm_available=False, hunks_total=1, hunks_failed=1, hunk_results=results
        )

        result = _pr(outcome, [], "--json", env=self._env(target))

        assert result.exit_code == 3
        recorded = json.loads(target.read_text(encoding="utf-8"))
        assert recorded["exit_code"] == 3
        assert recorded["hunks"] == [
            {"file": "a.py", "line": 1, "status": "error", "detail": "not_configured"}
        ]

    def test_a_clean_review_records_exit_zero_and_no_hunks(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"

        result = _pr(_outcome(4, 0), [], "--json", env=self._env(target))

        assert result.exit_code == 0
        recorded = json.loads(target.read_text(encoding="utf-8"))
        assert (recorded["exit_code"], recorded["hunks"], recorded["hunks_reviewed"]) == (0, [], 4)

    def test_a_tolerated_partial_review_records_exit_zero_with_its_hunks(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "outcome.json"
        env = {**self._env(target), "TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": "1"}

        result = _pr(_outcome(4, 1), [_COMMENT], "--json", env=env)

        assert result.exit_code == 0
        recorded = json.loads(target.read_text(encoding="utf-8"))
        assert recorded["exit_code"] == 0
        assert recorded["hunks_unreviewed"] == 1

    def test_the_local_path_writes_it_too(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"

        result = _local(tmp_path, _outcome(3, 1), [_COMMENT], env=self._env(target))

        assert result.exit_code == 4
        assert json.loads(target.read_text(encoding="utf-8"))["exit_code"] == 4

    def test_nothing_is_written_without_the_variable(self) -> None:
        with patch("trelix.review.outcome_file.write_outcome_file") as writer:
            result = _pr(_outcome(4, 1), [_COMMENT], "--json")

        assert result.exit_code == 4
        writer.assert_not_called()

    def test_a_blank_variable_writes_nothing(self) -> None:
        with patch("trelix.review.outcome_file.write_outcome_file") as writer:
            _pr(_outcome(4, 1), [_COMMENT], "--json", env={"TRELIX_REVIEW_OUTCOME_FILE": "  "})

        writer.assert_not_called()

    def test_the_record_exists_before_comments_are_posted(self, tmp_path: Path) -> None:
        # A post that hangs until the caller's timeout kills the process must not cost the record.
        target = tmp_path / "outcome.json"
        seen: list[bool] = []

        def _post(*args: object, **kwargs: object) -> None:
            seen.append(target.exists())

        with (
            patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
            patch("trelix.review.github.GitHubPRClient.get_pr_head_sha", return_value="abc123"),
            patch("trelix.review.github.GitHubPRClient.post_review", side_effect=_post),
            patch.object(DiffReviewer, "review", _fake_review(_outcome(5, 2), [_COMMENT])),
        ):
            result = runner.invoke(
                app,
                ["review", "--pr", "owner/repo#1", "--post-comments"],
                env={**_ENV, **self._env(target)},
            )

        assert result.exit_code == 4
        assert seen == [True]

    @pytest.mark.parametrize("path", [".", "/", "./"])
    def test_a_path_with_no_file_name_warns_and_changes_no_exit_code(self, path: str) -> None:
        result = _pr(_outcome(4, 1), [_COMMENT], "--json", env={"TRELIX_REVIEW_OUTCOME_FILE": path})

        assert result.exit_code == 4, result.output
        assert "could not write the review outcome file" in result.stderr
        assert "Traceback" not in result.output

    def test_a_review_of_no_hunks_records_zeros_and_exits_zero(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"

        result = _pr(ReviewOutcome(), [], "--json", env=self._env(target))

        assert result.exit_code == 0
        recorded = json.loads(target.read_text(encoding="utf-8"))
        assert (recorded["hunks_total"], recorded["hunks_unreviewed"], recorded["exit_code"]) == (
            0,
            0,
            0,
        )

    def test_the_local_path_records_exit_three_too(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"
        results = (HunkResult("a.py", 1, HunkStatus.ERROR, "not_configured"),)
        outcome = ReviewOutcome(
            llm_available=False, hunks_total=1, hunks_failed=1, hunk_results=results
        )

        result = _local(tmp_path, outcome, [], "--json", env=self._env(target))

        assert result.exit_code == 3
        assert json.loads(target.read_text(encoding="utf-8"))["exit_code"] == 3

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_variable_means_no_file(self, tmp_path: Path, blank: str) -> None:
        result = _pr(
            _outcome(4, 1), [_COMMENT], "--json", env={"TRELIX_REVIEW_OUTCOME_FILE": blank}
        )

        assert result.exit_code == 4
        assert "outcome file" not in result.stderr

    def test_a_file_that_cannot_be_written_warns_and_changes_no_exit_code(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "no-such-directory" / "outcome.json"

        result = _pr(_outcome(4, 1), [_COMMENT], "--json", env=self._env(target))

        assert result.exit_code == 4
        assert "could not write the review outcome file" in result.stderr
        assert json.loads(result.stdout) == _FINDING_JSON

    def test_the_unwritable_file_does_not_turn_a_clean_review_red(self, tmp_path: Path) -> None:
        target = tmp_path / "no-such-directory" / "outcome.json"

        result = _pr(_outcome(4, 0), [], "--json", env=self._env(target))

        assert result.exit_code == 0

    def test_a_long_list_is_capped_in_the_file(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"

        _pr(_outcome(150, 120), [_COMMENT], "--json", env=self._env(target))

        recorded = json.loads(target.read_text(encoding="utf-8"))
        assert len(recorded["hunks"]) == 100
        assert recorded["hunks_omitted"] == 20
        assert recorded["hunks_unreviewed"] == 120
