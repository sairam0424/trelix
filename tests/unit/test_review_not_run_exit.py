"""
`trelix review` must not report success when nothing was reviewed.

With no usable LLM key, or when every hunk's LLM call failed, the command used to
print "No issues found." (or `[]`) and exit 0, and the repo's own PR workflow then
published a green "found 0 issue(s)" check for a review that never ran. These tests
pin the dedicated exit code, the stdout/stderr split, and the workflow that consumes
the code.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tests.unit.review_workflow_harness import (
    NODE,
    OUTCOME_FILE_PATH,
    findings_json,
    load_cases,
    outcome_text,
    publish_script,
    run_publish_script,
    stdout_text,
    workflow_step,
)
from trelix.cli.main import REVIEW_INCOMPLETE_EXIT_CODE, REVIEW_NOT_RUN_EXIT_CODE, app
from trelix.core.config import LLMConfig
from trelix.llm.providers.openai_backend import OpenAIBackend
from trelix.review.github import PRFile
from trelix.review.hunk_status import HunkResult, HunkStatus
from trelix.review.reviewer import DiffReviewer, ReviewComment, ReviewOutcome

runner = CliRunner()

_ENV = {"GITHUB_TOKEN": "fake-token-for-test"}

_COMMENT = ReviewComment(
    file_path="src/foo.py", line_start=1, line_end=1, severity="WARN", comment="smell here"
)


def _pr_files() -> list[PRFile]:
    return [
        PRFile(
            filename="src/foo.py",
            status="modified",
            additions=1,
            deletions=0,
            patch="@@ -1,1 +1,2 @@\n+added line",
        )
    ]


def _fake_review(outcome: ReviewOutcome, comments: list[ReviewComment]) -> Any:
    def _review(self: DiffReviewer, *args: object, **kwargs: object) -> list[ReviewComment]:
        self.last_outcome = outcome
        return comments

    return _review


def _run_pr_review(
    outcome: ReviewOutcome,
    comments: list[ReviewComment],
    *extra: str,
    env_extra: dict[str, str] | None = None,
) -> Any:
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch.object(DiffReviewer, "review", _fake_review(outcome, comments)),
    ):
        return runner.invoke(
            app, ["review", "--pr", "owner/repo#1", *extra], env={**_ENV, **(env_extra or {})}
        )


class TestExitCodeConstant:
    def test_code_does_not_collide_with_existing_meanings(self) -> None:
        # 0 = ok, 1 = error, 2 = click usage error / audit "could not check".
        assert REVIEW_NOT_RUN_EXIT_CODE not in (0, 1, 2)
        # The workflow script and the GitHub App hard-code the same number.
        assert REVIEW_NOT_RUN_EXIT_CODE == 3

    def test_the_incomplete_review_code_is_its_own_number(self) -> None:
        # Distinct from 3, so a wrapper can tell "no review" from "a partial review".
        assert REVIEW_INCOMPLETE_EXIT_CODE not in (0, 1, 2, 3)
        # The workflow script and the GitHub App hard-code the same number.
        assert REVIEW_INCOMPLETE_EXIT_CODE == 4


class TestCutOffHunksThatKeptFindings:
    """A hunk cut off after some complete findings is a partial review, not one that did not run.

    Without this, a review in which every hunk was cut off would exit 3 and print `[]` whenever
    no finding survived, and would silently drop the findings whenever one did.
    """

    def test_every_hunk_cut_off_but_one_finding_kept_shows_it_and_warns(self) -> None:
        outcome = ReviewOutcome(
            llm_available=True,
            hunks_total=3,
            hunks_failed=3,
            hunk_results=(
                HunkResult("src/foo.py", 1, HunkStatus.TRUNCATED, "length_after_retry", 1),
                HunkResult("src/bar.py", 2, HunkStatus.TRUNCATED, "length_after_retry", 0),
                HunkResult("src/baz.py", 3, HunkStatus.PARSE_FAILED, "no_review_array", 0),
            ),
        )

        result = _run_pr_review(outcome, [_COMMENT], "--json")

        # Findings exist, so this is not "did not run" (3); it is an incomplete review (4).
        assert result.exit_code == 4, result.stderr
        assert json.loads(result.stdout) == [
            {"file": "src/foo.py", "lines": "1-1", "severity": "WARN", "comment": "smell here"}
        ]
        assert "3 of 3 hunks could not be fully reviewed" in result.stderr
        assert "did not run" not in result.stderr

    def test_the_same_outcome_with_no_finding_kept_is_still_a_review_that_did_not_run(self) -> None:
        outcome = ReviewOutcome(
            llm_available=True,
            hunks_total=2,
            hunks_failed=2,
            hunk_results=(
                HunkResult("src/foo.py", 1, HunkStatus.TRUNCATED, "length", 0),
                HunkResult("src/bar.py", 2, HunkStatus.REFUSED, "refusal", 0),
            ),
        )

        result = _run_pr_review(outcome, [], "--json")

        assert result.exit_code == 3
        assert json.loads(result.stdout) == []


class TestPrPathExitsNonZeroWhenNotReviewed:
    def test_no_llm_json_mode_exits_with_dedicated_code_and_empty_array(self) -> None:
        result = _run_pr_review(
            ReviewOutcome(llm_available=False, hunks_total=3, hunks_failed=3), [], "--json"
        )

        assert result.exit_code == 3, result.stderr
        assert json.loads(result.stdout) == []
        assert "not configured" in result.stderr.lower()

    def test_no_llm_text_mode_does_not_claim_no_issues(self) -> None:
        result = _run_pr_review(
            ReviewOutcome(llm_available=False, hunks_total=3, hunks_failed=3), []
        )

        assert result.exit_code == 3
        assert "No issues found" not in result.stdout
        assert "not configured" in result.stderr.lower()

    def test_every_hunk_failed_exits_with_dedicated_code(self) -> None:
        result = _run_pr_review(
            ReviewOutcome(llm_available=True, hunks_total=4, hunks_failed=4), [], "--json"
        )

        assert result.exit_code == 3
        assert json.loads(result.stdout) == []
        assert "4 of 4" in result.stderr

    def test_partial_failure_exits_four_keeps_the_findings_and_warns_with_counts(self) -> None:
        result = _run_pr_review(
            ReviewOutcome(llm_available=True, hunks_total=5, hunks_failed=2),
            [_COMMENT],
            "--json",
        )

        assert result.exit_code == 4, result.stderr
        assert json.loads(result.stdout) == [
            {"file": "src/foo.py", "lines": "1-1", "severity": "WARN", "comment": "smell here"}
        ]
        assert "2 of 5" in result.stderr

    def test_the_escape_hatch_restores_exit_zero_for_a_partial_review(self) -> None:
        result = _run_pr_review(
            ReviewOutcome(llm_available=True, hunks_total=5, hunks_failed=2),
            [_COMMENT],
            "--json",
            env_extra={"TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": "1"},
        )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == [
            {"file": "src/foo.py", "lines": "1-1", "severity": "WARN", "comment": "smell here"}
        ]
        assert "2 of 5" in result.stderr

    def test_clean_review_is_unchanged(self) -> None:
        result = _run_pr_review(
            ReviewOutcome(llm_available=True, hunks_total=2, hunks_failed=0), [], "--json"
        )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == []
        assert "not configured" not in result.stderr.lower()
        assert "hunk" not in result.stderr.lower()

    def test_findings_with_unavailable_llm_cannot_happen_but_are_not_discarded(self) -> None:
        """If comments exist they were produced by a model; never drop them."""
        result = _run_pr_review(
            ReviewOutcome(llm_available=True, hunks_total=1, hunks_failed=0), [_COMMENT], "--json"
        )

        assert result.exit_code == 0
        assert len(json.loads(result.stdout)) == 1

    def test_end_to_end_with_a_keyless_backend(self, tmp_path: Path) -> None:
        """No mocks below the client: the real placeholder path from a real backend."""
        keyless = LLMConfig(provider="openai", _env_file=None)  # type: ignore[call-arg]
        backend = OpenAIBackend(keyless.model_copy(update={"openai_api_key": None}))

        with (
            patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
            patch.object(DiffReviewer, "_get_client", return_value=backend),
            patch.object(DiffReviewer, "_get_retriever"),
        ):
            result = runner.invoke(
                app, ["review", str(tmp_path), "--pr", "owner/repo#1", "--json"], env=_ENV
            )

        assert result.exit_code == 3, result.stderr
        assert json.loads(result.stdout) == []


class TestLocalDiffPath:
    _DIFF = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,1 +1,2 @@\n"
        " old\n"
        "+added line\n"
    )

    def _invoke(self, tmp_path: Path, outcome: ReviewOutcome, *extra: str) -> Any:
        diff_file = tmp_path / "changes.diff"
        diff_file.write_text(self._DIFF)
        with patch.object(DiffReviewer, "review", _fake_review(outcome, [])):
            return runner.invoke(app, ["review", str(tmp_path), "--diff", str(diff_file), *extra])

    def test_not_reviewed_exits_with_dedicated_code(self, tmp_path: Path) -> None:
        result = self._invoke(
            tmp_path, ReviewOutcome(llm_available=False, hunks_total=1, hunks_failed=1), "--json"
        )

        assert result.exit_code == 3, result.stderr
        assert "No issues found" not in result.stdout
        assert "not configured" in result.stderr.lower()

    def test_clean_review_still_exits_zero(self, tmp_path: Path) -> None:
        result = self._invoke(
            tmp_path, ReviewOutcome(llm_available=True, hunks_total=1, hunks_failed=0)
        )

        assert result.exit_code == 0
        assert "No issues found" in result.stdout


# ---------------------------------------------------------------------------
# The workflow that consumes the exit code
# ---------------------------------------------------------------------------


_CASES = load_cases()
_ROWS = [c for c in _CASES["cases"] if "workflow" in c.get("applies_to", ["workflow", "app"])]

_VALID_OUTCOME = json.dumps(_CASES["valid_outcome"])


class TestWorkflowDoesNotSwallowTheExitCode:
    def test_review_step_no_longer_swallows_failure(self) -> None:
        run = workflow_step("review")["run"]

        assert "|| true" not in run
        assert "review_exit_code" in run

    def _run_review_step(
        self, tmp_path: Path, fake_exit: int, *, trelix_body: str = ""
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "trelix"
        fake.write_text(
            f'#!/bin/sh\necho "[]"\necho "review stderr" >&2\n{trelix_body}\nexit {fake_exit}\n'
        )
        fake.chmod(0o755)
        github_output = tmp_path / "github_output"
        github_output.write_text("")
        # The step writes to fixed /tmp paths; point them into tmp_path.
        step = workflow_step("review")
        script = step["run"].replace("/tmp/", f"{tmp_path}/")
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "GITHUB_OUTPUT": str(github_output),
            "GITHUB_TOKEN": "fake-token-for-test",
            "TRELIX_PR_NUMBER": "7",
            "TRELIX_REPO": "owner/repo",
            # What the workflow's own `env:` block gives the step, with the path moved.
            "TRELIX_REVIEW_OUTCOME_FILE": step["env"]["TRELIX_REVIEW_OUTCOME_FILE"].replace(
                "/tmp/", f"{tmp_path}/"
            ),
        }
        proc = subprocess.run(  # noqa: S603
            ["bash", "-e", "-c", script], env=env, capture_output=True, text=True, check=False
        )
        return proc, github_output.read_text()

    @pytest.mark.parametrize("fake_exit", [0, 3, 4, 1])
    def test_step_records_the_exit_code_and_still_succeeds(
        self, tmp_path: Path, fake_exit: int
    ) -> None:
        proc, outputs = self._run_review_step(tmp_path, fake_exit)

        assert proc.returncode == 0, proc.stderr
        assert f"review_exit_code={fake_exit}" in outputs.splitlines()
        assert "review stderr" in proc.stdout  # stderr log is still echoed to the job log

    def test_the_review_is_told_where_to_write_its_outcome_record(self, tmp_path: Path) -> None:
        # The fake trelix writes to the path it was given, as the real one does.
        proc, _ = self._run_review_step(
            tmp_path, 4, trelix_body='echo "recorded" > "$TRELIX_REVIEW_OUTCOME_FILE"'
        )

        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "trelix-review-outcome.json").read_text() == "recorded\n"

    def test_a_record_left_at_that_path_by_an_earlier_step_is_removed_first(
        self, tmp_path: Path
    ) -> None:
        stale = tmp_path / "trelix-review-outcome.json"
        stale.write_text("planted before the review ran")

        # A review that writes nothing (an old trelix, or one that stopped early).
        proc, _ = self._run_review_step(tmp_path, 4)

        assert proc.returncode == 0, proc.stderr
        assert not stale.exists()

    def test_both_steps_use_the_same_outcome_path(self) -> None:
        # Literals, not the constant from the harness: the path is the contract between the
        # review step's `env:` and the fixed path the publishing script reads.
        assert workflow_step("review")["env"]["TRELIX_REVIEW_OUTCOME_FILE"] == (
            "/tmp/trelix-review-outcome.json"
        )
        assert "const outcomeFile = '/tmp/trelix-review-outcome.json';" in publish_script()
        assert OUTCOME_FILE_PATH == "/tmp/trelix-review-outcome.json"


@pytest.mark.skipif(NODE is None, reason="node is required to run the script")
class TestWorkflowPublishesHonestCheck:
    """Runs the github-script body under node with a recording fake `github`."""

    def _publish(
        self, tmp_path: Path, exit_code: str | None, payload: str | None, outcome: str | None = None
    ) -> dict[str, Any]:
        return run_publish_script(tmp_path, exit_code, payload, outcome=outcome).call

    def test_dedicated_exit_code_publishes_neutral_not_success(self, tmp_path: Path) -> None:
        call = self._publish(tmp_path, "3", "[]")

        assert call["conclusion"] == "neutral"
        assert "0 issue" not in call["output"]["title"]
        assert "did not run" in call["output"]["title"].lower()
        assert "LLM" in call["output"]["summary"]

    def test_the_did_not_run_text_matches_what_exit_3_now_means(self, tmp_path: Path) -> None:
        summary = self._publish(tmp_path, "3", "[]")["output"]["summary"]

        # The stale claim was "every LLM call failed": a cut-off, refused or unparseable reply
        # with no finding kept is also exit 3, and a hunk cut off after findings is not.
        assert "every LLM call failed" not in summary
        assert "no hunk got a usable review and none kept a finding" in summary

    def test_other_nonzero_exit_is_also_not_a_success(self, tmp_path: Path) -> None:
        call = self._publish(tmp_path, "1", "")

        assert call["conclusion"] == "neutral"
        assert "exit code 1" in call["output"]["summary"]

    def test_missing_exit_code_is_not_a_success(self, tmp_path: Path) -> None:
        call = self._publish(tmp_path, None, "[]")

        assert call["conclusion"] == "neutral"

    def test_successful_run_behaves_exactly_as_before(self, tmp_path: Path) -> None:
        finding = {"file": "a.py", "lines": "3-4", "severity": "WARN", "comment": "x"}

        clean = self._publish(tmp_path, "0", "[]")
        found = self._publish(tmp_path, "0", json.dumps([finding]))

        assert clean["conclusion"] == "success"
        assert clean["output"]["title"] == "trelix found 0 issue(s)"
        assert found["conclusion"] == "success"
        assert found["output"]["annotations"][0]["annotation_level"] == "warning"

    @pytest.mark.parametrize("case", _ROWS, ids=[c["id"] for c in _ROWS])
    def test_every_row_of_the_shared_table_gets_its_conclusion_and_title(
        self, tmp_path: Path, case: dict[str, Any]
    ) -> None:
        """The same rows the GitHub App's tests run (tests/review-outcome.test.ts)."""
        call = self._publish(
            tmp_path,
            case["exit_code"],
            stdout_text(case),
            outcome_text(case.get("outcome"), _CASES),
        )

        assert call["conclusion"] == case["conclusion"]
        assert call["output"]["title"] == case["title"]

    def test_the_shared_table_is_not_vacuous(self) -> None:
        """A cut-down or emptied fixture must not turn the row test above into a pass."""
        assert len(_ROWS) >= 14
        assert {c["conclusion"] for c in _ROWS} == {"success", "failure", "neutral"}
        assert {c["exit_code"] for c in _ROWS} >= {"0", "1", "2", "3", "4", "5", None}
        assert {c.get("outcome") for c in _ROWS} == {None, "valid", "missing", "corrupt"}
        # Rows for output that is not a findings array, after a clean exit and after exit 4.
        unreadable = [c for c in _ROWS if "stdout_text" in c]
        assert {c["exit_code"] for c in unreadable} == {"0", "4"}
        assert len(unreadable) >= 6
        assert {c["conclusion"] for c in unreadable} == {"neutral"}

    @pytest.mark.parametrize(
        ("exit_code", "payload", "outcome"),
        [
            ("3", "[]", None),  # did not run
            ("1", "", None),  # did not complete
            ("0", "[]", None),  # clean
            ("0", '[{"file": "a.py", "lines": "1-1", "severity": "ERROR", "comment": "x"}]', None),
            ("4", '[{"file": "a.py", "lines": "1-1", "severity": "WARN", "comment": "x"}]', "ok"),
            ("4", "[]", None),  # incomplete, no record
        ],
        ids=["did-not-run", "did-not-complete", "clean", "failure", "incomplete", "no-record"],
    )
    def test_every_check_is_posted_on_the_pull_request_head_commit(
        self, tmp_path: Path, exit_code: str, payload: str, outcome: str | None
    ) -> None:
        call = self._publish(
            tmp_path, exit_code, payload, _VALID_OUTCOME if outcome == "ok" else None
        )

        # Not context.sha ('merge000'): the merge commit of a pull_request event is on no PR page.
        assert call["head_sha"] == "head111"

    @pytest.mark.parametrize(
        "stdout",
        [
            None,  # no file at all
            "",
            "not json",
            '[{"file": "a.py"',  # cut off
            "{}",
            '{"findings": []}',
            "null",
            '"x"',
            "5",
            "true",
            "[null]",
            "[1]",
            '["x"]',
            "[[]]",
            '[{"file": "a.py", "lines": "1-1", "severity": "ERROR", "comment": "x"}, null]',
        ],
        ids=[
            "no-file",
            "empty",
            "text",
            "cut-off",
            "object",
            "object-with-findings",
            "null",
            "string",
            "number",
            "true",
            "array-of-null",
            "array-of-number",
            "array-of-string",
            "array-of-array",
            "error-finding-then-null",
        ],
    )
    def test_a_clean_exit_with_findings_that_cannot_be_read_is_not_a_success(
        self, tmp_path: Path, stdout: str | None
    ) -> None:
        """Exit 0 says the review ran. If what it printed is not a findings array, "found 0
        issue(s)" would be a claim nobody checked (the GitHub App posts "did not complete" for
        the same output, and the step used to throw on an object, null or a number)."""
        call = self._publish(tmp_path, "0", stdout)

        assert call["conclusion"] == "neutral"
        assert call["output"]["title"] == "trelix review did not complete"
        assert "issue(s)" not in call["output"]["title"]
        assert "annotations" not in call["output"]
        assert call["output"]["summary"] == (
            "trelix review finished on PR #7, but the findings it printed could not be read, so "
            "no review result is available. This is not a clean result. "
            'See the "Run trelix review" step log.'
        )
        assert call["head_sha"] == "head111"

    @pytest.mark.parametrize("stdout", ["[]", " [ ] ", "[]\n"])
    def test_an_empty_findings_array_is_still_a_clean_result(
        self, tmp_path: Path, stdout: str
    ) -> None:
        call = self._publish(tmp_path, "0", stdout)

        assert call["conclusion"] == "success"
        assert call["output"]["title"] == "trelix found 0 issue(s)"

    def test_error_finding_still_fails_the_check(self, tmp_path: Path) -> None:
        finding = {"file": "a.py", "lines": "3-4", "severity": "ERROR", "comment": "x"}

        call = self._publish(tmp_path, "0", json.dumps([finding]))

        assert call["conclusion"] == "failure"

    def test_an_error_beyond_the_fiftieth_finding_still_fails_the_check(
        self, tmp_path: Path
    ) -> None:
        # GitHub takes 50 annotations per request; the verdict is about every finding.
        severities = ["INFO"] * 55 + ["ERROR"]

        complete = self._publish(tmp_path, "0", findings_json(severities))
        incomplete = self._publish(tmp_path, "4", findings_json(severities), _VALID_OUTCOME)

        assert complete["conclusion"] == "failure"
        assert incomplete["conclusion"] == "failure"
        assert len(complete["output"]["annotations"]) == 50
        assert len(incomplete["output"]["annotations"]) == 50


@pytest.mark.skipif(NODE is None, reason="node is required to run the script")
class TestWorkflowPublishesAnIncompleteReview:
    """Exit 4: the findings there are, and what was left out, in a neutral or failed check."""

    def _publish(
        self, tmp_path: Path, severities: list[str], outcome: str | None = _VALID_OUTCOME
    ) -> dict[str, Any]:
        return run_publish_script(tmp_path, "4", findings_json(severities), outcome=outcome).call

    def test_the_summary_says_what_was_covered_and_lists_what_was_not(self, tmp_path: Path) -> None:
        call = self._publish(tmp_path, ["WARN"])

        assert call["conclusion"] == "neutral"
        assert call["output"]["title"] == "trelix review incomplete"
        assert call["output"]["summary"] == (
            "trelix reviewed only part of PR #7: 3 of 5 hunks were reviewed and 2 were not. "
            "This is not a clean result.\n"
            "\n"
            "1 issue(s) found in the hunks that were reviewed.\n"
            "\n"
            "Hunks that were not reviewed:\n"
            "- `src/a.py:10` (truncated)\n"
            "- `src/b.py:20` (refused)\n"
            "\n"
            'See the "Run trelix review" step log for the reason.'
        )

    def test_the_findings_are_still_posted_as_annotations(self, tmp_path: Path) -> None:
        call = self._publish(tmp_path, ["INFO", "WARN"])

        assert call["output"]["annotations"] == [
            {
                "path": "src/f0.py",
                "start_line": 1,
                "end_line": 1,
                "annotation_level": "notice",
                "message": "c",
                "title": "trelix review",
            },
            {
                "path": "src/f1.py",
                "start_line": 2,
                "end_line": 2,
                "annotation_level": "warning",
                "message": "c",
                "title": "trelix review",
            },
        ]

    def test_an_error_finding_fails_the_check_and_is_annotated_as_a_failure(
        self, tmp_path: Path
    ) -> None:
        call = self._publish(tmp_path, ["WARN", "ERROR"])

        assert call["conclusion"] == "failure"
        assert call["output"]["title"] == "trelix review incomplete"
        assert [a["annotation_level"] for a in call["output"]["annotations"]] == [
            "warning",
            "failure",
        ]

    def test_no_findings_is_said_in_words_and_is_not_a_clean_result(self, tmp_path: Path) -> None:
        call = self._publish(tmp_path, [])

        assert call["conclusion"] == "neutral"
        assert "No findings in the hunks that were reviewed." in call["output"]["summary"]
        assert "This is not a clean result." in call["output"]["summary"]
        assert call["output"]["annotations"] == []

    def test_no_model_prose_reaches_the_summary(self, tmp_path: Path) -> None:
        finding = {
            "file": "a.py",
            "lines": "1-1",
            "severity": "WARN",
            "comment": "canary-model-prose [click](https://attacker.example/canary)",
        }

        call = run_publish_script(tmp_path, "4", json.dumps([finding]), outcome=_VALID_OUTCOME).call

        assert "canary-model-prose" not in call["output"]["summary"]
        assert "attacker.example" not in call["output"]["summary"]

    def test_the_detail_strings_of_the_record_are_not_posted(self, tmp_path: Path) -> None:
        record = json.loads(_VALID_OUTCOME)
        record["hunks"][0]["detail"] = "canary-detail-string"

        call = self._publish(tmp_path, [], json.dumps(record))

        assert "canary-detail-string" not in call["output"]["summary"]

    def test_a_record_without_the_findings_file_says_the_findings_could_not_be_read(
        self, tmp_path: Path
    ) -> None:
        call = run_publish_script(tmp_path, "4", "not json at all", outcome=_VALID_OUTCOME).call

        assert call["conclusion"] == "neutral"
        assert (
            "The list of findings could not be read, so none are shown."
            in (call["output"]["summary"])
        )
        assert "No findings in the hunks that were reviewed." not in call["output"]["summary"]
        assert call["output"]["annotations"] == []

    def test_findings_that_are_not_an_array_are_unreadable_too(self, tmp_path: Path) -> None:
        call = run_publish_script(
            tmp_path, "4", '{"severity": "ERROR"}', outcome=_VALID_OUTCOME
        ).call

        assert call["conclusion"] == "neutral"
        assert "could not be read" in call["output"]["summary"]

    @pytest.mark.parametrize("stdout", ["[null]", "[1]", '["x"]', "[[]]", "null"])
    def test_an_array_holding_something_that_is_not_an_object_is_unreadable_and_still_posts(
        self, tmp_path: Path, stdout: str
    ) -> None:
        # The step used to throw on the null, and no check was posted at all.
        finding = {"file": "a.py", "lines": "1-1", "severity": "ERROR", "comment": "x"}
        mixed = json.dumps([finding])[:-1] + ", null]"

        for text in (stdout, mixed):
            call = run_publish_script(tmp_path, "4", text, outcome=_VALID_OUTCOME).call

            assert call["conclusion"] == "neutral"
            assert call["output"]["title"] == "trelix review incomplete"
            assert (
                "The list of findings could not be read, so none are shown."
                in (call["output"]["summary"])
            )
            assert call["output"]["annotations"] == []
