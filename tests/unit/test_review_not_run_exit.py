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
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from trelix.cli.main import REVIEW_NOT_RUN_EXIT_CODE, app
from trelix.core.config import LLMConfig
from trelix.llm.providers.openai_backend import OpenAIBackend
from trelix.review.github import PRFile
from trelix.review.reviewer import DiffReviewer, ReviewComment, ReviewOutcome

runner = CliRunner()

_ENV = {"GITHUB_TOKEN": "fake-token-for-test"}
_REPO_ROOT = Path(__file__).parents[2]
_WORKFLOW = _REPO_ROOT / ".github/workflows/trelix-review.yml"

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


def _run_pr_review(outcome: ReviewOutcome, comments: list[ReviewComment], *extra: str) -> Any:
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch.object(DiffReviewer, "review", _fake_review(outcome, comments)),
    ):
        return runner.invoke(app, ["review", "--pr", "owner/repo#1", *extra], env=_ENV)


class TestExitCodeConstant:
    def test_code_does_not_collide_with_existing_meanings(self) -> None:
        # 0 = ok, 1 = error, 2 = click usage error / audit "could not check".
        assert REVIEW_NOT_RUN_EXIT_CODE not in (0, 1, 2)
        # The workflow script and the GitHub App hard-code the same number.
        assert REVIEW_NOT_RUN_EXIT_CODE == 3


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

    def test_partial_failure_keeps_exit_zero_and_warns_with_counts(self) -> None:
        result = _run_pr_review(
            ReviewOutcome(llm_available=True, hunks_total=5, hunks_failed=2),
            [_COMMENT],
            "--json",
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


def _steps() -> list[dict[str, Any]]:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["trelix-review"]["steps"]


def _step(step_id_or_name: str) -> dict[str, Any]:
    for step in _steps():
        if step.get("id") == step_id_or_name or step.get("name") == step_id_or_name:
            return step
    raise AssertionError(f"no step {step_id_or_name!r} in {_WORKFLOW}")


class TestWorkflowDoesNotSwallowTheExitCode:
    def test_review_step_no_longer_swallows_failure(self) -> None:
        run = _step("review")["run"]

        assert "|| true" not in run
        assert "review_exit_code" in run

    def _run_review_step(
        self, tmp_path: Path, fake_exit: int
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "trelix"
        fake.write_text(f'#!/bin/sh\necho "[]"\necho "review stderr" >&2\nexit {fake_exit}\n')
        fake.chmod(0o755)
        github_output = tmp_path / "github_output"
        github_output.write_text("")
        # The step writes to fixed /tmp paths; point them into tmp_path.
        script = _step("review")["run"].replace("/tmp/", f"{tmp_path}/")
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "GITHUB_OUTPUT": str(github_output),
            "GITHUB_TOKEN": "fake-token-for-test",
            "TRELIX_PR_NUMBER": "7",
            "TRELIX_REPO": "owner/repo",
        }
        proc = subprocess.run(  # noqa: S603
            ["bash", "-e", "-c", script], env=env, capture_output=True, text=True, check=False
        )
        return proc, github_output.read_text()

    @pytest.mark.parametrize("fake_exit", [0, 3, 1])
    def test_step_records_the_exit_code_and_still_succeeds(
        self, tmp_path: Path, fake_exit: int
    ) -> None:
        proc, outputs = self._run_review_step(tmp_path, fake_exit)

        assert proc.returncode == 0, proc.stderr
        assert f"review_exit_code={fake_exit}" in outputs.splitlines()
        assert "review stderr" in proc.stdout  # stderr log is still echoed to the job log


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the script")
class TestWorkflowPublishesHonestCheck:
    """Runs the github-script body under node with a recording fake `github`."""

    _HARNESS = """
const fs = require('fs');
const [scriptFile, reviewFile] = process.argv.slice(2);
const script = fs.readFileSync(scriptFile, 'utf8');
const calls = [];
const github = { rest: { checks: { create: async (args) => { calls.push(args); } } } };
// sha is the merge commit of a pull_request event; the PR page shows head.sha.
const context = {
  repo: { owner: 'o', repo: 'r' },
  sha: 'merge000',
  payload: { pull_request: { head: { sha: 'head111' } } },
};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction('github', 'context', 'require', 'process', script)(
  github, context, require, process
).then(() => { console.log(JSON.stringify(calls)); });
"""

    def _publish(self, tmp_path: Path, exit_code: str | None, payload: str) -> dict[str, Any]:
        review_file = tmp_path / "trelix-review.json"
        review_file.write_text(payload)
        script = _step("Post review as Check annotations")["with"]["script"]
        script = script.replace("/tmp/trelix-review.json", str(review_file))
        script_file = tmp_path / "script.js"
        script_file.write_text(script)
        harness = tmp_path / "harness.js"
        harness.write_text(self._HARNESS)
        env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "TRELIX_PR_NUMBER": "7"}
        if exit_code is not None:
            env["TRELIX_REVIEW_EXIT_CODE"] = exit_code
        node = shutil.which("node")
        assert node is not None
        proc = subprocess.run(  # noqa: S603
            [node, str(harness), str(script_file), str(review_file)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        calls = json.loads(proc.stdout.strip().splitlines()[-1])
        assert len(calls) == 1
        return calls[0]

    def test_dedicated_exit_code_publishes_neutral_not_success(self, tmp_path: Path) -> None:
        call = self._publish(tmp_path, "3", "[]")

        assert call["conclusion"] == "neutral"
        assert "0 issue" not in call["output"]["title"]
        assert "did not run" in call["output"]["title"].lower()
        assert "LLM" in call["output"]["summary"]

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

    @pytest.mark.parametrize(
        ("exit_code", "payload"),
        [
            ("3", "[]"),  # did not run
            ("1", ""),  # did not complete
            ("0", "[]"),  # clean
            ("0", '[{"file": "a.py", "lines": "1-1", "severity": "ERROR", "comment": "x"}]'),
        ],
        ids=["did-not-run", "did-not-complete", "clean", "failure"],
    )
    def test_every_check_is_posted_on_the_pull_request_head_commit(
        self, tmp_path: Path, exit_code: str, payload: str
    ) -> None:
        call = self._publish(tmp_path, exit_code, payload)

        # Not context.sha ('merge000'): the merge commit of a pull_request event is on no PR page.
        assert call["head_sha"] == "head111"

    def test_error_finding_still_fails_the_check(self, tmp_path: Path) -> None:
        finding = {"file": "a.py", "lines": "3-4", "severity": "ERROR", "comment": "x"}

        call = self._publish(tmp_path, "0", json.dumps([finding]))

        assert call["conclusion"] == "failure"
