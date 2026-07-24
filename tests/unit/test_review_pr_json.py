"""Regression tests for `trelix review --pr ... --json`'s stdout purity.

Callers (the PR-review CI workflow, and any future GitHub App) redirect
stdout to a file and parse it as JSON. Every status/progress message in the
--pr path must go to stderr instead — a single stray stdout line before the
JSON array corrupts the whole payload.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.review.github import PRFile
from trelix.review.reviewer import ReviewComment

runner = CliRunner()

_ENV = {"GITHUB_TOKEN": "fake-token-for-test"}


def _pr_files(patch_text: str = "@@ -1,1 +1,2 @@\n+added line") -> list[PRFile]:
    return [
        PRFile(
            filename="src/foo.py",
            status="modified",
            additions=1,
            deletions=0,
            patch=patch_text,
        )
    ]


def test_json_mode_stdout_is_valid_json_with_comments():
    comments = [
        ReviewComment(
            file_path="src/foo.py", line_start=1, line_end=1, severity="ERROR", comment="bug here"
        ),
        ReviewComment(
            file_path="src/foo.py", line_start=2, line_end=2, severity="WARN", comment="smell here"
        ),
    ]
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch("trelix.review.reviewer.DiffReviewer.review", return_value=comments),
    ):
        result = runner.invoke(app, ["review", "--pr", "owner/repo#1", "--json"], env=_ENV)

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed == [
        {"file": "src/foo.py", "lines": "1-1", "severity": "ERROR", "comment": "bug here"},
        {"file": "src/foo.py", "lines": "2-2", "severity": "WARN", "comment": "smell here"},
    ]
    # The "Fetching PR diff..." status line must not have leaked onto stdout.
    assert "Fetching PR diff" not in result.stdout


def test_json_mode_stdout_is_empty_json_array_when_no_comments():
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch("trelix.review.reviewer.DiffReviewer.review", return_value=[]),
    ):
        result = runner.invoke(app, ["review", "--pr", "owner/repo#1", "--json"], env=_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert "No issues found" not in result.stdout


def test_json_mode_stdout_is_empty_json_array_when_no_textual_changes():
    with patch(
        "trelix.review.github.GitHubPRClient.get_pr_files",
        return_value=[
            PRFile(filename="image.png", status="modified", additions=0, deletions=0, patch=None)
        ],
    ):
        result = runner.invoke(app, ["review", "--pr", "owner/repo#1", "--json"], env=_ENV)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_non_json_mode_still_prints_status_messages_to_stdout():
    """Confirms the stderr-routing fix is --json-gated, not a blanket behavior change."""
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch("trelix.review.reviewer.DiffReviewer.review", return_value=[]),
    ):
        result = runner.invoke(app, ["review", "--pr", "owner/repo#1"], env=_ENV)

    assert result.exit_code == 0, result.output
    assert "Fetching PR diff" in result.stdout
    assert "No issues found" in result.stdout
