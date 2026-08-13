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


# ---------------------------------------------------------------------------
# stdout purity, part two: Rich must not touch the payload it is handed
# ---------------------------------------------------------------------------
#
# The tests above pin that nothing EXTRA reaches stdout. These pin that the
# payload itself arrives unaltered, which is a different failure: this array was
# emitted with console.print(), so Rich parsed and re-flowed it on the way out.
# Both modes below are reached by ordinary review output, no hostile input
# needed, and both are invisible to the tests above because those use short,
# bracket-free comments. See _print_json() in trelix/cli/main.py.


def test_json_mode_survives_markup_shaped_comment_and_path():
    """A "[/red]" in a comment or path is an unmatched Rich closing tag.

    Review comments are LLM prose quoting the diff, so they inherit every
    bracket in the reviewed code, and file paths may hold brackets legally.
    console.print() raised MarkupError on the first one, so `review --json`
    exited nonzero having written nothing at all to the file CI parses.
    """
    hostile_path = "src/w[/red]/handler.py"
    hostile_comment = "Guard this: re.sub(r\"^//[/!]?\\s?\", '', raw)"
    comments = [
        ReviewComment(
            file_path=hostile_path,
            line_start=1,
            line_end=1,
            severity="ERROR",
            comment=hostile_comment,
        )
    ]
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch("trelix.review.reviewer.DiffReviewer.review", return_value=comments),
    ):
        result = runner.invoke(app, ["review", "--pr", "owner/repo#1", "--json"], env=_ENV)

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    # Byte-for-byte, not merely parseable: escaping instead of disabling markup
    # would round-trip valid JSON while writing stray backslashes into the
    # consumer's strings, which is the reason _print_json exists.
    assert parsed[0]["file"] == hostile_path
    assert parsed[0]["comment"] == hostile_comment
    assert "\\[" not in result.stdout


def test_json_mode_survives_long_comment_without_wrap_corruption():
    """A long comment must not be hard-wrapped into unparseable JSON.

    Rich wraps at the console width (80 under CliRunner, since stdout is not a
    terminal). A wrap landing inside a JSON *string* injects a raw newline and
    json.loads rejects it as "Invalid control character". Whitespace BETWEEN
    JSON tokens is semantically free, which is why the short-comment tests above
    never caught this — it takes one line longer than the terminal, which any
    review comment quoting a long symbol name produces.
    """
    long_comment = (
        "Rename this before merging: "
        "AbstractSingletonProxyFactoryBeanConfigurationProviderRegistry "
        "is referenced in 14 call sites and none of them need the indirection."
    )
    comments = [
        ReviewComment(
            file_path="src/foo.py",
            line_start=1,
            line_end=1,
            severity="WARN",
            comment=long_comment,
        )
    ]
    with (
        patch("trelix.review.github.GitHubPRClient.get_pr_files", return_value=_pr_files()),
        patch("trelix.review.reviewer.DiffReviewer.review", return_value=comments),
    ):
        result = runner.invoke(app, ["review", "--pr", "owner/repo#1", "--json"], env=_ENV)

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed[0]["comment"] == long_comment, "Rich re-flowed the comment text"
