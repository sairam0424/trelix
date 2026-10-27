"""`trelix review` on a repository that was never indexed: still reviews, creates nothing.

`DiffReviewer` builds its Retriever lazily, on the first hunk. On a repository with no index
that construction created an empty `.trelix/index.db`, retrieved nothing, and the review went
ahead from the diff alone. The PR workflow and the GitHub App both tolerate a failed
`trelix index` and run `trelix review` anyway, so the review must keep working there. The
only change is that it no longer leaves an empty index behind: a reviewed repository must not
start looking indexed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tests.fixtures.indexed import mark_indexed
from trelix.cli.main import app
from trelix.core.config import IndexConfig
from trelix.llm.client import ChatResponse
from trelix.review.diff_parser import DiffHunk
from trelix.review.reviewer import DiffReviewer

runner = CliRunner()

_ONE_COMMENT = (
    '[{"line_start": 1, "line_end": 2, "severity": "WARN", "comment": "sign flip in add()"}]'
)
_DIFF = (
    "diff --git a/calc.py b/calc.py\n"
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a + b\n"
    "+    return a - b\n"
)


def _hunk() -> DiffHunk:
    return DiffHunk(
        file_path="calc.py",
        old_start=1,
        new_start=1,
        old_lines=2,
        new_lines=2,
        added=["    return a - b"],
        removed=["    return a + b"],
        context=["def add(a, b):"],
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return root


def _reviewer(repo: Path) -> DiffReviewer:
    reviewer = DiffReviewer(IndexConfig(repo_path=str(repo)))
    reviewer._llm_client = MagicMock()
    reviewer._llm_client.complete.return_value = ChatResponse(
        content=_ONE_COMMENT, model="gpt-test", finish_reason="stop"
    )
    return reviewer


def _prompt_sent(reviewer: DiffReviewer) -> str:
    message = reviewer._llm_client.complete.call_args.kwargs["messages"][0]
    return str(message.content)


def test_an_unindexed_repo_is_still_reviewed_from_the_diff_alone(repo: Path) -> None:
    reviewer = _reviewer(repo)

    with patch("trelix.retrieval.retriever.Retriever") as retriever:
        comments = reviewer.review([_hunk()])

    assert [c.comment for c in comments] == ["sign flip in add()"]
    assert reviewer.last_outcome.hunks_failed == 0
    assert "Related codebase context" not in _prompt_sent(reviewer)
    retriever.assert_not_called()
    assert not (repo / ".trelix").exists()


def test_an_indexed_repo_still_gets_retrieved_context(repo: Path) -> None:
    mark_indexed(repo)
    reviewer = _reviewer(repo)
    context = MagicMock()
    context.context_text = "def total(xs): return sum(xs)"

    with patch("trelix.retrieval.retriever.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = context
        comments = reviewer.review([_hunk()])

    assert [c.comment for c in comments] == ["sign flip in add()"]
    retriever.assert_called_once()
    assert "def total(xs): return sum(xs)" in _prompt_sent(reviewer)


def test_the_retriever_is_built_once_per_review_not_per_hunk(repo: Path) -> None:
    mark_indexed(repo)
    reviewer = _reviewer(repo)
    context = MagicMock()
    context.context_text = "ctx"

    with patch("trelix.retrieval.retriever.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = context
        reviewer.review([_hunk(), _hunk()])

    assert retriever.call_count == 1
    assert retriever.return_value.retrieve.call_count == 2


def _stub_client(self: Any) -> MagicMock:  # noqa: ANN401 - replaces a bound method
    client = MagicMock()
    client.complete.return_value = ChatResponse(
        content=_ONE_COMMENT, model="gpt-test", finish_reason="stop"
    )
    return client


def test_cli_local_review_on_an_unindexed_repo_exits_0_and_creates_nothing(
    repo: Path, tmp_path: Path
) -> None:
    diff_file = tmp_path / "change.diff"
    diff_file.write_text(_DIFF, encoding="utf-8")

    with patch("trelix.review.reviewer.DiffReviewer._get_client", _stub_client):
        result = runner.invoke(app, ["review", str(repo), "--diff", str(diff_file), "--json"])

    assert result.exit_code == 0, result.output
    assert [row["comment"] for row in json.loads(result.stdout)] == ["sign flip in add()"]
    assert not (repo / ".trelix").exists()


def test_cli_pr_review_on_an_unindexed_repo_exits_0_and_creates_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-k")
    pr_file = MagicMock()
    pr_file.filename = "calc.py"
    pr_file.previous_filename = None
    pr_file.patch = "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b"
    github = MagicMock()
    github.get_pr_files.return_value = [pr_file]

    with (
        patch("trelix.review.github.GitHubPRClient", return_value=github),
        patch("trelix.review.reviewer.DiffReviewer._get_client", _stub_client),
    ):
        result = runner.invoke(app, ["review", str(repo), "--pr", "acme/widgets#7", "--json"])

    assert result.exit_code == 0, result.output
    assert [row["comment"] for row in json.loads(result.stdout)] == ["sign flip in add()"]
    assert not (repo / ".trelix").exists()
