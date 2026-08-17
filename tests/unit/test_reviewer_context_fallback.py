"""
What a review looks like when retrieval died under it.

DiffReviewer is retrieval-augmented by construction: the whole claim of the
feature is that comments are grounded in the surrounding codebase. When
`retriever.retrieve()` raises, the hunk is still reviewed — from the diff alone
— and the resulting comments are indistinguishable from grounded ones. These
tests pin the two things that make that survivable: the failure is audible in
the log, and the degraded comments say so in the text the reader actually sees.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from trelix.review.diff_parser import DiffHunk
from trelix.review.reviewer import _NO_CONTEXT_LABEL, DiffReviewer

_RETRIEVAL_BOOM = RuntimeError("vector store unreachable: no such table: chunks")


def _hunk(file_path: str = "src/auth.py") -> DiffHunk:
    return DiffHunk(
        file_path=file_path,
        old_start=10,
        new_start=10,
        old_lines=3,
        new_lines=4,
        added=["    return True"],
        removed=["    return self._check(user, password)"],
        context=["def login(self, user, password):"],
    )


def _reviewer(tmp_path: Path, *, retrieval_raises: bool, comments_json: str) -> DiffReviewer:
    from trelix.core.config import IndexConfig

    reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))
    reviewer._retriever = MagicMock()
    if retrieval_raises:
        reviewer._retriever.retrieve.side_effect = _RETRIEVAL_BOOM
    else:
        ctx = MagicMock()
        ctx.context_text = "def login(self, user, password): ..."
        ctx.results = []
        reviewer._retriever.retrieve.return_value = ctx

    reviewer._llm_client = MagicMock()
    reviewer._llm_client.complete.return_value = MagicMock(content=comments_json)
    return reviewer


_ONE_COMMENT = (
    '[{"line_start": 10, "line_end": 12, "severity": "ERROR",'
    ' "comment": "This bypasses the auth check added in login()."}]'
)
_TWO_COMMENTS = (
    '[{"line_start": 10, "line_end": 12, "severity": "ERROR", "comment": "first"},'
    ' {"line_start": 11, "line_end": 11, "severity": "INFO", "comment": "second"}]'
)


class TestRetrievalFailureIsAudible:
    def test_retrieval_failure_is_logged_at_warning(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        """A swallowed `except: pass` is invisible: the CLI configures WARNING."""
        reviewer = _reviewer(tmp_path, retrieval_raises=True, comments_json=_ONE_COMMENT)
        with caplog.at_level(logging.WARNING, logger="trelix.review.reviewer"):
            reviewer.review([_hunk()])

        assert any(
            "no such table" in r.message or "no such table" in str(r.args) for r in caplog.records
        ), f"the retrieval exception was never surfaced at WARNING: {caplog.records}"

    def test_the_log_line_names_the_file_it_degraded(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        """One bad hunk in a 40-hunk PR is only actionable if the log says which."""
        reviewer = _reviewer(tmp_path, retrieval_raises=True, comments_json=_ONE_COMMENT)
        with caplog.at_level(logging.WARNING, logger="trelix.review.reviewer"):
            reviewer.review([_hunk("src/payments/charge.py")])

        assert any(
            "src/payments/charge.py" in r.message or "src/payments/charge.py" in str(r.args)
            for r in caplog.records
        ), f"the degraded hunk was not identified: {caplog.records}"

    def test_successful_retrieval_logs_no_warning(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        """The warning has to stay rare enough to mean something."""
        reviewer = _reviewer(tmp_path, retrieval_raises=False, comments_json=_ONE_COMMENT)
        with caplog.at_level(logging.WARNING, logger="trelix.review.reviewer"):
            reviewer.review([_hunk()])
        assert caplog.records == []


class TestContextFreeCommentsAreLabelled:
    def test_every_comment_from_a_failed_hunk_is_labelled(self, tmp_path: Path) -> None:
        """`comment` is the only field every renderer prints — CLI table, --json, GitHub."""
        reviewer = _reviewer(tmp_path, retrieval_raises=True, comments_json=_TWO_COMMENTS)
        comments = reviewer.review([_hunk()])

        assert len(comments) == 2
        for c in comments:
            assert c.comment.endswith(_NO_CONTEXT_LABEL), c.comment
        # The model's own text survives ahead of the label.
        assert comments[0].comment.startswith("first")

    def test_grounded_comments_are_not_labelled(self, tmp_path: Path) -> None:
        reviewer = _reviewer(tmp_path, retrieval_raises=False, comments_json=_ONE_COMMENT)
        comments = reviewer.review([_hunk()])
        assert len(comments) == 1
        assert _NO_CONTEXT_LABEL not in comments[0].comment
        assert comments[0].comment == "This bypasses the auth check added in login()."

    def test_the_label_is_per_hunk_not_per_review(self, tmp_path: Path) -> None:
        """A PR where retrieval fails on one file must not smear the label over the rest."""
        from trelix.core.config import IndexConfig

        reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))
        ok_ctx = MagicMock()
        ok_ctx.context_text = "def login(self, user, password): ..."
        ok_ctx.results = []

        reviewer._retriever = MagicMock()
        reviewer._retriever.retrieve.side_effect = [_RETRIEVAL_BOOM, ok_ctx]
        reviewer._llm_client = MagicMock()
        reviewer._llm_client.complete.return_value = MagicMock(content=_ONE_COMMENT)

        comments = reviewer.review([_hunk("broken.py"), _hunk("fine.py")])

        by_file = {c.file_path: c.comment for c in comments}
        assert by_file["broken.py"].endswith(_NO_CONTEXT_LABEL)
        assert _NO_CONTEXT_LABEL not in by_file["fine.py"]


class TestDegradedReviewStillHappens:
    def test_retrieval_failure_does_not_drop_the_review(self, tmp_path: Path) -> None:
        """Surfacing the degradation must not turn a usable review into silence."""
        reviewer = _reviewer(tmp_path, retrieval_raises=True, comments_json=_ONE_COMMENT)
        comments = reviewer.review([_hunk()])
        assert len(comments) == 1
        assert comments[0].severity == "ERROR"
        assert reviewer._llm_client.complete.called

    def test_no_context_section_reaches_the_model(self, tmp_path: Path) -> None:
        """Pins the precondition for the label: the prompt really was context-free."""
        reviewer = _reviewer(tmp_path, retrieval_raises=True, comments_json=_ONE_COMMENT)
        reviewer.review([_hunk()])
        prompt = str(reviewer._llm_client.complete.call_args.kwargs["messages"][0].content)
        assert "Related codebase context" not in prompt
