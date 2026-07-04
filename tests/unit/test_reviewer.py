"""Tests for DiffReviewer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from trelix.review.diff_parser import DiffHunk
from trelix.review.reviewer import DiffReviewer, ReviewComment


def _make_hunk(file_path: str = "src/auth.py") -> DiffHunk:
    return DiffHunk(
        file_path=file_path,
        old_start=10,
        new_start=10,
        old_lines=3,
        new_lines=4,
        added=["    if not user:", "        raise ValueError('missing')"],
        removed=["    return self._check(user, password)"],
        context=["def login(self, user, password):"],
    )


class TestReviewComment:
    def test_dataclass_fields(self) -> None:
        comment = ReviewComment(
            file_path="src/auth.py",
            line_start=10,
            line_end=12,
            severity="WARN",
            comment="This raises ValueError without logging.",
        )
        assert comment.severity == "WARN"
        assert comment.file_path == "src/auth.py"

    def test_severity_values(self) -> None:
        for sev in ["INFO", "WARN", "ERROR"]:
            c = ReviewComment("f.py", 1, 2, sev, "msg")
            assert c.severity == sev


class TestDiffReviewer:
    def test_review_returns_list(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        reviewer = DiffReviewer(cfg)
        hunks = [_make_hunk()]
        # No indexed repo — reviewer should return [] gracefully, not raise
        result = reviewer.review(hunks)
        assert isinstance(result, list)

    def test_review_never_raises(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        reviewer = DiffReviewer(cfg)
        # Even with malformed hunks, never raises
        bad_hunk = DiffHunk("bad.py", 0, 0, 0, 0)
        result = reviewer.review([bad_hunk])
        assert isinstance(result, list)

    def test_review_with_mocked_retriever_and_llm(self, tmp_path: Path) -> None:

        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        reviewer = DiffReviewer(cfg)

        mock_ctx = MagicMock()
        mock_ctx.context_text = "def login(user, password): ..."
        mock_ctx.results = []

        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(
            content=(
                '[{"line_start": 10, "line_end": 12, "severity": "WARN",'
                ' "comment": "Consider logging before raise."}]'
            )
        )

        reviewer._retriever = MagicMock()
        reviewer._retriever.retrieve.return_value = mock_ctx
        reviewer._llm_client = mock_llm

        hunks = [_make_hunk()]
        result = reviewer.review(hunks)
        assert isinstance(result, list)
        # If LLM returned valid JSON, we get a ReviewComment
        if result:
            assert isinstance(result[0], ReviewComment)

    def test_review_diff_text_empty_returns_empty(self, tmp_path: Path) -> None:
        """review(diff_text='') returns [] without raising."""
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        reviewer = DiffReviewer(cfg)
        result = reviewer.review(diff_text="")
        assert result == []

    def test_review_diff_text_none_returns_empty(self, tmp_path: Path) -> None:
        """review() with no args returns []."""
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        reviewer = DiffReviewer(cfg)
        result = reviewer.review()
        assert result == []

    def test_review_diff_text_parsed_into_hunks(self, tmp_path: Path) -> None:
        """review(diff_text=...) parses text into hunks and forwards to LLM pipeline."""
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        reviewer = DiffReviewer(cfg)

        # Minimal but valid unified diff
        diff_text = (
            "diff --git a/src/auth.py b/src/auth.py\n"
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -10,3 +10,4 @@\n"
            " def login():\n"
            "-    return False\n"
            "+    return True\n"
        )

        mock_ctx = MagicMock()
        mock_ctx.context_text = ""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(
            content='[{"line_start": 10, "line_end": 11, "severity": "INFO", "comment": "ok"}]'
        )

        reviewer._retriever = MagicMock()
        reviewer._retriever.retrieve.return_value = mock_ctx
        reviewer._llm_client = mock_llm

        result = reviewer.review(diff_text=diff_text)
        assert isinstance(result, list)
        # LLM was invoked because hunks were parsed from diff_text
        assert mock_llm.complete.called
        if result:
            assert isinstance(result[0], ReviewComment)
            assert result[0].file_path == "src/auth.py"

    def test_review_diff_text_preferred_over_empty_hunks(self, tmp_path: Path) -> None:
        """When hunks=None and diff_text is provided, diff_text is parsed."""
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        reviewer = DiffReviewer(cfg)

        diff_text = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,1 +1,2 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )

        mock_ctx = MagicMock()
        mock_ctx.context_text = ""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(content="[]")

        reviewer._retriever = MagicMock()
        reviewer._retriever.retrieve.return_value = mock_ctx
        reviewer._llm_client = mock_llm

        result = reviewer.review(hunks=None, diff_text=diff_text)
        assert isinstance(result, list)
        assert mock_llm.complete.called
