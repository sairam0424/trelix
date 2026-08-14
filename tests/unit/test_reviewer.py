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

        cfg = IndexConfig(repo_path=str(tmp_path))
        reviewer = DiffReviewer(cfg)
        hunks = [_make_hunk()]
        # No indexed repo — reviewer should return [] gracefully, not raise
        result = reviewer.review(hunks)
        assert isinstance(result, list)

    def test_review_never_raises(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path))
        reviewer = DiffReviewer(cfg)
        # Even with malformed hunks, never raises
        bad_hunk = DiffHunk("bad.py", 0, 0, 0, 0)
        result = reviewer.review([bad_hunk])
        assert isinstance(result, list)

    def test_review_with_mocked_retriever_and_llm(self, tmp_path: Path) -> None:

        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path))
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

        cfg = IndexConfig(repo_path=str(tmp_path))
        reviewer = DiffReviewer(cfg)
        result = reviewer.review(diff_text="")
        assert result == []

    def test_review_diff_text_none_returns_empty(self, tmp_path: Path) -> None:
        """review() with no args returns []."""
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path))
        reviewer = DiffReviewer(cfg)
        result = reviewer.review()
        assert result == []

    def test_review_diff_text_parsed_into_hunks(self, tmp_path: Path) -> None:
        """review(diff_text=...) parses text into hunks and forwards to LLM pipeline."""
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path))
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

        cfg = IndexConfig(repo_path=str(tmp_path))
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


def _capture_user_content(tmp_path: Path, hunk: DiffHunk, context_text: str) -> str:
    """Run one hunk review against a fake client and return the user prompt."""
    from trelix.core.config import IndexConfig

    reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))
    mock_ctx = MagicMock()
    mock_ctx.context_text = context_text
    mock_ctx.results = []
    reviewer._retriever = MagicMock()
    reviewer._retriever.retrieve.return_value = mock_ctx

    mock_llm = MagicMock()
    mock_llm.complete.return_value = MagicMock(content="[]")
    reviewer._llm_client = mock_llm

    reviewer.review([hunk])
    assert mock_llm.complete.call_count == 1
    messages = mock_llm.complete.call_args.kwargs["messages"]
    return str(messages[0].content)


class TestReviewHunkFencing:
    """Both fenced blocks in the review prompt must survive backticks in the payload."""

    def test_plain_hunk_is_byte_identical_to_legacy_prompt(self, tmp_path: Path) -> None:
        """Guarantees the fence change is a no-op for payloads without backticks."""
        hunk = _make_hunk()
        content = _capture_user_content(tmp_path, hunk, "def login(user, password): ...")
        diff_text = "\n".join(
            [f"- {line}" for line in hunk.removed] + [f"+ {line}" for line in hunk.added]
        )
        expected = (
            f"File: {hunk.file_path} (lines {hunk.new_start}–"
            f"{hunk.new_start + hunk.new_lines})\n\n"
            f"Changed code:\n```\n{diff_text}\n```\n\n"
            f"Related codebase context:\n```\ndef login(user, password): ...\n```\n\n"
            "Provide review comments as a JSON array."
        )
        assert content == expected

    def test_diff_containing_a_fence_is_still_well_formed(self, tmp_path: Path) -> None:
        # Reviewing a change to a markdown file puts ``` straight into the diff.
        hunk = DiffHunk(
            file_path="README.md",
            old_start=1,
            new_start=1,
            old_lines=1,
            new_lines=3,
            added=["```bash", "pip install trelix", "```"],
            removed=["old text"],
        )
        content = _capture_user_content(tmp_path, hunk, "")
        block = content.split("Changed code:\n", 1)[1].split("\n\nProvide review")[0]
        lines = block.split("\n")
        assert lines[0] == "````"
        assert lines[-1] == "````"
        assert block == "````\n- old text\n+ ```bash\n+ pip install trelix\n+ ```\n````"
        # No inner line may open a fence long enough to close the block early.
        for line in lines[1:-1]:
            assert not line.lstrip().startswith("````")

    def test_context_containing_a_fence_is_still_well_formed(self, tmp_path: Path) -> None:
        # Retrieved context is a symbol body; markdown symbols carry fences.
        context = "Verify a release:\n```bash\npip install pypi-attestations\n```"
        content = _capture_user_content(tmp_path, _make_hunk(), context)
        block = content.split("Related codebase context:\n", 1)[1].split("\n\nProvide review")[0]
        assert block == f"````\n{context}\n````"
        for line in context.split("\n"):
            assert not line.lstrip().startswith("````")

    def test_the_two_fences_are_derived_independently(self, tmp_path: Path) -> None:
        """A fence in the context must not inflate the diff's fence, or vice versa."""
        hunk = DiffHunk(
            file_path="README.md",
            old_start=1,
            new_start=1,
            old_lines=1,
            new_lines=1,
            added=["plain added line"],
            removed=[],
        )
        content = _capture_user_content(tmp_path, hunk, "`````\nlong run\n`````")
        diff_block = content.split("Changed code:\n", 1)[1].split("\n\n", 1)[0]
        context_block = content.split("Related codebase context:\n", 1)[1].split(
            "\n\nProvide review"
        )[0]
        assert diff_block == "```\n+ plain added line\n```"
        assert context_block == "``````\n`````\nlong run\n`````\n``````"
