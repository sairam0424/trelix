"""
DiffReviewer must say whether a review actually happened.

`review()` returns a plain list, so "the model looked and found nothing" and
"nothing was looked at" were both `[]`. With no usable LLM key the OpenAI/Anthropic/
Vertex backends answer with a placeholder response instead of raising, the reviewer
then failed to parse it as JSON (debug log only), and per-hunk exceptions were
swallowed, so a review that never ran read as a clean one. These tests pin the
`last_outcome` record that lets a caller tell the two apart, without changing what
`review()` returns.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from trelix.core.config import IndexConfig, LLMConfig
from trelix.llm.client import UNCONFIGURED_MODEL, ChatResponse
from trelix.llm.providers.openai_backend import OpenAIBackend
from trelix.review.diff_parser import DiffHunk
from trelix.review.reviewer import DiffReviewer, ReviewOutcome

_ONE_COMMENT = (
    '[{"line_start": 10, "line_end": 12, "severity": "WARN", "comment": "check the null case"}]'
)


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


def _reviewer(tmp_path: Path, client: object) -> DiffReviewer:
    reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))
    reviewer._retriever = MagicMock()
    reviewer._retriever.retrieve.return_value = MagicMock(context_text="ctx", results=[])
    reviewer._llm_client = client
    return reviewer


def _ok_client() -> MagicMock:
    client = MagicMock()
    client.complete.return_value = ChatResponse(
        content=_ONE_COMMENT, model="gpt-test", finish_reason="stop"
    )
    return client


class TestOutcomeBeforeAnyReview:
    def test_fresh_reviewer_reports_nothing_wrong(self, tmp_path: Path) -> None:
        reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))

        assert reviewer.last_outcome == ReviewOutcome(
            llm_available=True, hunks_total=0, hunks_failed=0
        )
        assert not reviewer.last_outcome.could_not_review


class TestUnconfiguredLlm:
    def test_placeholder_response_is_detected_explicitly(self, tmp_path: Path) -> None:
        """The key-less backend answers with a placeholder, not an exception."""
        keyless = LLMConfig(provider="openai", _env_file=None)  # type: ignore[call-arg]
        backend = OpenAIBackend(keyless.model_copy(update={"openai_api_key": None}))
        reviewer = _reviewer(tmp_path, backend)

        result = reviewer.review([_hunk(), _hunk("src/b.py")])

        assert result == []
        assert reviewer.last_outcome.llm_available is False
        assert reviewer.last_outcome.hunks_total == 2
        assert reviewer.last_outcome.hunks_failed == 2
        assert reviewer.last_outcome.could_not_review

    def test_detection_does_not_depend_on_json_parsing(self, tmp_path: Path) -> None:
        """Even a placeholder that happens to parse as a JSON array is not a review."""
        client = MagicMock()
        client.complete.return_value = ChatResponse(
            content='[{"line_start": 1, "comment": "looks like a finding"}]',
            model=UNCONFIGURED_MODEL,
            finish_reason="stop",
        )
        reviewer = _reviewer(tmp_path, client)

        assert reviewer.review([_hunk()]) == []
        assert reviewer.last_outcome.llm_available is False

    def test_no_client_at_all_is_unavailable(self, tmp_path: Path) -> None:
        reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))

        with patch("trelix.llm.factory.build_chat_client", side_effect=ValueError("bad provider")):
            result = reviewer.review([_hunk()])

        assert result == []
        assert reviewer.last_outcome == ReviewOutcome(
            llm_available=False, hunks_total=1, hunks_failed=1
        )


class TestHunkFailures:
    def test_every_hunk_failing_is_reported_as_not_reviewed(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.complete.side_effect = RuntimeError("401 unauthorized")
        reviewer = _reviewer(tmp_path, client)

        assert reviewer.review([_hunk(), _hunk("src/b.py")]) == []

        assert reviewer.last_outcome == ReviewOutcome(
            llm_available=True, hunks_total=2, hunks_failed=2
        )
        assert reviewer.last_outcome.could_not_review

    def test_partial_failure_keeps_comments_and_counts_failures(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.complete.side_effect = [
            RuntimeError("429 rate limited"),
            ChatResponse(content=_ONE_COMMENT, model="gpt-test", finish_reason="stop"),
        ]
        reviewer = _reviewer(tmp_path, client)

        result = reviewer.review([_hunk(), _hunk("src/b.py")])

        assert [c.file_path for c in result] == ["src/b.py"]
        assert reviewer.last_outcome == ReviewOutcome(
            llm_available=True, hunks_total=2, hunks_failed=1
        )
        assert not reviewer.last_outcome.could_not_review


class TestCleanReview:
    def test_clean_review_is_distinguishable_from_no_review(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.complete.return_value = ChatResponse(
            content="[]", model="gpt-test", finish_reason="stop"
        )
        reviewer = _reviewer(tmp_path, client)

        assert reviewer.review([_hunk()]) == []

        assert reviewer.last_outcome == ReviewOutcome(
            llm_available=True, hunks_total=1, hunks_failed=0
        )
        assert not reviewer.last_outcome.could_not_review

    def test_no_hunks_is_not_a_failure(self, tmp_path: Path) -> None:
        reviewer = _reviewer(tmp_path, _ok_client())

        assert reviewer.review([]) == []

        assert not reviewer.last_outcome.could_not_review

    def test_outcome_is_replaced_on_every_call(self, tmp_path: Path) -> None:
        """A failed review must not poison the next one on the same reviewer."""
        client = MagicMock()
        client.complete.side_effect = [
            RuntimeError("boom"),
            ChatResponse(content=_ONE_COMMENT, model="gpt-test", finish_reason="stop"),
        ]
        reviewer = _reviewer(tmp_path, client)

        reviewer.review([_hunk()])
        assert reviewer.last_outcome.could_not_review
        reviewer.review([_hunk()])

        assert not reviewer.last_outcome.could_not_review
        assert reviewer.last_outcome.hunks_total == 1
