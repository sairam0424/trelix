"""A hunk whose prompt the local server cut is `truncated` with detail `prompt_truncated`: not
retried, nothing salvaged (R-C4-02, the reviewer's half).

`OpenAIBackend.complete()` puts `prompt_truncated` in `ChatResponse.signals` when a local
OpenAI-compatible server (`TRELIX_LLM_BASE_URL`) reported fewer prompt tokens than trelix sent
(tests/unit/test_llm_openai_local_server.py). The reviewer is the first reader of that signal.
The helpers are those of test_reviewer_hunk_status.py.

Mutations each test was checked against (every one fails a test below): keep the retry on a
cut-off reply that also carries the signal (call_count 2); map the signal after the
finish-reason branches (detail `length_after_retry`); salvage the findings of a prompt-truncated
reply (comments kept); read `signals` with `in` on whatever object it is (a string `signals`
then counts as the signal); report the hunk as `error` or `refused` instead of `truncated`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from trelix.core.config import IndexConfig
from trelix.llm.client import ChatResponse
from trelix.review.diff_parser import DiffHunk
from trelix.review.hunk_status import HunkResult, HunkStatus
from trelix.review.reviewer import DiffReviewer

_ONE = '[{"line_start": 10, "line_end": 11, "severity": "WARN", "comment": "check this"}]'
_TRUNCATED = HunkResult("src/auth.py", 10, HunkStatus.TRUNCATED, "prompt_truncated")


def _hunk(path: str = "src/auth.py", start: int = 10) -> DiffHunk:
    return DiffHunk(
        file_path=path,
        old_start=start,
        new_start=start,
        old_lines=3,
        new_lines=4,
        added=["    return True"],
        removed=["    return self._check()"],
        context=["def login(self):"],
    )


def _reply(
    content: str,
    finish: str = "stop",
    signals: list[str] | None = None,
    output_tokens: int = 0,
) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="gpt-test",
        finish_reason=finish,
        output_tokens=output_tokens,
        signals=list(signals or []),
    )


def _reviewer(tmp_path: Path, *replies: object) -> tuple[DiffReviewer, MagicMock]:
    reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))  # type: ignore[arg-type]
    reviewer._retriever = MagicMock()
    reviewer._retriever.retrieve.return_value = MagicMock(context_text="ctx", results=[])
    client = MagicMock()
    client.complete.side_effect = list(replies)
    reviewer._llm_client = client
    return reviewer, client


def _only_result(reviewer: DiffReviewer) -> HunkResult:
    (result,) = reviewer.last_outcome.hunk_results
    return result


class TestPromptTruncatedHunk:
    def test_the_signal_makes_the_hunk_truncated_with_nothing_kept(self, tmp_path: Path) -> None:
        """The reply parses as one finding, and none of it is kept: the model never saw the
        whole hunk, so a line number in that finding is not one of this diff's."""
        reviewer, client = _reviewer(tmp_path, _reply(_ONE, signals=["prompt_truncated"]))

        comments = reviewer.review([_hunk()])

        assert comments == []
        assert _only_result(reviewer) == _TRUNCATED
        assert _only_result(reviewer).kept_comments == 0
        assert reviewer.last_outcome.hunks_failed == 1
        assert client.complete.call_count == 1

    def test_a_cut_prompt_is_not_retried_even_when_the_reply_was_cut_too(
        self, tmp_path: Path
    ) -> None:
        """`finish_reason="length"` alone is retried once at a larger limit; with the signal it
        is not (a bigger output cap cannot help when the input was cut), and the detail is
        `prompt_truncated`, never `length_after_retry`."""
        reviewer, client = _reviewer(
            tmp_path,
            _reply("[", finish="length", signals=["prompt_truncated"]),
            _reply(_ONE),  # would be the retry's reply
        )

        comments = reviewer.review([_hunk()])

        assert comments == []
        assert _only_result(reviewer) == _TRUNCATED
        assert _only_result(reviewer).detail == "prompt_truncated"
        assert client.complete.call_count == 1

    def test_a_stop_that_used_every_token_is_not_retried_either(self, tmp_path: Path) -> None:
        """`_effective_finish` reads a `stop` at the full limit as `length`; the signal still
        wins over that retry path."""
        reviewer, client = _reviewer(
            tmp_path,
            _reply(_ONE, output_tokens=4096, signals=["prompt_truncated"]),
            _reply(_ONE),
        )

        reviewer.review([_hunk()])

        assert _only_result(reviewer) == _TRUNCATED
        assert client.complete.call_count == 1

    def test_a_filter_outage_alone_is_still_reviewed(self, tmp_path: Path) -> None:
        """The other signal, `content_filter_error`, keeps its 3.4.3 meaning: the reply is
        complete and the hunk is reviewed."""
        reviewer, _ = _reviewer(tmp_path, _reply(_ONE, signals=["content_filter_error"]))

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["check this"]
        assert _only_result(reviewer) == HunkResult("src/auth.py", 10, HunkStatus.REVIEWED, "")

    def test_a_reply_whose_signals_is_not_a_list_is_reviewed(self, tmp_path: Path) -> None:
        """A test double's `signals` is a string here, and `"prompt_truncated" in <that str>` is
        true: only a real list carries the signal."""
        reply = MagicMock()
        reply.content = _ONE
        reply.model = "gpt-test"
        reply.finish_reason = "stop"
        reply.raw_finish_reason = None
        reply.output_tokens = 0
        reply.signals = "prompt_truncated"
        reviewer, _ = _reviewer(tmp_path, reply)

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["check this"]
        assert _only_result(reviewer).status is HunkStatus.REVIEWED

    def test_as_dict_is_the_outcome_file_record(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply("[]", signals=["prompt_truncated"]))

        reviewer.review([_hunk()])

        assert _only_result(reviewer).as_dict() == {
            "file": "src/auth.py",
            "line": 10,
            "status": "truncated",
            "detail": "prompt_truncated",
        }

    def test_every_hunk_cut_is_a_review_that_did_not_run(self, tmp_path: Path) -> None:
        """Nothing kept and nothing reviewed: `could_not_review` (the CLI's exit 3), rather
        than a clean `[]`."""
        reviewer, client = _reviewer(
            tmp_path,
            _reply(_ONE, signals=["prompt_truncated"]),
            _reply("[]", signals=["prompt_truncated"]),
        )

        comments = reviewer.review([_hunk("a.py", 1), _hunk("b.py", 1)])

        assert comments == []
        assert reviewer.last_outcome.hunks_failed == 2
        assert reviewer.last_outcome.could_not_review is True
        assert [r.detail for r in reviewer.last_outcome.hunk_results] == [
            "prompt_truncated",
            "prompt_truncated",
        ]
        assert client.complete.call_count == 2
