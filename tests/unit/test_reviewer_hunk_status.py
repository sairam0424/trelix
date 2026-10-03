"""Each hunk gets a status, and only a parsed array after a clean stop counts as reviewed.

`review()` used to return [] for "the model found nothing" and for "the reply was cut off,
refused, filtered or not JSON", and the CLI exited 0 for both. These tests pin the five
statuses, the single retry at a larger limit, the salvage of a cut-off array, and the fixed
vocabulary of the `detail` strings (they end up in a published GitHub Check).

Mutations each group was checked against (every one fails a test below):
- accept an empty array found inside prose as a review (the prose-echo tests fail);
- drop the retry on a cut-off reply (the length-then-stop test fails);
- skip the is_complete() check so an unknown stop is parsed (the error-status tests fail);
- make salvage return nothing (the salvage tests fail);
- count only raised exceptions in hunks_failed (the outcome tests fail).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from trelix.core.config import IndexConfig
from trelix.llm.client import ChatResponse
from trelix.review.diff_parser import DiffHunk
from trelix.review.hunk_status import (
    HunkResult,
    HunkStatus,
    extract_review_items,
    safe_token,
    salvage_review_items,
)
from trelix.review.reviewer import DiffReviewer, ReviewOutcome

_ONE = '[{"line_start": 10, "line_end": 11, "severity": "WARN", "comment": "check this"}]'
_TWO_THEN_CUT = (
    '[{"line_start": 1, "line_end": 1, "severity": "INFO", "comment": "first"},'
    ' {"line_start": 2, "line_end": 2, "severity": "WARN", "comment": "second"},'
    ' {"line_start": 3, "line_end": 3, "severity": "ERR'
)


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
    raw: str | None = None,
    model: str = "gpt-test",
    output_tokens: int = 0,
) -> ChatResponse:
    return ChatResponse(
        content=content,
        model=model,
        finish_reason=finish,
        raw_finish_reason=raw,
        output_tokens=output_tokens,
    )


def _reviewer(tmp_path: Path, *replies: object, **config: object) -> tuple[DiffReviewer, MagicMock]:
    reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path), **config))  # type: ignore[arg-type]
    reviewer._retriever = MagicMock()
    reviewer._retriever.retrieve.return_value = MagicMock(context_text="ctx", results=[])
    client = MagicMock()
    client.complete.side_effect = list(replies)
    reviewer._llm_client = client
    return reviewer, client


def _only_result(reviewer: DiffReviewer) -> HunkResult:
    (result,) = reviewer.last_outcome.hunk_results
    return result


class TestEachStatus:
    def test_reviewed_with_a_finding(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(_ONE))

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["check this"]
        assert _only_result(reviewer) == HunkResult("src/auth.py", 10, HunkStatus.REVIEWED, "")

    def test_reviewed_with_no_findings_is_still_reviewed(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply("[]"))

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer).status is HunkStatus.REVIEWED

    def test_an_array_in_a_code_fence_is_reviewed(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(f"```json\n{_ONE}\n```"))

        assert len(reviewer.review([_hunk()])) == 1
        assert _only_result(reviewer).status is HunkStatus.REVIEWED

    def test_prose_around_a_non_empty_array_is_reviewed(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(f"Here is my review:\n{_ONE}\nHope it helps."))

        assert len(reviewer.review([_hunk()])) == 1
        assert _only_result(reviewer).status is HunkStatus.REVIEWED

    @pytest.mark.parametrize(
        ("content", "detail"),
        [
            ("", "empty_reply"),
            ("   \n", "empty_reply"),
            ("I could not find any problems in this change.", "no_review_array"),
            ("I would return [] if there were nothing to say, but there is.", "no_review_array"),
            ('{"line_start": 1, "comment": "an object, not an array"}', "no_review_array"),
            ("[1, 2, 3]", "no_review_array"),
            ("[note] see below", "no_review_array"),
            # A non-empty array with no item that has a text comment is not a review.
            ("[{}]", "no_review_array"),
            ('[{"comment": ""}]', "no_review_array"),
            ('[{"comment": 7}]', "no_review_array"),
            ('[{"line": 1, "message": "a real bug, under the wrong key"}]', "no_review_array"),
            ('{"error": "too large", "details": [{"code": 1}]}', "no_review_array"),
            # A review cut off behind a clean-looking stop (LiteLLM can hide a truncation): the
            # inner array of the first finding must not stand in for the review.
            (
                '[{"comment": "a", "refs": [{"file": "f", "line": 3}]}, {"comment": "b", "x',
                "no_review_array",
            ),
            ('[{"comment": "a"', "no_review_array"),
            # The same, where the inner array even has a text comment: it is a child of the
            # finding that was cut off, not a review of its own.
            (
                '[{"comment": "a", "children": [{"comment": "inner"}]}, {"comment": "b", "x',
                "no_review_array",
            ),
        ],
    )
    def test_a_clean_stop_without_a_review_array_is_parse_failed(
        self, tmp_path: Path, content: str, detail: str
    ) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(content))

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer).status is HunkStatus.PARSE_FAILED
        assert _only_result(reviewer).detail == detail

    @pytest.mark.parametrize(
        "line_start", ['"ten"', "null", "[]", "{}", "1e999", "NaN", "true", "false"]
    )
    def test_an_unusable_line_number_falls_back_to_the_hunk_range(
        self, tmp_path: Path, line_start: str
    ) -> None:
        reply = f'[{{"line_start": {line_start}, "line_end": null, "comment": "x"}}]'
        reviewer, _ = _reviewer(tmp_path, _reply(reply))

        (comment,) = reviewer.review([_hunk()])

        assert (comment.line_start, comment.line_end) == (10, 14)
        assert _only_result(reviewer).status is HunkStatus.REVIEWED

    @pytest.mark.parametrize(("raw", "expected"), [('"12"', 12), ("12", 12), ("12.9", 12)])
    def test_a_numeric_line_number_is_used(self, tmp_path: Path, raw: str, expected: int) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(f'[{{"line_start": {raw}, "comment": "x"}}]'))

        (comment,) = reviewer.review([_hunk()])

        assert comment.line_start == expected

    def test_one_bad_item_does_not_cost_the_hunk_its_other_findings(self, tmp_path: Path) -> None:
        reply = '[{"comment": "x", "line_start": null}, {"comment": "good", "line_start": 12}]'
        reviewer, _ = _reviewer(tmp_path, _reply(reply))

        comments = reviewer.review([_hunk()])

        assert [(c.comment, c.line_start) for c in comments] == [("x", 10), ("good", 12)]

    def test_an_integer_too_long_to_parse_is_parse_failed_not_error(self, tmp_path: Path) -> None:
        reply = '[{"comment": "x", "line_start": ' + "9" * 5000 + "}]"
        reviewer, _ = _reviewer(tmp_path, _reply(reply))

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer).status is HunkStatus.PARSE_FAILED

    def test_items_without_a_text_comment_are_dropped_from_a_mixed_array(
        self, tmp_path: Path
    ) -> None:
        reply = '[{"line_start": 1}, {"comment": "kept"}, {"comment": ""}]'
        reviewer, _ = _reviewer(tmp_path, _reply(reply))

        assert [c.comment for c in reviewer.review([_hunk()])] == ["kept"]

    def test_a_reply_that_is_not_text_is_parse_failed_with_its_own_label(
        self, tmp_path: Path
    ) -> None:
        odd = MagicMock(content=None, model="gpt-test", finish_reason="stop", output_tokens=0)
        reviewer, _ = _reviewer(tmp_path, odd)

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer).detail == "non_text_reply"

    @pytest.mark.parametrize(
        ("finish", "detail"), [("refusal", "refusal"), ("content_filter", "content_filter")]
    )
    def test_a_refusal_or_filter_is_refused_even_if_the_content_looks_like_a_review(
        self, tmp_path: Path, finish: str, detail: str
    ) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(_ONE, finish=finish))

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer) == HunkResult("src/auth.py", 10, HunkStatus.REFUSED, detail)

    @pytest.mark.parametrize(
        ("finish", "raw", "detail"),
        [
            ("error", "malformed_model_output", "error:malformed_model_output"),
            ("unknown", "a_value_added_next_year", "unknown:a_value_added_next_year"),
            ("unknown", None, "unknown:none"),
            # The reviewer offers no tools, so this is as suspect as an unclassified stop.
            ("tool_calls", "tool_use", "tool_calls:tool_use"),
        ],
    )
    def test_an_error_or_unclassified_stop_is_error_not_parsed(
        self, tmp_path: Path, finish: str, raw: str | None, detail: str
    ) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(_ONE, finish=finish, raw=raw))

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer) == HunkResult("src/auth.py", 10, HunkStatus.ERROR, detail)

    def test_a_reply_whose_stop_reason_is_not_a_string_is_error(self, tmp_path: Path) -> None:
        bogus = MagicMock(content=_ONE, model="gpt-test", finish_reason=MagicMock())
        reviewer, _ = _reviewer(tmp_path, bogus)

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer).status is HunkStatus.ERROR

    def test_a_call_that_raises_is_error_with_the_exception_class_only(
        self, tmp_path: Path
    ) -> None:
        reviewer, _ = _reviewer(tmp_path, RuntimeError("401 credential canary rejected"))

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer) == HunkResult(
            "src/auth.py", 10, HunkStatus.ERROR, "exception:RuntimeError"
        )


class TestCutOffReplies:
    def test_a_cut_off_reply_is_retried_once_at_four_times_the_limit(self, tmp_path: Path) -> None:
        reviewer, client = _reviewer(tmp_path, _reply(_TWO_THEN_CUT, "length"), _reply(_ONE))

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["check this"]
        assert _only_result(reviewer) == HunkResult(
            "src/auth.py", 10, HunkStatus.REVIEWED, "retried"
        )
        assert [call.kwargs["max_tokens"] for call in client.complete.call_args_list] == [
            4096,
            16384,
        ]

    def test_a_paused_reply_is_truncated_and_not_retried_at_a_larger_limit(
        self, tmp_path: Path
    ) -> None:
        # A pause needs a continuation, which a bigger limit does not give.
        reviewer, client = _reviewer(tmp_path, _reply(_TWO_THEN_CUT, "paused"))

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["first", "second"]
        assert _only_result(reviewer) == HunkResult(
            "src/auth.py", 10, HunkStatus.TRUNCATED, "paused"
        )
        assert client.complete.call_count == 1

    def test_the_retry_sends_exactly_the_same_prompt(self, tmp_path: Path) -> None:
        reviewer, client = _reviewer(tmp_path, _reply("[", "length"), _reply("[]"))

        reviewer.review([_hunk()])

        first, second = (
            call.kwargs["messages"][0].content for call in client.complete.call_args_list
        )
        assert first == second

    def test_a_failing_retry_keeps_the_findings_of_the_first_reply(self, tmp_path: Path) -> None:
        reviewer, client = _reviewer(
            tmp_path, _reply(_TWO_THEN_CUT, "length"), RuntimeError("timed out")
        )

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["first", "second"]
        assert _only_result(reviewer) == HunkResult(
            "src/auth.py", 10, HunkStatus.TRUNCATED, "length_retry_failed"
        )
        assert client.complete.call_count == 2

    @pytest.mark.parametrize(
        ("retry_reply", "status", "detail"),
        [
            (_reply(_ONE, finish="refusal"), HunkStatus.REFUSED, "refusal"),
            (
                _reply(_ONE, finish="unknown", raw="new_value"),
                HunkStatus.ERROR,
                "unknown:new_value",
            ),
            (_reply("[", finish="paused"), HunkStatus.TRUNCATED, "paused_after_retry"),
        ],
    )
    def test_the_retry_reply_is_classified_like_any_other(
        self, tmp_path: Path, retry_reply: ChatResponse, status: HunkStatus, detail: str
    ) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply("[", "length"), retry_reply)

        reviewer.review([_hunk()])

        assert (_only_result(reviewer).status, _only_result(reviewer).detail) == (status, detail)

    def test_a_placeholder_answer_on_the_retry_still_means_not_configured(
        self, tmp_path: Path
    ) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply("[", "length"), _reply(_ONE, model="none"))

        reviewer.review([_hunk()])

        assert not reviewer.last_outcome.llm_available
        assert _only_result(reviewer).detail == "not_configured"

    def test_salvaged_comments_carry_the_no_context_label(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(
            tmp_path, _reply(_TWO_THEN_CUT, "length"), _reply(_TWO_THEN_CUT, "length")
        )
        reviewer._retriever.retrieve.side_effect = RuntimeError("store gone")

        comments = reviewer.review([_hunk()])

        assert len(comments) == 2
        assert all("no codebase context" in c.comment for c in comments)

    def test_a_clean_stop_that_used_every_token_is_read_as_cut_off(self, tmp_path: Path) -> None:
        # LiteLLM can report a truncation as "stop"; a reply that used the whole limit was cut.
        reviewer, client = _reviewer(
            tmp_path,
            _reply(_TWO_THEN_CUT, "stop", output_tokens=4096),
            _reply(_ONE, "stop", output_tokens=300),
        )

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["check this"]
        assert _only_result(reviewer).detail == "retried"
        assert client.complete.call_count == 2

    def test_a_clean_stop_below_the_limit_is_not_retried(self, tmp_path: Path) -> None:
        reviewer, client = _reviewer(tmp_path, _reply(_ONE, "stop", output_tokens=4095))

        reviewer.review([_hunk()])

        assert client.complete.call_count == 1
        assert _only_result(reviewer).status is HunkStatus.REVIEWED

    def test_the_retry_reply_is_judged_against_the_larger_limit(self, tmp_path: Path) -> None:
        # 9000 tokens would be "every token used" at the first limit (4096) but not at the retry's.
        reviewer, _ = _reviewer(
            tmp_path, _reply("[", "length"), _reply(_ONE, "stop", output_tokens=9000)
        )

        reviewer.review([_hunk()])

        assert _only_result(reviewer) == HunkResult(
            "src/auth.py", 10, HunkStatus.REVIEWED, "retried"
        )

    def test_a_retry_that_used_the_whole_larger_limit_is_still_cut_off(
        self, tmp_path: Path
    ) -> None:
        reviewer, _ = _reviewer(
            tmp_path,
            _reply("[", "length"),
            _reply(_TWO_THEN_CUT, "stop", output_tokens=16384),
        )

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["first", "second"]
        assert _only_result(reviewer).detail == "length_after_retry"

    def test_a_token_count_that_is_not_a_number_is_ignored(self, tmp_path: Path) -> None:
        odd = MagicMock(content=_ONE, model="gpt-test", finish_reason="stop", output_tokens="lots")
        reviewer, client = _reviewer(tmp_path, odd)

        reviewer.review([_hunk()])

        assert client.complete.call_count == 1
        assert _only_result(reviewer).status is HunkStatus.REVIEWED

    def test_still_cut_off_after_the_retry_is_truncated_and_keeps_the_complete_findings(
        self, tmp_path: Path
    ) -> None:
        reviewer, client = _reviewer(
            tmp_path, _reply(_TWO_THEN_CUT, "length"), _reply(_TWO_THEN_CUT, "length")
        )

        comments = reviewer.review([_hunk()])

        assert [c.comment for c in comments] == ["first", "second"]
        assert _only_result(reviewer) == HunkResult(
            "src/auth.py", 10, HunkStatus.TRUNCATED, "length_after_retry"
        )
        assert client.complete.call_count == 2
        assert reviewer.last_outcome.hunks_failed == 1

    def test_the_retry_limit_is_capped(self, tmp_path: Path) -> None:
        reviewer, client = _reviewer(
            tmp_path, _reply("[", "length"), _reply("[", "length"), review_max_tokens=10000
        )

        reviewer.review([_hunk()])

        assert [call.kwargs["max_tokens"] for call in client.complete.call_args_list] == [
            10000,
            16384,
        ]

    def test_no_retry_when_the_limit_is_already_at_the_ceiling(self, tmp_path: Path) -> None:
        reviewer, client = _reviewer(tmp_path, _reply("[", "length"), review_max_tokens=16384)

        reviewer.review([_hunk()])

        assert client.complete.call_count == 1
        assert _only_result(reviewer) == HunkResult(
            "src/auth.py", 10, HunkStatus.TRUNCATED, "length"
        )


class TestTheLimit:
    def test_the_default_limit_reaches_the_model(self, tmp_path: Path) -> None:
        reviewer, client = _reviewer(tmp_path, _reply("[]"))

        reviewer.review([_hunk()])

        assert client.complete.call_args.kwargs["max_tokens"] == 4096

    def test_the_code_default_limit_is_4096(self) -> None:
        # The test environment pins the variable to its default (tests/_env_isolation.py), so
        # an instance would always say 4096; read the field's own default instead.
        assert IndexConfig.model_fields["review_max_tokens"].default == 4096

    def test_the_environment_variable_sets_the_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_REVIEW_MAX_TOKENS", "8192")
        reviewer, client = _reviewer(tmp_path, _reply("[]"))

        reviewer.review([_hunk()])

        assert client.complete.call_args.kwargs["max_tokens"] == 8192

    @pytest.mark.parametrize("value", [255, 16385, 65536, 0, -1])
    def test_an_out_of_range_limit_is_rejected(self, tmp_path: Path, value: int) -> None:
        with pytest.raises(ValidationError):
            IndexConfig(repo_path=str(tmp_path), review_max_tokens=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [256, 16384])
    def test_the_range_boundaries_are_accepted(self, tmp_path: Path, value: int) -> None:
        config = IndexConfig(repo_path=str(tmp_path), review_max_tokens=value)  # type: ignore[arg-type]

        assert config.review_max_tokens == value


class TestOutcome:
    def test_results_follow_the_hunks_and_hunks_failed_counts_every_unreviewed_one(
        self, tmp_path: Path
    ) -> None:
        reviewer, _ = _reviewer(
            tmp_path,
            _reply(_ONE),
            _reply(_ONE, finish="refusal"),
            _reply("no array here"),
            RuntimeError("boom"),
            _reply("[]"),
        )
        hunks = [
            _hunk("a.py", 1),
            _hunk("b.py", 2),
            _hunk("c.py", 3),
            _hunk("d.py", 4),
            _hunk("e.py", 5),
        ]

        reviewer.review(hunks)

        outcome = reviewer.last_outcome
        assert [(r.file_path, r.line, r.status.value) for r in outcome.hunk_results] == [
            ("a.py", 1, "reviewed"),
            ("b.py", 2, "refused"),
            ("c.py", 3, "parse_failed"),
            ("d.py", 4, "error"),
            ("e.py", 5, "reviewed"),
        ]
        assert outcome.hunks_total == 5
        assert outcome.hunks_failed == 3
        assert outcome.hunks_reviewed == 2
        assert outcome.unreviewed_fraction == pytest.approx(0.6)
        assert not outcome.could_not_review

    def test_every_hunk_cut_off_is_a_review_that_did_not_run(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, *[_reply("[", "length")] * 4)

        reviewer.review([_hunk("a.py"), _hunk("b.py")])

        assert reviewer.last_outcome.could_not_review
        assert reviewer.last_outcome.hunks_failed == 2

    def test_a_cut_off_hunk_that_kept_findings_is_a_partial_review_not_a_review_that_did_not_run(
        self, tmp_path: Path
    ) -> None:
        reviewer, _ = _reviewer(
            tmp_path, _reply(_TWO_THEN_CUT, "length"), _reply(_TWO_THEN_CUT, "length")
        )

        comments = reviewer.review([_hunk()])

        assert len(comments) == 2
        assert reviewer.last_outcome.hunks_failed == 1
        assert _only_result(reviewer).kept_comments == 2
        assert not reviewer.last_outcome.could_not_review

    def test_hunk_two_unconfigured_after_hunk_one_reviewed_marks_every_hunk(
        self, tmp_path: Path
    ) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(_ONE), _reply(_ONE, model="none"))

        reviewer.review([_hunk("a.py"), _hunk("b.py")])

        assert not reviewer.last_outcome.llm_available
        assert [r.detail for r in reviewer.last_outcome.hunk_results] == [
            "not_configured",
            "not_configured",
        ]

    def test_a_cut_off_hunk_does_not_hide_the_findings_of_the_others(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(
            tmp_path, _reply(_ONE), _reply("[", "length"), _reply("[", "length")
        )

        comments = reviewer.review([_hunk("a.py"), _hunk("b.py")])

        assert [c.file_path for c in comments] == ["a.py"]
        assert reviewer.last_outcome.hunks_failed == 1
        assert not reviewer.last_outcome.could_not_review

    def test_no_hunks_means_zero_fraction(self) -> None:
        assert ReviewOutcome().unreviewed_fraction == 0.0

    def test_the_placeholder_backend_marks_every_hunk_not_configured(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply(_ONE, model="none"))

        assert reviewer.review([_hunk("a.py"), _hunk("b.py")]) == []

        assert not reviewer.last_outcome.llm_available
        assert [(r.status.value, r.detail) for r in reviewer.last_outcome.hunk_results] == [
            ("error", "not_configured"),
            ("error", "not_configured"),
        ]

    def test_no_client_marks_every_hunk(self, tmp_path: Path) -> None:
        reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))
        reviewer._get_client = lambda: None  # type: ignore[method-assign]

        reviewer.review([_hunk()])

        assert _only_result(reviewer).detail == "no_llm_client"

    def test_equality_still_compares_the_counters_only(self) -> None:
        with_results = ReviewOutcome(
            hunks_total=1,
            hunks_failed=1,
            hunk_results=(HunkResult("a.py", 1, HunkStatus.ERROR, "x"),),
        )

        assert with_results == ReviewOutcome(hunks_total=1, hunks_failed=1)


class TestDetailVocabulary:
    _ALLOWED = re.compile(r"[A-Za-z0-9_.-]{0,40}(:[A-Za-z0-9_.-]{1,40})?")

    def test_no_detail_carries_refusal_text_or_model_prose(self, tmp_path: Path) -> None:
        prose = "CANARY model prose that must never reach a check"
        reviewer, _ = _reviewer(
            tmp_path,
            _reply(prose, finish="refusal"),
            _reply(prose),
            _reply(prose, finish="unknown", raw="not a token!"),
            _reply(prose, finish="error", raw="x" * 200),
            RuntimeError(prose),
        )
        hunks = [_hunk(f"f{i}.py", i + 1) for i in range(5)]

        reviewer.review(hunks)

        details = [r.detail for r in reviewer.last_outcome.hunk_results]
        assert details == [
            "refusal",
            "no_review_array",
            "unknown:none",
            "error:none",
            "exception:RuntimeError",
        ]
        assert all(self._ALLOWED.fullmatch(d) for d in details)
        assert all("CANARY" not in d for d in details)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("max_tokens", "max_tokens"),
            ("a" * 40, "a" * 40),
            ("a" * 41, "none"),
            ("has space", "none"),
            ("semi;colon", "none"),
            ("", "none"),
            (None, "none"),
            (7, "none"),
        ],
    )
    def test_safe_token_keeps_only_short_plain_tokens(self, value: object, expected: str) -> None:
        assert safe_token(value) == expected

    def test_as_dict_uses_the_outcome_file_keys(self) -> None:
        result = HunkResult("a.py", 3, HunkStatus.TRUNCATED, "length")

        assert result.as_dict() == {
            "file": "a.py",
            "line": 3,
            "status": "truncated",
            "detail": "length",
        }


class TestExtractAndSalvage:
    @pytest.mark.parametrize(
        ("content", "expected_comments"),
        [
            ("[]", []),
            ("  [ ]  ", []),
            ("```\n[]\n```", []),
            # "No issues", said in the ways models say it.
            ("No issues found.\n```json\n[]\n```", []),
            ("[]\n\nThe diff looks fine.", []),
            ("```json\n[]\n```\nNothing else to add.", []),
            ("```json\r\n[]\r\n```", []),
            ("~~~\n[]\n~~~", []),
            ("````\n[]\n````", []),
            ("``` json\n[]\n```", []),
            ("\ufeff[]", []),
            ("ok " + "x[i] " * 30 + '[{"comment": "a"}]', ["a"]),
            ('[{"comment": "a"}]', ["a"]),
            ('text [{"comment": "a"}] more text', ["a"]),
            ('[{"comment": "a"}, 5, "x"]', ["a"]),
            ("", None),
            ("no brackets", None),
            ("[1, 2]", None),
            ("Return [] when clean.", None),
            ("see [1] and [2]", None),
            ("[{}]", None),
            ('[{"comment": ""}]', None),
            # A fence must sit on its own lines, and `[]` must end its line.
            ("Found bugs. ```json\n[]\n``` was the sample", None),
            ("[] (but I found bugs in foo)", None),
            ("The result is []\nplus caveats about foo.", None),
            # An array of objects with no comment next to an empty one is not "no issues".
            ('```json\n[]\n```\n[{"message": "x"}]', None),
            ('[]\n[{"line": 1}]', None),
            # Fail-loud choice: `[]` must end its line, so prose on the same line is not accepted.
            ("[] No issues found.", None),
            ("[]. The diff is fine.", None),
            # Accepted trade-offs, pinned so they stay deliberate: prose around an empty array on
            # its own line, or in a fence, is not read for meaning.
            ("[]\nActually line 3 has a SQL injection.", []),
            ("```python\n[]\n```", []),
            # Only a leading byte-order mark is dropped; one inside a comment is kept.
            ('[{"comment": "a\ufeffb"}]', ["a\ufeffb"]),
            # Two marks are not a fence.
            ("``\n[]\n``", None),
            ("~~\n[]\n~~", None),
            # Accepted trade-off: prose before a fenced `[]` is not read for meaning.
            ("Issue in line 5 is severe.\n```json\n[]\n```", []),
            # A non-JSON sample in a preamble makes the whole reply unusable: loud, never green.
            ('Format: [{line_start: int}]. Review:\n[{"comment": "real"}]', None),
            # An integer too long for Python to parse is a bad reply, not an exception.
            ('[{"comment": "x", "line_start": ' + "9" * 5000 + "}]", None),
            ('[{"comment": "a", "children": [{"comment": "inner"}]}, {"comment": "b", "x', None),
        ],
    )
    def test_extract_review_items(self, content: str, expected_comments: list[str] | None) -> None:
        items = extract_review_items(content)

        if expected_comments is None:
            assert items is None
        else:
            assert items is not None
            assert [i["comment"] for i in items] == expected_comments

    def test_salvage_keeps_the_complete_objects_and_drops_the_cut_one(self) -> None:
        assert [i["comment"] for i in salvage_review_items(_TWO_THEN_CUT)] == ["first", "second"]

    @pytest.mark.parametrize(
        "content",
        ["", "no array", "[", "[ {", '[{"comment": "cut', "[1, 2]"],
    )
    def test_salvage_of_nothing_usable_is_empty(self, content: str) -> None:
        assert salvage_review_items(content) == []

    def test_salvage_survives_an_integer_too_long_to_parse(self) -> None:
        text = '[{"comment": "a"}, {"comment": "b", "line_start": ' + "9" * 5000 + "}]"

        assert [i["comment"] for i in salvage_review_items(text)] == ["a"]

    @pytest.mark.parametrize(
        "reply",
        [
            "`" * 50_000,
            "`" * 50_000 + "x",
            "~" * 50_000 + "!",
            "```json\n[]" + " " * 50_000,
            "```json\n[]" + "\n" * 50_000,
            "```" + " " * 50_000 + "!",
            "[" + " " * 50_000,
            "[ " * 25_000,
            "```\n" * 25_000,
            "```json\n" + "[] \n" * 10_000,
            "[" + "\n" * 50_000 + "x",
        ],
        ids=[
            "backticks",
            "backticks-then-x",
            "tildes-then-bang",
            "fence-empty-then-spaces",
            "fence-empty-then-newlines",
            "fence-then-spaces-then-bang",
            "open-bracket-then-spaces",
            "many-open-brackets",
            "many-fences",
            "many-empty-arrays",
            "open-bracket-then-newlines",
        ],
    )
    def test_hostile_reply_shapes_are_handled_in_linear_time(self, reply: str) -> None:
        # Model output can be tens of thousands of backticks or whitespace characters, and a
        # pattern that rescans it per position is quadratic: seconds at 8,000, minutes at 50,000.
        started = time.perf_counter()
        extract_review_items(reply)
        salvage_review_items(reply)
        elapsed = time.perf_counter() - started

        assert elapsed < 2.0, f"{elapsed:.1f}s on a {len(reply)}-character reply"

    def test_deeply_nested_model_output_is_not_an_exception(self) -> None:
        # json raises RecursionError on this; a prompt-injected diff could ask for it.
        assert extract_review_items("[" * 100_000) is None
        assert extract_review_items('[{"a":' * 50_000) is None
        assert salvage_review_items('[{"a":' * 50_000) == []

    def test_deeply_nested_reply_is_parse_failed_not_error(self, tmp_path: Path) -> None:
        reviewer, _ = _reviewer(tmp_path, _reply('[{"a":' * 50_000))

        assert reviewer.review([_hunk()]) == []
        assert _only_result(reviewer).status is HunkStatus.PARSE_FAILED

    def test_salvage_starts_at_the_array_of_objects_not_at_an_earlier_bracket(self) -> None:
        text = 'Issue [1]: [{"comment": "a"}, {"com'

        assert [i["comment"] for i in salvage_review_items(text)] == ["a"]

    def test_salvage_stops_at_the_first_thing_that_is_not_an_object(self) -> None:
        text = '[{"comment": "a"}, 7, {"comment": "b"}]'

        assert [i["comment"] for i in salvage_review_items(text)] == ["a"]
