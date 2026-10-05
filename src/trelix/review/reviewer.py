"""
DiffReviewer — retrieval-augmented PR review.

For each changed hunk:
1. Build a search query from changed lines (identifier extraction)
2. Retrieve relevant context via trelix hybrid search
3. Call LLM with hunk + context -> structured review comments
4. Parse the reply into ReviewComment objects and record a status for the hunk

Crash-safe: any failure returns [] and logs a warning. `last_outcome` says what became of each
hunk, because `[]` alone cannot tell "the model found nothing" from "the reply was cut off".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from trelix.core.index_check import IndexNotFoundError, require_index
from trelix.llm.finish_reasons import CONTENT_FILTER, LENGTH, PAUSED, REFUSAL, STOP
from trelix.llm.prompt import fenced_block
from trelix.review.hunk_status import (
    HunkResult,
    HunkStatus,
    extract_review_items,
    safe_token,
    salvage_review_items,
)

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig
    from trelix.llm.client import ChatResponse
    from trelix.review.diff_parser import DiffHunk

logger = logging.getLogger("trelix.review.reviewer")

_REVIEW_SYSTEM = """\
You are an expert code reviewer. Given a code diff and its surrounding context,
provide concise, actionable review comments.

Return a JSON array of review comments, each with:
  line_start: int (line number in the new file)
  line_end: int
  severity: "INFO" | "WARN" | "ERROR"
  comment: str (concise, specific, actionable — no platitudes)

Return [] if no issues are found. Do not explain your reasoning outside the JSON array.
"""

# A hunk whose retrieval raised is still reviewed, from the diff alone — a
# diff-only comment is often still right, so dropping the hunk would be worse.
# But the fallback used to be `except Exception: pass`, so a vector store that
# had gone missing produced a full page of confident comments written against
# context_text="" with not one log line and nothing in the output to tell them
# apart from grounded ones. `comment` is the only field every renderer prints
# (CLI table, --json, posted GitHub inline comments), so the label goes there.
_NO_CONTEXT_LABEL = " [trelix: no codebase context — retrieval failed for this hunk]"

# A reply cut off at the limit is retried once at this multiple of it, up to the ceiling. The
# ceiling is also the largest `review_max_tokens`: the Anthropic SDK refuses a non-streaming
# request above 21333 tokens (it would run longer than its 10-minute limit), and some models
# refuse less, so a retry above this could raise before producing anything.
_RETRY_FACTOR = 4
_RETRY_CEILING = 16384
_RETRYABLE = frozenset({LENGTH})
_CUT_OFF = frozenset({LENGTH, PAUSED})


@dataclass
class ReviewComment:
    """A single review comment on a code change."""

    file_path: str
    line_start: int
    line_end: int
    severity: str  # "INFO" | "WARN" | "ERROR"
    comment: str


@dataclass(frozen=True)
class ReviewOutcome:
    """What happened during a review, separate from what it found.

    `review()` returns `[]` both when the model looked and found nothing and when
    nothing was looked at; this record is how a caller tells the two apart.

    `hunks_failed` counts the hunks that were not reviewed: every status except `reviewed`,
    so a reply that was cut off, refused, filtered or unparseable counts, not only a call that
    raised. `hunk_results` has the status of each hunk, in order.

    `could_not_review` is stricter than "nothing was reviewed": a cut-off hunk that still kept
    complete findings is a partial review, not a review that did not run.
    """

    llm_available: bool = True
    hunks_total: int = 0
    hunks_failed: int = 0
    # Left out of ==, so two outcomes are equal when their counters are; the per-hunk detail is
    # for display and the outcome file, and tests assert it directly.
    hunk_results: tuple[HunkResult, ...] = field(default=(), compare=False)

    @property
    def could_not_review(self) -> bool:
        """True when nothing came of the review: no usable LLM, or no hunk was reviewed and
        none kept a finding."""
        if not self.llm_available:
            return True
        kept_findings = any(r.kept_comments for r in self.hunk_results)
        return self.hunks_total > 0 and self.hunks_failed >= self.hunks_total and not kept_findings

    @property
    def hunks_reviewed(self) -> int:
        return max(self.hunks_total - self.hunks_failed, 0)

    @property
    def unreviewed_fraction(self) -> float:
        """The share of hunks that were not reviewed, 0.0 when there were no hunks."""
        if self.hunks_total <= 0:
            return 0.0
        return min(self.hunks_failed / self.hunks_total, 1.0)


@dataclass(frozen=True)
class _HunkReview:
    comments: list[ReviewComment]
    result: HunkResult


class LLMNotConfiguredError(RuntimeError):
    """A backend answered with its "no credentials" placeholder instead of a review."""


def _error_result(hunk: DiffHunk, detail: str) -> HunkResult:
    return HunkResult(hunk.file_path, hunk.new_start, HunkStatus.ERROR, detail)


class DiffReviewer:
    """
    Retrieve-augmented code reviewer for git diffs.

    Usage:
        reviewer = DiffReviewer(config)
        hunks = DiffParser().from_git(repo_path)
        comments = reviewer.review(hunks)
    """

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._retriever: Any = None
        self._llm_client: Any = None
        self.last_outcome = ReviewOutcome()

    def _get_retriever(self) -> Any:
        """The Retriever for this repository, or None when it has no index yet.

        A repository nobody indexed is still reviewed, from the diff alone: there is
        nothing to retrieve. Building a Retriever there would only create an empty index
        (see `trelix.core.index_check`), so it is skipped.
        """
        if self._retriever is None:
            try:
                require_index(self._config)
            except IndexNotFoundError:
                return None
            from trelix.retrieval.retriever import Retriever

            self._retriever = Retriever(self._config)
        return self._retriever

    def _get_client(self) -> Any:
        if self._llm_client is None:
            from trelix.llm.factory import build_chat_client

            try:
                self._llm_client = build_chat_client(self._config.llm)
            except Exception as exc:
                logger.debug("DiffReviewer: could not build LLM client: %s", exc)
                return None
        return self._llm_client

    def review(
        self,
        hunks: list[DiffHunk] | None = None,
        diff_text: str | None = None,
    ) -> list[ReviewComment]:
        """
        Review a list of diff hunks (or a raw diff string). Returns [] on any failure.

        `[]` does not say whether the model looked and found nothing or nothing was
        looked at: read `last_outcome` after the call to tell the two apart.

        Args:
            hunks:     DiffHunk objects from DiffParser. If omitted, diff_text is parsed.
            diff_text: Raw unified diff string. Parsed into hunks when hunks is None/empty.

        Returns:
            list[ReviewComment] — empty list if no issues or any error
        """
        if hunks is None:
            hunks = []

        if not hunks and diff_text:
            from trelix.review.diff_parser import DiffParser

            hunks = DiffParser().parse(diff_text)

        if not hunks:
            self.last_outcome = ReviewOutcome()
            return []

        comments: list[ReviewComment] = []
        client = self._get_client()
        if client is None:
            logger.warning("DiffReviewer: no LLM client available")
            self.last_outcome = ReviewOutcome(
                llm_available=False,
                hunks_total=len(hunks),
                hunks_failed=len(hunks),
                hunk_results=tuple(_error_result(h, "no_llm_client") for h in hunks),
            )
            return []

        results: list[HunkResult] = []
        llm_available = True
        for hunk in hunks:
            try:
                review = self._review_hunk(hunk, client)
            except LLMNotConfiguredError as exc:
                # Every remaining hunk would get the same placeholder answer.
                logger.warning("DiffReviewer: %s", exc)
                llm_available = False
                results = [_error_result(h, "not_configured") for h in hunks]
                break
            except Exception as exc:
                logger.warning("DiffReviewer: hunk review failed (non-fatal): %s", exc)
                results.append(
                    _error_result(
                        hunk, f"exception:{safe_token(type(exc).__name__, default='unknown')}"
                    )
                )
                continue
            comments.extend(review.comments)
            results.append(review.result)
            if not review.result.reviewed:
                logger.warning(
                    "DiffReviewer: %s:%d was not reviewed (%s: %s)",
                    review.result.file_path,
                    review.result.line,
                    review.result.status.value,
                    review.result.detail or "no detail",
                )

        self.last_outcome = ReviewOutcome(
            llm_available=llm_available,
            hunks_total=len(hunks),
            hunks_failed=sum(1 for r in results if not r.reviewed),
            hunk_results=tuple(results),
        )
        return comments

    def _call(self, client: Any, user_content: str, max_tokens: int) -> Any:
        from trelix.llm.client import UNCONFIGURED_MODEL, ChatMessage

        response = client.complete(
            messages=[ChatMessage(role="user", content=user_content)],
            max_tokens=max_tokens,
            temperature=0.0,
            system=_REVIEW_SYSTEM,
        )
        if response.model == UNCONFIGURED_MODEL:
            raise LLMNotConfiguredError(response.content)
        return response

    def _review_hunk(self, hunk: DiffHunk, client: Any) -> _HunkReview:
        """Review a single hunk with retrieved context."""
        # Retrieve context for this hunk
        query = hunk.to_search_query()
        context_text = ""
        retrieval_failed = False
        try:
            retriever = self._get_retriever()
            if retriever is not None:
                ctx = retriever.retrieve(query)
                context_text = ctx.context_text[:3000]  # cap context size
        except Exception as exc:
            # WARNING, matching the sibling per-hunk handler in review(): the CLI
            # configures WARNING, so a DEBUG record would be as silent as the
            # `pass` this replaced. The file name is in the message because one
            # degraded hunk in a 40-hunk PR is only actionable if you know which.
            retrieval_failed = True
            logger.warning(
                "DiffReviewer: context retrieval failed for %s — reviewing the diff alone: %s",
                hunk.file_path,
                exc,
            )

        # Build diff text for the hunk
        diff_lines = []
        for line in hunk.removed:
            diff_lines.append(f"- {line}")
        for line in hunk.added:
            diff_lines.append(f"+ {line}")
        diff_text = "\n".join(diff_lines)

        # Each fence length is derived from its own payload. Reviewing a change
        # to a markdown file puts ``` directly into diff_text, and retrieved
        # context is a symbol body that can carry one too; a hard-coded three-
        # backtick fence closes early on either and the remainder of the payload
        # reaches the model as instructions rather than as quoted code.
        user_content = (
            f"File: {hunk.file_path} (lines {hunk.new_start}–"
            f"{hunk.new_start + hunk.new_lines})\n\n"
            f"Changed code:\n{fenced_block(diff_text)}\n\n"
        )
        if context_text:
            user_content += f"Related codebase context:\n{fenced_block(context_text)}\n\n"
        user_content += "Provide review comments as a JSON array."

        limit = self._config.review_max_tokens
        response = self._call(client, user_content, limit)
        used, retried, retry_failed = limit, False, False
        if self._effective_finish(response, used) in _RETRYABLE:
            larger = min(limit * _RETRY_FACTOR, _RETRY_CEILING)
            if larger > limit:
                logger.info(
                    "DiffReviewer: reply for %s was cut off at %d tokens; retrying at %d",
                    hunk.file_path,
                    limit,
                    larger,
                )
                try:
                    retry = self._call(client, user_content, larger)
                except LLMNotConfiguredError:
                    raise
                except Exception as exc:
                    # The first reply still holds the complete findings written before it was
                    # cut off; a failed retry must not throw them away.
                    retry_failed = True
                    logger.warning(
                        "DiffReviewer: retry for %s at %d tokens failed, keeping first reply: %s",
                        hunk.file_path,
                        larger,
                        exc,
                    )
                else:
                    response, used, retried = retry, larger, True

        return self._classify_reply(
            response,
            hunk,
            used=used,
            context_failed=retrieval_failed,
            retried=retried,
            retry_failed=retry_failed,
        )

    @staticmethod
    def _effective_finish(response: ChatResponse, limit: int) -> str:
        """The reply's finish reason, read as `length` when it says stop but used every token.

        LiteLLM turns some provider stop values into "stop" before trelix sees them, and a reply
        that consumed the whole limit it was given was cut off whatever the provider called it.
        """
        finish = response.finish_reason
        tokens = response.output_tokens
        if finish == STOP and isinstance(tokens, int) and not isinstance(tokens, bool):
            if tokens >= limit:
                return LENGTH
        return finish

    def _classify_reply(
        self,
        response: ChatResponse,
        hunk: DiffHunk,
        *,
        used: int,
        context_failed: bool,
        retried: bool,
        retry_failed: bool,
    ) -> _HunkReview:
        """Turn a model reply into comments and a status; only a clean, parsed array is reviewed."""
        # Labelled per hunk, not per review: retrieval failing on one file must
        # not cast doubt on the grounded comments for the rest of the PR.
        suffix = _NO_CONTEXT_LABEL if context_failed else ""

        def outcome(status: HunkStatus, detail: str = "", kept: int = 0) -> HunkResult:
            return HunkResult(hunk.file_path, hunk.new_start, status, detail, kept)

        finish = self._effective_finish(response, used)
        is_text = isinstance(response.content, str)
        content = response.content if is_text else ""
        if finish in (REFUSAL, CONTENT_FILTER):
            return _HunkReview([], outcome(HunkStatus.REFUSED, finish))
        if finish in _CUT_OFF:
            kept = self._comments(salvage_review_items(content), hunk, suffix)
            detail = finish
            if retry_failed:
                detail = f"{finish}_retry_failed"
            elif retried:
                detail = f"{finish}_after_retry"
            return _HunkReview(kept, outcome(HunkStatus.TRUNCATED, detail, len(kept)))
        if finish != STOP:
            # Only a plain stop is a finished review. The reviewer offers no tools, so a
            # tool_calls stop is as suspect as an error or an unclassified value.
            token = safe_token(getattr(response, "raw_finish_reason", None))
            return _HunkReview([], outcome(HunkStatus.ERROR, f"{safe_token(finish)}:{token}"))

        items = extract_review_items(content)
        if items is None:
            if not is_text:
                detail = "non_text_reply"
            else:
                detail = "empty_reply" if not content.strip() else "no_review_array"
            return _HunkReview([], outcome(HunkStatus.PARSE_FAILED, detail))
        parsed = self._comments(items, hunk, suffix)
        detail = "retried" if retried else ""
        return _HunkReview(parsed, outcome(HunkStatus.REVIEWED, detail, len(parsed)))

    @staticmethod
    def _comments(items: list[dict[str, Any]], hunk: DiffHunk, suffix: str) -> list[ReviewComment]:
        default_end = hunk.new_start + hunk.new_lines
        return [
            ReviewComment(
                file_path=hunk.file_path,
                line_start=_line_number(item.get("line_start"), hunk.new_start),
                line_end=_line_number(item.get("line_end"), default_end),
                severity=str(item.get("severity", "INFO")),
                comment=str(item["comment"]) + suffix,
            )
            for item in items
        ]


def _line_number(value: object, default: int) -> int:
    """A line number from model output, or `default` when it is missing or not a usable number.

    One bad number must not cost the hunk its other findings, and a finding with an approximate
    location is worth more than none.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    try:
        return int(value)
    except (ValueError, OverflowError):
        return default
