"""
DiffReviewer — retrieval-augmented PR review.

For each changed hunk:
1. Build a search query from changed lines (identifier extraction)
2. Retrieve relevant context via trelix hybrid search
3. Call LLM with hunk + context -> structured review comments
4. Parse and return ReviewComment objects

Crash-safe: any failure returns [] and logs a warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from trelix.llm.prompt import fenced_block

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig
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


@dataclass
class ReviewComment:
    """A single review comment on a code change."""

    file_path: str
    line_start: int
    line_end: int
    severity: str  # "INFO" | "WARN" | "ERROR"
    comment: str


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

    def _get_retriever(self) -> Any:
        if self._retriever is None:
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
            return []

        comments: list[ReviewComment] = []
        client = self._get_client()
        if client is None:
            logger.warning("DiffReviewer: no LLM client available")
            return []

        for hunk in hunks:
            try:
                hunk_comments = self._review_hunk(hunk, client)
                comments.extend(hunk_comments)
            except Exception as exc:
                logger.warning("DiffReviewer: hunk review failed (non-fatal): %s", exc)

        return comments

    def _review_hunk(self, hunk: DiffHunk, client: Any) -> list[ReviewComment]:
        """Review a single hunk with retrieved context."""
        from trelix.llm.client import ChatMessage

        # Retrieve context for this hunk
        query = hunk.to_search_query()
        context_text = ""
        retrieval_failed = False
        try:
            retriever = self._get_retriever()
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

        response = client.complete(
            messages=[ChatMessage(role="user", content=user_content)],
            max_tokens=512,
            temperature=0.0,
            system=_REVIEW_SYSTEM,
        )

        return self._parse_response(response.content, hunk, context_failed=retrieval_failed)

    def _parse_response(
        self, content: str, hunk: DiffHunk, *, context_failed: bool = False
    ) -> list[ReviewComment]:
        """Parse LLM JSON response into ReviewComment objects."""
        # Labelled per hunk, not per review: retrieval failing on one file must
        # not cast doubt on the grounded comments for the rest of the PR.
        suffix = _NO_CONTEXT_LABEL if context_failed else ""
        try:
            # Extract JSON array from response (LLM may add prose before/after)
            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            items = json.loads(content[start:end])
            return [
                ReviewComment(
                    file_path=hunk.file_path,
                    line_start=int(item.get("line_start", hunk.new_start)),
                    line_end=int(item.get("line_end", hunk.new_start + hunk.new_lines)),
                    severity=str(item.get("severity", "INFO")),
                    comment=str(item.get("comment", "")) + suffix,
                )
                for item in items
                if isinstance(item, dict) and item.get("comment")
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("DiffReviewer._parse_response failed: %s", exc)
            return []
