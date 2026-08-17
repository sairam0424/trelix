"""
File-level LLM summarizer for RAPTOR-style multi-granularity indexing.

Generates a 2-4 sentence description of a file's purpose from its symbol list.
The summary is stored in `file_summaries` DB table and embedded as a
high-level retrieval entry — enabling queries like "how does this codebase
handle authentication" to retrieve file-level context rather than scattered
symbol chunks.

This is Phase 2 of multi-granularity indexing.  Phase 1 (symbol-level) is
the existing chunker. Phase 2 adds file-level summaries.

Research basis: RAPTOR (arXiv 2401.18059, ICLR 2024) — 82.6% on QuALITY
benchmark (+20.3pp absolute) via recursive hierarchical summarization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.models import Language, Symbol
    from trelix.llm.client import TrelixChatClient

logger = logging.getLogger("trelix.indexing.file_summarizer")

_SYSTEM_PROMPT = (
    "You are a senior engineer writing concise file-level documentation. "
    "Summarize what a source file does in 2-4 sentences. "
    "Focus on: what problem it solves, key classes/functions, and its role "
    "in the broader codebase. Be specific, not generic."
)

_PROMPT_TEMPLATE = """\
File: {rel_path}
Language: {language}

Top symbols:
{symbols_text}

Write a 2-4 sentence summary of what this file does."""


class FileSummarizer:
    """
    Generates LLM-based file-level summaries for multi-granularity indexing.

    Safe to use without an LLM client — returns empty string on any failure.
    The indexer treats empty summaries as "no file-level entry" and skips them.

    "" is a FAILURE signal, not a neutral one, and the caller is responsible for
    counting it: the only other way to get "" is an empty symbol list, which the
    caller can test for itself. The indexer's `if summary:` had no `else` and no
    counter, so on the live index 442 of 467 summaries succeeded and the 25 that
    failed were indistinguishable from 467 successes — from the stats dict, the
    console, and (because the failure path logged at DEBUG while the CLI runs at
    WARNING) the logs too.
    """

    def __init__(
        self,
        client: TrelixChatClient,
        max_symbols: int = 30,
        max_tokens: int = 150,
    ) -> None:
        self._client = client
        self._max_symbols = max_symbols
        self._max_tokens = max_tokens

    def summarize(
        self,
        rel_path: str,
        symbols: list[Symbol],
        language: Language,
    ) -> str:
        """
        Generate a file-level summary. Returns "" on any failure.

        Args:
            rel_path: repo-relative file path (shown to LLM for context)
            symbols: parsed symbols from the file (truncated to max_symbols)
            language: file language enum

        Returns:
            summary string, or "" if LLM unavailable / failed
        """
        if not symbols:
            return ""

        top = symbols[: self._max_symbols]
        sym_lines = "\n".join(
            f"- {s.kind.value} {s.qualified_name}: {s.signature[:80]}" for s in top
        )
        prompt = _PROMPT_TEMPLATE.format(
            rel_path=rel_path,
            language=language.value,
            symbols_text=sym_lines,
        )

        try:
            from trelix.llm.client import ChatMessage

            response = self._client.complete(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=self._max_tokens,
                temperature=0.0,
                system=_SYSTEM_PROMPT,
            )
            summary = response.content.strip()
            if not summary:
                # A model that answers with whitespace produced no summary; without
                # this branch it took the success path and returned "", which the
                # caller cannot tell apart from a 429.
                logger.warning(
                    "File summary for %s came back blank — the file gets no "
                    "file-level retrieval entry",
                    rel_path,
                )
            return summary
        except Exception as exc:
            # WARNING, not DEBUG: this is the only place the REASON for the "" exists,
            # and the CLI runs at WARNING. At DEBUG, a whole run's summaries could fail
            # on expired credentials and the only visible trace was a file_summaries
            # table that happened to be empty.
            logger.warning(
                "File summary failed for %s (%s) — the file gets no file-level retrieval entry",
                rel_path,
                exc,
            )
            return ""
