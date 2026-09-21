"""
AbstractiveCompressor — LLM-summarized body compression (v3.4).

Layers ON TOP OF, not instead of, the same must-keep contract every
compressor honors: the declaration line + docstring are always kept
VERBATIM, exactly as ExtractiveCompressor keeps them. Only the REMAINING
body is replaced — with a synthesized natural-language summary, never
claimed as verbatim source. ``kept_spans`` therefore only ever names lines
that are genuinely, byte-for-byte retained; the summary itself is not a
"span" of anything and is rendered under an explicit header saying so, so no
downstream renderer or citation can mistake it for quoted source.

Design, per research (arXiv:2410.02741 SigExt; qodo-ai/pr-agent's PR-diff
compression practice): explicitly inject the already-available code facts
(signature, docstring, qualified_name) into the summarization prompt rather
than trust the LLM to re-derive them from the body alone — LLMs do not
reliably extract these implicitly (arXiv:2304.06815, ICSE 2024). One
``TrelixChatClient.complete()`` call per compressed unit; never a provider
SDK call directly (house rule, see llm/client.py's module docstring).

Same graceful-degradation contract as ExtractiveCompressor: never raises
into the retrieval path — an LLM error, timeout, or empty response all
degrade to passthrough (the body unchanged), never a worse-without-a-trace
failure.
"""

from __future__ import annotations

import logging

import tiktoken

from trelix.compression.base import CompressionResult, CompressionUnit, Compressor
from trelix.llm.client import ChatMessage, TrelixChatClient

logger = logging.getLogger("trelix.compression.abstractive")

#: Marks the synthesized block so it is never mistaken for verbatim source —
#: mirrors ExtractiveCompressor's elision markers in spirit (never claim text
#: the reader cannot verify against the file), but for prose rather than a
#: line-count gap.
_SUMMARY_HEADER = "# --- abstractive summary of the remaining body (not verbatim source) ---"

_PROMPT_TEMPLATE = """\
Summarize the BEHAVIOR of this code symbol for a code-review retrieval context.
Focus on what it does, side effects, error handling, and anything relevant to
the query below. Do NOT repeat the signature or docstring verbatim -- they are
already shown separately and will be included alongside your summary. Output
ONLY the summary text: no preamble, no markdown code fences, target roughly
{target_tokens} tokens or fewer.

Query: {query}

Symbol: {qualified_name}
Signature: {signature}
Docstring: {docstring}

Body:
{body}
"""


class AbstractiveCompressor(Compressor):
    """See module docstring. Constructed via ``make_compressor``."""

    def __init__(self, chat_client: TrelixChatClient) -> None:
        self._chat_client = chat_client
        self._encoder = tiktoken.get_encoding("cl100k_base")  # matches ExtractiveCompressor

    def compress(
        self,
        query: str,
        unit: CompressionUnit,
        *,
        target_ratio: float,
        query_embedding: list[float] | None = None,
    ) -> CompressionResult:
        original_tokens = self._count(unit.body)

        # Passthrough: caller asked for no shrink (or an invalid ratio) — same
        # contract as ExtractiveCompressor, and it also means no LLM call is
        # ever made for a unit that didn't need compressing in the first place.
        if target_ratio >= 1.0:
            return self._passthrough(unit, original_tokens)

        try:
            return self._compress(query, unit, target_ratio, original_tokens)
        except Exception as exc:  # noqa: BLE001 — graceful degradation (extractive/reranker contract)
            logger.warning(
                "AbstractiveCompressor failed for %s (%s); returning body unchanged",
                unit.qualified_name,
                exc,
            )
            return self._passthrough(unit, original_tokens)

    def _passthrough(self, unit: CompressionUnit, original_tokens: int) -> CompressionResult:
        """Return the body unchanged. Same span-derivation rationale as
        ExtractiveCompressor._passthrough: a stored body is not always a
        line-faithful copy of the declared ``[line_start, line_end]`` range
        (extractors truncate it while keeping the full AST span), so the kept
        span is derived from how many lines ``body`` actually has."""
        n_lines = len(unit.body.splitlines())
        end = unit.line_start + n_lines - 1 if n_lines > 0 else unit.line_start
        lo = unit.line_start
        hi = unit.line_end if unit.line_end >= lo else lo
        a, b = max(lo, min(unit.line_start, hi)), max(lo, min(end, hi))
        return CompressionResult(
            text=unit.body,
            token_count=original_tokens,
            original_token_count=original_tokens,
            kept_spans=[(a, b)] if a <= b else [],
            provider="abstractive",
        )

    def _compress(
        self,
        query: str,
        unit: CompressionUnit,
        target_ratio: float,
        original_tokens: int,
    ) -> CompressionResult:
        if not unit.body.strip():
            return self._passthrough(unit, original_tokens)

        target_tokens = max(1, int(target_ratio * original_tokens))
        must_keep_text, must_keep_span = self._must_keep(unit)

        prompt = _PROMPT_TEMPLATE.format(
            target_tokens=target_tokens,
            query=query or "(none)",
            qualified_name=unit.qualified_name,
            signature=unit.signature,
            docstring=unit.docstring or "(none)",
            body=unit.body,
        )
        response = self._chat_client.complete(
            [ChatMessage(role="user", content=prompt)],
            # Headroom, not a hard cap on the summary's target length — a
            # summarizer that's cut off mid-sentence by max_tokens is worse
            # than one that ran a bit long, so this only guards against a
            # truly runaway response, not against exceeding target_tokens.
            max_tokens=max(256, target_tokens * 4),
        )
        summary = (response.content or "").strip()
        if not summary:
            return self._passthrough(unit, original_tokens)

        text = f"{must_keep_text}\n{_SUMMARY_HEADER}\n{summary}" if must_keep_text else summary
        # RESULT-LOSSLESS: recomputed, never inherited — same contract as
        # ExtractiveCompressor, even though nothing here is selected spans.
        return CompressionResult(
            text=text,
            token_count=self._count(text),
            original_token_count=original_tokens,
            kept_spans=must_keep_span,
            provider="abstractive",
        )

    def _must_keep(self, unit: CompressionUnit) -> tuple[str, list[tuple[int, int]]]:
        """The declaration line, verbatim, plus its absolute span.

        Deliberately independent of ExtractiveCompressor's must-keep
        machinery (no shared body-line-index helpers) — abstractive doesn't
        do span-based selection over the rest of the body, so it only needs
        this one span, not the general merge/clamp apparatus extractive uses
        for many candidate spans.
        """
        body_lines = unit.body.splitlines()
        if not body_lines:
            return "", []
        sig_end = 0
        for i, line in enumerate(body_lines[:8]):
            if line.rstrip().endswith((":", "{")):
                sig_end = i
                break
        lo = unit.line_start
        hi = unit.line_end if unit.line_end >= lo else lo
        abs_start = max(lo, min(unit.line_start, hi))
        abs_end = max(lo, min(unit.line_start + sig_end, hi))
        text = "\n".join(body_lines[: sig_end + 1])
        return text, ([(abs_start, abs_end)] if abs_start <= abs_end else [])

    def _count(self, text: str) -> int:
        return len(self._encoder.encode(text))
