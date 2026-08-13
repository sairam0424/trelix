"""
Assembler-side compression packing — the wave-based, result-lossless packer that
lets an oversized result be INCLUDED in a shrunk form instead of dropped.

Split out of ``assembler.py`` so both files stay small: ``ContextAssembler``
owns the budget and the output format; this module owns the compression waves
and the citation-faithful block rendering.

Two waves, and the wave order is the whole safety argument:

  Wave 1  the EXACT uncompressed pack the assembler already does (greedy /
          proportional / breadth-first). The compressor is not called at all.
  Wave 2  ONLY the candidates wave 1 could not fit — the ones today silently
          DROPS — are compressed and re-offered against the leftover budget.

So the compressed selection is always a SUPERSET of the uncompressed selection:
Recall / MRR / nDCG can only go up, never down, by construction. Compression is
also lazy — a result that already fits is never touched, so its text stays
byte-identical and the common case costs zero compression work.

Per candidate, wave 2 tries a ladder and stops at the first rung that fits:
  1. target ratio, tightened to what the leftover budget can actually hold
  2. the compressor's floor (signature + docstring only)
  3. skip — recorded in the stats so the trace shows it was not silent
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from trelix.core.models import Chunk, SearchResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trelix.compression.base import CompressionResult, Compressor

logger = logging.getLogger(__name__)

# Rendered between kept blocks so the text never implies contiguity.
# Must match trelix.compression.extractive._ELISION.
_ELISION = "# ... {n} lines elided ..."

# Recognizes a rendered elision marker. Built from _ELISION itself so the two
# cannot drift apart: a block's line-range header covers only the lines up to
# the next marker, so the renderer must be able to spot one.
_ELISION_RE = re.compile("^" + re.escape(_ELISION).replace(r"\{n\}", r"\d+") + "$")

# "Keep essentially nothing" — the compressor's must-keep spans (signature +
# docstring) are unconditional, so this asks for its smallest legal output.
_FLOOR_RATIO = 0.01


def pack_compressed(
    *,
    query: str,
    eligible: list[SearchResult],
    wave1: list[SearchResult],
    token_budget: int,
    compressor: Compressor,
    target_ratio: float,
    min_tokens: int,
    query_embedding: list[float] | None,
) -> tuple[list[SearchResult], dict[int, CompressionResult], dict[str, object]]:
    """
    Extend an uncompressed pack with compressed versions of what it could not fit.

    Args:
        query:           raw user query — the lexical path scores segments against it.
        eligible:        the ordered candidate pool the assembler considered. For
                         breadth-first this is ALREADY truncated to max_per_file,
                         so that cap is preserved through wave 2 for free.
        wave1:           the uncompressed selection (a subset of ``eligible``).
        token_budget:    the same hard ceiling the uncompressed pack respected.
        compressor:      any :class:`Compressor`; never raises by contract.
        target_ratio:    per-intent target fraction of the original body to keep.
        min_tokens:      bodies smaller than this are not worth compressing.
        query_embedding: an embedding the retriever ALREADY computed, or None.
                         Never triggers a new embedding/network call either way.

    Returns:
        ``(selected, compressed, stats)`` where ``selected`` is a superset of
        ``wave1`` restored to ``eligible`` order, ``compressed`` maps
        ``chunk.symbol_id`` -> :class:`CompressionResult` for the wave-2 entries
        only (so the formatter knows which results need span-split rendering),
        and ``stats`` is a trace-friendly summary.
    """
    position = {id(r): i for i, r in enumerate(eligible)}
    wave1_ids = {id(r) for r in wave1}
    tokens_used = sum(r.chunk.token_count for r in wave1)

    selected = list(wave1)
    compressed: dict[int, CompressionResult] = {}
    counts = {"compressed": 0, "floor": 0, "skipped": 0, "below_min_tokens": 0, "errors": 0}
    paths: list[str] = []

    for result in eligible:
        if id(result) in wave1_ids:
            continue
        remaining = token_budget - tokens_used
        if remaining <= 0:
            counts["skipped"] += 1
            continue
        original = result.chunk.token_count
        if original <= 0 or original < min_tokens:
            counts["below_min_tokens"] += 1
            continue

        outcome = _compress_to_fit(
            query=query,
            result=result,
            compressor=compressor,
            target_ratio=target_ratio,
            remaining=remaining,
            query_embedding=query_embedding,
        )
        if outcome is None:
            counts["skipped"] += 1
            continue
        cresult, rung = outcome

        counts[rung] += 1
        path = getattr(compressor, "last_path", None)
        if path and path not in paths:
            paths.append(path)

        shrunk = _rebuild(result, cresult)
        position[id(shrunk)] = position[id(result)]
        selected.append(shrunk)
        compressed[shrunk.chunk.symbol_id] = cresult
        tokens_used += cresult.token_count

    selected.sort(key=lambda r: position[id(r)])
    stats: dict[str, object] = {
        "wave1_kept": len(wave1),
        "wave2_added": counts["compressed"] + counts["floor"],
        **counts,
        "paths": paths,
        "tokens_used": tokens_used,
    }
    return selected, compressed, stats


def _compress_to_fit(
    *,
    query: str,
    result: SearchResult,
    compressor: Compressor,
    target_ratio: float,
    remaining: int,
    query_embedding: list[float] | None,
) -> tuple[CompressionResult, str] | None:
    """Walk the ladder; return ``(result, rung)`` or None when nothing fits."""
    from trelix.compression.base import CompressionUnit

    symbol = result.symbol
    body = symbol.body or ""
    if not body.strip():
        return None
    unit = CompressionUnit(
        symbol_id=result.chunk.symbol_id,
        body=body,
        signature=symbol.signature or "",
        docstring=symbol.docstring,
        line_start=symbol.line_start,
        line_end=symbol.line_end,
        qualified_name=symbol.qualified_name,
    )
    original = result.chunk.token_count

    # Tighten the target to what the leftover budget can actually hold, so a
    # 5000-token body with 200 tokens left aims at 200, not at 0.45 * 5000.
    effective = min(target_ratio, remaining / original)
    for ratio, rung in ((effective, "compressed"), (_FLOOR_RATIO, "floor")):
        try:
            candidate = compressor.compress(
                query, unit, target_ratio=ratio, query_embedding=query_embedding
            )
        except Exception as exc:  # noqa: BLE001 — graceful degradation contract
            logger.warning("Compressing %s failed (%s); skipping", symbol.qualified_name, exc)
            return None
        # Must both fit AND actually be smaller — a passthrough (ratio clamped
        # to >= 1.0, or nothing elidable) would put us right back over budget.
        if candidate.token_count <= remaining and candidate.token_count < original:
            return candidate, rung
    return None


def _rebuild(result: SearchResult, cresult: CompressionResult) -> SearchResult:
    """
    Build a NEW SearchResult wrapping a NEW Chunk — the original is never mutated
    (immutability rule; same posture as reranker's result rebuild).

    ``symbol_id`` is carried through unchanged so ``Retriever._dedup()`` and any
    downstream symbol-keyed lookup behave identically. ``embedding`` is dropped:
    the text changed, so the stored vector no longer describes it.
    """
    return SearchResult(
        chunk=Chunk(
            symbol_id=result.chunk.symbol_id,
            chunk_text=cresult.text,
            token_count=cresult.token_count,
            embedding=None,
            id=result.chunk.id,
        ),
        symbol=result.symbol,
        file=result.file,
        score=result.score,
        rank=result.rank,
        source=result.source,
    )


def format_compressed_blocks(result: SearchResult, cresult: CompressionResult) -> str:
    """
    Render a partially-kept body as SEPARATE line-range blocks.

    Citation fidelity is the whole point: one header claiming ``[Lines 42-99]``
    over text that no longer contains lines 55-90 is a lie the LLM will cite. So
    every kept span gets its own truthful header, and every gap gets an explicit
    ``# ... N lines elided ...`` marker::

        [Lines 42-45] LoginView.authenticate_user
        def authenticate_user(self, username, password):
            \"\"\"Authenticate and return the user.\"\"\"
        # ... 9 lines elided ...
        [Lines 55-56] LoginView.authenticate_user
            return user

    Falls back to a single block over the kept span envelope if the stored spans
    cannot be mapped onto the body (never raises, never renders nothing).
    """
    symbol = result.symbol
    body_lines = (symbol.body or "").splitlines()
    spans = [
        (a, b)
        for a, b in cresult.kept_spans
        if symbol.line_start <= a <= b <= symbol.line_start + len(body_lines) - 1
    ]
    if not spans or not body_lines:
        return _fallback_block(symbol.line_start, symbol.line_end, symbol.qualified_name, cresult)

    parts: list[str] = []
    head_gap = spans[0][0] - symbol.line_start
    if head_gap > 0:
        parts.append(_ELISION.format(n=head_gap))
    previous_end: int | None = None
    for start, end in spans:
        if previous_end is not None:
            gap = start - previous_end - 1
            if gap > 0:
                parts.append(_ELISION.format(n=gap))
        i0 = start - symbol.line_start
        i1 = end - symbol.line_start
        text = "\n".join(body_lines[i0 : i1 + 1])
        parts.append(f"[Lines {start}-{end}] {symbol.qualified_name}\n{text}")
        previous_end = end
    # Measured against the body we actually HAVE, not symbol.line_end. An elision
    # marker means "compression removed lines we held"; when an extractor already
    # truncated the stored body (body=...[:2000] while keeping the full AST span),
    # the missing tail was never ours to elide, and counting it would both invent
    # a bogus number and make the block show more lines than its header claims.
    body_end = symbol.line_start + len(body_lines) - 1
    tail_gap = min(symbol.line_end, body_end) - spans[-1][1]
    if tail_gap > 0:
        parts.append(_ELISION.format(n=tail_gap))
    return "\n".join(parts)


def _envelope(cresult: CompressionResult, line_start: int, line_end: int) -> tuple[int, int]:
    """Smallest [start, end] covering the kept spans, clamped to the symbol.

    Never returns an inverted range. Clamping the two ends independently used to
    produce nonsense like ``(9000, 122)`` when every kept span lay outside the
    symbol (e.g. a stale/bogus span): ``max(122, 9000)`` paired with
    ``min(122, 9000)``. When no span overlaps the symbol we fall back to the
    declared range rather than emitting a negative-length citation.
    """
    hi = line_end if line_end >= line_start else line_start
    if not cresult.kept_spans:
        return (line_start, hi)
    start = max(line_start, min(a for a, _ in cresult.kept_spans))
    end = min(hi, max(b for _, b in cresult.kept_spans))
    if start > end:  # spans lie entirely outside the symbol
        return (line_start, hi)
    return (start, end)


def _fallback_block(
    line_start: int, line_end: int, qualified_name: str, cresult: CompressionResult
) -> str:
    """One truthful block for when stored spans cannot be mapped onto the body.

    The header is derived from the TEXT ACTUALLY RENDERED — never from the
    unmappable spans — so it can neither invert nor over-claim. Only the leading
    run of real source lines is covered, because a block ends at the first
    elision marker: lines after it are not under this header's range.
    """
    hi = line_end if line_end >= line_start else line_start
    leading = 0
    for line in cresult.text.splitlines():
        if _ELISION_RE.match(line):
            break
        leading += 1
    end = line_start + leading - 1 if leading > 0 else line_start
    end = min(max(end, line_start), hi)  # stay inside the symbol, never inverted
    return f"[Lines {line_start}-{end}] {qualified_name}\n{cresult.text}"
