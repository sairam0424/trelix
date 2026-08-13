"""
ExtractiveCompressor — zero-inference, result-lossless body compression.

Two scoring paths, chosen at runtime and logged once:
  sub_chunk — index has ``sub_chunks`` rows + stored vectors + the query
              embedding was passed in: score each sub-chunk by
              cosine(query_embedding, sub_chunk_vector), keep top spans. Reuses
              vectors from index time (chunk_id = id + _SUB_CHUNK_OFFSET); no
              new embedding / API / network call.
  lexical   — otherwise (non-Python, multi-granularity off, no query embedding,
              or no stored vectors): split on blank lines + brace balance and
              score segments by query-token overlap. Zero inference, any language.

Both paths ALWAYS keep the signature line + docstring, emit kept spans as
absolute subranges of the unit's span, render elided gaps with explicit
markers, and recompute ``token_count`` via the assembler's tiktoken encoder.
"""

from __future__ import annotations

import logging
import re
import struct

import numpy as np
import tiktoken

from trelix.compression.base import CompressionResult, CompressionUnit, Compressor

logger = logging.getLogger("trelix.compression.extractive")

# Mirrors trelix.store.vector.SQLiteVectorStore._SUB_CHUNK_OFFSET: sub-chunk
# vectors live in chunk_embeddings at chunk_id = sub_chunk_id + this offset.
_SUB_CHUNK_OFFSET = 10_000_000

# Marker rendered between kept blocks so the text never implies contiguity.
_ELISION = "# ... {n} lines elided ..."

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# Line-relative (0-indexed) inclusive span within a body.
_Span = tuple[int, int]


class ExtractiveCompressor(Compressor):
    """See module docstring. Constructed via ``make_compressor``."""

    def __init__(self, db: object, embedder: object | None = None) -> None:
        self._db = db
        # Stored for reserved future use / trace; the extractive path never
        # calls the embedder (the query embedding is passed into compress()).
        self._embedder = embedder
        self._encoder = tiktoken.get_encoding("cl100k_base")  # matches assembler.count_tokens
        self._sub_chunks_available: bool | None = None  # cached runtime probe
        self._logged_path = False
        #: Last scoring path used ("sub_chunk" | "lexical" | "passthrough") —
        #: surfaced so the retriever can record it in the query trace.
        self.last_path: str | None = None

    # --- Public API ---

    def compress(
        self,
        query: str,
        unit: CompressionUnit,
        *,
        target_ratio: float,
        query_embedding: list[float] | None = None,
    ) -> CompressionResult:
        original_tokens = self._count(unit.body)

        # Passthrough: caller asked for no shrink (or an invalid ratio) — return
        # the body verbatim (byte-identical text, full-span citation).
        if target_ratio >= 1.0:
            return self._passthrough(unit, original_tokens)

        try:
            return self._compress(query, unit, target_ratio, query_embedding, original_tokens)
        except Exception as exc:  # noqa: BLE001 — graceful degradation (reranker contract)
            logger.warning(
                "ExtractiveCompressor failed for %s (%s); returning body unchanged",
                unit.qualified_name,
                exc,
            )
            return self._passthrough(unit, original_tokens)

    def _passthrough(self, unit: CompressionUnit, original_tokens: int) -> CompressionResult:
        """Return the body unchanged — used for ratio>=1.0, empty, or any failure.

        The kept span is derived from how many lines ``body`` ACTUALLY has, not
        from the declared ``[line_start, line_end]``. A stored body is often not
        a line-faithful copy of its declared range: ~23 extractor sites truncate
        it (e.g. ``body=self._txt(node, src)[:2000]``) while keeping the full AST
        span, so claiming the whole range would cite lines the text lacks.
        """
        self.last_path = "passthrough"
        n_lines = len(unit.body.splitlines())
        end = unit.line_start + n_lines - 1 if n_lines > 0 else unit.line_start
        return CompressionResult(
            text=unit.body,
            token_count=original_tokens,
            original_token_count=original_tokens,
            kept_spans=self._clamp_spans(unit, [(unit.line_start, end)]),
            provider="extractive",
        )

    @staticmethod
    def _clamp_spans(unit: CompressionUnit, spans: list[_Span]) -> list[_Span]:
        """Clamp absolute spans into the unit's declared ``[line_start, line_end]``.

        Enforces CompressionResult's documented invariant ("every span is a
        subrange of the unit's original range") at the point of creation, so no
        downstream renderer can be handed an out-of-range or inverted span.
        Needed because ``line_start + body_index`` can escape the declared range
        when a parser prepends a synthetic line to ``body`` (making it longer
        than the range) or truncates it (making it shorter).
        """
        lo = unit.line_start
        hi = unit.line_end if unit.line_end >= lo else lo  # defensive: malformed span
        out: list[_Span] = []
        for a, b in spans:
            a2 = max(lo, min(a, hi))
            b2 = max(lo, min(b, hi))
            if a2 <= b2:
                out.append((a2, b2))
        return out

    # --- Core ---

    def _compress(
        self,
        query: str,
        unit: CompressionUnit,
        target_ratio: float,
        query_embedding: list[float] | None,
        original_tokens: int,
    ) -> CompressionResult:
        body_lines = unit.body.splitlines()
        n = len(body_lines)
        if n == 0:
            return self._passthrough(unit, original_tokens)

        target_tokens = max(1, int(target_ratio * original_tokens))

        # --- choose scoring path -------------------------------------------------
        scored: list[tuple[_Span, float]] | None = None
        path = "lexical"
        if query_embedding is not None and self._has_sub_chunks():
            scored = self._score_sub_chunks(unit, body_lines, query_embedding)
            if scored:
                path = "sub_chunk"
        if scored is None:
            scored = self._score_lexical(query, body_lines)
        self._log_path_once(path)
        self.last_path = path

        # --- must-keep (signature line + docstring), unconditional ---------------
        selected: list[_Span] = self._must_keep_spans(unit, body_lines)

        # Greedy add top-scored spans that still fit (mirrors assembler._pack_greedy:
        # scan all, skip oversized, no break). must-keep stays unconditional.
        for span, _score in sorted(scored, key=lambda s: (-s[1], s[0][0])):
            candidate = _merge_spans([*selected, span])
            if self._span_tokens(body_lines, candidate) <= target_tokens:
                selected = candidate

        merged = _merge_spans(selected)
        # RESULT-LOSSLESS: never empty — degrade to the signature line at worst.
        if not merged:
            merged = [(0, self._signature_end_idx(body_lines))]

        # Nothing elided -> whole body kept: return it verbatim (byte-identical).
        if merged == [(0, n - 1)]:
            return self._passthrough(unit, original_tokens)

        text = self._render(body_lines, merged)
        # Clamped: line_start + body_index can escape [line_start, line_end] when a
        # parser prepends a synthetic line to body (see _clamp_spans).
        abs_spans = self._clamp_spans(
            unit, [(unit.line_start + i0, unit.line_start + i1) for (i0, i1) in merged]
        )
        return CompressionResult(
            text=text,
            token_count=self._count(text),  # ALWAYS recomputed
            original_token_count=original_tokens,
            kept_spans=abs_spans,
            provider="extractive",
        )

    # --- Sub-chunk (cosine) path ---

    def _score_sub_chunks(
        self,
        unit: CompressionUnit,
        body_lines: list[str],
        query_embedding: list[float],
    ) -> list[tuple[_Span, float]] | None:
        """Score each stored sub-chunk vector by cosine vs the query embedding."""
        getter = getattr(self._db, "get_sub_chunks_for_symbol", None)
        if getter is None:
            return None
        subs = getter(unit.symbol_id)
        if not subs:
            return None

        ids = [s.id for s in subs if getattr(s, "id", None) is not None]
        vectors = self._fetch_sub_chunk_vectors(ids)
        if not vectors:
            return None

        q = np.asarray(query_embedding, dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return None

        n = len(body_lines)
        scored: list[tuple[_Span, float]] = []
        for sub in subs:
            sub_id = getattr(sub, "id", None)
            if sub_id is None:
                continue
            vec = vectors.get(int(sub_id))
            if vec is None:
                continue
            span = self._sub_to_span(sub, unit, n)
            if span is None:
                continue
            scored.append((span, _cosine(q, q_norm, vec)))
        return scored or None

    def _sub_to_span(self, sub: object, unit: CompressionUnit, n: int) -> _Span | None:
        """Convert an (absolute) sub-chunk line range to a clamped body span."""
        i0 = max(0, int(sub.line_start) - unit.line_start)  # type: ignore[attr-defined]
        i1 = min(n - 1, int(sub.line_end) - unit.line_start)  # type: ignore[attr-defined]
        if i1 < i0:
            return None
        return (i0, i1)

    def _fetch_sub_chunk_vectors(self, sub_chunk_ids: list[int]) -> dict[int, np.ndarray]:
        """Read stored sub-chunk vectors from ``chunk_embeddings`` (offset path).

        Vectors and the ``sub_chunks`` table share the index DB file, so we read
        them via the db object's connection. Any failure returns what we have
        (possibly empty) -> graceful fall back to lexical.
        """
        conn = getattr(self._db, "_conn", None)
        if conn is None or not sub_chunk_ids:
            return {}
        out: dict[int, np.ndarray] = {}
        for sid in sub_chunk_ids:
            try:
                row = conn.execute(
                    "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?",
                    (sid + _SUB_CHUNK_OFFSET,),
                ).fetchone()
            except Exception:  # noqa: BLE001
                return out
            if row is None:
                continue
            blob = row[0]
            if isinstance(blob, (bytes, bytearray)):
                count = len(blob) // 4
                vec = np.asarray(struct.unpack(f"{count}f", blob), dtype=np.float64)
            else:
                vec = np.asarray(list(blob), dtype=np.float64)
            out[sid] = vec
        return out

    # --- Lexical path (zero inference, any language) ---

    def _score_lexical(self, query: str, body_lines: list[str]) -> list[tuple[_Span, float]]:
        """Split on blank lines + brace balance; score by query-token overlap."""
        query_tokens = {t.lower() for t in _WORD_RE.findall(query or "")}
        scored: list[tuple[_Span, float]] = []
        for i0, i1 in self._segments(body_lines):
            seg_text = "\n".join(body_lines[i0 : i1 + 1])
            seg_tokens = {t.lower() for t in _WORD_RE.findall(seg_text)}
            overlap = len(query_tokens & seg_tokens)
            # Density-normalised so a short, on-topic segment beats a long,
            # incidentally-matching one; +1 avoids div-by-zero on empty segs.
            score = overlap / (len(seg_tokens) + 1) ** 0.5
            scored.append(((i0, i1), score))
        return scored

    def _segments(self, body_lines: list[str]) -> list[_Span]:
        """Blank-line-delimited runs, then merged until brace-balanced."""
        raw: list[_Span] = []
        start: int | None = None
        for i, line in enumerate(body_lines):
            if line.strip() == "":
                if start is not None:
                    raw.append((start, i - 1))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            raw.append((start, len(body_lines) - 1))

        merged: list[_Span] = []
        i = 0
        while i < len(raw):
            s, e = raw[i]
            balance = self._brace_balance(body_lines, s, e)
            j = i
            while balance > 0 and j + 1 < len(raw):
                j += 1
                e = raw[j][1]
                balance += self._brace_balance(body_lines, raw[j][0], raw[j][1])
            merged.append((s, e))
            i = j + 1
        return merged

    @staticmethod
    def _brace_balance(body_lines: list[str], i0: int, i1: int) -> int:
        text = "\n".join(body_lines[i0 : i1 + 1])
        return text.count("{") - text.count("}")

    # --- Must-keep spans (signature + docstring) ---

    def _must_keep_spans(self, unit: CompressionUnit, body_lines: list[str]) -> list[_Span]:
        spans: list[_Span] = [(0, self._signature_end_idx(body_lines))]
        if unit.docstring:
            doc = _locate_text(body_lines, unit.docstring)
            if doc is not None:
                spans.append(doc)
        return _merge_spans(spans)

    @staticmethod
    def _signature_end_idx(body_lines: list[str]) -> int:
        """Last body-line index of the declaration header (0 for single-line)."""
        for i, line in enumerate(body_lines[:8]):
            if line.rstrip().endswith((":", "{")):
                return i
        return 0

    # --- Rendering ---

    def _render(self, body_lines: list[str], merged: list[_Span]) -> str:
        n = len(body_lines)
        parts: list[str] = []
        prev_end: int | None = None
        first_start = merged[0][0]
        if first_start > 0:
            parts.append(_ELISION.format(n=first_start))
        for i0, i1 in merged:
            if prev_end is not None:
                gap = i0 - prev_end - 1
                if gap > 0:
                    parts.append(_ELISION.format(n=gap))
            parts.append("\n".join(body_lines[i0 : i1 + 1]))
            prev_end = i1
        last_end = merged[-1][1]
        if last_end < n - 1:
            parts.append(_ELISION.format(n=n - 1 - last_end))
        return "\n".join(parts)

    # --- Runtime detection / helpers ---

    def _has_sub_chunks(self) -> bool:
        if self._sub_chunks_available is not None:
            return self._sub_chunks_available
        conn = getattr(self._db, "_conn", None)
        available = False
        if conn is not None:
            try:
                available = conn.execute("SELECT 1 FROM sub_chunks LIMIT 1").fetchone() is not None
            except Exception:  # noqa: BLE001
                available = False
        self._sub_chunks_available = available
        return available

    def _log_path_once(self, path: str) -> None:
        if not self._logged_path:
            logger.info("ExtractiveCompressor: using %s scoring path", path)
            self._logged_path = True

    def _count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _span_tokens(self, body_lines: list[str], spans: list[_Span]) -> int:
        if not spans:
            return 0
        merged = _merge_spans(spans)
        text = "\n".join("\n".join(body_lines[i0 : i1 + 1]) for i0, i1 in merged)
        return self._count(text)


# --- Module-level helpers ---


def _cosine(q: np.ndarray, q_norm: float, vec: np.ndarray) -> float:
    if vec.shape != q.shape:
        return 0.0
    v_norm = float(np.linalg.norm(vec))
    if v_norm == 0.0:
        return 0.0
    return float(np.dot(q, vec) / (q_norm * v_norm))


def _merge_spans(spans: list[_Span]) -> list[_Span]:
    """Sort and merge overlapping/adjacent inclusive spans."""
    if not spans:
        return []
    ordered = sorted(spans)
    out: list[_Span] = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = out[-1]
        if s <= le + 1:  # overlapping or adjacent
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def _locate_text(body_lines: list[str], text: str) -> _Span | None:
    """Smallest contiguous body span whose lines cover ``text``.

    Matches by stripped-line membership (each direction) so it is robust to
    quote/indentation differences between the stored docstring and the verbatim
    body lines. Returns None when nothing matches.
    """
    wanted = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not wanted:
        return None
    hits: list[int] = []
    for i, line in enumerate(body_lines):
        stripped = line.strip()
        if not stripped:
            continue
        for w in wanted:
            if stripped == w or stripped in w or w in stripped:
                hits.append(i)
                break
    if not hits:
        return None
    return (min(hits), max(hits))
