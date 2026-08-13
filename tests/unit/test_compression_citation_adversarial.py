"""
ADVERSARIAL: try to FALSIFY "a compressed body never renders a header claiming
lines it lacks".

Three probes:

A. MAINLINE — a 42-120 function that keeps only 42-50 and 110-120 must render
   TWO truthful blocks + an elision marker, never one "[Lines 42-120]".
B. SPAN ESCAPE — the subrange invariant (kept_spans within
   [line_start, line_end]) is only ever exercised by fixtures that DERIVE
   line_end from len(body.splitlines()). Real extractors do not guarantee that
   equality (python.py:607 PREPENDS a synthetic "Exports: ..." line to __all__
   bodies while line_start/line_end stay the AST node's span). Probe what the
   compressor + renderer do when body has MORE lines than the declared span.
C. FALLBACK HEADER — format_compressed_blocks' unmappable-spans fallback emits
   ONE header over cresult.text. Probe whether that header can claim lines the
   text lacks (including an inverted range).

Zero inference: lexical path only (no sub_chunks, no query embedding).
"""

from __future__ import annotations

import re
from datetime import datetime

import tiktoken

from trelix.compression import CompressionResult, CompressionUnit, ExtractiveCompressor
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.assembler import ContextAssembler
from trelix.retrieval.context_compression import format_compressed_blocks

_ENC = tiktoken.get_encoding("cl100k_base")
_HEADER_RE = re.compile(r"^\[Lines (\d+)-(\d+)\] (\S+)$")
_ELISION_RE = re.compile(r"^# \.\.\. (\d+) lines elided \.\.\.$")

REL_PATH = "src/pack/service.py"
FILE = IndexedFile(
    path=f"/repo/{REL_PATH}",
    rel_path=REL_PATH,
    language=Language.PYTHON,
    hash="sha-1",
    size_bytes=4096,
    id=1,
    indexed_at=datetime(2024, 1, 1),
)


class _NoSubChunkDB:
    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int) -> list[object]:  # noqa: ARG002
        return []


def _compressor() -> ExtractiveCompressor:
    return ExtractiveCompressor(db=_NoSubChunkDB(), embedder=None)


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


def _headers(text: str) -> list[tuple[int, int, str]]:
    out = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return out


def _blocks(text: str) -> list[tuple[int, int, str, list[str]]]:
    out: list[tuple[int, int, str, list[str]]] = []
    current: tuple[int, int, str] | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            while buffer and buffer[-1] == "":
                buffer.pop()
            out.append((current[0], current[1], current[2], list(buffer)))

    for line in text.splitlines():
        h = _HEADER_RE.match(line)
        if h:
            flush()
            current = (int(h.group(1)), int(h.group(2)), h.group(3))
            buffer = []
            continue
        if _ELISION_RE.match(line) or line.startswith("=== "):
            flush()
            current = None
            buffer = []
            continue
        if current is not None:
            buffer.append(line)
    flush()
    return out


# ---------------------------------------------------------------------------
# PROBE A — mainline: 42-120 keeping 42-50 + 110-120
# ---------------------------------------------------------------------------


def _long_function(line_start: int = 42, stanzas: int = 26) -> SearchResult:
    """A function whose declared span EQUALS its body line count (the good case)."""
    lines = [
        "def authenticate_user(self, username, password, session, retries):",
        '    """Authenticate and return the user."""',
    ]
    for b in range(stanzas):
        lines.append(f"    step_{b} = compute_intermediate_value_{b}(username, retries)")
        lines.append(f"    audit_log_{b} = record_audit_entry(step_{b}, session)")
        lines.append("")
    lines.append("    return build_session_token(username, password, session)")
    body = "\n".join(lines)
    line_end = line_start + len(body.splitlines()) - 1
    symbol = Symbol(
        file_id=1,
        name="authenticate_user",
        qualified_name="LoginView.authenticate_user",
        kind=SymbolKind.FUNCTION,
        line_start=line_start,
        line_end=line_end,
        signature="def authenticate_user(self, username, password, session, retries)",
        body=body,
        docstring="Authenticate and return the user.",
        id=7,
    )
    return SearchResult(
        chunk=Chunk(symbol_id=7, chunk_text=body, token_count=_tok(body), id=7),
        symbol=symbol,
        file=FILE,
        score=0.9,
        rank=1,
        source="vector",
    )


class TestMainlineSplitBlocks:
    """The claim, on the path the extractive compressor actually takes."""

    def _rendered(self) -> tuple[str, SearchResult]:
        target = _long_function()
        # Force a partial keep: hand the renderer the exact 42-50 / 110-120 split
        # the adversarial question asks about.
        cresult = CompressionResult(
            text="unused-by-the-span-path",
            token_count=40,
            original_token_count=target.chunk.token_count,
            kept_spans=[(42, 50), (110, 120)],
            provider="extractive",
        )
        return format_compressed_blocks(target, cresult), target

    def test_two_blocks_not_one_envelope_header(self) -> None:
        text, target = self._rendered()
        hdrs = _headers(text)
        assert (42, 120, "LoginView.authenticate_user") not in hdrs, (
            f"one header claimed the whole 42-120 envelope: {hdrs}"
        )
        assert [(a, b) for a, b, _ in hdrs] == [(42, 50), (110, 120)], hdrs

    def test_elision_marker_between_and_around_blocks(self) -> None:
        text, _ = self._rendered()
        gaps = [int(m.group(1)) for m in map(_ELISION_RE.match, text.splitlines()) if m]
        # 51..109 = 59 interior lines elided; tail 121..line_end.
        assert 59 in gaps, gaps

    def test_each_header_matches_the_lines_it_shows(self) -> None:
        text, target = self._rendered()
        body_lines = target.symbol.body.splitlines()
        for start, end, _name, rendered in _blocks(text):
            i0 = start - target.symbol.line_start
            i1 = end - target.symbol.line_start
            assert rendered == body_lines[i0 : i1 + 1], f"[Lines {start}-{end}] lies"
            assert len(rendered) == end - start + 1

    def test_end_to_end_through_the_assembler(self) -> None:
        """Same claim, but via ContextAssembler + the real ExtractiveCompressor."""
        first = _long_function(line_start=42)
        second = _long_function(line_start=400)
        second.symbol.qualified_name = "LoginView.refresh_token"
        second.symbol.id = 8
        second.chunk.symbol_id = 8
        budget = first.chunk.token_count + second.chunk.token_count // 2
        ctx = ContextAssembler(
            token_budget=budget,
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble("audit log entry for the session token", [first, second])
        assert len(ctx.results) == 2, "premise: second result kept, shrunk"
        target = second
        body_lines = target.symbol.body.splitlines()
        blocks = [b for b in _blocks(ctx.context_text) if b[2] == target.symbol.qualified_name]
        # NOTE: >=2 blocks is NOT required — the floor rung (signature+docstring)
        # is one contiguous span, so a single block is legitimate there. What IS
        # required is that every header tells the truth about what it shows.
        assert blocks, ctx.context_text
        for start, end, _n, rendered in blocks:
            assert target.symbol.line_start <= start <= end <= target.symbol.line_end, (
                f"[Lines {start}-{end}] escapes "
                f"[{target.symbol.line_start},{target.symbol.line_end}]"
            )
            i0 = start - target.symbol.line_start
            assert rendered == body_lines[i0 : i0 + len(rendered)]
            assert len(rendered) == end - start + 1


# ---------------------------------------------------------------------------
# PROBE B — body has MORE lines than the declared span (python.py:607 shape)
# ---------------------------------------------------------------------------


def _all_dunder_symbol() -> tuple[CompressionUnit, int, int]:
    """
    Mirrors trelix/indexing/parser/extractors/python.py:607 exactly:

        body = f"Exports: {', '.join(exported)}\\n{body}"

    line_start/line_end stay the AST node's span, so body gains ONE line the
    declared span does not cover.
    """
    exported = [f"dispatch_payload_{i}" for i in range(14)]
    source_lines = ["__all__ = ["]
    for i, name in enumerate(exported):
        source_lines.append(f'    "{name}",')
        if i % 4 == 3:
            source_lines.append("")  # blank line -> lexical segment boundary
    source_lines.append("]")
    line_start = 100
    line_end = line_start + len(source_lines) - 1  # the TRUE AST span
    body = f"Exports: {', '.join(exported)}\n" + "\n".join(source_lines)
    unit = CompressionUnit(
        symbol_id=11,
        body=body,
        signature=body.split("\n")[0][:200],
        docstring=None,
        line_start=line_start,
        line_end=line_end,
        qualified_name="__all__",
    )
    return unit, line_start, line_end


class TestBodyLongerThanDeclaredSpan:
    def test_body_really_is_longer_than_the_span(self) -> None:
        unit, line_start, line_end = _all_dunder_symbol()
        span_lines = line_end - line_start + 1
        body_lines = len(unit.body.splitlines())
        assert body_lines == span_lines + 1, (body_lines, span_lines)

    def test_kept_spans_stay_inside_the_declared_span(self) -> None:
        """The invariant CompressionResult documents: every span is a subrange."""
        unit, line_start, line_end = _all_dunder_symbol()
        comp = _compressor()
        violations: list[tuple[float, tuple[int, int]]] = []
        for ratio in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            res = comp.compress("dispatch_payload_13", unit, target_ratio=ratio)
            for a, b in res.kept_spans:
                if not (line_start <= a <= b <= line_end):
                    violations.append((ratio, (a, b)))
        assert not violations, f"kept_spans escaped [{line_start},{line_end}]: {violations}"

    def test_rendered_header_never_claims_a_line_outside_the_symbol(self) -> None:
        """End-to-end: does a [Lines a-b] header cite a line past line_end?"""
        unit, line_start, line_end = _all_dunder_symbol()
        symbol = Symbol(
            file_id=1,
            name="__all__",
            qualified_name="__all__",
            kind=SymbolKind.CONSTANT,
            line_start=line_start,
            line_end=line_end,
            signature=unit.signature,
            body=unit.body,
            docstring=None,
            id=11,
        )
        result = SearchResult(
            chunk=Chunk(symbol_id=11, chunk_text=unit.body, token_count=_tok(unit.body), id=11),
            symbol=symbol,
            file=FILE,
            score=0.9,
            rank=1,
            source="vector",
        )
        comp = _compressor()
        offenders: list[tuple[float, list[tuple[int, int, str]]]] = []
        for ratio in (0.2, 0.4, 0.6, 0.8):
            res = comp.compress("dispatch_payload_13", unit, target_ratio=ratio)
            rendered = format_compressed_blocks(result, res)
            bad = [(a, b, n) for a, b, n in _headers(rendered) if a < line_start or b > line_end]
            if bad:
                offenders.append((ratio, bad))
        assert not offenders, f"header cited lines outside [{line_start},{line_end}]: {offenders}"


# ---------------------------------------------------------------------------
# PROBE C — the unmappable-spans fallback header
# ---------------------------------------------------------------------------


class TestFallbackHeaderTruthfulness:
    def test_fallback_header_is_a_valid_non_inverted_range(self) -> None:
        target = _long_function()
        bogus = CompressionResult(
            text="def authenticate_user(self, username, password, session, retries):",
            token_count=12,
            original_token_count=target.chunk.token_count,
            kept_spans=[(9_000, 9_001)],  # unmappable -> fallback branch
            provider="extractive",
        )
        rendered = format_compressed_blocks(target, bogus)
        hdrs = _headers(rendered)
        assert hdrs, rendered
        for start, end, _n in hdrs:
            assert start <= end, f"inverted range [Lines {start}-{end}] in: {rendered!r}"
            assert target.symbol.line_start <= start <= end <= target.symbol.line_end, (
                f"[Lines {start}-{end}] escapes the symbol span"
            )

    def test_fallback_header_does_not_claim_lines_the_text_lacks(self) -> None:
        """A partial text under ONE header must not claim more lines than it shows."""
        target = _long_function()
        partial = CompressionResult(
            text=(
                "def authenticate_user(self, username, password, session, retries):\n"
                "# ... 59 lines elided ...\n"
                "    return build_session_token(username, password, session)"
            ),
            token_count=30,
            original_token_count=target.chunk.token_count,
            kept_spans=[(9_000, 9_001)],  # unmappable -> fallback branch
            provider="extractive",
        )
        rendered = format_compressed_blocks(target, partial)
        for start, end, _n, body in _blocks(rendered):
            claimed = end - start + 1
            real = len([ln for ln in body if not _ELISION_RE.match(ln)])
            assert claimed == real, (
                f"[Lines {start}-{end}] claims {claimed} lines but shows {real}: {rendered!r}"
            )
