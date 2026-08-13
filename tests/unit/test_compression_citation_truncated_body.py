"""
ADVERSARIAL probe D: body SHORTER than the declared span (the common real case).

35 extractor call sites cap the stored body (``body=self._txt(node, src)[:2000]``,
``[:500]``, ``[:1000]``) while ``line_start``/``line_end`` keep the full AST node
span. Module symbols are worse: ``body = module_doc`` with
``line_end = root.end_point[0] + 1`` (EOF).

ExtractiveCompressor._passthrough hardcodes
``kept_spans=[(unit.line_start, unit.line_end)]`` regardless of how many lines the
body actually has. format_compressed_blocks then filters spans against
``symbol.line_start + len(body_lines) - 1`` -> the span is rejected -> the
fallback emits ONE header over the declared envelope.

So: does a truncated body plus the documented graceful-degradation passthrough
render a header claiming lines the text lacks?
"""

from __future__ import annotations

import re
from datetime import datetime

import tiktoken

from trelix.compression import CompressionUnit, ExtractiveCompressor
from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol, SymbolKind
from trelix.retrieval.assembler import ContextAssembler
from trelix.retrieval.context_compression import format_compressed_blocks

_ENC = tiktoken.get_encoding("cl100k_base")
_HEADER_RE = re.compile(r"^\[Lines (\d+)-(\d+)\] (\S+)$")

REL_PATH = "src/pack/service.py"
FILE = IndexedFile(
    path=f"/repo/{REL_PATH}",
    rel_path=REL_PATH,
    language=Language.PYTHON,
    hash="sha-1",
    size_bytes=90_000,
    id=1,
    indexed_at=datetime(2024, 1, 1),
)

LINE_START = 100
DECLARED_LINE_END = 399  # the real AST span: 300 source lines


class _NoSubChunkDB:
    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int) -> list[object]:  # noqa: ARG002
        return []


class _FailingScorer(ExtractiveCompressor):
    """
    A real internal failure (db/numpy/scoring error) inside _compress.

    compress() is NOT overridden, so this exercises the SHIPPED
    graceful-degradation handler (extractive.py:79-85) -> _passthrough.
    """

    def _score_lexical(self, query, body_lines):  # type: ignore[no-untyped-def]
        raise RuntimeError("scoring backend unavailable")


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


def _headers(text: str) -> list[tuple[int, int, str]]:
    return [
        (int(m.group(1)), int(m.group(2)), m.group(3))
        for m in (_HEADER_RE.match(ln) for ln in text.splitlines())
        if m
    ]


def _truncated_body() -> str:
    """What an extractor stores after ``[:2000]`` — a PREFIX of a 300-line symbol."""
    lines = [
        "def rebuild_dispatch_index(self, session, retries, stage):",
        '    """Rebuild the dispatch index for the session."""',
    ]
    for b in range(19):
        lines.append(f"    payload_{b} = build_payload(session, stage={b})")
        lines.append(f"    dispatched_{b} = dispatch(payload_{b}, timeout={b + 1})")
    return "\n".join(lines)  # 40 lines, NOT 300


def _result() -> SearchResult:
    body = _truncated_body()
    symbol = Symbol(
        file_id=1,
        name="rebuild_dispatch_index",
        qualified_name="Service.rebuild_dispatch_index",
        kind=SymbolKind.METHOD,
        line_start=LINE_START,
        line_end=DECLARED_LINE_END,
        signature="def rebuild_dispatch_index(self, session, retries, stage)",
        body=body,
        docstring="Rebuild the dispatch index for the session.",
        id=21,
    )
    # chunk_text is what the chunker builds (path + signature + context summary +
    # body), so chunk.token_count is strictly LARGER than tiktoken(body).
    chunk_text = f"# {REL_PATH}\n# {symbol.signature}\n{body}\n" + "# trailing context\n" * 40
    return SearchResult(
        chunk=Chunk(symbol_id=21, chunk_text=chunk_text, token_count=_tok(chunk_text), id=21),
        symbol=symbol,
        file=FILE,
        score=0.9,
        rank=1,
        source="vector",
    )


def _unit(result: SearchResult) -> CompressionUnit:
    s = result.symbol
    return CompressionUnit(
        symbol_id=result.chunk.symbol_id,
        body=s.body,
        signature=s.signature,
        docstring=s.docstring,
        line_start=s.line_start,
        line_end=s.line_end,
        qualified_name=s.qualified_name,
    )


class TestTruncatedBodyPassthrough:
    def test_premise_body_is_shorter_than_the_declared_span(self) -> None:
        result = _result()
        body_lines = len(result.symbol.body.splitlines())
        span_lines = DECLARED_LINE_END - LINE_START + 1
        assert body_lines < span_lines, (body_lines, span_lines)

    def test_premise_passthrough_is_smaller_than_the_chunk(self) -> None:
        """So _compress_to_fit accepts it and it lands in the `compressed` dict."""
        result = _result()
        assert _tok(result.symbol.body) < result.chunk.token_count

    def test_passthrough_kept_spans_exceed_the_body_it_returns(self) -> None:
        result = _result()
        comp = _FailingScorer(db=_NoSubChunkDB(), embedder=None)
        res = comp.compress("dispatch payload stage", _unit(result), target_ratio=0.45)
        body_lines = len(result.symbol.body.splitlines())
        claimed = sum(b - a + 1 for a, b in res.kept_spans)
        assert claimed == body_lines, (
            f"kept_spans claim {claimed} lines but the returned text has {body_lines}: "
            f"{res.kept_spans}"
        )

    def test_fallback_header_does_not_overclaim(self) -> None:
        result = _result()
        comp = _FailingScorer(db=_NoSubChunkDB(), embedder=None)
        res = comp.compress("dispatch payload stage", _unit(result), target_ratio=0.45)
        rendered = format_compressed_blocks(result, res)
        hdrs = _headers(rendered)
        assert hdrs, rendered
        shown = len(res.text.splitlines())
        for start, end, _n in hdrs:
            assert end - start + 1 == shown, (
                f"[Lines {start}-{end}] claims {end - start + 1} lines but shows {shown}"
            )

    def test_end_to_end_assembler_header_matches_rendered_line_count(self) -> None:
        """The whole pipeline: assemble() -> context_text header vs its own body."""
        big = _result()
        # A first result that consumes most of the budget so `big` misses wave 1
        # and is re-offered to wave 2 compressed.
        filler_body = "\n".join(f"    filler_line_{i} = compute({i})" for i in range(60))
        filler = SearchResult(
            chunk=Chunk(symbol_id=1, chunk_text=filler_body, token_count=_tok(filler_body), id=1),
            symbol=Symbol(
                file_id=1,
                name="filler",
                qualified_name="Service.filler",
                kind=SymbolKind.METHOD,
                line_start=1,
                line_end=60,
                signature="def filler()",
                body=filler_body,
                docstring=None,
                id=1,
            ),
            file=FILE,
            score=0.99,
            rank=1,
            source="vector",
        )
        budget = filler.chunk.token_count + _tok(big.symbol.body) + 5
        ctx = ContextAssembler(
            token_budget=budget,
            compressor=_FailingScorer(db=_NoSubChunkDB(), embedder=None),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble("dispatch payload stage", [filler, big])
        assert len(ctx.results) == 2, f"premise: wave 2 kept the big result: {ctx.results}"

        text = ctx.context_text
        hdrs = [h for h in _headers(text) if h[2] == big.symbol.qualified_name]
        assert hdrs, text
        lines = text.splitlines()
        for start, end, name in hdrs:
            i = lines.index(f"[Lines {start}-{end}] {name}")
            shown = 0
            for ln in lines[i + 1 :]:
                if _HEADER_RE.match(ln) or ln.startswith("=== "):
                    break
                if ln.strip():
                    shown += 1
            assert end - start + 1 == shown, (
                f"[Lines {start}-{end}] claims {end - start + 1} lines but shows {shown}"
            )
