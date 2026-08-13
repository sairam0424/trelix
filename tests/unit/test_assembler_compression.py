"""
Unit tests for compression-aware context assembly.

Three things are load-bearing here and each gets a dedicated gate:

1. DISABLED == byte-identical. Compression is additive and default-OFF, so with
   no compressor (or ratio 1.0) the assembled context must match the
   pre-compression output byte for byte — and the compressor must not even be
   consulted.
2. ENABLED is result-lossless. Compression may SHRINK a body; it may never DROP
   a SearchResult the uncompressed pack would have kept, and the total must stay
   inside the budget. That is what keeps Recall/MRR/nDCG unchanged.
3. Citation fidelity. A partially-kept body renders one truthful
   ``[Lines a-b]`` block PER kept span plus an explicit elision marker — never
   one header claiming lines the text no longer contains.

Everything runs headless: the extractive compressor's zero-inference lexical
path is used (no sub_chunks table, no query embedding), so no test here can make
an embedding, API, or network call.
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

_ENC = tiktoken.get_encoding("cl100k_base")

QUERY = "how does dispatch build the payload for stage 3"

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

_HEADER_RE = re.compile(r"^\[Lines (\d+)-(\d+)\] (\S+)$")
_ELISION_RE = re.compile(r"^# \.\.\. (\d+) lines elided \.\.\.$")


# ---------------------------------------------------------------------------
# Helpers — a realistic multi-block Python body so there is something to elide
# ---------------------------------------------------------------------------


def _body(name: str, blocks: int = 8) -> str:
    """Signature + one-line docstring + `blocks` blank-line-delimited stanzas."""
    lines = [
        f"def {name}(request, session, retries):",
        f'    """Handle {name} for the request."""',
    ]
    for b in range(blocks):
        lines.append("")
        lines.append(f"    # step {b}: prepare the payload for stage {b}")
        lines.append(f"    payload_{b} = build_payload(request, stage={b})")
        lines.append(f"    validated_{b} = validate(payload_{b}, session, retries)")
        lines.append(f"    dispatched_{b} = dispatch(validated_{b}, timeout={b + 1})")
    lines.append("")
    lines.append("    return aggregate(dispatched_0)")
    return "\n".join(lines)


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


def _make_result(
    *,
    sym_id: int,
    name: str,
    line_start: int,
    score: float,
    rank: int,
    file: IndexedFile = FILE,
    blocks: int = 8,
) -> SearchResult:
    """A result whose chunk_text IS its body, so token accounting is exact."""
    body = _body(name, blocks=blocks)
    line_end = line_start + len(body.splitlines()) - 1
    symbol = Symbol(
        file_id=file.id or 1,
        name=name,
        qualified_name=f"Service.{name}",
        kind=SymbolKind.FUNCTION,
        line_start=line_start,
        line_end=line_end,
        signature=f"def {name}(request, session, retries)",
        body=body,
        docstring=f"Handle {name} for the request.",
        id=sym_id,
    )
    return SearchResult(
        chunk=Chunk(symbol_id=sym_id, chunk_text=body, token_count=_tok(body), id=sym_id),
        symbol=symbol,
        file=file,
        score=score,
        rank=rank,
        source="vector",
    )


class _NoSubChunkDB:
    """An index with no sub_chunks table — forces the lexical (zero-inference) path."""

    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int) -> list[object]:  # noqa: ARG002
        return []


class _SpyCompressor(ExtractiveCompressor):
    """Counts compress() calls so 'disabled means untouched' is provable."""

    def __init__(self) -> None:
        super().__init__(db=_NoSubChunkDB(), embedder=None)
        self.calls = 0

    def compress(self, query, unit, *, target_ratio, query_embedding=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().compress(
            query, unit, target_ratio=target_ratio, query_embedding=query_embedding
        )


class _ExplodingCompressor(ExtractiveCompressor):
    """Every compress() raises — exercises the graceful-degradation contract."""

    def __init__(self) -> None:
        super().__init__(db=_NoSubChunkDB(), embedder=None)

    def compress(self, query, unit, *, target_ratio, query_embedding=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")


def _compressor() -> ExtractiveCompressor:
    return ExtractiveCompressor(db=_NoSubChunkDB(), embedder=None)


def _parse_blocks(context_text: str) -> list[tuple[int, int, str, list[str]]]:
    """Extract every ``[Lines a-b] name`` block and the raw lines under it."""
    out: list[tuple[int, int, str, list[str]]] = []
    current: tuple[int, int, str] | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            while buffer and buffer[-1] == "":
                buffer.pop()
            out.append((current[0], current[1], current[2], list(buffer)))

    for line in context_text.splitlines():
        header = _HEADER_RE.match(line)
        if header:
            flush()
            current = (int(header.group(1)), int(header.group(2)), header.group(3))
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
# Gate 1: compression DISABLED is byte-identical
# ---------------------------------------------------------------------------


class TestDisabledIsByteIdentical:
    def _fixture(self) -> list[SearchResult]:
        return [
            _make_result(sym_id=1, name="alpha", line_start=10, score=0.9, rank=1),
            _make_result(sym_id=2, name="beta", line_start=60, score=0.8, rank=2),
        ]

    def _expected(self, results: list[SearchResult]) -> str:
        """The pre-compression format, restated independently of the implementation."""
        first, second = results
        return (
            f"=== {REL_PATH} ===\n"
            "\n"
            f"[Lines {first.symbol.line_start}-{first.symbol.line_end}] "
            f"{first.symbol.qualified_name}\n{first.chunk.chunk_text}\n"
            "\n"
            f"[Lines {second.symbol.line_start}-{second.symbol.line_end}] "
            f"{second.symbol.qualified_name}\n{second.chunk.chunk_text}\n"
        )

    def test_no_compressor_matches_frozen_format(self) -> None:
        results = self._fixture()
        context = ContextAssembler(token_budget=100_000).assemble(QUERY, results)
        assert context.context_text == self._expected(results)

    def test_compressor_supplied_but_ratio_one_is_byte_identical(self) -> None:
        results = self._fixture()
        baseline = ContextAssembler(token_budget=100_000).assemble(QUERY, results)
        spy = _SpyCompressor()
        disabled = ContextAssembler(
            token_budget=100_000, compressor=spy, compression_ratio=1.0
        ).assemble(QUERY, results)
        assert disabled.context_text == baseline.context_text
        assert spy.calls == 0, "ratio 1.0 must not consult the compressor at all"

    def test_disabled_leaves_no_compression_stats(self) -> None:
        assembler = ContextAssembler(token_budget=100_000)
        assembler.assemble(QUERY, self._fixture())
        assert assembler.last_compression_stats is None

    def test_disabled_is_byte_identical_under_tight_budget(self) -> None:
        """The drop-on-overflow path must also stay untouched when disabled."""
        results = self._fixture()
        budget = results[0].chunk.token_count + 5
        baseline = ContextAssembler(token_budget=budget).assemble(QUERY, results)
        with_compressor = ContextAssembler(
            token_budget=budget, compressor=_SpyCompressor(), compression_ratio=1.0
        ).assemble(QUERY, results)
        assert len(baseline.results) == 1  # second result did not fit -> dropped
        assert with_compressor.context_text == baseline.context_text

    def test_disabled_breadth_first_is_byte_identical(self) -> None:
        results = self._fixture()
        budget = results[0].chunk.token_count + 5
        baseline = ContextAssembler(token_budget=budget).assemble(
            QUERY, results, assembly_mode="breadth_first"
        )
        with_compressor = ContextAssembler(
            token_budget=budget, compressor=_SpyCompressor(), compression_ratio=1.0
        ).assemble(QUERY, results, assembly_mode="breadth_first")
        assert with_compressor.context_text == baseline.context_text


# ---------------------------------------------------------------------------
# Gate 2: compression ENABLED is result-lossless and stays inside the budget
# ---------------------------------------------------------------------------


class TestEnabledIsResultLossless:
    def _fixture(self) -> list[SearchResult]:
        return [
            _make_result(sym_id=1, name="alpha", line_start=10, score=0.9, rank=1),
            _make_result(sym_id=2, name="beta", line_start=60, score=0.8, rank=2),
            _make_result(sym_id=3, name="gamma", line_start=110, score=0.7, rank=3),
        ]

    def _tight_budget(self, results: list[SearchResult]) -> int:
        """Room for the first two verbatim plus half of the third."""
        sizes = [r.chunk.token_count for r in results]
        return sizes[0] + sizes[1] + sizes[2] // 2

    def test_uncompressed_baseline_drops_the_third_result(self) -> None:
        """Guards the premise: without compression this fixture loses a result."""
        results = self._fixture()
        baseline = ContextAssembler(token_budget=self._tight_budget(results)).assemble(
            QUERY, results
        )
        assert len(baseline.results) == 2

    def test_no_result_is_dropped(self) -> None:
        results = self._fixture()
        context = ContextAssembler(
            token_budget=self._tight_budget(results),
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        assert len(context.results) == len(results)
        assert {r.chunk.symbol_id for r in context.results} == {1, 2, 3}

    def test_total_tokens_within_budget(self) -> None:
        results = self._fixture()
        budget = self._tight_budget(results)
        context = ContextAssembler(
            token_budget=budget,
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        assert context.total_tokens <= budget

    def test_selection_is_a_superset_of_the_uncompressed_selection(self) -> None:
        results = self._fixture()
        budget = self._tight_budget(results)
        baseline = ContextAssembler(token_budget=budget).assemble(QUERY, results)
        compressed = ContextAssembler(
            token_budget=budget,
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        baseline_ids = {r.chunk.symbol_id for r in baseline.results}
        assert baseline_ids <= {r.chunk.symbol_id for r in compressed.results}

    def test_wave1_results_are_untouched_objects(self) -> None:
        """A result that already fits keeps its verbatim text — no needless work."""
        results = self._fixture()
        context = ContextAssembler(
            token_budget=self._tight_budget(results),
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        kept = {r.chunk.symbol_id: r for r in context.results}
        assert kept[1] is results[0]
        assert kept[2] is results[1]

    def test_originals_are_never_mutated(self) -> None:
        results = self._fixture()
        before = [(r.chunk.chunk_text, r.chunk.token_count) for r in results]
        ContextAssembler(
            token_budget=self._tight_budget(results),
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        assert [(r.chunk.chunk_text, r.chunk.token_count) for r in results] == before

    def test_compressed_result_is_a_new_object_with_recomputed_tokens(self) -> None:
        results = self._fixture()
        context = ContextAssembler(
            token_budget=self._tight_budget(results),
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        shrunk = next(r for r in context.results if r.chunk.symbol_id == 3)
        assert shrunk is not results[2]
        assert shrunk.chunk.token_count < results[2].chunk.token_count
        assert shrunk.chunk.token_count == _tok(shrunk.chunk.chunk_text)
        assert shrunk.chunk.embedding is None

    def test_stats_report_the_wave_split(self) -> None:
        results = self._fixture()
        assembler = ContextAssembler(
            token_budget=self._tight_budget(results),
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        )
        assembler.assemble(QUERY, results)
        stats = assembler.last_compression_stats
        assert stats is not None
        assert stats["wave1_kept"] == 2
        assert stats["wave2_added"] == 1
        assert stats["ratio"] == 0.45
        assert stats["paths"] == ["lexical"], "no sub_chunks -> zero-inference path"

    def test_below_min_tokens_is_skipped_not_compressed(self) -> None:
        """A body too small to be worth eliding is handled exactly as today."""
        results = self._fixture()
        assembler = ContextAssembler(
            token_budget=self._tight_budget(results),
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10_000,
        )
        context = assembler.assemble(QUERY, results)
        assert len(context.results) == 2
        stats = assembler.last_compression_stats
        assert stats is not None
        assert stats["below_min_tokens"] == 1
        assert stats["wave2_added"] == 0

    def test_compressor_failure_degrades_to_uncompressed(self) -> None:
        results = self._fixture()
        budget = self._tight_budget(results)
        baseline = ContextAssembler(token_budget=budget).assemble(QUERY, results)
        context = ContextAssembler(
            token_budget=budget,
            compressor=_ExplodingCompressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        assert context.context_text == baseline.context_text

    def test_breadth_first_max_per_file_still_enforced(self) -> None:
        other = IndexedFile(
            path="/repo/src/pack/other.py",
            rel_path="src/pack/other.py",
            language=Language.PYTHON,
            hash="sha-2",
            size_bytes=4096,
            id=2,
            indexed_at=datetime(2024, 1, 1),
        )
        results = [
            _make_result(sym_id=1, name="alpha", line_start=10, score=0.90, rank=1),
            _make_result(sym_id=2, name="beta", line_start=60, score=0.85, rank=2),
            _make_result(sym_id=3, name="gamma", line_start=110, score=0.80, rank=3),
            _make_result(sym_id=4, name="delta", line_start=10, score=0.75, rank=4, file=other),
            _make_result(sym_id=5, name="epsilon", line_start=60, score=0.70, rank=5, file=other),
            _make_result(sym_id=6, name="zeta", line_start=110, score=0.65, rank=6, file=other),
        ]
        context = ContextAssembler(
            token_budget=100_000,  # deliberately huge: only the cap may limit us
            compressor=_compressor(),
            compression_ratio=0.30,
            compression_min_tokens=10,
        ).assemble(QUERY, results, assembly_mode="breadth_first")
        per_file: dict[str, int] = {}
        for r in context.results:
            per_file[r.file.rel_path] = per_file.get(r.file.rel_path, 0) + 1
        assert per_file == {REL_PATH: 2, "src/pack/other.py": 2}


# ---------------------------------------------------------------------------
# Gate 3: citation fidelity of a partial keep
# ---------------------------------------------------------------------------


class TestCitationFidelity:
    def _compressed_context(self) -> tuple[str, SearchResult]:
        results = [
            _make_result(sym_id=1, name="alpha", line_start=10, score=0.9, rank=1),
            _make_result(sym_id=2, name="beta", line_start=60, score=0.8, rank=2),
        ]
        budget = results[0].chunk.token_count + results[1].chunk.token_count // 2
        context = ContextAssembler(
            token_budget=budget,
            compressor=_compressor(),
            compression_ratio=0.45,
            compression_min_tokens=10,
        ).assemble(QUERY, results)
        assert len(context.results) == 2, "premise: the second result was kept, shrunk"
        return context.context_text, results[1]

    def test_partial_keep_renders_multiple_line_range_blocks(self) -> None:
        text, target = self._compressed_context()
        blocks = [b for b in _parse_blocks(text) if b[2] == target.symbol.qualified_name]
        assert len(blocks) >= 2, "a partially-kept body needs one header per kept span"

    def test_partial_keep_emits_an_elision_marker(self) -> None:
        text, _target = self._compressed_context()
        assert any(_ELISION_RE.match(line) for line in text.splitlines())

    def test_every_header_range_matches_the_real_source_lines(self) -> None:
        text, target = self._compressed_context()
        body_lines = target.symbol.body.splitlines()
        blocks = [b for b in _parse_blocks(text) if b[2] == target.symbol.qualified_name]
        for start, end, _name, rendered in blocks:
            assert target.symbol.line_start <= start <= end <= target.symbol.line_end
            i0 = start - target.symbol.line_start
            i1 = end - target.symbol.line_start
            assert rendered == body_lines[i0 : i1 + 1], (
                f"[Lines {start}-{end}] does not match the real body lines"
            )

    def test_no_header_claims_more_lines_than_it_shows(self) -> None:
        text, target = self._compressed_context()
        blocks = [b for b in _parse_blocks(text) if b[2] == target.symbol.qualified_name]
        for start, end, _name, rendered in blocks:
            assert len(rendered) == end - start + 1

    def test_kept_spans_are_strictly_inside_the_symbol_range(self) -> None:
        """Guard against an off-by-one that would cite a neighbouring symbol."""
        text, target = self._compressed_context()
        blocks = [b for b in _parse_blocks(text) if b[2] == target.symbol.qualified_name]
        covered = sum(end - start + 1 for start, end, _n, _r in blocks)
        assert 0 < covered < target.symbol.line_end - target.symbol.line_start + 1

    def test_signature_line_is_always_kept(self) -> None:
        text, target = self._compressed_context()
        blocks = [b for b in _parse_blocks(text) if b[2] == target.symbol.qualified_name]
        assert blocks[0][0] == target.symbol.line_start


# ---------------------------------------------------------------------------
# Fallback rendering: unmappable spans must still render, never raise
# ---------------------------------------------------------------------------


class TestRenderFallback:
    def test_out_of_range_spans_fall_back_to_a_single_clamped_block(self) -> None:
        from trelix.retrieval.context_compression import format_compressed_blocks

        result = _make_result(sym_id=1, name="alpha", line_start=10, score=0.9, rank=1)
        bogus = CompressionResult(
            text="def alpha(...): ...",
            token_count=7,
            original_token_count=400,
            kept_spans=[(9_000, 9_001)],  # cannot be mapped onto the body
            provider="extractive",
        )
        rendered = format_compressed_blocks(result, bogus)
        assert "Service.alpha" in rendered
        assert bogus.text in rendered

    def test_compression_unit_is_built_from_the_symbol_verbatim(self) -> None:
        """The unit the assembler hands the compressor must preserve symbol_id."""
        from trelix.retrieval.context_compression import _compress_to_fit

        result = _make_result(sym_id=42, name="alpha", line_start=10, score=0.9, rank=1)
        captured: list[CompressionUnit] = []

        class _Capture(ExtractiveCompressor):
            def compress(self, query, unit, *, target_ratio, query_embedding=None):  # type: ignore[no-untyped-def]
                captured.append(unit)
                return super().compress(
                    query, unit, target_ratio=target_ratio, query_embedding=query_embedding
                )

        _compress_to_fit(
            query=QUERY,
            result=result,
            compressor=_Capture(db=_NoSubChunkDB(), embedder=None),
            target_ratio=0.45,
            remaining=result.chunk.token_count // 2,
            query_embedding=None,
        )
        assert captured, "the compressor must be consulted"
        assert captured[0].symbol_id == 42
        assert captured[0].body == result.symbol.body
        assert captured[0].line_start == result.symbol.line_start


# ---------------------------------------------------------------------------
# Retriever wiring — config + per-intent ratio -> compressor, zero inference
# ---------------------------------------------------------------------------


class _StubConfig:
    """Just enough of IndexConfig for Retriever._make_compressor()."""

    def __init__(self, **retrieval_kwargs: object) -> None:
        from trelix.core.config import RetrievalConfig

        self.retrieval = RetrievalConfig(**retrieval_kwargs)  # type: ignore[arg-type]


class _StubRetriever:
    """Borrows the real methods so the wiring, not a copy of it, is tested."""

    def __init__(self, config: _StubConfig, embedder: object = None) -> None:
        self.config = config
        self.db = _NoSubChunkDB()
        self.embedder = embedder

    from trelix.retrieval.retriever import Retriever

    _make_compressor = Retriever._make_compressor
    _cached_query_embedding = Retriever._cached_query_embedding
    del Retriever


class TestRetrieverWiring:
    def test_disabled_yields_no_compressor_and_ratio_one(self) -> None:
        retriever = _StubRetriever(_StubConfig(compression_enabled=False))
        compressor, ratio = retriever._make_compressor("feature_flow")
        assert compressor is None
        assert ratio == 1.0

    def test_enabled_uses_the_per_intent_ratio(self) -> None:
        retriever = _StubRetriever(_StubConfig(compression_enabled=True))
        compressor, ratio = retriever._make_compressor("blast_radius")
        assert isinstance(compressor, ExtractiveCompressor)
        assert ratio == 0.30

    def test_opted_out_intent_disables_compression_even_when_enabled(self) -> None:
        retriever = _StubRetriever(_StubConfig(compression_enabled=True))
        compressor, ratio = retriever._make_compressor("symbol_lookup")
        assert compressor is None, "ratio 1.0 must short-circuit before construction"
        assert ratio == 1.0

    def test_unknown_intent_falls_back_to_the_config_ratio(self) -> None:
        retriever = _StubRetriever(
            _StubConfig(compression_enabled=True, compression_target_ratio=0.6)
        )
        _compressor_obj, ratio = retriever._make_compressor(None)
        assert ratio == 0.6

    def test_cached_query_embedding_is_a_peek_not_a_call(self) -> None:
        class _Embedder:
            def __init__(self) -> None:
                self._cache = {"how does it work": [0.1, 0.2]}

            def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
                raise AssertionError("assembly must never trigger an embedding call")

        retriever = _StubRetriever(_StubConfig(), embedder=_Embedder())
        assert retriever._cached_query_embedding("  How Does It Work  ") == [0.1, 0.2]

    def test_cache_miss_returns_none_for_the_lexical_fallback(self) -> None:
        class _Embedder:
            _cache: dict[str, list[float]] = {}

        retriever = _StubRetriever(_StubConfig(), embedder=_Embedder())
        assert retriever._cached_query_embedding("anything") is None

    def test_uncached_embedder_returns_none(self) -> None:
        retriever = _StubRetriever(_StubConfig(), embedder=object())
        assert retriever._cached_query_embedding("anything") is None
