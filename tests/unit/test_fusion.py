"""
Unit tests for trelix.retrieval.fusion (RRF fusion).

All tests use in-memory mock SearchResult objects — no DB required.
"""

from __future__ import annotations

from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.fusion import reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(symbol_id: int, score: float, source: str = "vector") -> SearchResult:
    """Build a minimal SearchResult with the given symbol_id and score."""
    chunk = Chunk(symbol_id=symbol_id, chunk_text=f"body_{symbol_id}", token_count=10)
    symbol = Symbol(
        id=symbol_id,
        file_id=1,
        name=f"sym_{symbol_id}",
        qualified_name=f"mod.sym_{symbol_id}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=5,
        signature=f"def sym_{symbol_id}()",
        body=f"def sym_{symbol_id}(): pass",
    )
    file = IndexedFile(
        id=1,
        path="/repo/mod.py",
        rel_path="mod.py",
        language=Language.PYTHON,
        hash="abc",
        size_bytes=100,
    )
    return SearchResult(chunk=chunk, symbol=symbol, file=file, score=score, rank=1, source=source)


# ---------------------------------------------------------------------------
# Empty and trivial inputs
# ---------------------------------------------------------------------------


class TestRRFEdgeCases:
    def test_empty_input_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_single_empty_list_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([[]]) == []

    def test_single_list_single_item(self) -> None:
        r = _make_result(symbol_id=1, score=0.9)
        result = reciprocal_rank_fusion([[r]])
        assert len(result) == 1
        assert result[0].chunk.symbol_id == 1

    def test_single_list_preserves_order(self) -> None:
        """A single list: items ranked 1,2,3 → RRF scores 1/61 > 1/62 > 1/63."""
        results = [_make_result(i, score=1.0 - i * 0.1) for i in range(1, 4)]
        fused = reciprocal_rank_fusion([results])
        ids = [r.chunk.symbol_id for r in fused]
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestRRFDeduplication:
    def test_same_symbol_in_both_lists_deduplicated(self) -> None:
        """Symbol appearing in both vector and BM25 lists should appear only once."""
        list_a = [_make_result(1, 0.9, "vector"), _make_result(2, 0.8, "vector")]
        list_b = [_make_result(1, 0.7, "bm25"), _make_result(3, 0.6, "bm25")]
        fused = reciprocal_rank_fusion([list_a, list_b])

        ids = [r.chunk.symbol_id for r in fused]
        assert len(ids) == len(set(ids)), "Duplicate symbol_ids found in fused output"
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids
        assert len(ids) == 3

    def test_three_lists_with_overlapping_symbols(self) -> None:
        """Symbol_id 42 appears in all three lists — must appear once in output."""
        shared = _make_result(42, 0.95)
        list_a = [shared, _make_result(1, 0.8)]
        list_b = [shared, _make_result(2, 0.7)]
        list_c = [shared, _make_result(3, 0.6)]
        fused = reciprocal_rank_fusion([list_a, list_b, list_c])

        ids = [r.chunk.symbol_id for r in fused]
        assert ids.count(42) == 1


# ---------------------------------------------------------------------------
# RRF score correctness — ranked highest from multiple lists
# ---------------------------------------------------------------------------


class TestRRFScoring:
    def test_top_item_in_all_lists_gets_highest_fused_score(self) -> None:
        """
        Symbol 1 ranked 1st in both lists → RRF = 1/61 + 1/61 = 2/61 ≈ 0.03279
        Symbol 2 ranked 2nd in list_a only → RRF = 1/62 ≈ 0.01613
        Symbol 1 must have strictly higher fused score.
        """
        list_a = [_make_result(1, 0.9), _make_result(2, 0.8)]
        list_b = [_make_result(1, 0.7), _make_result(3, 0.6)]
        fused = reciprocal_rank_fusion([list_a, list_b])

        # Symbol 1 should be first
        assert fused[0].chunk.symbol_id == 1

    def test_rrf_formula_k60_applied_correctly(self) -> None:
        """
        Manually verify the RRF formula for a 2-item, 2-list scenario:
          symbol_a: rank 1 in list_a, rank 2 in list_b → score = 1/61 + 1/62
          symbol_b: rank 2 in list_a, rank 1 in list_b → score = 1/62 + 1/61
          Both equal → order can be either, but scores must match formula exactly.
        """
        k = 60
        list_a = [_make_result(10, 0.9), _make_result(20, 0.5)]
        list_b = [_make_result(20, 0.9), _make_result(10, 0.5)]
        fused = reciprocal_rank_fusion([list_a, list_b], k=k)

        fused_map = {r.chunk.symbol_id: r.score for r in fused}

        expected_10 = 1.0 / (k + 1) + 1.0 / (k + 2)
        expected_20 = 1.0 / (k + 2) + 1.0 / (k + 1)

        assert abs(fused_map[10] - expected_10) < 1e-10
        assert abs(fused_map[20] - expected_20) < 1e-10

    def test_rrf_formula_custom_k(self) -> None:
        """k parameter is respected — lower k amplifies rank differences."""
        k = 1
        result = _make_result(99, 1.0)
        fused = reciprocal_rank_fusion([[result]], k=k)
        # rank=1, k=1 → score = 1/(1+1) = 0.5
        assert abs(fused[0].score - 0.5) < 1e-10

    def test_scores_are_positive(self) -> None:
        """All fused scores must be strictly positive."""
        lists = [
            [_make_result(i, 1.0 / i) for i in range(1, 6)],
            [_make_result(i, 0.5 / i) for i in range(3, 8)],
        ]
        fused = reciprocal_rank_fusion(lists)
        for r in fused:
            assert r.score > 0.0

    def test_fused_scores_are_monotonically_decreasing(self) -> None:
        """Output list must be sorted best-first (scores descending or equal)."""
        list_a = [_make_result(i, 1.0 / i) for i in range(1, 6)]
        list_b = [_make_result(i, 0.9 / i) for i in range(2, 7)]
        fused = reciprocal_rank_fusion([list_a, list_b])

        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score, (
                f"Score at rank {i} ({fused[i].score}) < rank {i + 1} ({fused[i + 1].score})"
            )

    def test_rank_field_is_1_indexed_sequential(self) -> None:
        """rank fields on fused results must be 1, 2, 3, ..."""
        list_a = [_make_result(i, 1.0) for i in range(1, 4)]
        list_b = [_make_result(i, 0.9) for i in range(2, 5)]
        fused = reciprocal_rank_fusion([list_a, list_b])

        for expected_rank, result in enumerate(fused, start=1):
            assert result.rank == expected_rank

    def test_item_only_in_one_list_still_included(self) -> None:
        """A symbol appearing in only one of many lists must still appear in output."""
        list_a = [_make_result(1, 0.9), _make_result(2, 0.8)]
        list_b = [_make_result(3, 0.9), _make_result(4, 0.8)]
        fused = reciprocal_rank_fusion([list_a, list_b])

        ids = {r.chunk.symbol_id for r in fused}
        assert ids == {1, 2, 3, 4}

    def test_symbol_in_more_lists_beats_same_rank_in_one_list(self) -> None:
        """
        Symbol A ranked #1 in all 3 lists beats Symbol B ranked #1 in only one list.
        RRF(A) = 3 * 1/61 ≈ 0.0492
        RRF(B) = 1/61 ≈ 0.0164
        """
        sym_a = _make_result(100, 0.9)
        sym_b = _make_result(200, 0.85)
        # A appears 1st in all three; B appears 1st in one but only A appears in others
        list_a = [sym_a, sym_b]
        list_b = [sym_a, _make_result(300, 0.7)]
        list_c = [sym_a, _make_result(400, 0.6)]
        fused = reciprocal_rank_fusion([list_a, list_b, list_c])
        assert fused[0].chunk.symbol_id == 100
