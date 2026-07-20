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


# ---------------------------------------------------------------------------
# File-type weight multiplier
# ---------------------------------------------------------------------------


def _make_result_lang(
    symbol_id: int,
    score: float,
    language: Language,
    source: str = "bm25",
    file_id: int | None = None,
) -> SearchResult:
    """Build a SearchResult with a specific language — for weighting tests."""
    chunk = Chunk(symbol_id=symbol_id, chunk_text=f"body_{symbol_id}", token_count=10)
    symbol = Symbol(
        id=symbol_id,
        file_id=file_id or symbol_id,
        name=f"sym_{symbol_id}",
        qualified_name=f"mod.sym_{symbol_id}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=5,
        signature=f"def sym_{symbol_id}()",
        body=f"def sym_{symbol_id}(): pass",
    )
    file = IndexedFile(
        id=file_id or symbol_id,
        path=f"/repo/file_{symbol_id}.py",
        rel_path=f"file_{symbol_id}.py",
        language=language,
        hash="abc",
        size_bytes=100,
    )
    return SearchResult(chunk=chunk, symbol=symbol, file=file, score=score, rank=1, source=source)


class TestFileTypeWeighting:
    def test_weights_none_produces_identical_output_to_unweighted(self) -> None:
        """weights=None must give bit-for-bit identical scores to calling without weights."""
        k = 60
        list_a = [_make_result(1, 0.9), _make_result(2, 0.8), _make_result(3, 0.7)]
        list_b = [_make_result(2, 0.85), _make_result(1, 0.75), _make_result(4, 0.6)]

        fused_unweighted = reciprocal_rank_fusion([list_a, list_b], k=k)
        fused_none = reciprocal_rank_fusion([list_a, list_b], k=k, weights=None)

        assert len(fused_unweighted) == len(fused_none)
        for a, b in zip(fused_unweighted, fused_none):
            assert a.chunk.symbol_id == b.chunk.symbol_id
            assert a.score == b.score  # exact float equality — no arithmetic difference

    def test_empty_weights_dict_produces_identical_output(self) -> None:
        """weights={} (empty dict) is falsy → multiplier step skipped, same as weights=None."""
        k = 60
        results = [_make_result(i, 1.0 / i) for i in range(1, 5)]
        fused_none = reciprocal_rank_fusion([results], k=k, weights=None)
        fused_empty = reciprocal_rank_fusion([results], k=k, weights={})
        for a, b in zip(fused_none, fused_empty):
            assert a.chunk.symbol_id == b.chunk.symbol_id
            assert a.score == b.score

    def test_weight_multiplier_applied_to_rrf_score_python(self) -> None:
        """Python at rank 1, weight 1.0 → score = 1/(60+1) * 1.0."""
        py_result = _make_result_lang(symbol_id=1, score=0.9, language=Language.PYTHON)
        fused = reciprocal_rank_fusion([[py_result]], k=60, weights={"python": 1.0})
        expected = (1.0 / (60 + 1)) * 1.0
        assert abs(fused[0].score - expected) < 1e-12

    def test_weight_multiplier_applied_to_rrf_score_markdown(self) -> None:
        """Markdown at rank 1, weight 0.3 → score = 1/(60+1) * 0.3."""
        md_result = _make_result_lang(symbol_id=2, score=0.9, language=Language.MARKDOWN)
        fused = reciprocal_rank_fusion([[md_result]], k=60, weights={"markdown": 0.3})
        expected = (1.0 / (60 + 1)) * 0.3
        assert abs(fused[0].score - expected) < 1e-12

    def test_markdown_downweighted_below_python(self) -> None:
        """
        README.md (markdown) outranks the Python file in raw BM25 — rank 1 vs rank 2.
        After file-type weighting, the Python file must rank above the README.

        Without weights:
          markdown_score = 1/(60+1) ≈ 0.01639
          python_score   = 1/(60+2) ≈ 0.01613
          → markdown wins

        With weights={markdown: 0.3, python: 1.0}:
          markdown_score = 1/61 * 0.3 ≈ 0.00492
          python_score   = 1/62 * 1.0 ≈ 0.01613
          → python wins
        """
        md = _make_result_lang(symbol_id=10, score=0.95, language=Language.MARKDOWN)
        py = _make_result_lang(symbol_id=20, score=0.80, language=Language.PYTHON)

        # BM25 leg: markdown at rank 1, python at rank 2
        bm25_leg = [md, py]

        weights = {"python": 1.0, "markdown": 0.3}
        fused = reciprocal_rank_fusion([bm25_leg], k=60, weights=weights)

        ranked_ids = [r.chunk.symbol_id for r in fused]
        assert ranked_ids[0] == 20, (
            f"Expected python (id=20) at rank 1 after weighting, "
            f"but got {ranked_ids[0]} — markdown still winning"
        )

    def test_missing_language_key_defaults_to_1_0(self) -> None:
        """
        A Language value not present in the weights dict must NOT be penalised.
        weights.get(lang, 1.0) must return 1.0 for unknown languages.
        """
        # Use a language explicitly absent from the weights dict
        go_result = _make_result_lang(symbol_id=5, score=0.9, language=Language.GO)
        # weights dict has no "go" key
        weights = {"python": 1.0, "markdown": 0.3}
        fused = reciprocal_rank_fusion([[go_result]], k=60, weights=weights)
        expected = 1.0 / (60 + 1) * 1.0  # fallback multiplier = 1.0
        assert abs(fused[0].score - expected) < 1e-12

    def test_multiple_legs_weights_applied_after_accumulation(self) -> None:
        """
        Python chunk appears in both vector (rank 1) and BM25 (rank 2).
        Markdown chunk appears only in BM25 (rank 1).

        Accumulated RRF before weighting:
          python  = 1/61 + 1/62 ≈ 0.03252
          markdown = 1/61      ≈ 0.01639

        After weights {python: 1.0, markdown: 0.3}:
          python  = 0.03252 * 1.0 ≈ 0.03252
          markdown = 0.01639 * 0.3 ≈ 0.00492

        Python must rank first.
        """
        py = _make_result_lang(symbol_id=1, score=0.9, language=Language.PYTHON)
        md = _make_result_lang(symbol_id=2, score=0.85, language=Language.MARKDOWN)

        # vector leg: python at rank 1 only
        # bm25 leg: markdown at rank 1, python at rank 2
        vector_leg = [py]
        bm25_leg = [md, py]

        weights = {"python": 1.0, "markdown": 0.3}
        fused = reciprocal_rank_fusion([vector_leg, bm25_leg], k=60, weights=weights)

        assert fused[0].chunk.symbol_id == 1, "Python must outrank Markdown after weighting"

        # Verify exact scores
        k = 60
        expected_python = (1.0 / (k + 1) + 1.0 / (k + 2)) * 1.0
        expected_markdown = (1.0 / (k + 1)) * 0.3
        fused_map = {r.chunk.symbol_id: r.score for r in fused}
        assert abs(fused_map[1] - expected_python) < 1e-12
        assert abs(fused_map[2] - expected_markdown) < 1e-12

    def test_html_css_downweighted_below_source(self) -> None:
        """HTML at rank 1 (weight 0.4) must fall below Python at rank 2 (weight 1.0)."""
        html = _make_result_lang(symbol_id=30, score=0.9, language=Language.HTML)
        py = _make_result_lang(symbol_id=31, score=0.8, language=Language.PYTHON)
        bm25_leg = [html, py]
        weights = {"html": 0.4, "python": 1.0}
        fused = reciprocal_rank_fusion([bm25_leg], k=60, weights=weights)
        # html: 1/61 * 0.4 = 0.00656   python: 1/62 * 1.0 = 0.01613 → python wins
        assert fused[0].chunk.symbol_id == 31

    def test_scores_still_positive_after_weighting(self) -> None:
        """All weighted scores must remain strictly positive."""
        md = _make_result_lang(symbol_id=100, score=0.9, language=Language.MARKDOWN)
        fused = reciprocal_rank_fusion([[md]], k=60, weights={"markdown": 0.3})
        assert fused[0].score > 0.0

    def test_rank_field_updated_after_weighting(self) -> None:
        """rank fields on weighted fused output must be 1-indexed and sequential."""
        py = _make_result_lang(symbol_id=1, score=0.9, language=Language.PYTHON)
        md = _make_result_lang(symbol_id=2, score=0.8, language=Language.MARKDOWN)
        fused = reciprocal_rank_fusion([[md, py]], k=60, weights={"python": 1.0, "markdown": 0.3})
        for expected_rank, result in enumerate(fused, start=1):
            assert result.rank == expected_rank


# ---------------------------------------------------------------------------
# Per-list weight multiplier (federated search — weight one source list above
# another, e.g. one repo above another). Orthogonal to the language `weights`.
# ---------------------------------------------------------------------------


class TestListWeights:
    def test_list_weights_none_is_backward_compatible(self) -> None:
        """list_weights=None must give bit-for-bit identical scores to omitting it."""
        list_a = [_make_result(1, 0.9), _make_result(2, 0.8)]
        list_b = [_make_result(1, 0.7), _make_result(3, 0.6)]

        fused_default = reciprocal_rank_fusion([list_a, list_b])
        fused_explicit_none = reciprocal_rank_fusion([list_a, list_b], list_weights=None)

        assert len(fused_default) == len(fused_explicit_none)
        for a, b in zip(fused_default, fused_explicit_none):
            assert a.chunk.symbol_id == b.chunk.symbol_id
            assert a.score == b.score

    def test_list_weights_scales_contribution(self) -> None:
        """
        Two single-item lists, symbol_id 1 at rank 1 in list_a (weight 1.0),
        symbol_id 2 at rank 1 in list_b (weight 5.0).

        Without weighting both would tie at 1/61. With list_weights=[1.0, 5.0],
        symbol 2's contribution is 5x — it must rank first with an exact score.
        """
        k = 60
        list_a = [_make_result(1, 0.9)]
        list_b = [_make_result(2, 0.9)]

        fused = reciprocal_rank_fusion([list_a, list_b], k=k, list_weights=[1.0, 5.0])

        fused_map = {r.chunk.symbol_id: r.score for r in fused}
        assert abs(fused_map[1] - (1.0 / (k + 1))) < 1e-12
        assert abs(fused_map[2] - (5.0 / (k + 1))) < 1e-12
        assert fused[0].chunk.symbol_id == 2, "Higher-weighted list's result must rank first"

    def test_list_weights_orthogonal_to_language_weights(self) -> None:
        """Both list_weights and weights can apply simultaneously without conflict."""
        py = _make_result_lang(symbol_id=1, score=0.9, language=Language.PYTHON)
        fused = reciprocal_rank_fusion(
            [[py]], k=60, weights={"python": 2.0}, list_weights=[3.0]
        )
        expected = (3.0 / (60 + 1)) * 2.0
        assert abs(fused[0].score - expected) < 1e-12
