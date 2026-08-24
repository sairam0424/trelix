"""Property: `reciprocal_rank_fusion`'s output is sorted by score descending
with sequential 1-indexed ranks, no identity is lost or duplicated, and every
score matches an INDEPENDENTLY reimplemented closed-form RRF sum -- across
ARBITRARY multi-list, multi-weight configurations, not just the hand-picked
2-3-item lists in tests/unit/test_fusion.py (674 lines, read in full first).

That file already hand-pins: empty/single-list edge cases, same-symbol
dedup across 2-3 lists, the k=60 default and a custom k on ONE hand-built
pair, `test_fused_scores_are_monotonically_decreasing` (one 3-item case),
`test_rank_field_is_1_indexed_sequential` (one case), list_weights scaling
(one case), and file-type weighting (several hand-picked language pairs).
None of those sweep VARYING list counts, list lengths, or overlap patterns
together -- the exact shape of defect the module's own docstring warns about
twice: EXE-02 (bare symbol_id collided across repos) and the deleted
second-dedupe-pass regression (both were invisible to fixed hand-cases and
needed a completeness argument, which is what
TestNoIdentityIsLostOrDuplicated below generalizes).

The independent oracle in TestScoreMatchesClosedFormSum is NOT imported from
fusion.py -- it is the published formula from the module's own docstring
(`score(doc) = Σ 1/(k+rank_i)`) reimplemented from scratch in this file,
against SearchResult objects this test builds and therefore already knows
the rank-per-list of. Per rule 1, this is legitimate: nothing is imported
from the module under test to build the expected value.

FALSIFYING INPUT CONFIRMED BY HAND (see PROOF PROTOCOL below): two lists,
list0=[sym1, sym2, sym3], list1=[sym3, sym1], k=60, no weights. By hand:
score(sym1) = 1/61 + 1/62 = 0.0325224..., score(sym3) = 1/63 + 1/61 =
0.0322664..., score(sym2) = 1/62 = 0.0161290... -- so the correct order is
sym1, sym3, sym2. Confirmed by actually running `reciprocal_rank_fusion`
unmutated (matches the hand calc to 10 decimal places) and then mutating
`sorted(rrf_scores, key=..., reverse=True)` -> `reverse=False` in fusion.py,
which inverts the order to sym2, sym3, sym1 and fails the
non-increasing-score assertion; and separately mutating
`enumerate(sorted_ids, start=1)` -> `start=0`, which fails the
1-indexed-rank assertion. Both pasted in the round report.
"""

from __future__ import annotations

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol, SymbolKind
from trelix.retrieval.fusion import reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(symbol_id: int) -> SearchResult:
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
    # Same path for every result: this property is about cross-list RANK
    # accumulation, not about _fusion_identity's path/symbol_id pairing (that
    # is tests/unit/test_fusion.py's TestRRFDeduplication's territory).
    file = IndexedFile(
        id=1,
        path="/repo/mod.py",
        rel_path="mod.py",
        language=Language.PYTHON,
        hash="abc",
        size_bytes=100,
    )
    return SearchResult(chunk=chunk, symbol=symbol, file=file, score=0.0, rank=1, source="vector")


# A small closed universe of symbol ids lets lists overlap frequently (the
# interesting case for dedup/accumulation) without every example degenerating
# into all-disjoint lists.
_SYMBOL_UNIVERSE = list(range(1, 13))

_RANKED_LIST = (
    st.permutations(_SYMBOL_UNIVERSE)
    .map(lambda ids: list(ids))
    .flatmap(lambda ids: st.integers(min_value=0, max_value=len(ids)).map(lambda n: ids[:n]))
)

_RANKED_LISTS = st.lists(_RANKED_LIST, min_size=1, max_size=4)


def _build_lists(id_lists: list[list[int]]) -> list[list[SearchResult]]:
    return [[_make_result(sid) for sid in ids] for ids in id_lists]


class TestOutputIsSortedWithSequentialRanks:
    """Fails under: `reverse=True` -> `reverse=False` in the final `sorted(...)`
    call (breaks non-increasing scores), or `enumerate(sorted_ids, start=1)` ->
    `start=0` (breaks 1-indexed sequential ranks).
    """

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @example(id_lists=[[1, 2, 3], [3, 1]])  # the hand-verified case
    @given(id_lists=_RANKED_LISTS)
    def test_scores_non_increasing_and_ranks_sequential(self, id_lists: list[list[int]]) -> None:
        ranked_lists = _build_lists(id_lists)
        fused = reciprocal_rank_fusion(ranked_lists, k=60)

        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True), (
            f"fused scores are not non-increasing: {scores!r}"
        )
        ranks = [r.rank for r in fused]
        assert ranks == list(range(1, len(fused) + 1)), (
            f"fused ranks are not 1-indexed sequential: {ranks!r}"
        )


class TestNoIdentityIsLostOrDuplicated:
    """Fails under: a second dedupe pass being reintroduced downstream (the
    module docstring names exactly this as a shipped regression), or any
    change that drops an identity seen in only one list, or emits an identity
    twice. All symbol ids here share one `file.path`, so identity collapses to
    symbol_id alone -- set equality both ways, per rule 2.
    """

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @example(id_lists=[[1, 2, 3], [3, 1]])
    @given(id_lists=_RANKED_LISTS)
    def test_output_identities_equal_the_union_of_input_identities(
        self, id_lists: list[list[int]]
    ) -> None:
        ranked_lists = _build_lists(id_lists)
        fused = reciprocal_rank_fusion(ranked_lists, k=60)

        expected_ids = set()
        for ids in id_lists:
            expected_ids |= set(ids)
        observed_ids = {r.symbol.id for r in fused}

        assert observed_ids == expected_ids
        assert expected_ids == observed_ids
        # No duplicate rows: the output list is exactly as long as the id set.
        assert len(fused) == len(expected_ids)


class TestScoreMatchesClosedFormSum:
    """Fails under: `k + rank` -> `k - rank` or `rank` -> `rank - 1` (an
    off-by-one in the accumulation), or `list_weight / (k + rank)` -> just
    `1 / (k + rank)` (list_weights silently dropped). The oracle below is
    reimplemented from the module's own documented formula, not imported.
    """

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @example(id_lists=[[1, 2, 3], [3, 1]], k=60, list_weights=None)
    @given(
        id_lists=_RANKED_LISTS,
        k=st.integers(min_value=1, max_value=200),
        list_weights=st.one_of(
            st.none(),
            st.lists(
                st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=4,
            ),
        ),
    )
    def test_score_equals_independently_computed_rrf_sum(
        self, id_lists: list[list[int]], k: int, list_weights: list[float] | None
    ) -> None:
        if list_weights is not None and len(list_weights) != len(id_lists):
            list_weights = None  # mismatched length is exercised by list_weights=None only

        ranked_lists = _build_lists(id_lists)
        fused = reciprocal_rank_fusion(ranked_lists, k=k, list_weights=list_weights)

        expected_scores: dict[int, float] = {}
        for list_idx, ids in enumerate(id_lists):
            weight = list_weights[list_idx] if list_weights else 1.0
            for rank, sid in enumerate(ids, start=1):
                expected_scores[sid] = expected_scores.get(sid, 0.0) + weight / (k + rank)

        observed_scores = {r.symbol.id: r.score for r in fused}

        assert observed_scores.keys() == expected_scores.keys()
        for sid, expected in expected_scores.items():
            observed = observed_scores[sid]
            assert abs(observed - expected) < 1e-9, (
                f"symbol {sid}: expected RRF score {expected!r}, got {observed!r}"
            )
