"""Known-answer arithmetic tests for `trelix.eval.ndcg.ndcg_at_k`.

Every retrieval A/B decision in this project is read off nDCG@10, but the metric had
no pinned arithmetic: the pre-existing tests assert `== 1.0`, `== 0.0`, or
`0.0 < score < 1.0`, and *all three of those shapes are invariant* to the discount
function. nDCG is a ratio DCG/IDCG computed with one shared discount, so replacing

    1.0 / math.log2(rank + 2)      # correct: rank 0 -> 1/log2(2) = 1.0

with any other monotone decreasing discount still returns exactly 1.0 for a perfect
ranking, exactly 0.0 for a miss, and something strictly between for a partial hit.
Measured: `1.0 / math.log2(rank + 2)` -> `1.0 / math.log2(rank + 3)` survived all 46
tests that touch this file.

The only assertions that can see the discount are exact interior values. Those are
written below as decimal literals, hand-derived in each docstring from the standard
binary-log constants

    log2(2) = 1
    log2(3) = 1.584962500721156
    log2(4) = 2
    log2(5) = 2.321928094887362
    log2(6) = 2.584962500721156

so a reader can check each number against the closed form without executing, or
trusting, the module under test. No expected value here is computed with the
formula the implementation uses.
"""

from __future__ import annotations

import pytest

from trelix.eval.ndcg import ndcg_at_k

# Tight enough that the ~10-25% shifts produced by the target mutation cannot hide,
# loose enough to absorb float summation order.
_REL = 1e-12


class TestNdcgDiscountIsPinnedToLog2RankPlus2:
    """The discount denominator itself, not just the [0, 1] envelope."""

    def test_single_relevant_document_scores_the_bare_positional_discount(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ndcg_at_k.dcg:
            1.0 / math.log2(rank + 2)  ->  1.0 / math.log2(rank + 3)

        Hand arithmetic. With exactly one relevant document, IDCG is that document
        placed at rank 1, i.e. 1/log2(0 + 2) = 1/1 = 1.0 exactly. So nDCG collapses to
        the raw discount of whatever position the document actually occupies:

            document at 1-indexed rank r  ->  nDCG = 1 / log2(r + 1)

            r = 1 : 1 / log2(2) = 1 / 1                 = 1.0
            r = 2 : 1 / log2(3) = 1 / 1.584962500721156 = 0.6309297535714575
            r = 3 : 1 / log2(4) = 1 / 2                 = 0.5
            r = 4 : 1 / log2(5) = 1 / 2.321928094887362 = 0.43067655807339306

        Under the mutation the whole curve becomes log2(3)/log2(r + 2):
            r = 1 -> 1.0 (UNCHANGED -- this is why "perfect ranking == 1.0" cannot
                          catch the mutation and why r >= 2 is required here)
            r = 2 -> 1.584962500721156 / 2                 = 0.792481250360578
            r = 3 -> 1.584962500721156 / 2.321928094887362 = 0.6826061944859853
            r = 4 -> 1.584962500721156 / 2.584962500721156 = 0.6131471927654584
        """
        relevant = {7}
        # Precondition (rule 4): the assertions below are only the bare discount
        # because there is exactly one relevant document, which forces IDCG == 1.0.
        # If this fixture ever grows a second relevant id, IDCG stops being 1.0 and
        # the literals below stop meaning what the docstring says they mean.
        assert len(relevant) == 1
        assert ndcg_at_k([7], relevant, k=10) == pytest.approx(1.0, rel=_REL)

        # Explicit, non-iterated expected table: one flat assertion per rank, so
        # deleting any line visibly deletes a case instead of silently shrinking a loop.
        assert ndcg_at_k([7, 0, 1, 2], relevant, k=10) == pytest.approx(1.0, rel=_REL)
        assert ndcg_at_k([0, 7, 1, 2], relevant, k=10) == pytest.approx(
            0.6309297535714575, rel=_REL
        )
        assert ndcg_at_k([0, 1, 7, 2], relevant, k=10) == pytest.approx(0.5, rel=_REL)
        assert ndcg_at_k([0, 1, 2, 7], relevant, k=10) == pytest.approx(
            0.43067655807339306, rel=_REL
        )

    def test_two_relevant_documents_at_ranks_two_and_four(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ndcg_at_k.dcg:
            1.0 / math.log2(rank + 2)  ->  1.0 / math.log2(rank + 3)

        Hand arithmetic. relevant = {0, 1}; ranking [9, 0, 8, 1] puts relevant docs at
        1-indexed ranks 2 and 4, so the ideal ordering (both relevant docs at ranks 1
        and 2) differs from the actual one and IDCG is load-bearing.

            DCG  = 1/log2(3) + 1/log2(5)
                 = 0.6309297535714575 + 0.43067655807339306 = 1.0616063116448506
            IDCG = 1/log2(2) + 1/log2(3)
                 = 1.0               + 0.6309297535714575   = 1.6309297535714575
            nDCG = 1.0616063116448506 / 1.6309297535714575  = 0.6509209298071326

        Under the mutation:
            DCG  = 1/log2(4) + 1/log2(6) = 0.5 + 0.38685280723454163 = 0.8868528072345416
            IDCG = 1/log2(3) + 1/log2(4) = 0.6309297535714575 + 0.5  = 1.1309297535714575
            nDCG = 0.7841802768331765
        """
        relevant = {0, 1}
        ranked = [9, 0, 8, 1]
        # Precondition (rule 4): if either relevant doc sat at rank 1 the actual and
        # ideal orderings would coincide at the head and IDCG would stop discriminating.
        assert ranked[0] not in relevant
        assert len(relevant) == 2

        assert ndcg_at_k(ranked, relevant, k=10) == pytest.approx(0.6509209298071326, rel=_REL)

    def test_one_of_two_relevant_documents_found_at_rank_one(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ndcg_at_k.dcg:
            1.0 / math.log2(rank + 2)  ->  1.0 / math.log2(rank + 3)

        Hand arithmetic. relevant = {0, 1}; ranking [0, 8, 9] finds only doc 0, at
        rank 1. The missing second relevant doc lives only in IDCG, so the score is
        the ratio of a one-term DCG to a two-term IDCG:

            DCG  = 1/log2(2) = 1.0
            IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309297535714575 = 1.6309297535714575
            nDCG = 1.0 / 1.6309297535714575 = 0.6131471927654584

        Under the mutation:
            DCG  = 1/log2(3) = 0.6309297535714575
            IDCG = 1/log2(3) + 1/log2(4) = 1.1309297535714575
            nDCG = 0.5578858913022597
        """
        assert ndcg_at_k([0, 8, 9], {0, 1}, k=10) == pytest.approx(0.6131471927654584, rel=_REL)

    def test_relevant_document_past_the_cutoff_counts_in_idcg_only(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ndcg_at_k.dcg:
            1.0 / math.log2(rank + 2)  ->  1.0 / math.log2(rank + 3)

        Hand arithmetic. relevant = {0, 5}, ranking [9, 8, 0, 5], k = 3. The k cutoff
        keeps only [9, 8, 0], so doc 0 at rank 3 is the sole DCG contributor while both
        relevant docs still count toward IDCG:

            DCG  = 1/log2(4) = 0.5
            IDCG = 1/log2(2) + 1/log2(3) = 1.6309297535714575
            nDCG = 0.5 / 1.6309297535714575 = 0.3065735963827292

        Under the mutation:
            DCG  = 1/log2(5) = 0.43067655807339306
            IDCG = 1/log2(3) + 1/log2(4) = 1.1309297535714575
            nDCG = 0.3808163652192575
        """
        ranked = [9, 8, 0, 5]
        relevant = {0, 5}
        # Precondition (rule 4): the cutoff must actually bite -- doc 5 has to sit
        # outside the top k, otherwise this is just another partial-hit case.
        assert 5 in ranked[3:]

        assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(0.3065735963827292, rel=_REL)

    def test_perfect_ranking_is_exactly_one(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ndcg_at_k.dcg:
            `if doc_id in rel`  ->  `if doc_id not in rel`   (score becomes 0.0)

        Documented anchor, kept explicit: both relevant docs at the top two ranks is
        DCG == IDCG == 1/log2(2) + 1/log2(3) = 1.6309297535714575, hence exactly 1.0.

        Stated plainly so nobody mistakes this for discount coverage: this assertion
        is INVARIANT to `rank + 2 -> rank + 3`, because that change scales DCG and
        IDCG by the same factors. It cannot kill the target mutation. The interior
        values above are what do that.
        """
        assert ndcg_at_k([1, 2, 3, 4, 5], {1, 2}, k=5) == pytest.approx(1.0, rel=_REL)

    def test_empty_result_list_scores_zero(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ndcg_at_k:
            `return actual / ideal if ideal > 0 else 0.0`
            ->  `return 1.0 - actual / ideal if ideal > 0 else 0.0`

        Retrieving nothing must score 0.0, not "no penalty": DCG over an empty ranking
        is the empty sum 0.0, IDCG is 1/log2(2) = 1.0, so nDCG = 0.0 / 1.0 = 0.0.
        """
        assert ndcg_at_k([], {1}, k=10) == 0.0

    def test_no_ground_truth_scores_zero(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ndcg_at_k:
            `if not relevant_ids: return 0.0`  ->  `if not relevant_ids: return 1.0`

        Documented guard clause: an empty `relevant_ids` returns 0.0 rather than
        dividing by an empty IDCG.
        """
        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0
