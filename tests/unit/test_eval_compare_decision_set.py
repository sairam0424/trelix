"""The eval-compare judge: pairing by id, the decision set, and errors in either run.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. pairing by position instead of by id            (test_f12_queries_are_paired_by_id_...)
2. the decision set including dev records          (F13, example 2)
3. `min_queries` compared with the record count    (test_f13b_min_queries_counts_...)
4. candidate errors ignored                        (test_f15_a_candidate_query_that_raised_...)
5. base errors ignored                             (test_f15_the_same_error_in_the_baseline_...)
6. the decision set taken from the candidate's labels  (example 2, with the labels refusal)
7. the bootstrap fed the queries in file order, not sorted by id  (test_the_bootstrap_sees_...)
8. candidate errors looked for in the decision set only   (test_f15_a_candidate_error_on_a_dev_...)
9. base errors looked for in the decision set only        (test_f15_a_baseline_error_on_a_dev_...)
10. row 2 (too few queries) tried before row 1 (errors)  (test_f15_a_candidate_error_is_judged_...)
11. the MDE computed with the record count, not the size of the decision set  (test_the_mde_...)
12. the number of candidate errors reported as a constant  (test_f15_every_query_that_raised_...)
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.eval_compare_fixtures import (
    alternating,
    base_doc,
    cand_doc,
    query_ids,
    rows,
    verdict_of,
)
from trelix.eval.compare import Verdict


class TestPairingAndTheDecisionSet:
    def test_f12_queries_are_paired_by_id_not_by_position(self) -> None:
        ids = query_ids(20)
        base_scores = [(number + 1) / 32 for number in range(20)]
        # The candidate file lists the queries in the opposite order; every query gains 1/16.
        reversed_ids = list(reversed(ids))
        cand_scores = [score + 1 / 16 for score in reversed(base_scores)]

        verdict = verdict_of(
            base_doc(base_scores, ids=ids), cand_doc(cand_scores, ids=reversed_ids)
        )

        # Paired by position the deltas would be spread over [-0.59, 0.66] with mean 0.0625
        # and an interval of about [-0.09, 0.22]: INCONCLUSIVE.
        assert verdict.outcome == "PASS"
        assert (
            "ndcg@10: base 0.3281 cand 0.3906 delta +0.0625 ci [+0.0625, +0.0625] "
            "at 95.00% confidence"
        ) in verdict.lines

    def test_f13_dev_queries_never_enter_the_decision(self) -> None:
        # 30 dev queries gain 0.5 and 30 test queries do not change: the test set has no gain.
        ids = query_ids(60)
        splits = ["dev"] * 30 + ["test"] * 30
        base = base_doc([0.25] * 30 + [0.5] * 30, ids=ids, splits=splits)
        cand = cand_doc([0.75] * 30 + [0.5] * 30, ids=ids, splits=splits)

        verdict = verdict_of(base, cand, expected_effect=0.10)

        assert verdict.outcome == "INCONCLUSIVE"
        assert rows(verdict) == ["R6", "R7"]
        assert (
            "queries: 30 in the decision set (split labels: test queries only), min_queries 20"
        ) in verdict.lines

    def test_f13b_min_queries_counts_the_decision_set_not_the_records(self) -> None:
        # 60 records but 30 test queries, and the pre-registration asks for 40. Counting records
        # would let the 30 test queries (gain 0.0625 on each) through to PASS.
        ids = query_ids(60)
        splits = ["dev"] * 30 + ["test"] * 30
        base = base_doc(0.5, ids=ids, splits=splits)
        cand = cand_doc([0.5] * 30 + [0.5625] * 30, ids=ids, splits=splits)

        verdict = verdict_of(base, cand, min_queries=40)

        assert verdict.outcome == "INCONCLUSIVE"
        assert verdict.reasons == ("the decision set has 30 queries, fewer than min_queries 40",)
        assert rows(verdict) == ["R2"]

    def test_example_2_the_base_labels_decide_which_queries_count(self) -> None:
        ids = query_ids(30)
        splits = ["dev"] * 10 + ["test"] * 20
        base = base_doc(0.5, ids=ids, splits=splits)
        cand = cand_doc([0.75] * 10 + [0.5] * 20, ids=ids, splits=splits)

        verdict = verdict_of(base, cand, expected_effect=0.10)

        assert verdict.outcome == "INCONCLUSIVE"
        assert rows(verdict) == ["R6", "R7"]

    def test_example_2_a_candidate_cannot_relabel_its_way_to_a_pass(self) -> None:
        # The candidate marks q01..q20 as test (its gain is all in q01..q10) and q21..q30 as dev.
        ids = query_ids(30)
        base = base_doc(0.5, ids=ids, splits=["dev"] * 10 + ["test"] * 20)
        cand = cand_doc([0.75] * 10 + [0.5] * 20, ids=ids, splits=["test"] * 20 + ["dev"] * 10)

        verdict = verdict_of(base, cand, expected_effect=0.10)

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (
            "the runs label queries differently: q01, q02, q03, q04, q05 (and 15 more)",
        )

    @pytest.mark.parametrize(
        ("min_queries", "outcome"), [(54, "INCONCLUSIVE"), (38, "PASS"), (39, "INCONCLUSIVE")]
    )
    def test_example_3_min_queries_against_the_size_of_the_test_split(
        self, min_queries: int, outcome: str
    ) -> None:
        # 54 golden queries, 16 dev and 38 test; the candidate gains 0.0625 on every test query.
        ids = query_ids(54)
        splits = ["dev"] * 16 + ["test"] * 38
        base = base_doc(0.5, ids=ids, splits=splits)
        cand = cand_doc([0.5] * 16 + [0.5625] * 38, ids=ids, splits=splits)

        verdict = verdict_of(base, cand, min_queries=min_queries)

        assert verdict.outcome == outcome
        if outcome == "INCONCLUSIVE":
            assert verdict.reasons == (
                f"the decision set has 38 queries, fewer than min_queries {min_queries}",
            )

    def test_the_mde_is_computed_from_the_decision_set_not_from_every_record(self) -> None:
        # 30 dev queries that do not change, then 30 test queries whose gain alternates 0.25 and
        # 0. On the 30 test queries sigma_d is 0.1271 and the MDE 2.8 * 0.1271 / sqrt(30) =
        # 0.0650, above the expected effect 0.05. Divided by sqrt(60) it would be 0.0460, and
        # the same files would PASS.
        ids = query_ids(60)
        splits = ["dev"] * 30 + ["test"] * 30
        base = base_doc(0.5, ids=ids, splits=splits)
        cand = cand_doc([0.5] * 30 + alternating(0.75, 0.5, 30), ids=ids, splits=splits)

        verdict = verdict_of(base, cand, expected_effect=0.05, min_queries=30)

        assert verdict.outcome == "INCONCLUSIVE"
        assert verdict.reasons == (
            "expected_effect 0.0500 is below the minimum detectable effect 0.0650 "
            "(exploratory: cannot change a default)",
        )
        assert "mde: 0.0650 (sigma_d 0.1271, n 30); expected_effect 0.0500" in verdict.lines

    def test_an_all_dev_baseline_has_an_empty_decision_set(self) -> None:
        base = base_doc(0.5, splits=["dev"] * 20)
        cand = cand_doc(0.5625, splits=["dev"] * 20)

        verdict = verdict_of(base, cand)

        assert verdict.reasons == ("the decision set has 0 queries, fewer than min_queries 20",)

    def test_the_order_of_the_records_in_either_file_changes_nothing(self) -> None:
        ids = query_ids(24)
        gain = [1.0] * 4 + [0.0] * 20
        forward = verdict_of(base_doc(0.0, ids=ids), cand_doc(gain, ids=ids), expected_effect=0.25)
        shuffled_ids = ids[12:] + ids[:12]
        shuffled_gain = gain[12:] + gain[:12]
        backward = verdict_of(
            base_doc(0.0, ids=list(reversed(ids))),
            cand_doc(shuffled_gain, ids=shuffled_ids),
            expected_effect=0.25,
        )

        assert forward == backward

    def test_the_bootstrap_sees_the_queries_in_sorted_id_order(self) -> None:
        # 24 distinct gains, so the resampled means, and the interval ends to four places,
        # depend on which gain sits at which position. The 0/1 gains of the test above cannot
        # see the order. Only equality across orders is asserted: the ends themselves come
        # from numpy's generator, which this suite does not pin.
        ids = query_ids(24)
        gain = {query: 0.5 + ((number * 7919) % 97) / 400 for number, query in enumerate(ids, 1)}

        def judged(order: list[str]) -> Verdict:
            return verdict_of(
                base_doc(0.5, ids=order),
                cand_doc([gain[query] for query in order], ids=order),
                expected_effect=0.05,
            )

        forward = judged(ids)

        assert forward.outcome == "PASS"
        for order in (ids[::-1], ids[8:] + ids[:8], ids[16:] + ids[:16], ids[::2] + ids[1::2]):
            assert judged(order) == forward


def _dev_and_test_pair(
    *, base_errors: dict[str, str] | None = None, cand_errors: dict[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """30 dev queries (q01-q30) then 30 test ones; the candidate gains 0.0625 on each test query."""
    ids = query_ids(60)
    splits = ["dev"] * 30 + ["test"] * 30
    base = base_doc(0.5, ids=ids, splits=splits, errors=base_errors)
    cand = cand_doc([0.5] * 30 + [0.5625] * 30, ids=ids, splits=splits, errors=cand_errors)
    return base, cand


class TestCandidateAndBaselineErrors:
    def test_f15_a_candidate_query_that_raised_fails_the_run(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5625, errors={"q05": "boom"}))

        assert verdict.outcome == "FAIL"
        assert verdict.reasons == (
            "the candidate run has 1 record(s) with an error (first: q05: boom)",
        )
        assert not any(line.startswith("ndcg@10:") for line in verdict.lines)

    @pytest.mark.parametrize("message", [" ", "\n"], ids=["space", "newline"])
    def test_f15_an_error_message_of_only_whitespace_is_still_an_error(self, message: str) -> None:
        failed = verdict_of(base_doc(), cand_doc(0.5625, errors={"q05": message}))
        refused = verdict_of(base_doc(errors={"q05": message}), cand_doc(0.5625))

        assert failed.outcome == "FAIL"
        assert failed.reasons[0].startswith("the candidate run has 1 record(s) with an error")
        assert refused.outcome == "REFUSED"
        assert refused.reasons[0].startswith("base run has 1 record(s) with an error")

    def test_f15_every_query_that_raised_is_counted_and_the_first_is_named(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5625, errors={"q09": "bang", "q05": "boom"}))

        assert verdict.outcome == "FAIL"
        assert verdict.reasons == (
            "the candidate run has 2 record(s) with an error (first: q05: boom)",
        )

    def test_f15_a_candidate_error_is_judged_before_the_number_of_queries(self) -> None:
        # Row 1 stops the evaluation before row 2: with 20 queries and min_queries 21 the run
        # is a FAIL for its error, not an INCONCLUSIVE for its size.
        verdict = verdict_of(base_doc(), cand_doc(0.5625, errors={"q05": "boom"}), min_queries=21)

        assert verdict.outcome == "FAIL"
        assert verdict.reasons == (
            "the candidate run has 1 record(s) with an error (first: q05: boom)",
        )

    def test_f15_the_same_error_in_the_baseline_is_a_refusal(self) -> None:
        verdict = verdict_of(base_doc(errors={"q05": "boom"}), cand_doc(0.5625))

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (
            "base run has 1 record(s) with an error (first: q05: boom): "
            "a baseline that raised is not an instrument",
        )
        assert verdict.lines == ("verdict: REFUSED",)

    def test_f15_a_candidate_error_on_a_dev_query_still_fails_the_run(self) -> None:
        # q05 is a dev query, outside the decision set, but a candidate that raised is broken
        # whichever query it broke on. The same files without the error are a PASS.
        assert verdict_of(*_dev_and_test_pair()).outcome == "PASS"

        verdict = verdict_of(*_dev_and_test_pair(cand_errors={"q05": "boom"}))

        assert verdict.outcome == "FAIL"
        assert verdict.reasons == (
            "the candidate run has 1 record(s) with an error (first: q05: boom)",
        )

    def test_f15_a_baseline_error_on_a_dev_query_is_still_a_refusal(self) -> None:
        verdict = verdict_of(*_dev_and_test_pair(base_errors={"q05": "boom"}))

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (
            "base run has 1 record(s) with an error (first: q05: boom): "
            "a baseline that raised is not an instrument",
        )
