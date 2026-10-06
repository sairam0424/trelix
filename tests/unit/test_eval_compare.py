"""The eval-compare judge (trelix.eval.compare): the decision table and its boundaries.

Every expected verdict was worked out on paper first (scores are dyadic fractions, so a
constant delta gives a one-point bootstrap interval and an alternating delta a binomial one)
and then confirmed against the real `paired_bootstrap`. The numbers in the comments were
measured with numpy 2.5.0. The borderline fixtures keep a margin of about 2x to the tail they
sit near, so a different numpy random stream does not move them; none pins a stream. The
seed and the number of resamples that the README and CHANGELOG state are held by a spy on
`paired_bootstrap`, not by numpy's output.

The rows are named R1 to R8 as in the module docstring of `trelix.eval.compare`; `rows()` (in
`eval_compare_fixtures`) maps a verdict's reason lines back to them, so a test states WHICH rows
fired, not just the outcome. The pairing, the decision set, the refusals and the notes are in
`test_eval_compare_decision_set.py`, `test_eval_compare_refusals.py` and
`test_eval_compare_notes.py`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. R4 `Hr < G` -> `<=`, R8 `Lr < G` -> `<=`     (test_recall_at_exactly_the_guard_is_not_...)
   the guard -0.02 -> -0.2                       (F9a, F9b)
2. R3 `H < 0` -> `<= 0`                          (F3)
3. R6 `L <= 0` -> `< 0`                          (F3, F11b)
4. R7 `delta < h` -> `<=`                        (test_a_delta_exactly_at_the_hurdle_passes)
5. R5 `expected_effect < MDE` -> `<=` or `>`     (test_an_effect_exactly_at_the_mde_..., F5a, F5b)
6. R2 `n < min_queries` -> `<=`                  (F1 against F1b)
7. `alpha / family_size` -> `alpha` or `* family_size`  (F11a against F11b)
8. base and cand swapped in `paired_bootstrap`   (F1)
9. `stdev` -> `pstdev` (sample sd -> population sd)   (F5c)
10. `EXIT_CODES` permuted                          (test_exit_codes)
11. R3 `H < 0` -> `delta < 0` (the point estimate)   (ndcg-point-estimate-negative-...)
12. R4 `Hr < G` -> `mean < G` (the point estimate)    (recall-point-estimate-below-guard-...)
13. `BOOTSTRAP_SEED` or `BOOTSTRAP_RESAMPLES` changed    (test_the_documented_seed_...)
14. the printed recall@10 base or candidate mean taken from the other run
                                                     (test_the_recall_line_prints_both_means_...)
"""

from __future__ import annotations

from typing import Any

import pytest

import trelix.eval.compare as compare_module
import trelix.eval.prereg as prereg_module
from tests.unit.eval_compare_fixtures import (
    alternating,
    base_doc,
    cand_doc,
    query_ids,
    rows,
    verdict_of,
)


class TestTheDecisionTable:
    @pytest.mark.parametrize(
        ("cand_ndcg", "cand_recall", "prereg", "outcome", "fired"),
        [
            pytest.param(0.5625, 0.5, {}, "PASS", [], id="F1-a-steady-gain-passes"),
            pytest.param(
                0.5625, 0.5, {"min_queries": 21}, "INCONCLUSIVE", ["R2"], id="F1b-too-few-queries"
            ),
            pytest.param(0.4375, 0.5, {}, "FAIL", ["R3"], id="F2-confidently-worse"),
            pytest.param(0.5, 0.5, {}, "INCONCLUSIVE", ["R6", "R7"], id="F3-identical-arms"),
            pytest.param(
                alternating(1.0, 0.0), 0.5, {}, "INCONCLUSIVE", ["R5", "R6", "R7"], id="F4-noise"
            ),
            pytest.param(
                alternating(0.75, 0.5),
                0.5,
                {"expected_effect": 0.05},
                "INCONCLUSIVE",
                ["R5"],
                id="F5a-expected-effect-below-the-mde",
            ),
            pytest.param(
                alternating(0.75, 0.5),
                0.5,
                {"expected_effect": 0.10},
                "PASS",
                [],
                id="F5b-expected-effect-above-the-mde",
            ),
            pytest.param(
                alternating(0.75, 0.5),
                0.5,
                {"expected_effect": 0.079},
                "INCONCLUSIVE",
                ["R5"],
                id="F5c-the-sample-standard-deviation",
            ),
            pytest.param(0.5078125, 0.5, {}, "INCONCLUSIVE", ["R7"], id="F7-below-the-hurdle"),
            pytest.param(0.5625, 0.4375, {}, "FAIL", ["R4"], id="F9a-recall-down-6-points"),
            pytest.param(0.5625, 0.46875, {}, "FAIL", ["R4"], id="F9b-recall-down-3-points"),
            pytest.param(0.5625, 0.484375, {}, "PASS", [], id="F9c-recall-down-1.5-points"),
            pytest.param(
                0.5625,
                alternating(0.75, 0.25),
                {},
                "INCONCLUSIVE",
                ["R8"],
                id="F10-recall-guard-unresolved",
            ),
            pytest.param(0.4375, 0.4375, {}, "FAIL", ["R3", "R4"], id="both-fail-rows-print"),
            # Two queries drop to 0.0 (nDCG) or 0.2 (recall), the rest are level: the point
            # estimate is negative (-0.05, -0.03), but a resample misses both queries 12% of
            # the time (0.9^20), about 5x the 2.5% tail, so the interval still reaches 0 and
            # R3 / R4 (a confident interval, not a point estimate) do not fire.
            pytest.param(
                [0.0, 0.0] + [0.5] * 18,
                0.5,
                {},
                "INCONCLUSIVE",
                ["R5", "R6", "R7"],
                id="ndcg-point-estimate-negative-interval-reaches-0",
            ),
            pytest.param(
                0.5625,
                [0.2, 0.2] + [0.5] * 18,
                {},
                "INCONCLUSIVE",
                ["R8"],
                id="recall-point-estimate-below-guard-interval-reaches-0",
            ),
        ],
    )
    def test_the_fixture_table(
        self,
        cand_ndcg: Any,
        cand_recall: Any,
        prereg: dict[str, Any],
        outcome: str,
        fired: list[str],
    ) -> None:
        verdict = verdict_of(base_doc(), cand_doc(cand_ndcg, cand_recall), **prereg)

        assert verdict.outcome == outcome
        assert rows(verdict) == fired
        assert verdict.lines[-1] == f"verdict: {outcome}"

    def test_a_fail_class_row_hides_the_inconclusive_class_rows(self) -> None:
        # F2 also has L <= 0 (R6) and delta < h (R7); only the winning class prints.
        verdict = verdict_of(base_doc(), cand_doc(0.4375))

        assert rows(verdict) == ["R3"]

    def test_f1_prints_the_numbers_the_rule_read(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5625))

        assert verdict.lines == (
            "comparison: baseline..flag-on",
            "experiment: EXP-example",
            "suite: demo golden_version v1 repo_sha " + "a" * 40,
            "runs: base 2026-10-05T12:00:00+00:00 cand 2026-10-05T12:00:00+00:00",
            "queries: 20 in the decision set (no split labels: all queries), min_queries 20",
            "note: pipeline.config identical",
            "ndcg@10: base 0.5000 cand 0.5625 delta +0.0625 ci [+0.0625, +0.0625] "
            "at 95.00% confidence",
            "recall@10: base 0.5000 cand 0.5000 delta +0.0000 ci [+0.0000, +0.0000] "
            "at 95.00% confidence",
            "mde: 0.0000 (sigma_d 0.0000, n 20); expected_effect 0.0300",
            "hurdle: 0.0100 (cost_class flag)",
            "verdict: PASS",
        )

    def test_the_recall_line_prints_both_means_when_they_differ(self) -> None:
        # Recall falls by exactly 1/64 on every query, inside the -0.02 guard: a PASS, and the
        # line must show the baseline mean 0.5 and the candidate mean 31/64.
        verdict = verdict_of(base_doc(), cand_doc(0.5625, 0.484375))

        assert verdict.outcome == "PASS"
        assert (
            "recall@10: base 0.5000 cand 0.4844 delta -0.0156 ci [-0.0156, -0.0156] "
            "at 95.00% confidence"
        ) in verdict.lines

    def test_example_1_the_mde_from_the_sample_standard_deviation(self) -> None:
        # 20 queries, base 0.5, candidate 0.75 on 10 and 0.5 on 10; deltas 0.25 x 10 and 0 x 10:
        # mean 0.125, sum of squares 0.3125, sample sd sqrt(0.3125 / 19) = 0.128247,
        # MDE = 2.8 * 0.128247 / sqrt(20) = 0.080296. The interval is binomial: [0.075, 0.175].
        verdict = verdict_of(base_doc(), cand_doc(alternating(0.75, 0.5)), expected_effect=0.079)

        assert verdict.outcome == "INCONCLUSIVE"
        assert (
            "ndcg@10: base 0.5000 cand 0.6250 delta +0.1250 ci [+0.0750, +0.1750] "
            "at 95.00% confidence"
        ) in verdict.lines
        assert "mde: 0.0803 (sigma_d 0.1282, n 20); expected_effect 0.0790" in verdict.lines
        assert verdict.reasons == (
            "expected_effect 0.0790 is below the minimum detectable effect 0.0803 "
            "(exploratory: cannot change a default)",
        )

    def test_a_degenerate_interval_prints_its_single_point(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5), expected_effect=0.03)

        assert verdict.reasons == (
            "ndcg@10 improvement not demonstrated: the interval's lower end +0.0000 is not above 0",
            "ndcg@10 delta +0.0000 is below the hurdle 0.0100 (cost_class flag)",
        )

    def test_the_four_fail_and_inconclusive_messages_are_pinned(self) -> None:
        worse = verdict_of(base_doc(), cand_doc(0.4375, 0.4375))
        unresolved = verdict_of(base_doc(), cand_doc(0.5625, alternating(0.75, 0.25)))

        assert worse.reasons == (
            "ndcg@10 is confidently worse: the interval's upper end -0.0625 is below 0",
            "recall@10 is confidently down by more than 0.02: the interval's upper end -0.0625 "
            "is below -0.0200",
        )
        assert unresolved.reasons == (
            "recall@10 guard not resolved: the interval's lower end -0.1000 is below -0.0200",
        )


class TestBoundaries:
    def test_a_delta_exactly_at_the_hurdle_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The delta is exactly 0.0625, so the hurdle can be set to exactly that.
        monkeypatch.setitem(prereg_module.HURDLES, "flag", 0.0625)
        at_the_hurdle = verdict_of(base_doc(), cand_doc(0.5625))
        monkeypatch.setitem(prereg_module.HURDLES, "flag", 0.06251)
        just_above = verdict_of(base_doc(), cand_doc(0.5625))

        assert at_the_hurdle.outcome == "PASS"
        assert just_above.outcome == "INCONCLUSIVE"
        assert rows(just_above) == ["R7"]

    def test_an_effect_exactly_at_the_mde_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Float equality cannot be built from real data, so the MDE is pinned to 0.05.
        monkeypatch.setattr(compare_module, "mde", lambda sigma_d, n: 0.05)
        at_the_mde = verdict_of(base_doc(), cand_doc(alternating(0.75, 0.5)), expected_effect=0.05)
        just_below = verdict_of(
            base_doc(), cand_doc(alternating(0.75, 0.5)), expected_effect=0.0499
        )

        assert at_the_mde.outcome == "PASS"
        assert just_below.outcome == "INCONCLUSIVE"
        assert rows(just_below) == ["R5"]

    def test_recall_at_exactly_the_guard_is_not_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Recall falls by exactly 1/32 on every query, so the interval is the point -0.03125;
        # with the guard set to that, Hr == G == Lr: neither R4 (Hr < G) nor R8 (Lr < G) fires.
        monkeypatch.setattr(compare_module, "RECALL_GUARD", -0.03125)

        verdict = verdict_of(base_doc(), cand_doc(0.5625, 0.46875))

        assert verdict.outcome == "PASS"

    def test_n_equal_to_min_queries_is_enough(self) -> None:
        enough = verdict_of(base_doc(), cand_doc(0.5625), min_queries=20)
        one_short = verdict_of(base_doc(), cand_doc(0.5625), min_queries=21)

        assert enough.outcome == "PASS"
        assert one_short.reasons == ("the decision set has 20 queries, fewer than min_queries 21",)
        assert one_short.lines[-2:] == (
            "reason: the decision set has 20 queries, fewer than min_queries 21",
            "verdict: INCONCLUSIVE",
        )
        assert not any(line.startswith("ndcg@10:") for line in one_short.lines)


class TestMultiplicity:
    IDS = query_ids(24)
    GAIN = [1.0] * 4 + [0.0] * 20  # 4 of 24 queries gain 1.0

    def test_f11a_one_comparison_resolves_the_gain(self) -> None:
        # 4 of 24 deltas are 1: the chance that every draw of a replicate misses them is
        # (20/24)^24 = 0.0126, above a 0.625% tail and below a 2.5% tail, each by a factor 2.
        verdict = verdict_of(
            base_doc(0.0, ids=self.IDS),
            cand_doc(self.GAIN, ids=self.IDS),
            expected_effect=0.25,
        )

        assert verdict.outcome == "PASS"
        assert (
            "ndcg@10: base 0.0000 cand 0.1667 delta +0.1667 ci [+0.0417, +0.3333] "
            "at 95.00% confidence"
        ) in verdict.lines
        assert "mde: 0.2176 (sigma_d 0.3807, n 24); expected_effect 0.2500" in verdict.lines

    def test_f11b_four_comparisons_widen_the_interval_to_98_75_percent(self) -> None:
        verdict = verdict_of(
            base_doc(0.0, ids=self.IDS),
            cand_doc(self.GAIN, ids=self.IDS),
            expected_effect=0.25,
            family_size=4,
        )

        assert verdict.outcome == "INCONCLUSIVE"
        assert rows(verdict) == ["R6"]
        assert (
            "ndcg@10: base 0.0000 cand 0.1667 delta +0.1667 ci [+0.0000, +0.3750] "
            "at 98.75% confidence"
        ) in verdict.lines


class TestTheVerdictObject:
    def test_exit_codes(self) -> None:
        passed = verdict_of(base_doc(), cand_doc(0.5625))
        failed = verdict_of(base_doc(), cand_doc(0.4375))
        unproven = verdict_of(base_doc(), cand_doc(0.5))
        refused = verdict_of(
            base_doc(), cand_doc(0.5625, trelix_version="3.4.4", ids=query_ids(19))
        )

        assert [v.exit_code for v in (passed, failed, unproven, refused)] == [0, 1, 2, 3]
        assert [v.outcome for v in (passed, failed, unproven, refused)] == [
            "PASS",
            "FAIL",
            "INCONCLUSIVE",
            "REFUSED",
        ]

    def test_the_same_inputs_give_the_same_verdict(self) -> None:
        first = verdict_of(base_doc(), cand_doc(alternating(0.75, 0.5)), expected_effect=0.079)
        second = verdict_of(base_doc(), cand_doc(alternating(0.75, 0.5)), expected_effect=0.079)

        assert first == second
        assert first.lines == second.lines


class TestTheBootstrapSettings:
    def test_the_documented_seed_and_number_of_resamples_are_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = compare_module.paired_bootstrap
        calls: list[dict[str, Any]] = []

        def spy(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - forwards paired_bootstrap's
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(compare_module, "paired_bootstrap", spy)

        verdict_of(base_doc(), cand_doc(0.5625))

        # One call for the nDCG@10 interval and one for the recall@10 guard.
        assert [(call["seed"], call["n_resamples"]) for call in calls] == [(0, 10000), (0, 10000)]
