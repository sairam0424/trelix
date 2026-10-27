"""Paired bootstrap, minimum detectable effect and Holm step-down (trelix.eval.stats).

Every expected value is a literal, worked out by hand or from a closed form, never read
back from the module: the bootstrap cases use fixtures whose exact resampling
distribution is known (0/1 deltas are binomial; a single nonzero delta out of three has
P(all draws zero) = (2/3)^3), so they test the arithmetic rather than a frozen random
stream.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from trelix.eval.stats import holm, mde, paired_bootstrap

# 50 query scores in {0.00, 0.05, ..., 0.95} that differ a lot from query to query, so a
# paired comparison and an unpaired one give very different intervals.
_VARIED_SCORES = [((i * 7) % 20) / 20 for i in range(50)]


class TestIdenticalArms:
    def test_delta_is_zero_and_the_interval_is_the_point_zero(self) -> None:
        result = paired_bootstrap(_VARIED_SCORES, _VARIED_SCORES)
        assert result.mean_delta == 0.0
        assert result.ci_low == 0.0
        assert result.ci_high == 0.0
        assert result.n == 50

    def test_p_value_is_one_not_two(self) -> None:
        # Every replicate mean is exactly 0, so both shares are 1.0 and 2 * 1.0 is capped.
        assert paired_bootstrap(_VARIED_SCORES, _VARIED_SCORES).p_value == 1.0


class TestPlantedDelta:
    def test_a_constant_gain_is_found_and_the_interval_excludes_zero(self) -> None:
        cand = [score + 0.05 for score in _VARIED_SCORES]
        result = paired_bootstrap(_VARIED_SCORES, cand)
        assert result.mean_delta == pytest.approx(0.05, abs=1e-12)
        # The pairing cancels the query-to-query spread, so the interval is a point.
        # Unpaired resampling of the same arms gives roughly [-0.07, 0.17] instead.
        assert result.ci_low == pytest.approx(0.05, abs=1e-9)
        assert result.ci_high == pytest.approx(0.05, abs=1e-9)
        assert result.ci_low > 0.0
        assert result.p_value == 0.0
        assert result.n == 50

    def test_a_constant_loss_is_found_too(self) -> None:
        cand = [score - 0.05 for score in _VARIED_SCORES]
        result = paired_bootstrap(_VARIED_SCORES, cand)
        assert result.mean_delta == pytest.approx(-0.05, abs=1e-12)
        assert result.ci_high < 0.0
        assert result.p_value == 0.0


class TestIntervalAndPValueArithmetic:
    # 100 queries, 50 of which gain exactly 1.0 and 50 of which do not. Each replicate
    # mean is Binomial(100, 0.5) / 100: sd 0.05, and the exact 2.5th and 97.5th
    # percentiles are 0.40 and 0.60 (the 5th and 95th are 0.42 and 0.58).
    _BASE = [0.0] * 100
    _CAND = [1.0] * 50 + [0.0] * 50

    def test_default_interval_is_the_95_percent_percentile_interval(self) -> None:
        result = paired_bootstrap(self._BASE, self._CAND)
        assert result.mean_delta == 0.5
        # Replicate means move in steps of 0.01, so a tolerance of half a step notices a
        # bound that is one step off (the 97th percentile gives 0.59, not 0.60).
        assert result.ci_low == pytest.approx(0.40, abs=0.005)
        assert result.ci_high == pytest.approx(0.60, abs=0.005)

    def test_confidence_sets_the_interval_width(self) -> None:
        # Binomial(100, 0.5) quartiles: 47 and 53 (the 25th and 75th percentiles).
        result = paired_bootstrap(self._BASE, self._CAND, confidence=0.5)
        assert result.ci_low == pytest.approx(0.47, abs=0.005)
        assert result.ci_high == pytest.approx(0.53, abs=0.005)

    def test_p_value_is_twice_the_smaller_tail(self) -> None:
        # Deltas [1, 0, 0]: a replicate mean is <= 0 only if all three draws are the
        # zeros, probability (2/3)^3 = 8/27, and >= 0 always. p = 2 * 8/27 = 0.5926.
        result = paired_bootstrap([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result.p_value == pytest.approx(0.5926, abs=0.05)

    def test_p_value_uses_the_other_tail_for_a_loss(self) -> None:
        result = paired_bootstrap([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        assert result.p_value == pytest.approx(0.5926, abs=0.05)

    def test_resample_size_is_the_number_of_queries(self) -> None:
        # With n - 1 draws per replicate the same fixture gives (2/3)^2 * 2 = 0.889.
        result = paired_bootstrap([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result.p_value < 0.7

    def test_n_resamples_sets_the_resolution_of_the_p_value(self) -> None:
        # p = 2 * count / n_resamples, so p * 250 is a whole number for 250 resamples.
        result = paired_bootstrap([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], n_resamples=250)
        scaled = result.p_value * 250
        assert scaled == pytest.approx(round(scaled), abs=1e-9)
        assert round(scaled) % 2 == 0

    def test_the_default_is_ten_thousand_resamples(self) -> None:
        # Continuous scores, so a different number of resamples moves the interval.
        base, cand = TestSeed._noisy_arms()
        assert paired_bootstrap(base, cand) == paired_bootstrap(base, cand, n_resamples=10_000)


class TestSeed:
    @staticmethod
    def _noisy_arms() -> tuple[list[float], list[float]]:
        rng = np.random.default_rng(42)
        base = rng.random(60)
        cand = base + rng.normal(0.02, 0.2, 60)
        return base.tolist(), cand.tolist()

    def test_the_same_seed_gives_the_same_result(self) -> None:
        base, cand = self._noisy_arms()
        assert paired_bootstrap(base, cand, seed=7) == paired_bootstrap(base, cand, seed=7)

    def test_the_default_seed_is_zero(self) -> None:
        base, cand = self._noisy_arms()
        assert paired_bootstrap(base, cand) == paired_bootstrap(base, cand, seed=0)

    def test_a_different_seed_gives_a_different_resampling(self) -> None:
        base, cand = self._noisy_arms()
        first = paired_bootstrap(base, cand, seed=0)
        second = paired_bootstrap(base, cand, seed=1)
        assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high)
        assert first.mean_delta == second.mean_delta


class TestQueryOrder:
    def test_a_joint_permutation_changes_neither_the_mean_nor_the_interval(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.random(50)
        cand = base + rng.normal(0.02, 0.2, 50)
        order = np.random.default_rng(11).permutation(50)

        straight = paired_bootstrap(base.tolist(), cand.tolist(), n_resamples=20_000)
        shuffled = paired_bootstrap(base[order].tolist(), cand[order].tolist(), n_resamples=20_000)

        assert shuffled.mean_delta == straight.mean_delta  # exact: the mean is order-free
        # The resamples differ (the same seed picks different pairs), so the interval
        # moves by sampling noise only. Here the standard error is ~0.025 and the
        # endpoints move by ~0.001 at 20,000 resamples; 0.005 is 20 percent of the SE.
        assert shuffled.ci_low == pytest.approx(straight.ci_low, abs=0.005)
        assert shuffled.ci_high == pytest.approx(straight.ci_high, abs=0.005)
        assert shuffled.n == straight.n == 50


class TestResultShape:
    def test_n_counts_the_pairs(self) -> None:
        assert paired_bootstrap([0.1, 0.2, 0.3], [0.2, 0.2, 0.4]).n == 3

    def test_accepts_tuples_and_arrays(self) -> None:
        from_tuples = paired_bootstrap((0.1, 0.2, 0.3), (0.2, 0.2, 0.4))
        from_arrays = paired_bootstrap(np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.2, 0.4]))
        assert from_tuples == from_arrays

    def test_the_result_is_immutable(self) -> None:
        result = paired_bootstrap([0.1, 0.2], [0.3, 0.4])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.mean_delta = 1.0  # type: ignore[misc]

    def test_field_names(self) -> None:
        result = paired_bootstrap([0.1, 0.2], [0.3, 0.4])
        assert [f.name for f in dataclasses.fields(result)] == [
            "mean_delta",
            "ci_low",
            "ci_high",
            "p_value",
            "n",
        ]


class TestInputValidation:
    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="3 != 2"):
            paired_bootstrap([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_no_queries_raises(self) -> None:
        with pytest.raises(ValueError, match="no queries"):
            paired_bootstrap([], [])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_a_non_finite_score_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            paired_bootstrap([0.1, bad], [0.1, 0.2])
        with pytest.raises(ValueError, match="NaN or infinite"):
            paired_bootstrap([0.1, 0.2], [0.1, bad])

    def test_a_nested_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="flat sequence"):
            paired_bootstrap([[0.1, 0.2]], [[0.1, 0.2]])

    @pytest.mark.parametrize("n_resamples", [0, -5])
    def test_fewer_than_one_resample_raises(self, n_resamples: int) -> None:
        with pytest.raises(ValueError, match="n_resamples"):
            paired_bootstrap([0.1], [0.2], n_resamples=n_resamples)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, 95.0, -0.1, float("nan")])
    def test_confidence_outside_zero_one_raises(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            paired_bootstrap([0.1], [0.2], confidence=confidence)


class TestMde:
    def test_known_values(self) -> None:
        assert mde(0.5, 4) == pytest.approx(0.7)  # 2.8 * 0.5 / 2
        assert mde(0.2, 25) == pytest.approx(0.112)  # 2.8 * 0.2 / 5
        assert mde(1.0, 1) == pytest.approx(2.8)

    def test_zero_spread_detects_anything(self) -> None:
        assert mde(0.0, 10) == 0.0

    def test_quadrupling_the_queries_halves_it(self) -> None:
        assert mde(0.3, 400) == pytest.approx(mde(0.3, 100) / 2)

    def test_the_multiplier_is_the_sum_of_the_two_normal_quantiles(self) -> None:
        # Closed form: z(0.975) = 1.959964, z(0.80) = 0.841621; their sum rounds to 2.8.
        assert mde(1.0, 1) == pytest.approx(1.959964 + 0.841621, abs=0.005)

    def test_sqrt_of_n_not_n(self) -> None:
        assert mde(1.0, 9) == pytest.approx(2.8 / math.sqrt(9))

    @pytest.mark.parametrize("sigma_d", [-0.1, float("nan")])
    def test_a_negative_or_nan_spread_raises(self, sigma_d: float) -> None:
        with pytest.raises(ValueError, match="sigma_d"):
            mde(sigma_d, 10)

    @pytest.mark.parametrize("n", [0, -3])
    def test_no_queries_raises(self, n: int) -> None:
        with pytest.raises(ValueError, match="n must be"):
            mde(0.1, n)


class TestHolm:
    def test_a_hand_worked_case_in_input_order(self) -> None:
        # m = 3, alpha = .05: thresholds .0167, .025, .05 against the sorted
        # .01, .03, .04. .01 passes; .03 > .025 stops the walk.
        assert holm([0.01, 0.04, 0.03]) == [True, False, False]

    def test_step_down_rejects_what_plain_bonferroni_would_not(self) -> None:
        # Bonferroni uses .0167 throughout and keeps .02 and .04. Holm's thresholds
        # .0167, .025, .05 take all three.
        assert holm([0.001, 0.02, 0.04]) == [True, True, True]

    def test_the_walk_stops_at_the_first_failure(self) -> None:
        # Sorted .001 .02 .021 .04 against .0125 .0167 .025 .05. .02 > .0167 fails, and
        # .021 and .04 would each pass their own threshold but are kept anyway.
        assert holm([0.001, 0.02, 0.021, 0.04]) == [True, False, False, False]

    def test_tied_p_values_get_the_same_answer_when_they_pass(self) -> None:
        assert holm([0.01, 0.01, 0.01]) == [True, True, True]

    def test_tied_p_values_get_the_same_answer_when_they_fail(self) -> None:
        assert holm([0.03, 0.03]) == [False, False]
        # Sorted .005 .02 .02 .5 against .0125 .0167 .025 .05: .02 fails at its first
        # copy, so the second copy is kept too.
        assert holm([0.02, 0.005, 0.02, 0.5]) == [False, True, False, False]

    def test_a_p_value_exactly_at_its_threshold_is_rejected(self) -> None:
        # alpha / m = .05 / 2 = .025 exactly (a division by a power of two is exact).
        assert holm([0.025, 0.9]) == [True, False]

    def test_alpha_is_a_parameter(self) -> None:
        assert holm([0.02, 0.9], alpha=0.1) == [True, False]
        assert holm([0.02, 0.9], alpha=0.01) == [False, False]

    def test_nothing_is_rejected_when_the_smallest_fails(self) -> None:
        assert holm([0.5, 0.6, 0.7]) == [False, False, False]

    def test_a_single_p_value_is_tested_at_alpha(self) -> None:
        assert holm([0.05]) == [True]
        assert holm([0.051]) == [False]

    def test_no_p_values(self) -> None:
        assert holm([]) == []

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, float("nan")])
    def test_alpha_outside_zero_one_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            holm([0.01], alpha=alpha)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
    def test_a_p_value_outside_zero_one_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="p-value"):
            holm([0.01, bad])
