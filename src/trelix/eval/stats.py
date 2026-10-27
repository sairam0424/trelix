"""Paired statistics for comparing two retrieval runs over the same golden queries.

Three small functions, numpy only:

* `paired_bootstrap` — the mean paired difference between two runs, with a percentile
  bootstrap confidence interval and a two-sided bootstrap p-value.
* `mde` — the smallest true difference a test on `n` queries can reliably detect.
* `holm` — Holm-Bonferroni step-down, for when several comparisons are made at once.

The inputs are per-query scores paired by position: element `i` of each run is the
same golden query. See `EvalHarness.run_detailed`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Minimum detectable effect multiplier: z(1 - alpha/2) + z(power) = 1.96 + 0.84 = 2.80,
# for a two-sided test at alpha = 0.05 with 80 percent power.
_MDE_Z_SUM = 2.8


@dataclass(frozen=True)
class BootstrapResult:
    """`cand - base` averaged over the queries, with its uncertainty.

    `p_value` is a bootstrap p-value, so it is a multiple of `1 / n_resamples` and 0.0
    means that no resample reached zero, i.e. p < `1 / n_resamples`.
    """

    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def _as_scores(name: str, scores: Sequence[float]) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a flat sequence of scores, got {array.ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a NaN or infinite score")
    return array


def paired_bootstrap(
    base: Sequence[float],
    cand: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Compare two runs scored on the same queries: `cand` against `base`.

    Each replicate draws `n` query indices with replacement, once, and applies the
    same indices to both runs, so the pairing survives and the variance of the shared
    query difficulty cancels. The interval is the percentile interval of the replicate
    mean deltas; `p_value` is `2 * min(share of replicates <= 0, share >= 0)`, capped
    at 1.0. The same `seed` gives the same result. Memory is `n_resamples * n` indices.

    Raises ValueError for sequences of unequal length, no queries, a NaN or infinite
    score, `n_resamples < 1`, or a `confidence` outside (0, 1).
    """
    base_scores = _as_scores("base", base)
    cand_scores = _as_scores("cand", cand)
    if len(base_scores) != len(cand_scores):
        raise ValueError(
            f"base and cand must pair query for query: {len(base_scores)} != {len(cand_scores)}"
        )
    n = len(base_scores)
    if n == 0:
        raise ValueError("no queries to compare")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    deltas = cand_scores - base_scores
    indices = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    replicate_means = deltas[indices].mean(axis=1)

    tail = 100.0 * (1.0 - confidence) / 2.0
    ci_low, ci_high = np.percentile(replicate_means, [tail, 100.0 - tail])
    p_value = 2.0 * min(np.mean(replicate_means <= 0.0), np.mean(replicate_means >= 0.0))
    return BootstrapResult(
        # fsum rounds once whatever the order, so a reordering of the queries cannot
        # move the mean in the last digit the way a running float sum can.
        mean_delta=math.fsum(deltas.tolist()) / n,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=min(1.0, float(p_value)),
        n=n,
    )


def mde(sigma_d: float, n: int) -> float:
    """Minimum detectable effect: `2.8 * sigma_d / sqrt(n)`.

    `sigma_d` is the standard deviation of one paired difference and `n` the number of
    pairs averaged. A true mean difference smaller than this is missed more than one
    time in five by a two-sided test at the 5 percent level. The 2.8 is
    `1.96 + 0.84`: the normal quantiles for 5 percent two-sided error and 80 percent power.
    """
    if not sigma_d >= 0.0:
        raise ValueError(f"sigma_d must be >= 0, got {sigma_d}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return _MDE_Z_SUM * sigma_d / math.sqrt(n)


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: which hypotheses to reject, in input order.

    With `m` p-values sorted ascending, the k-th smallest (k from 0) is rejected while
    it is `<= alpha / (m - k)`; the first one that is not, and everything after it, is
    kept. Equal p-values always get the same answer. It controls the chance of any
    false rejection at `alpha`, without assuming the comparisons are independent.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if not all(0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("every p-value must be between 0 and 1")

    m = len(p_values)
    rejected = [False] * m
    for step, position in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        if p_values[position] > alpha / (m - step):
            break
        rejected[position] = True
    return rejected
