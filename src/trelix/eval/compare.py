"""`trelix eval-compare`: judge a candidate run against a baseline under a pre-registration.

This is the only producer of PASS in the retrieval-eval program. It reads two results files
(`trelix.eval.results`) and one pre-registration (`trelix.eval.prereg`) and needs no git, no
index and no embedder. The verdict is a pure function of those three files: the queries are
paired by `id` and fed to the bootstrap in sorted-id order, with a fixed seed.

The decision set `D` is taken from the BASE file's labels, so a candidate file cannot choose
its own: with no `split` on any base record `D` is every record, with one on every base record
`D` is the `test` records (dev queries never enter a decision), and a mixture is refused. The
candidate must label every query the same way. `n = |D|`, and `min_queries` is compared with it.

Rows are tried in this order, and rows 1 and 2 stop the evaluation (`h` is the hurdle of the
pre-registration's `cost_class`, `[L, H]` the interval of the nDCG@10 delta at confidence
`1 - alpha / family_size`, `[Lr, Hr]` the same for recall@10, `G` is -0.02):

    1  the candidate has a record with an error                      FAIL
    2  n < min_queries                                               INCONCLUSIVE
    3  H < 0           confidently worse                              FAIL
    4  Hr < G          recall confidently down by more than 0.02      FAIL
    5  expected_effect < MDE, MDE = 2.8 * sigma_d / sqrt(n)           INCONCLUSIVE
    6  L <= 0          improvement not demonstrated                   INCONCLUSIVE
    7  delta < h       below the hurdle                               INCONCLUSIVE
    8  Lr < G          recall guard not resolved                      INCONCLUSIVE
    9  none of the above                                              PASS

If any of rows 3 and 4 matches the verdict is FAIL, else if any of rows 5 to 8 matches it is
INCONCLUSIVE, and every matching row of the winning class is printed as a `reason:` line.
`sigma_d` is the sample standard deviation (n - 1) of the paired nDCG@10 deltas on `D`.
Holm is not applied: `family_size` widens every interval to `1 - alpha / family_size`
(Bonferroni), and `stats.mde` uses the fixed 2.8, so row 5 does not tighten with it.

Anything that makes the comparison invalid rather than unfavourable is REFUSED, with every
reason listed: a file that is unreadable or malformed, runs of different suites, queries or
labels, a run that was not frozen-plan and rerank-off, a comparison_id that does not name the
two arms, or a baseline with an error (a bad baseline is a bad instrument).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trelix.eval._loading import ProblemsError, clip
from trelix.eval.harness import QueryRecord
from trelix.eval.prereg import Prereg, load_prereg
from trelix.eval.results import FROZEN_FLAGS, SUITE_KEYS, Results, load_results
from trelix.eval.stats import BootstrapResult, mde, paired_bootstrap

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
# Recall@10 may not fall by more than this (confidently) without the candidate failing.
RECALL_GUARD = -0.02
EXIT_CODES = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2, "REFUSED": 3}

_LISTED_IDS = 5
_MAX_NOTE_LEAVES = 20
_ABSENT = object()  # a leaf one side does not have; never equal to a value read from a file


@dataclass(frozen=True)
class Verdict:
    """The outcome, and what to print.

    `lines` are the stdout lines and always end with `verdict: <outcome>`. For FAIL and
    INCONCLUSIVE they include one `reason:` line per entry of `reasons`. For REFUSED the
    `reasons` are the refusals, printed on stderr as `refused: ...`, and `lines` is only the
    verdict line.
    """

    outcome: str
    reasons: tuple[str, ...]
    lines: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.outcome]


def _refused(reasons: Sequence[str]) -> Verdict:
    return Verdict("REFUSED", tuple(reasons), ("verdict: REFUSED",))


def _decided(outcome: str, lines: Sequence[str], reasons: Sequence[str]) -> Verdict:
    reason_lines = [f"reason: {reason}" for reason in reasons]
    return Verdict(outcome, tuple(reasons), (*lines, *reason_lines, f"verdict: {outcome}"))


# ---------------------------------------------------------------------------
# files to a verdict
# ---------------------------------------------------------------------------


def compare_files(base_path: str, cand_path: str, prereg_path: str | None) -> Verdict:
    """Load the three files and judge. Never raises: every failure is a REFUSED verdict.

    An unexpected exception is reported as `internal error` and refused too, never left to
    end the process with a traceback and exit code 1, which would claim the candidate failed.
    """
    try:
        return _compare_files(base_path, cand_path, prereg_path)
    except Exception as exc:  # deliberately broad: the docstring says why
        return _refused([f"internal error: {type(exc).__name__}: {exc}"])


def _compare_files(base_path: str, cand_path: str, prereg_path: str | None) -> Verdict:
    if prereg_path is None:
        return _refused(["prereg: --prereg is required: the hypothesis is fixed before the run"])
    base, base_problems = _try_load(load_results, "base", base_path)
    cand, cand_problems = _try_load(load_results, "cand", cand_path)
    prereg, prereg_problems = _try_load(load_prereg, "prereg", prereg_path)
    problems = [*base_problems, *cand_problems, *prereg_problems]
    if base is None or cand is None or prereg is None:
        return _refused(problems)
    return compare(base, cand, prereg)


def _try_load[T](load: Callable[[Path], T], label: str, path: str) -> tuple[T | None, list[str]]:
    try:
        return load(Path(path)), []
    except ProblemsError as exc:
        return None, [f"{label} {path}: {problem}" for problem in exc.problems]


def compare(base: Results, cand: Results, prereg: Prereg) -> Verdict:
    """REFUSED when the two runs cannot be compared under `prereg`, else `judge`'s verdict."""
    problems = refusals(base, cand, prereg)
    return _refused(problems) if problems else judge(base, cand, prereg)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def refusals(base: Results, cand: Results, prereg: Prereg) -> list[str]:
    """Every reason the two runs cannot be compared; `[]` when `judge` may run."""
    return [
        *_identity_refusals(base, cand),
        *_pipeline_refusals(base, cand),
        *_query_set_refusals(base, cand),
        *_binding_refusals(base, cand, prereg),
        *_base_error_refusals(base),
    ]


def _identity_refusals(base: Results, cand: Results) -> list[str]:
    differing = [key for key in SUITE_KEYS if getattr(base.suite, key) != getattr(cand.suite, key)]
    return [f"suite identity differs in: {', '.join(differing)}"] if differing else []


def _pipeline_refusals(base: Results, cand: Results) -> list[str]:
    problems: list[str] = []
    for label, run in (("base", base), ("candidate", cand)):
        broken = [f"{flag} is true" for flag in FROZEN_FLAGS if getattr(run.pipeline, flag)]
        if run.pipeline.plans != "replayed":
            broken.append(f"plans is {clip(repr(run.pipeline.plans))}, not 'replayed'")
        if broken:
            problems.append(f"{label} is not a frozen-plan, rerank-off run: {', '.join(broken)}")
    return problems


def _listed(ids: Sequence[str]) -> str:
    """The first five ids, clipped, and how many more there are."""
    if not ids:
        return "none"
    shown = ", ".join(clip(item) for item in ids[:_LISTED_IDS])
    return shown + (f" (and {len(ids) - _LISTED_IDS} more)" if len(ids) > _LISTED_IDS else "")


def _split_scope(records: Sequence[QueryRecord]) -> str:
    """`all` (no record has a split), `test` (every record has one) or `mixed`."""
    labelled = sum(1 for record in records if record.split is not None)
    if labelled == 0:
        return "all"
    return "test" if labelled == len(records) else "mixed"


def _query_set_refusals(base: Results, cand: Results) -> list[str]:
    by_base = {record.id: record for record in base.records}
    by_cand = {record.id: record for record in cand.records}
    problems: list[str] = []
    only_base, only_cand = (
        sorted(by_base.keys() - by_cand.keys()),
        sorted(by_cand.keys() - by_base.keys()),
    )
    if only_base or only_cand:
        problems.append(
            f"the runs cover different queries: only in base: {_listed(only_base)}; "
            f"only in cand: {_listed(only_cand)}"
        )
    if _split_scope(base.records) == "mixed":
        problems.append(
            f"split labels are mixed in the base run: "
            f"{sum(1 for r in base.records if r.split is not None)} of {len(base.records)} "
            "records have one, so there is no decision set"
        )
    differing = sorted(
        key
        for key in by_base.keys() & by_cand.keys()
        if _labels(by_base[key]) != _labels(by_cand[key])
    )
    if differing:
        problems.append(f"the runs label queries differently: {_listed(differing)}")
    return problems


def _labels(record: QueryRecord) -> tuple[str | None, str | None, str | None]:
    return (record.split, record.kind, record.lang)


def _binding_refusals(base: Results, cand: Results, prereg: Prereg) -> list[str]:
    arms = f"{base.arm}..{cand.arm}"
    if prereg.comparison_id == arms:
        return []
    return [
        f"comparison_id {clip(prereg.comparison_id)} does not match the arms of the files given "
        f"as base and cand ({clip(arms)}): check the order of the arguments"
    ]


def _base_error_refusals(base: Results) -> list[str]:
    failed = [record for record in base.records if record.error is not None]
    if not failed:
        return []
    return [f"base run has {_error_summary(failed)}: a baseline that raised is not an instrument"]


def _error_summary(failed: Sequence[QueryRecord]) -> str:
    first = failed[0]
    return (
        f"{len(failed)} record(s) with an error (first: {clip(first.id)}: {clip(str(first.error))})"
    )


# ---------------------------------------------------------------------------
# the decision rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Numbers:
    """What the rule rows read: both metrics on the decision set, and the noise of the deltas."""

    n: int
    base_ndcg: float
    cand_ndcg: float
    primary: BootstrapResult
    base_recall: float
    cand_recall: float
    guard: BootstrapResult
    sigma_d: float
    minimum_effect: float


def judge(base: Results, cand: Results, prereg: Prereg) -> Verdict:
    """The verdict for two comparable runs. Precondition: `refusals(...)` is empty."""
    scope = _split_scope(base.records)
    ids = sorted(r.id for r in base.records if scope == "all" or r.split == "test")
    head = [*_header_lines(base, cand, prereg, len(ids), scope), *_notes(base, cand)]
    failed = [record for record in cand.records if record.error is not None]
    if failed:
        return _decided("FAIL", head, [f"the candidate run has {_error_summary(failed)}"])
    if len(ids) < prereg.min_queries:
        return _decided(
            "INCONCLUSIVE",
            head,
            [
                f"the decision set has {len(ids)} queries, "
                f"fewer than min_queries {prereg.min_queries}"
            ],
        )
    numbers = _measure(base, cand, ids, prereg)
    worse, unproven = _rows(numbers, prereg)
    lines = [*head, *_stat_lines(numbers, prereg)]
    if worse:
        return _decided("FAIL", lines, worse)
    if unproven:
        return _decided("INCONCLUSIVE", lines, unproven)
    return _decided("PASS", lines, [])


def _measure(base: Results, cand: Results, ids: Sequence[str], prereg: Prereg) -> _Numbers:
    by_base = {record.id: record for record in base.records}
    by_cand = {record.id: record for record in cand.records}
    before = [by_base[key] for key in ids]
    after = [by_cand[key] for key in ids]
    confidence = 1.0 - prereg.alpha_effective
    ndcg_before, ndcg_after = [r.ndcg for r in before], [r.ndcg for r in after]
    recall_before, recall_after = [r.recall for r in before], [r.recall for r in after]
    primary = paired_bootstrap(
        ndcg_before,
        ndcg_after,
        n_resamples=BOOTSTRAP_RESAMPLES,
        confidence=confidence,
        seed=BOOTSTRAP_SEED,
    )
    guard = paired_bootstrap(
        recall_before,
        recall_after,
        n_resamples=BOOTSTRAP_RESAMPLES,
        confidence=confidence,
        seed=BOOTSTRAP_SEED,
    )
    # Sample standard deviation (n - 1): `min_queries` is at least 20, so n >= 2 holds here.
    sigma_d = statistics.stdev(x - y for x, y in zip(ndcg_after, ndcg_before, strict=True))
    return _Numbers(
        n=len(ids),
        base_ndcg=statistics.fmean(ndcg_before),
        cand_ndcg=statistics.fmean(ndcg_after),
        primary=primary,
        base_recall=statistics.fmean(recall_before),
        cand_recall=statistics.fmean(recall_after),
        guard=guard,
        sigma_d=sigma_d,
        minimum_effect=mde(sigma_d, len(ids)),
    )


def _rows(x: _Numbers, prereg: Prereg) -> tuple[list[str], list[str]]:
    """The rows that match, as (FAIL-class reasons, INCONCLUSIVE-class reasons)."""
    low, high, delta = x.primary.ci_low, x.primary.ci_high, x.primary.mean_delta
    worse: list[str] = []
    unproven: list[str] = []
    if high < 0:
        worse.append(
            f"ndcg@10 is confidently worse: the interval's upper end {_signed(high)} is below 0"
        )
    if x.guard.ci_high < RECALL_GUARD:
        worse.append(
            f"recall@10 is confidently down by more than {-RECALL_GUARD}: "
            f"the interval's upper end {_signed(x.guard.ci_high)} is below {_signed(RECALL_GUARD)}"
        )
    if prereg.expected_effect < x.minimum_effect:
        unproven.append(
            f"expected_effect {_plain(prereg.expected_effect)} is below the minimum detectable "
            f"effect {_plain(x.minimum_effect)} (exploratory: cannot change a default)"
        )
    if low <= 0:
        unproven.append(
            "ndcg@10 improvement not demonstrated: "
            f"the interval's lower end {_signed(low)} is not above 0"
        )
    if delta < prereg.hurdle:
        unproven.append(
            f"ndcg@10 delta {_signed(delta)} is below the hurdle {_plain(prereg.hurdle)} "
            f"(cost_class {prereg.cost_class})"
        )
    if x.guard.ci_low < RECALL_GUARD:
        unproven.append(
            f"recall@10 guard not resolved: the interval's lower end {_signed(x.guard.ci_low)} "
            f"is below {_signed(RECALL_GUARD)}"
        )
    return worse, unproven


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _plain(value: float) -> str:
    return f"{value:.4f}"


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _header_lines(base: Results, cand: Results, prereg: Prereg, n: int, scope: str) -> list[str]:
    suite = base.suite
    which = "no split labels: all queries" if scope == "all" else "split labels: test queries only"
    return [
        f"comparison: {prereg.comparison_id}",
        f"experiment: {prereg.experiment_id}",
        f"suite: {suite.name} golden_version {suite.golden_version} repo_sha {suite.repo_sha}",
        f"runs: base {clip(base.created_at, 40)} cand {clip(cand.created_at, 40)}",
        f"queries: {n} in the decision set ({which}), min_queries {prereg.min_queries}",
    ]


def _stat_lines(x: _Numbers, prereg: Prereg) -> list[str]:
    percent = f"{(1.0 - prereg.alpha_effective) * 100:.2f}%"
    return [
        _metric_line("ndcg@10", x.base_ndcg, x.cand_ndcg, x.primary, percent),
        _metric_line("recall@10", x.base_recall, x.cand_recall, x.guard, percent),
        f"mde: {_plain(x.minimum_effect)} (sigma_d {_plain(x.sigma_d)}, n {x.n}); "
        f"expected_effect {_plain(prereg.expected_effect)}",
        f"hurdle: {_plain(prereg.hurdle)} (cost_class {prereg.cost_class})",
    ]


def _metric_line(
    name: str, before: float, after: float, result: BootstrapResult, percent: str
) -> str:
    return (
        f"{name}: base {_plain(before)} cand {_plain(after)} delta {_signed(result.mean_delta)} "
        f"ci [{_signed(result.ci_low)}, {_signed(result.ci_high)}] at {percent} confidence"
    )


def _notes(base: Results, cand: Results) -> list[str]:
    """Differences that are reported, not refused: a reviewer sees what the arms changed."""
    notes: list[str] = []
    if base.trelix_version != cand.trelix_version:
        notes.append(
            f"note: trelix_version differs: base {clip(base.trelix_version)}, "
            f"cand {clip(cand.trelix_version)}"
        )
    notes += [
        f"note: embedder differs: {difference}"
        for difference in _differences(asdict(base.embedder), asdict(cand.embedder))
    ]
    config = _differences(base.pipeline.config, cand.pipeline.config)
    if not config:
        return [*notes, "note: pipeline.config identical"]
    notes += [f"note: pipeline.config differs: {d}" for d in config[:_MAX_NOTE_LEAVES]]
    if len(config) > _MAX_NOTE_LEAVES:
        notes.append(
            f"note: pipeline.config differs in {len(config) - _MAX_NOTE_LEAVES} more values"
        )
    return notes


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    """`value` flattened to dotted paths; lists, scalars and an empty nested object are leaves."""
    if not isinstance(value, dict) or (not value and prefix):
        return {prefix: value}
    leaves: dict[str, object] = {}
    for key, item in value.items():
        leaves.update(_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
    return leaves


def _render(value: object) -> str:
    return "absent" if value is _ABSENT else json.dumps(value, sort_keys=True)


def _differences(base: Mapping[str, Any], cand: Mapping[str, Any]) -> list[str]:
    """`path (base X, cand Y)` for each leaf whose value differs, sorted by path."""
    before, after = _leaves(base), _leaves(cand)
    found: list[str] = []
    for path in sorted(before.keys() | after.keys()):
        left = _render(before.get(path, _ABSENT))
        right = _render(after.get(path, _ABSENT))
        if left != right:
            found.append(f"{clip(path)} (base {clip(left)}, cand {clip(right)})")
    return found
