"""The pre-registration file of one retrieval experiment: what is claimed, before the run.

A pre-registration is a small YAML file naming the hypothesis (`ndcg@10` goes up), the size of
effect expected, how many comparisons are made against the same baseline, and how costly the
change is. `trelix eval-compare` judges two results files against it, and nothing else may
produce a PASS. Every key is required and there are no defaults: a value nobody wrote down is a
value nobody committed to. Unknown keys, duplicate keys, YAML anchors and aliases are refused,
and the file is parsed with PyYAML's safe loader only.

The hurdle is a fixed table by `cost_class`, so a file cannot pick its own.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trelix.eval._loading import ProblemsError, clip, read_text_capped, show
from trelix.eval.results import ARM_PATTERN

PREREG_SCHEMA_VERSION = 1
MAX_PREREG_BYTES = 64 * 1024

# Smallest improvement in nDCG@10 worth shipping, by what the change costs to run: a flag flip,
# a change that needs a re-index, or a heavy model or service. Looked up at judging time.
HURDLES = {"flag": 0.01, "index": 0.02, "heavy": 0.03}

PRIMARY_METRIC = "ndcg@10"
DIRECTION = "increase"
# `alpha` may be stricter than 95 percent confidence, never looser.
ALPHA_MAX = 0.05
# Below this a tail of the 10,000-replicate bootstrap holds fewer than 5 replicates, so an
# interval end is one of the most extreme replicates and `1 - alpha / family_size` can round
# to 1.0, which `paired_bootstrap` rejects. With `ALPHA_MAX` it bounds `family_size` at 50.
MIN_ALPHA_EFFECTIVE = 0.001
MAX_FAMILY_SIZE = 50
# The fewest queries a decision may rest on; equals the per-kind minimum of `eval-validate`.
MIN_QUERIES_FLOOR = 20

PREREG_KEYS = (
    "schema_version",
    "experiment_id",
    "comparison_id",
    "primary_metric",
    "direction",
    "expected_effect",
    "alpha",
    "family_size",
    "min_queries",
    "cost_class",
)
_EXPERIMENT_ID_RE = re.compile(r"EXP-[A-Za-z0-9][A-Za-z0-9._-]{0,62}")
_COMPARISON_ID_RE = re.compile(rf"({ARM_PATTERN})\.\.({ARM_PATTERN})")


class PreregError(ProblemsError):
    """A pre-registration that cannot be used; `problems` lists every reason found."""


@dataclass(frozen=True)
class Prereg:
    experiment_id: str
    comparison_id: str
    primary_metric: str
    direction: str
    expected_effect: float
    alpha: float
    family_size: int
    min_queries: int
    cost_class: str

    @property
    def hurdle(self) -> float:
        """The smallest nDCG@10 gain that counts for this `cost_class`."""
        return HURDLES[self.cost_class]

    @property
    def alpha_effective(self) -> float:
        """`alpha / family_size`: the Bonferroni share of `alpha` each comparison gets."""
        return self.alpha / self.family_size


class _StrictLoader(yaml.SafeLoader):  # type: ignore[misc]
    """`SafeLoader` that refuses a duplicate key and any anchor or alias.

    PyYAML keeps the last of two equal keys silently, so a second `alpha:` would override the
    first without a word. Anchors and aliases have no use in a ten-line file.
    """

    def compose_node(self, parent: object, index: object) -> object:
        if getattr(self.peek_event(), "anchor", None) is not None:
            raise yaml.YAMLError("anchors and aliases are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                continue  # the base class refuses an unhashable key with its own message
            if key in seen:
                raise yaml.YAMLError(f"duplicate key {show(key)}")
            seen.add(key)
        mapping: dict[Any, Any] = super().construct_mapping(node, deep=deep)
        return mapping


def load_prereg(path: Path) -> Prereg:
    """Read and validate a pre-registration file. Raises PreregError, listing every problem."""
    try:
        text = read_text_capped(path, MAX_PREREG_BYTES)
    except (OSError, ValueError) as exc:
        raise PreregError([f"cannot read the pre-registration file: {exc}"]) from exc
    loader = _StrictLoader(text)
    try:
        raw = loader.get_single_data()
    except (yaml.YAMLError, ValueError) as exc:  # ValueError: a date PyYAML cannot build
        raise PreregError([f"not valid YAML: {clip(str(exc), 200)}"]) from exc
    finally:
        loader.dispose()
    return parse_prereg(raw)


def parse_prereg(raw: object) -> Prereg:
    """Validate an already-parsed document and return it as `Prereg`."""
    if not isinstance(raw, dict):
        raise PreregError(["the top level must be a mapping with the ten pre-registration keys"])
    problems = _version_problems(raw) + _key_problems(raw) + _identity_problems(raw)
    problems += _number_problems(raw)
    if problems:
        raise PreregError(problems)
    return Prereg(
        experiment_id=raw["experiment_id"],
        comparison_id=raw["comparison_id"],
        primary_metric=raw["primary_metric"],
        direction=raw["direction"],
        expected_effect=float(raw["expected_effect"]),
        alpha=float(raw["alpha"]),
        family_size=raw["family_size"],
        min_queries=raw["min_queries"],
        cost_class=raw["cost_class"],
    )


def _real(value: object, high: float) -> float | None:
    """`value` as a float when it is a number above 0 and at most `high`, else None.

    Written `0 < x <= high` so NaN and infinity fail like any out-of-range number; a bool (a
    subclass of int) is refused by the type test first. The range is checked before `float()`
    so an absurdly large integer is out of range rather than an OverflowError.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if 0 < value <= high else None


def _whole(value: object, low: int, high: int | None = None) -> int | None:
    """`value` when it is an integer (not a bool) from `low` to `high`, else None."""
    if type(value) is not int or value < low or (high is not None and value > high):
        return None
    return value


def _version_problems(raw: Mapping[str, Any]) -> list[str]:
    version = raw.get("schema_version")
    if (
        "schema_version" in raw
        and _whole(version, PREREG_SCHEMA_VERSION, PREREG_SCHEMA_VERSION) is None
    ):
        return [
            f"unsupported schema_version {show(version)}: "
            f"this reader understands {PREREG_SCHEMA_VERSION}"
        ]
    return []


def _key_problems(raw: Mapping[str, Any]) -> list[str]:
    unsupported = [f"unsupported key {show(key)}" for key in raw if key not in PREREG_KEYS]
    missing = [f"missing key {key!r}" for key in PREREG_KEYS if key not in raw]
    return unsupported + missing


def _identity_problems(raw: Mapping[str, Any]) -> list[str]:
    """experiment_id, comparison_id, primary_metric, direction and cost_class."""
    problems: list[str] = []
    experiment_id = raw.get("experiment_id")
    if "experiment_id" in raw and not (
        isinstance(experiment_id, str) and _EXPERIMENT_ID_RE.fullmatch(experiment_id)
    ):
        problems.append(
            "experiment_id must look like EXP-<id>, using letters, digits and . _ - "
            f"(got {show(experiment_id)})"
        )
    comparison_id = raw.get("comparison_id")
    if "comparison_id" in raw:
        match = (
            _COMPARISON_ID_RE.fullmatch(comparison_id) if isinstance(comparison_id, str) else None
        )
        if match is None or match.group(1) == match.group(2):
            problems.append(
                "comparison_id must be <baseline arm>..<candidate arm>, two different arm names "
                f"(got {show(comparison_id)})"
            )
    for key, allowed in (("primary_metric", PRIMARY_METRIC), ("direction", DIRECTION)):
        if key in raw and raw[key] != allowed:
            problems.append(f"{key} must be {allowed!r} (got {show(raw[key])})")
    cost_class = raw.get("cost_class")
    # A list or mapping is unhashable: looking it up in HURDLES would raise TypeError out of the
    # loader instead of listing this problem with the others.
    if "cost_class" in raw and not (isinstance(cost_class, str) and cost_class in HURDLES):
        problems.append(f"cost_class must be one of {', '.join(HURDLES)} (got {show(cost_class)})")
    return problems


def _number_problems(raw: Mapping[str, Any]) -> list[str]:
    """expected_effect, alpha, family_size and min_queries."""
    effect = _real(raw.get("expected_effect"), 1.0)
    alpha = _real(raw.get("alpha"), ALPHA_MAX)
    family = _whole(raw.get("family_size"), 1, MAX_FAMILY_SIZE)
    minimum = _whole(raw.get("min_queries"), MIN_QUERIES_FLOOR)
    problems: list[str] = []
    if "expected_effect" in raw and effect is None:
        problems.append(
            "expected_effect must be a number above 0 and at most 1 "
            f"(got {show(raw['expected_effect'])})"
        )
    if "alpha" in raw and alpha is None:
        problems.append(
            f"alpha must be a number above 0 and at most {ALPHA_MAX} (got {show(raw['alpha'])})"
        )
    if "family_size" in raw and family is None:
        problems.append(
            f"family_size must be an integer from 1 to {MAX_FAMILY_SIZE} "
            f"(got {show(raw['family_size'])})"
        )
    if alpha is not None and family is not None and alpha / family < MIN_ALPHA_EFFECTIVE:
        problems.append(
            f"alpha / family_size must be at least {MIN_ALPHA_EFFECTIVE} (got {alpha / family}): "
            "a tail of the bootstrap would hold fewer than 5 replicates"
        )
    if "min_queries" in raw and minimum is None:
        problems.append(
            f"min_queries must be an integer of at least {MIN_QUERIES_FLOOR} "
            f"(got {show(raw['min_queries'])})"
        )
    return problems
