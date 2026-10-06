"""Shared builders for the `eval-compare` tests: results documents, files and pre-registrations.

Not a test module (no ``test_`` prefix); imported as ``tests.unit.eval_compare_fixtures``.
The documents are written out by hand here, not with ``trelix.eval.results.build_results``, so
the tests of the reader do not lean on the writer they are meant to hold in place.

Every score the tests use is a dyadic fraction (0.5, 0.5625, k/32 ...), so a mean of equal
deltas is exact and a degenerate bootstrap interval is a single point, which is what lets the
expected verdicts be worked out on paper.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from trelix.eval.compare import Verdict, compare
from trelix.eval.prereg import parse_prereg
from trelix.eval.results import parse_results

SUITE: dict[str, str] = {
    "name": "demo",
    "golden_version": "v1",
    "repo_url": "/srv/demo.git",
    "repo_sha": "a" * 40,
    "license": "MIT",
    "golden_sha256": "b" * 64,
    "plans_sha256": "c" * 64,
}

PIPELINE: dict[str, Any] = {
    "rerank": False,
    "hyde_fallback_enabled": False,
    "multi_query_enabled": False,
    "flare_enabled": False,
    "plans": "replayed",
    "rerank_summary": "disabled",
    "config": {"retrieval": {"top_k": 10, "declaration_boost_enabled": False}},
}

EMBEDDER: dict[str, Any] = {
    "provider": "local",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "library_version": None,
}

PREREG_DEFAULTS: dict[str, str] = {
    "schema_version": "1",
    "experiment_id": "EXP-example",
    "comparison_id": "baseline..flag-on",
    "primary_metric": "ndcg@10",
    "direction": "increase",
    "expected_effect": "0.03",
    "alpha": "0.05",
    "family_size": "1",
    "min_queries": "20",
    "cost_class": "flag",
}


def query_ids(count: int = 20) -> list[str]:
    return [f"q{number:02d}" for number in range(1, count + 1)]


def scores(value: float | Sequence[float], count: int) -> list[float]:
    """`value` repeated `count` times, or the sequence itself (checked to be that long)."""
    if isinstance(value, int | float):
        return [float(value)] * count
    assert len(value) == count, f"{len(value)} scores for {count} queries"
    return [float(item) for item in value]


def record_doc(
    query_id: str,
    ndcg: float,
    recall: float,
    *,
    split: str | None = None,
    kind: str | None = None,
    lang: str | None = None,
    error: str | None = None,
    repo: str = "demo",
) -> dict[str, Any]:
    return {
        "id": query_id,
        "repo": repo,
        "kind": kind,
        "lang": lang,
        "split": split,
        "ndcg": ndcg,
        "recall": recall,
        "mrr": 0.5,
        "top10": ["src/app.py"],
        "error": error,
    }


def aggregate_doc(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The four means of `records` in file order, by the same sums `aggregate_metrics` uses."""
    count = len(records)
    return {
        "mrr": sum(item["mrr"] for item in records) / count,
        "n_queries": float(count),
        "ndcg@10": sum(item["ndcg"] for item in records) / count,
        "recall@10": sum(item["recall"] for item in records) / count,
    }


def records_doc(
    ids: Sequence[str],
    ndcg: float | Sequence[float],
    recall: float | Sequence[float],
    *,
    repo: str,
    splits: Sequence[str | None] | None = None,
    kinds: Sequence[str | None] | None = None,
    langs: Sequence[str | None] | None = None,
    errors: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """One record per id, in the order given; `splits`, `kinds` and `langs` are per record."""
    count = len(ids)
    ndcgs, recalls = scores(ndcg, count), scores(recall, count)
    label, kind, lang = (
        list(values) if values is not None else [None] * count for values in (splits, kinds, langs)
    )
    return [
        record_doc(
            ids[i],
            ndcgs[i],
            recalls[i],
            split=label[i],
            kind=kind[i],
            lang=lang[i],
            error=(errors or {}).get(ids[i]),
            repo=repo,
        )
        for i in range(count)
    ]


def results_doc(
    arm: str,
    ndcg: float | Sequence[float],
    recall: float | Sequence[float] = 0.5,
    *,
    ids: Sequence[str] | None = None,
    splits: Sequence[str | None] | None = None,
    kinds: Sequence[str | None] | None = None,
    langs: Sequence[str | None] | None = None,
    errors: Mapping[str, str] | None = None,
    suite: Mapping[str, str] = SUITE,
    pipeline: Mapping[str, Any] = PIPELINE,
    embedder: Mapping[str, Any] = EMBEDDER,
    trelix_version: str = "3.4.3",
    created_at: str = "2026-10-05T12:00:00+00:00",
) -> dict[str, Any]:
    """A schema 1 document with a consistent aggregate.

    `ids` is the order the records are written in. `splits`, `kinds` and `langs` are per record,
    in that order. `errors` maps a query id to the error message it raised with.
    """
    records = records_doc(
        list(ids) if ids is not None else query_ids(),
        ndcg,
        recall,
        repo=suite["name"],
        splits=splits,
        kinds=kinds,
        langs=langs,
        errors=errors,
    )
    return {
        "schema_version": 1,
        "arm": arm,
        "suite": dict(suite),
        "trelix_version": trelix_version,
        "embedder": dict(embedder),
        "pipeline": json.loads(json.dumps(pipeline)),
        "aggregate": aggregate_doc(records),
        "records": records,
        "run": {"created_at": created_at},
    }


def write_json(tmp_path: Path, name: str, doc: object) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def prereg_text(**overrides: object) -> str:
    """The sample pre-registration as YAML text: values as given, `None` drops a key."""
    fields = {**PREREG_DEFAULTS, **overrides}
    return "".join(f"{key}: {value}\n" for key, value in fields.items() if value is not None)


def write_prereg(tmp_path: Path, name: str = "exp.yaml", **overrides: object) -> str:
    path = tmp_path / name
    path.write_text(prereg_text(**overrides), encoding="utf-8")
    return str(path)


# A phrase from each row's reason line, to say which rows fired.
ROW_PHRASES = {
    "R1": "the candidate run has",
    "R2": "fewer than min_queries",
    "R3": "is confidently worse",
    "R4": "recall@10 is confidently down",
    "R5": "minimum detectable effect",
    "R6": "improvement not demonstrated",
    "R7": "below the hurdle",
    "R8": "recall@10 guard not resolved",
}


def pipeline_doc() -> dict[str, Any]:
    """A fresh copy of `PIPELINE`, safe to change."""
    return copy.deepcopy(PIPELINE)


def alternating(high: float, low: float, count: int = 20) -> list[float]:
    return [high if number % 2 == 0 else low for number in range(count)]


def verdict_of(base: dict[str, Any], cand: dict[str, Any], **prereg_overrides: object) -> Verdict:
    prereg = parse_prereg(yaml.safe_load(prereg_text(**prereg_overrides)))
    return compare(parse_results(base), parse_results(cand), prereg)


def rows(verdict: Verdict) -> list[str]:
    fired = []
    for reason in verdict.reasons:
        matching = [name for name, phrase in ROW_PHRASES.items() if phrase in reason]
        assert len(matching) == 1, f"reason matches rows {matching}: {reason}"
        fired.append(matching[0])
    return fired


def base_doc(ndcg: Any = 0.5, recall: Any = 0.5, **kwargs: Any) -> dict[str, Any]:
    return results_doc("baseline", ndcg, recall, **kwargs)


def cand_doc(ndcg: Any, recall: Any = 0.5, **kwargs: Any) -> dict[str, Any]:
    return results_doc("flag-on", ndcg, recall, **kwargs)
