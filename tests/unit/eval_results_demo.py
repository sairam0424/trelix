"""A hand-written two-query `results.json` and the helpers the results tests share.

Not a test module (no ``test_`` prefix); imported as ``tests.unit.eval_results_demo``.
``DEMO`` is written out by hand: its numbers are the per-query scores of a two-query run
(nDCG@10 1 and 1/log2(3), recall 1 and 1, MRR 1 and 1/2), averaged. Every expected value in the
results tests is a literal.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.unit.eval_compare_fixtures import write_json
from trelix.eval.harness import QueryRecord
from trelix.eval.results import ResultsError, build_results, load_results

DEMO: dict[str, Any] = {
    "aggregate": {
        "mrr": 0.75,
        "n_queries": 2.0,
        "ndcg@10": 0.8154648767857288,
        "recall@10": 1.0,
    },
    "arm": "baseline",
    "embedder": {
        "dimension": 384,
        "library_version": "5.1.0",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "provider": "local",
    },
    "pipeline": {
        "config": {"retrieval": {"declaration_boost_enabled": False, "top_k": 10}},
        "flare_enabled": False,
        "hyde_fallback_enabled": False,
        "multi_query_enabled": False,
        "plans": "replayed",
        "rerank": False,
        "rerank_summary": "disabled",
    },
    "records": [
        {
            "error": None,
            "id": "q0001",
            "kind": None,
            "lang": None,
            "mrr": 1.0,
            "ndcg": 1.0,
            "recall": 1.0,
            "repo": "demo",
            "split": None,
            "top10": ["src/app.py", "src/extra.py"],
        },
        {
            "error": None,
            "id": "q0002",
            "kind": "nl",
            "lang": "python",
            "mrr": 0.5,
            "ndcg": 0.6309297535714575,
            "recall": 1.0,
            "repo": "demo",
            "split": "test",
            "top10": ["src/app.py", "README.md"],
        },
    ],
    "run": {"created_at": "2026-10-05T12:00:00+00:00"},
    "schema_version": 1,
    "suite": {
        "golden_sha256": "b" * 64,
        "golden_version": "v1",
        "license": "MIT",
        "name": "demo",
        "plans_sha256": "c" * 64,
        "repo_sha": "a" * 40,
        "repo_url": "/srv/demo.git",
    },
    "trelix_version": "3.4.3",
}

FROZEN_CLOCK = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)


def demo_records() -> list[QueryRecord]:
    first = QueryRecord(
        "q0001", "demo", None, None, None, 1.0, 1.0, 1.0, ("src/app.py", "src/extra.py"), None
    )
    second = QueryRecord(
        "q0002",
        "demo",
        "nl",
        "python",
        "test",
        0.6309297535714575,
        1.0,
        0.5,
        ("src/app.py", "README.md"),
        None,
    )
    return [first, second]


def demo_build(now: Callable[[], datetime] = lambda: FROZEN_CLOCK) -> dict[str, Any]:
    return build_results(
        arm="baseline",
        suite=DEMO["suite"],
        trelix_version="3.4.3",
        embedder=DEMO["embedder"],
        pipeline=DEMO["pipeline"],
        records=demo_records(),
        now=now,
    )


def refusals_of(tmp_path: Path, doc: object) -> tuple[str, ...]:
    with pytest.raises(ResultsError) as caught:
        load_results(Path(write_json(tmp_path, "r.json", doc)))
    return caught.value.problems


def mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    doc = copy.deepcopy(DEMO)
    mutate(doc)
    return doc
