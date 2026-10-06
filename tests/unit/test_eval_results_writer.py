"""results.json schema 1: the writer (`build_results`, `write_results`) and the round trip.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `write_results` not validating the document first  (test_a_document_the_reader_...)
2. `build_results` moving the timestamp out of `run`  (test_two_runs_differ_only_in_run)
3. `top10` left a tuple in the document                                 (test_build_results_is_...)
4. `write_outcome_file` replaced by a plain `open(path, "w")`  (test_the_file_is_private_...)
5. the aggregate not computed from the records                          (test_build_results_is_...)
6. the JSON check removed, or its `allow_nan=False` / `sort_keys=True` dropped, or an
   exception type no longer caught                         (test_a_config_value_the_reader_...)
7. the records sorted by id instead of kept in the order of the run  (test_the_records_keep_...)
"""

from __future__ import annotations

import json
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.eval_results_demo import (
    DEMO,
    FROZEN_CLOCK,
    demo_build,
    demo_records,
)
from trelix.eval.harness import QueryRecord
from trelix.eval.results import ResultsError, build_results, load_results, write_results


class TestWriter:
    def test_build_results_is_the_documented_schema(self) -> None:
        assert demo_build() == DEMO

    def test_the_default_clock_is_utc_to_the_second(self) -> None:
        doc = build_results(
            arm="baseline",
            suite=DEMO["suite"],
            trelix_version="3.4.3",
            embedder=DEMO["embedder"],
            pipeline=DEMO["pipeline"],
            records=demo_records(),
        )

        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", doc["run"]["created_at"])

    def test_two_runs_differ_only_in_run(self) -> None:
        later = demo_build(lambda: datetime(2026, 10, 6, 1, 2, 3, tzinfo=UTC))
        first = demo_build()

        assert later["run"] == {"created_at": "2026-10-06T01:02:03+00:00"}
        assert first["run"] == {"created_at": "2026-10-05T12:00:00+00:00"}
        rest_first = {k: v for k, v in first.items() if k != "run"}
        rest_later = {k: v for k, v in later.items() if k != "run"}
        assert json.dumps(rest_first, sort_keys=True) == json.dumps(rest_later, sort_keys=True)

    def test_the_file_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"

        assert write_results(str(path), demo_build()) is None
        results = load_results(path)

        assert path.read_text(encoding="utf-8") == json.dumps(DEMO, indent=2, sort_keys=True) + "\n"
        assert results.arm == "baseline"
        assert results.records[0].top10 == ("src/app.py", "src/extra.py")
        assert results.records[1].ndcg == 0.6309297535714575
        assert results.records[1].split == "test"

    # `_failed_record` stores `str(exc)`, so `RuntimeError("\n")` leaves "\n": the file keeps it.
    @pytest.mark.parametrize("message", ["boom", " ", "\n"], ids=["text", "space", "newline"])
    def test_a_run_with_a_failed_query_round_trips(self, tmp_path: Path, message: str) -> None:
        failed = QueryRecord("q0001", "demo", None, None, None, 0.0, 0.0, 0.0, (), message)
        doc = build_results(
            arm="baseline",
            suite=DEMO["suite"],
            trelix_version="3.4.3",
            embedder=DEMO["embedder"],
            pipeline=DEMO["pipeline"],
            records=[failed],
            now=lambda: FROZEN_CLOCK,
        )
        path = tmp_path / "r.json"

        assert write_results(str(path), doc) is None

        loaded = load_results(path)
        assert loaded.records == (failed,)

    def test_the_records_keep_the_order_of_the_run(self, tmp_path: Path) -> None:
        # The ids are out of order on purpose: the file lists the records as the run scored them
        # and `Results.records` are in file order. This pins that order, not the arithmetic: the
        # compensated float `sum()` of Python 3.12 gives the same mean in any order in practice.
        run = [
            QueryRecord("q3", "demo", None, None, None, 0.3, 0.5, 0.5, (), None),
            QueryRecord("q1", "demo", None, None, None, 0.1, 0.5, 0.5, (), None),
            QueryRecord("q2", "demo", None, None, None, 0.2, 0.5, 0.5, (), None),
        ]
        doc = build_results(
            arm="baseline",
            suite=DEMO["suite"],
            trelix_version="3.4.3",
            embedder=DEMO["embedder"],
            pipeline=DEMO["pipeline"],
            records=run,
            now=lambda: FROZEN_CLOCK,
        )
        path = tmp_path / "r.json"

        assert write_results(str(path), doc) is None

        assert [item["id"] for item in doc["records"]] == ["q3", "q1", "q2"]
        assert [record.id for record in load_results(path).records] == ["q3", "q1", "q2"]

    def test_the_file_is_private_and_atomically_replaced(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text("old", encoding="utf-8")

        assert write_results(str(path), demo_build()) is None

        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert sorted(p.name for p in tmp_path.iterdir()) == ["r.json"]

    def test_a_document_the_reader_would_refuse_is_not_written(self, tmp_path: Path) -> None:
        doc = demo_build()
        doc["pipeline"] = {k: v for k, v in doc["pipeline"].items() if k != "config"}
        path = tmp_path / "r.json"

        with pytest.raises(ResultsError) as caught:
            write_results(str(path), doc)

        assert caught.value.problems == ("pipeline: missing key 'config'",)
        assert not path.exists()

    @pytest.mark.parametrize(
        ("value", "json_says"),
        [
            pytest.param(float("nan"), "not JSON compliant", id="nan"),
            pytest.param(float("inf"), "not JSON compliant", id="infinity"),
            pytest.param({1, 2}, "not JSON serializable", id="a-set"),
            pytest.param({1: "a", "b": 2}, "not supported between", id="keys-that-cannot-sort"),
        ],
    )
    def test_a_config_value_the_reader_could_not_parse_is_not_written(
        self, tmp_path: Path, value: object, json_says: str
    ) -> None:
        # `_validate` does not look inside pipeline.config; the JSON encoder (whose wording is
        # Python's, so only a fragment of it is pinned) is what refuses these.
        doc = demo_build()
        doc["pipeline"]["config"] = {"x": value}
        path = tmp_path / "r.json"

        with pytest.raises(ResultsError) as caught:
            write_results(str(path), doc)

        assert len(caught.value.problems) == 1
        assert caught.value.problems[0].startswith("the document cannot be written as JSON: ")
        assert json_says in caught.value.problems[0]
        assert not path.exists()

    def test_an_io_failure_is_returned_not_raised(self, tmp_path: Path) -> None:
        error = write_results(str(tmp_path / "missing-dir" / "r.json"), demo_build())

        assert error is not None
        assert error.startswith("FileNotFoundError: ")

    def test_an_empty_run_cannot_be_built(self) -> None:
        with pytest.raises(ValueError, match="no records to aggregate"):
            build_results(
                arm="baseline",
                suite=DEMO["suite"],
                trelix_version="3.4.3",
                embedder=DEMO["embedder"],
                pipeline=DEMO["pipeline"],
                records=[],
            )
