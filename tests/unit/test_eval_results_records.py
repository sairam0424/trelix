"""results.json schema 1: the records, the aggregate and the file-level refusals.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. the aggregate recomputation removed, or `!=` loosened to a tolerance  (TestAggregate)
   or `_is_number` accepting a bool (`True == 1.0`)                      (test_a_bool_aggregate_...)
   or any of its four keys left out of the comparison                 (test_every_aggregate_key_...)
2. the duplicate-id check removed                                       (test_a_duplicate_id_...)
3. the `repo == suite.name` check removed  (test_a_record_of_another_...)
4. the `split` vocabulary widened                                       (test_a_split_is_null_...)
5. `parse_constant` removed (NaN accepted)                              (test_nan_and_infinity_...)
6. `RecursionError` dropped from the JSON except  (test_deeply_nested_json_...)
7. the size cap removed                                                 (test_an_oversize_file_...)
8. the per-query hint removed                                           (test_a_per_query_file_...)
9. a blank `kind` or `lang` accepted (`_is_text` -> `isinstance(.., str)`)  (test_a_label_is_...)
10. the size cap no longer the documented 32 MiB                        (test_the_cap_is_32_mib)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import trelix.eval.results as results_module
from tests.unit.eval_compare_fixtures import results_doc, write_json
from tests.unit.eval_results_demo import (
    DEMO,
    mutated,
    refusals_of,
)
from trelix.eval.results import ResultsError, load_results


class TestRecords:
    @pytest.mark.parametrize("records", [[], None, {}, "q0001"])
    def test_records_must_be_a_non_empty_list(self, tmp_path: Path, records: object) -> None:
        doc = mutated(lambda d: d.update(records=records))

        assert refusals_of(tmp_path, doc) == ("records: must be a non-empty list",)

    def test_a_record_must_be_an_object(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["records"].append("q0003"))

        assert refusals_of(tmp_path, doc) == ("record #2: must be an object",)

    def test_a_record_key_is_missing_or_unsupported(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: (d["records"][0].pop("top10"), d["records"][0].update(score=1)))

        assert refusals_of(tmp_path, doc) == (
            "record q0001: unsupported key 'score'",
            "record q0001: missing key 'top10'",
        )

    @pytest.mark.parametrize("bad", ["", " ", 5, None])
    def test_a_bad_id_is_reported_by_position(self, tmp_path: Path, bad: object) -> None:
        doc = mutated(lambda d: d["records"][0].update(id=bad))

        assert refusals_of(tmp_path, doc) == ("record #0: 'id' must be a non-empty string",)

    def test_a_record_of_another_repo_is_refused(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["records"][1].update(repo="other"))

        assert refusals_of(tmp_path, doc) == (
            "record q0002: 'repo' is 'other' but the suite is named 'demo'",
        )

    @pytest.mark.parametrize("key", ["kind", "lang"])
    @pytest.mark.parametrize("label", [5, "", "  "])
    def test_a_label_is_text_or_null(self, tmp_path: Path, key: str, label: object) -> None:
        doc = mutated(lambda d: d["records"][0].update({key: label}))

        assert refusals_of(tmp_path, doc) == (
            f"record q0001: {key!r} must be a non-empty string or null",
        )

    @pytest.mark.parametrize("split", ["train", "", 1, True])
    def test_a_split_is_null_dev_or_test(self, tmp_path: Path, split: object) -> None:
        doc = mutated(lambda d: d["records"][0].update(split=split))

        problems = refusals_of(tmp_path, doc)

        assert len(problems) == 1
        assert problems[0].startswith("record q0001: 'split' must be null, 'dev' or 'test' (got ")

    @pytest.mark.parametrize("key", ["ndcg", "recall", "mrr"])
    @pytest.mark.parametrize("score", [1.0000001, -0.0000001, True, "0.5", None, [1]])
    def test_a_score_is_a_number_from_zero_to_one(
        self, tmp_path: Path, key: str, score: object
    ) -> None:
        doc = mutated(lambda d: d["records"][0].update({key: score}))

        problems = refusals_of(tmp_path, doc)

        assert len(problems) == 1
        assert problems[0].startswith(f"record q0001: {key!r} must be a number from 0 to 1 (got ")

    @pytest.mark.parametrize("top10", ["src/a.py", [1], [["a"]], None])
    def test_top10_is_a_list_of_strings(self, tmp_path: Path, top10: object) -> None:
        doc = mutated(lambda d: d["records"][0].update(top10=top10))

        assert refusals_of(tmp_path, doc) == (
            "record q0001: 'top10' must be a list of at most 10 strings",
        )

    @pytest.mark.parametrize("error", ["", 5, []])
    def test_an_error_is_null_or_text(self, tmp_path: Path, error: object) -> None:
        doc = mutated(lambda d: d["records"][0].update(error=error))

        assert refusals_of(tmp_path, doc) == (
            "record q0001: 'error' must be null or a non-empty string",
        )

    @pytest.mark.parametrize("error", [" ", "\n"], ids=["space", "newline"])
    def test_an_error_of_only_whitespace_is_still_an_error(
        self, tmp_path: Path, error: str
    ) -> None:
        # `RuntimeError("\n")` makes `error` "\n" (`_failed_record` only replaces ""), so the
        # reader must take what the harness can write.
        doc = mutated(lambda d: d["records"][0].update(error=error))

        results = load_results(Path(write_json(tmp_path, "r.json", doc)))

        assert results.records[0].error == error

    def test_a_duplicate_id_is_refused(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["records"][1].update(id="q0001"))

        assert refusals_of(tmp_path, doc) == ("duplicate query id 'q0001' (2 records)",)

    def test_a_query_that_raised_is_a_valid_record(self, tmp_path: Path) -> None:
        doc = results_doc(
            "baseline", [0.0, 0.5], [0.0, 0.5], ids=["q1", "q2"], errors={"q1": "boom"}
        )

        results = load_results(Path(write_json(tmp_path, "r.json", doc)))

        assert results.records[0].error == "boom"


class TestAggregate:
    def test_an_aggregate_off_by_one_part_in_a_trillion_is_refused(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["aggregate"].update({"ndcg@10": 0.8154648767867289}))

        assert refusals_of(tmp_path, doc) == (
            "aggregate does not match records: ndcg@10 is 0.8154648767867289 in the file "
            "but 0.8154648767857288 from its records",
        )

    @pytest.mark.parametrize(
        ("key", "wrong", "problem"),
        [
            ("ndcg@10", 0.5, "ndcg@10 is 0.5 in the file but 0.8154648767857288 from its records"),
            ("recall@10", 0.5, "recall@10 is 0.5 in the file but 1.0 from its records"),
            ("mrr", 0.5, "mrr is 0.5 in the file but 0.75 from its records"),
            ("n_queries", 3.0, "n_queries is 3.0 in the file but 2.0 from its records"),
        ],
    )
    def test_every_aggregate_key_is_held_to_its_records(
        self, tmp_path: Path, key: str, wrong: float, problem: str
    ) -> None:
        doc = mutated(lambda d: d["aggregate"].update({key: wrong}))

        assert refusals_of(tmp_path, doc) == (f"aggregate does not match records: {problem}",)

    def test_a_bool_aggregate_is_not_the_number_it_equals(self, tmp_path: Path) -> None:
        # recall@10 is 1.0 and True == 1.0, so only the type test can tell them apart.
        doc = mutated(lambda d: d["aggregate"].update({"recall@10": True}))

        problems = refusals_of(tmp_path, doc)

        assert problems == (
            "aggregate does not match records: recall@10 is True in the file "
            "but 1.0 from its records",
        )

    def test_an_aggregate_key_is_missing_or_unsupported(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: (d["aggregate"].pop("mrr"), d["aggregate"].update(f1=1.0)))

        assert refusals_of(tmp_path, doc) == (
            "aggregate: unsupported key 'f1'",
            "aggregate: missing key 'mrr'",
        )

    def test_an_aggregate_that_is_not_an_object_is_refused(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d.update(aggregate=[0.5]))

        assert refusals_of(tmp_path, doc) == ("aggregate: must be an object",)


class TestTheFile:
    def test_a_per_query_file_gets_a_pointed_message(self, tmp_path: Path) -> None:
        per_query = {
            "schema_version": 1,
            "records": DEMO["records"],
            "aggregate": DEMO["aggregate"],
        }

        problems = refusals_of(tmp_path, per_query)

        assert len(problems) == 1
        assert problems[0].startswith('not a results.json: it has no "suite" key. ')
        assert "`trelix eval --per-query-out`" in problems[0]

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_nan_and_infinity_are_not_json_here(self, tmp_path: Path, constant: str) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps(DEMO).replace("0.75", constant, 1), encoding="utf-8")

        with pytest.raises(ResultsError) as caught:
            load_results(path)

        assert caught.value.problems == (
            f"not a results.json: not valid JSON (the constant {constant} is not allowed)",
        )

    @pytest.mark.parametrize("text", ["", "{", "not json", "[]", "null", "42"])
    def test_text_that_is_not_an_object_is_refused(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "r.json"
        path.write_text(text, encoding="utf-8")

        with pytest.raises(ResultsError) as caught:
            load_results(path)

        assert len(caught.value.problems) == 1
        assert caught.value.problems[0].startswith("not a results.json: ")

    def test_deeply_nested_json_is_a_refusal_not_a_recursion_error(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text("[" * 200_000, encoding="utf-8")

        with pytest.raises(ResultsError) as caught:
            load_results(path)

        assert caught.value.problems[0].startswith("not a results.json: not valid JSON (")

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ResultsError) as caught:
            load_results(tmp_path / "absent.json")

        assert caught.value.problems[0].startswith("not a results.json: cannot read the file: ")

    def test_a_non_utf8_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_bytes(b'{"a": "\xff"}')

        with pytest.raises(ResultsError) as caught:
            load_results(path)

        assert caught.value.problems[0].startswith("not a results.json: cannot read the file: ")

    def test_the_cap_is_32_mib(self) -> None:
        assert results_module.MAX_RESULTS_BYTES == 32 * 1024 * 1024

    def test_an_oversize_file_is_refused_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(results_module, "MAX_RESULTS_BYTES", 100)
        path = Path(write_json(tmp_path, "r.json", DEMO))

        with pytest.raises(ResultsError) as caught:
            load_results(path)

        assert caught.value.problems == (
            "not a results.json: cannot read the file: is larger than 100 bytes",
        )
