"""Synthesis golden v2 and the records behind `trelix eval-synthesis` (roadmap C-5, PR 5a): the
three optional fields, the loader's refusals, the record ids and the aggregate over records.
Fixtures (`StubRetriever`, `make_harness`, `V1`, `ANSWER`, `AGGREGATE_KEYS`) are in
tests/unit/synthesis_eval_fixtures.py; `ANSWER` scores a V1 line at hallucination 0.0 and
completeness 1.0. Every expected value here is a literal.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied alone against the test named, with a
sha256-proved restore; the results are in the PR body)
 1. Drop a row of `_V2_FIELDS` -> that field's rows of TestValidateSynthesisEntry fail.
 2. Accept `null` (`item.get(name) is not None` for `name in item`) -> the `None` rows fail.
 3. Append a non-object JSON line as an entry -> the `[1]` row of TestTheLoader raises
    AttributeError instead of refusing with `got list`.
 4. Report distinct lines instead of `len(problems)` -> `test_two_problems_on_one_line...`.
 5. `enumerate(entries)` from 0 -> TestRecordIds reads `q0000`; 6. drop `:04d` -> `q1`;
 7. number by file line instead of loaded entry -> the `not json` row reads `q0003`.
 8. Exclude errored records from the four means -> `test_a_v1_file_with_one_errored_query...`
    reads `hallucination_rate 0.0`, not 0.3333.
 9. Average over all records -> `test_an_unanswerable_query...` reads completeness 0.5.
10. Count errored records in `n_unanswerable` -> `test_an_errored_unanswerable_query...`.
11. Drop `[:200]` in `failed_record` -> `test_failed_record_cuts_the_error...`.
12. Write the docstring's backslash unescaped (`__doc__` then holds a real newline) ->
    `test_docstring_item_9...`. 13. Reorder two record fields -> `test_the_record_has...`.
14. Drop `if not entries: return []` -> `test_an_empty_file...` sees a `Synthesizer` built.
15. In `_record`'s inner `except`, `raise` instead of `answer = ""` -> `test_a_model_failure...`
    reads `error 'model down'` and `unscoreable 1.0`; drop its `logger.warning` -> no record.
16. Drop `.strip()` in `_is_text` -> the `" "` rows of `test_a_wrong_field_gives...` validate.
17. Drop `.strip()` in `record_id` -> TestRecordIds reads `" "` for the last `_ID_LINES` id, not
    `q0004`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.unit.synthesis_eval_fixtures import (
    AGGREGATE_KEYS,
    ANSWER,
    V1,
    StubRetriever,
    make_harness,
    write_golden,
)
from trelix.core.models import RetrievedContext
from trelix.eval import synthesis
from trelix.eval.synthesis import validate_synthesis_entry
from trelix.eval.synthesis_records import (
    SynthesisRecord,
    aggregate_synthesis_metrics,
    failed_record,
)

_ANSWERABLE = '"answerable" must be true or false'
_GOLD = '"gold_answer" must be a non-empty string'
_CITES = '"expected_citations" must be a list of non-empty strings'
_MEANS = ("hallucination_rate", "completeness", "faithfulness", "overall")
_V1_LINES = [V1 % "q1", V1 % "q2", V1 % "q3"]
_NOT_JSON_BETWEEN = [V1 % "q1", "not json", V1 % "q2"]
_BAD_ANSWERABLE = '{"query": "q", "answerable": "no"}'
_ALL_WRONG = {"expected_citations": [1], "gold_answer": 3, "answerable": "no"}
_UNANSWERABLE_Q2 = '{"query": "q2", "answerable": false, "expected_answer_fragments": ["zzz"]}'
_ID_LINES = [
    '{"query": "a", "id": ""}',
    '{"query": "b", "id": 5}',
    '{"query": "c", "id": "e1"}',
    '{"query": "d", "id": " "}',
]
_SYNTHESIZER = "trelix.eval.synthesis.Synthesizer"
_MODEL_DOWN = "Synthesis failed for query 'q2', scoring an empty answer: model down"


def _model_down(context: RetrievedContext) -> str:
    """The patched `synthesize`: the model is down for q2 and answers every other query."""
    if context.query == "q2":
        raise RuntimeError("model down")
    return ANSWER


def _raise_q2() -> StubRetriever:
    """A retriever that raises for q2; built per test so no `calls` count leaks between them."""
    return StubRetriever(raise_for=frozenset({"q2"}))


def _records(tmp_path: Path, lines: list[str], retriever: StubRetriever | None = None) -> list:
    harness = make_harness(tmp_path, retriever or StubRetriever())
    golden = write_golden(tmp_path, lines)
    with patch(_SYNTHESIZER, **{"return_value.synthesize.return_value": ANSWER}):
        return harness.run_detailed(golden)


def _metrics(tmp_path: Path, lines: list[str], retriever: StubRetriever | None = None) -> dict:
    harness = make_harness(tmp_path, retriever or StubRetriever())
    golden = write_golden(tmp_path, lines)
    with patch(_SYNTHESIZER, **{"return_value.synthesize.return_value": ANSWER}):
        return harness.run(golden)


class TestValidateSynthesisEntry:
    @pytest.mark.parametrize(
        "item",
        [
            {"answerable": False},
            {"answerable": True},
            {"gold_answer": "text"},
            {"expected_citations": ["a.py"]},
            {"expected_citations": []},
            json.loads(V1 % "q"),
        ],
    )
    def test_a_well_typed_field_and_a_v1_line_have_no_problems(self, item: dict) -> None:
        assert validate_synthesis_entry(item) == []

    @pytest.mark.parametrize(
        ("item", "problems"),
        [
            ({"answerable": "no"}, [_ANSWERABLE]),
            ({"answerable": None}, [_ANSWERABLE]),
            ({"answerable": 1}, [_ANSWERABLE]),
            ({"gold_answer": 3}, [_GOLD]),
            ({"gold_answer": ""}, [_GOLD]),
            ({"gold_answer": " "}, [_GOLD]),
            ({"gold_answer": None}, [_GOLD]),
            ({"expected_citations": "a.py"}, [_CITES]),
            ({"expected_citations": [1]}, [_CITES]),
            ({"expected_citations": ["a.py", ""]}, [_CITES]),
            ({"expected_citations": ["a.py", " "]}, [_CITES]),
            ({"expected_citations": None}, [_CITES]),
            (_ALL_WRONG, [_ANSWERABLE, _GOLD, _CITES]),
        ],
    )
    def test_a_wrong_field_gives_its_literal_message(self, item: dict, problems: list) -> None:
        assert validate_synthesis_entry(item) == problems


class TestTheLoader:
    @pytest.mark.parametrize(
        ("lines", "needles"),
        [
            ([V1 % "q1", _BAD_ANSWERABLE], ["1 unusable golden entry:", f"line 2: {_ANSWERABLE}"]),
            (
                [_BAD_ANSWERABLE, V1 % "q2", '{"query": "q", "gold_answer": ""}'],
                ["2 unusable golden entries:", f"line 1: {_ANSWERABLE}", f"line 3: {_GOLD}"],
            ),
            ([V1 % "q1", "[1]"], ["line 2: expected a JSON object, got list"]),
            (['"x"'], ["line 1: expected a JSON object, got str"]),
            (["3"], ["line 1: expected a JSON object, got int"]),
        ],
    )
    def test_a_refusal_names_every_problem_and_runs_no_query(
        self, tmp_path: Path, lines: list[str], needles: list[str]
    ) -> None:
        retriever = StubRetriever()
        with pytest.raises(ValueError) as info:
            _records(tmp_path, lines, retriever)
        for needle in needles:
            assert needle in str(info.value)
        assert retriever.calls == 0

    def test_two_problems_on_one_line_count_as_two(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as info:
            _records(tmp_path, ['{"query": "q", "answerable": "no", "gold_answer": ""}'])
        assert "2 unusable golden entries:" in str(info.value)
        assert str(info.value).count("line 1: ") == 2

    def test_a_line_that_is_not_json_is_skipped_and_the_others_scored(self, tmp_path: Path) -> None:
        scored = [(r.query, r.completeness, r.error) for r in _records(tmp_path, _NOT_JSON_BETWEEN)]
        assert scored == [("q1", 1.0, None), ("q2", 1.0, None)]


class TestRecordIds:
    @pytest.mark.parametrize(
        ("lines", "ids"),
        [
            (_V1_LINES, ["q0001", "q0002", "q0003"]),
            (_NOT_JSON_BETWEEN, ["q0001", "q0002"]),
            (_ID_LINES, ["q0001", "q0002", "e1", "q0004"]),
        ],
    )
    def test_the_id_is_the_lines_or_its_position_among_loaded_entries(
        self, tmp_path: Path, lines: list[str], ids: list[str]
    ) -> None:
        assert [r.id for r in _records(tmp_path, lines)] == ids


def test_the_record_has_exactly_these_fields_in_this_order() -> None:
    names = ["id", "query", "answerable", "hallucination", "completeness", "faithfulness"]
    assert [f.name for f in dataclasses.fields(SynthesisRecord)] == [*names, "overall", "error"]


class TestTheAggregate:
    def test_a_v1_file_with_one_errored_query_keeps_todays_numbers(self, tmp_path: Path) -> None:
        metrics = _metrics(tmp_path, _V1_LINES, _raise_q2())
        assert metrics["hallucination_rate"] == pytest.approx(1 / 3)
        assert metrics["completeness"] == pytest.approx(2 / 3)
        assert (metrics["unscoreable"], metrics["n_unanswerable"]) == (1.0, 0.0)
        assert metrics["n_queries"] == 3.0
        assert 0.0 <= metrics["faithfulness"] <= 1.0
        assert 0.0 <= metrics["overall"] <= 1.0

    def test_the_errored_record_carries_the_error_and_the_placeholders(
        self, tmp_path: Path
    ) -> None:
        q1, q2, q3 = _records(tmp_path, _V1_LINES, _raise_q2())
        assert (q2.id, q2.error) == ("q0002", "index is unreadable")
        assert (q2.hallucination, q2.completeness, q2.faithfulness, q2.overall) == (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        for record in (q1, q3):
            assert (record.error, record.hallucination, record.completeness) == (None, 0.0, 1.0)

    def test_a_model_failure_is_an_empty_answer_not_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = make_harness(tmp_path, StubRetriever())
        golden = write_golden(tmp_path, [V1 % "q1", V1 % "q2"])
        with (
            patch(_SYNTHESIZER, **{"return_value.synthesize.side_effect": _model_down}),
            caplog.at_level(logging.WARNING, logger="trelix.eval.synthesis"),
        ):
            q1, q2 = harness.run_detailed(golden)
        assert (q2.id, q2.error) == ("q0002", None)
        assert (q2.hallucination, q2.completeness, q2.faithfulness, q2.overall) == (
            0.0,
            0.0,
            0.0,
            0.4,
        )
        assert (q1.error, q1.hallucination, q1.completeness) == (None, 0.0, 1.0)
        metrics = aggregate_synthesis_metrics([q1, q2])
        assert (metrics["unscoreable"], metrics["n_queries"]) == (0.0, 2.0)
        assert [record.getMessage() for record in caplog.records] == [_MODEL_DOWN]

    def test_an_unanswerable_query_is_counted_not_averaged(self, tmp_path: Path) -> None:
        metrics = _metrics(tmp_path, [V1 % "q1", _UNANSWERABLE_Q2])
        assert metrics["completeness"] == 1.0
        assert (metrics["n_unanswerable"], metrics["unscoreable"]) == (1.0, 0.0)
        assert metrics["n_queries"] == 2.0

    def test_an_errored_unanswerable_query_is_only_unscoreable(self, tmp_path: Path) -> None:
        metrics = _metrics(tmp_path, [V1 % "q1", _UNANSWERABLE_Q2], _raise_q2())
        assert (metrics["unscoreable"], metrics["n_unanswerable"]) == (1.0, 0.0)
        assert metrics["hallucination_rate"] == 0.0

    def test_only_unanswerable_queries_give_zero_means(self, tmp_path: Path) -> None:
        metrics = _metrics(tmp_path, ['{"query": "q1", "answerable": false}'])
        assert [metrics[key] for key in _MEANS] == [0.0, 0.0, 0.0, 0.0]
        assert (metrics["n_unanswerable"], metrics["n_queries"]) == (1.0, 1.0)

    def test_an_empty_file_gives_no_records_and_every_key_at_zero(self, tmp_path: Path) -> None:
        harness = make_harness(tmp_path, StubRetriever())
        with patch(_SYNTHESIZER) as synthesizer_class:
            assert harness.run_detailed(write_golden(tmp_path, [])) == []
        synthesizer_class.assert_not_called()
        metrics = _metrics(tmp_path, [])
        assert set(metrics) == AGGREGATE_KEYS
        assert all(value == 0.0 for value in metrics.values())
        assert aggregate_synthesis_metrics([]) == metrics

    def test_the_key_set_is_exactly_the_seven(self, tmp_path: Path) -> None:
        assert set(_metrics(tmp_path, _V1_LINES)) == AGGREGATE_KEYS


def test_failed_record_cuts_the_error_and_names_a_silent_one() -> None:
    assert len(failed_record({}, 1, RuntimeError("x" * 300)).error) == 200
    assert failed_record({}, 1, RuntimeError()).error == "RuntimeError"


def test_docstring_item_9_is_worded_for_line_out_of_range() -> None:
    assert (
        "(counted as line_out_of_range in citation_statuses; lines are counted as the "
        'extractors count them, count("\\n") + 1)'
    ) in synthesis.__doc__
