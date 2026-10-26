"""The abstention and citation metrics of `trelix eval-synthesis` (roadmap C-5, PR 5b):
`abstain_reason_for`, `expected_citation_paths`, the sixteen-key aggregate over hand-built records,
and the design's R5b example run end to end through `SynthesisEvalHarness.run_detailed` with a
tagged retrieval, F2 on disk and a patched Synthesizer. Fixtures (`TAGGED_SOURCES`, `G6_LINES`,
`G6_ANSWERS`, `metrics_records`, `write_f2`, `StubRetriever(tagged=True)`) are in
tests/unit/synthesis_eval_fixtures.py; every expected value here is a literal.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied alone against the test named, with a
sha256-proved restore; the results are in the PR body)
 1. Swap the precision and recall denominators -> 0.3333 and 0.5 swap in
    `test_the_r5b_aggregate_over_hand_built_records` and `test_the_r5b_example_end_to_end`.
 2. Average `citation_validity` over every scored record -> reads 2.5/6, not 0.625.
 3. Count expected citations against ALL cited paths -> e2 hits and `citation_recall` reads 1.0.
 4. `len(entry["expected_citations"])` instead of the set in `scored_record` -> the repeated-path
    assertion of `test_failed_record_and_a_bare_scored_record...`; dropping the list branch of
    `expected_citation_paths` -> the second row of `test_expected_citation_paths...`.
 5. Read `abstain_reason` from `synthesizer.last_abstain_reason` -> under the MagicMock every
    record reads `abstained True`; e1 of `test_the_r5b_example_end_to_end` fails.
 6. Swap the two branches of `abstain_reason_for` -> the `(False, "INSUFFICIENT_EVIDENCE: x")` row.
 7. Drop the `error is None` filter before `_trust_metrics` -> `test_an_errored_query_is_counted...`
    reads `abstention_recall 0.25`.
 8. Drop the B5(b) guard in `_verified_citations` -> `test_an_untagged_retrieval_verifies_nothing`
    reads e1 `citations_total 1` (every marker `unknown`).
 9. `Counter` over `c.marker` instead of `c.status` -> the `citation_statuses` literals fail.
10. Count `status != "unknown"` as valid -> e2 reads `citations_valid 1`.
11. Drop the `no_results` branch of `abstain_reason_for` -> `test_an_empty_retrieval...` and the
    first two rows of `test_abstain_reason...`.
12. Raise on `[]` in `aggregate_synthesis_metrics` -> `test_no_records_gives_every_key_at_zero`.
13. `failed_record` with `abstained=True` -> `test_failed_record_and_a_bare_scored_record...`.
14. `citation_statuses` as a list of pairs instead of a dict -> the `{"valid": 1}` literal of
    `test_the_r5b_example_end_to_end` fails (the records-file tests build records by hand, so
    only this file reaches `scored_record` with citations).
15. Count the ANSWERABLE records as the recall denominator, or multiply in `citation_recall`
    (`hit * total`) -> `test_a_ratio_runs_over_its_own_denominator`; the R5b records hide both
    (three answerable and three unanswerable, every expected total 1). mutmut found these two.
16. Read a missing or present `query` as `None` or another default in `scored_record` and
    `failed_record` -> the `.query` assertions of `test_failed_record_and_a_bare_scored_record...`.
17. Strip `./` or turn backslashes into slashes in `expected_citation_paths` -> the third row
    of `test_expected_citation_paths...`.
18. Return 0.0 from `score_completeness` for an empty fragment list ->
    `test_an_abstaining_answerable_line_scores_completeness_one_without_fragments`.
19. Pool `citation_recall` (sum of hits over sum of totals) instead of the per-line mean ->
    `test_a_ratio_runs_over_its_own_denominator` reads 0.3333, not 0.5.
20. Count only `insufficient_evidence` abstentions in `_trust_metrics` ->
    `test_a_no_results_abstention_counts_like_an_insufficient_evidence_one`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.unit.synthesis_eval_fixtures import (
    AGGREGATE_KEYS,
    ANSWER,
    G6_ANSWERS,
    G6_LINES,
    V1,
    StubRetriever,
    answers_from,
    make_harness,
    make_record,
    metrics_records,
    write_f2,
    write_golden,
)
from trelix.core.models import RetrievedContext
from trelix.eval.synthesis_records import (
    SynthesisRecord,
    abstain_reason_for,
    aggregate_synthesis_metrics,
    expected_citation_paths,
    failed_record,
    scored_record,
)

_SYNTHESIZER = "trelix.eval.synthesis.Synthesizer"
_NO_EVIDENCE = "insufficient_evidence"
# The fourteen literal values of the R5b aggregate; `faithfulness` and `overall` are lexical and
# asserted to lie in [0, 1] only, as the existing synthesis tests do.
_R5B = {
    "hallucination_rate": 0.0,
    "completeness": 2 / 3,
    "n_queries": 6.0,
    "unscoreable": 0.0,
    "n_unanswerable": 3.0,
    "n_abstained": 2.0,
    "abstention_precision": 0.5,
    "abstention_recall": 1 / 3,
    "citation_validity": 0.625,
    "n_cited_entries": 4.0,
    "citation_recall": 0.5,
    "n_expected_citation_entries": 2.0,
    "citations_total": 6.0,
    "citations_valid": 4.0,
}
# Per record: abstained, abstain_reason, citations_total, citations_valid, citation_statuses,
# expected_citations_hit, expected_citations_total.
_R5B_FACTS = {
    "e1": (False, None, 1, 1, {"valid": 1}, 1, 1),
    "e2": (False, None, 1, 0, {"file_missing": 1}, 0, 1),
    "e3": (True, _NO_EVIDENCE, 0, 0, {}, 0, 0),
    "e4": (False, None, 2, 1, {"unknown": 1, "valid": 1}, 0, 0),
    "e5": (True, _NO_EVIDENCE, 0, 0, {}, 0, 0),
    "e6": (False, None, 2, 2, {"valid": 2}, 0, 0),
}
_NOTHING = (False, None, 0, 0, {}, 0, 0)
_E7 = '{"id": "e7", "query": "g7", "answerable": false}'
_E8 = '{"id": "e8", "query": "g8", "expected_citations": ["src/auth/jwt.py"]}'
_E8_ANSWER = "Decoding happens in decode_token [C3]."
_SCORES = {"hallucination": 0.0, "completeness": 1.0, "faithfulness": 1.0, "overall": 1.0}
_COUNTS = ("n_queries", "unscoreable")


def _facts(record: SynthesisRecord) -> tuple:
    return (
        record.abstained,
        record.abstain_reason,
        record.citations_total,
        record.citations_valid,
        record.citation_statuses,
        record.expected_citations_hit,
        record.expected_citations_total,
    )


def _assert_r5b(metrics: dict[str, float]) -> None:
    assert set(metrics) == AGGREGATE_KEYS
    assert {key: metrics[key] for key in _R5B} == pytest.approx(_R5B)
    assert 0.0 <= metrics["faithfulness"] <= 1.0
    assert 0.0 <= metrics["overall"] <= 1.0


def _run(
    tmp_path: Path, lines: Sequence[str], answers: Mapping[str, str], retriever: StubRetriever
) -> list[SynthesisRecord]:
    """F2 on disk, a harness over `retriever`, the patched Synthesizer answering from `answers`."""
    write_f2(tmp_path)
    harness = make_harness(tmp_path, retriever)
    golden = write_golden(tmp_path, lines)
    with patch(_SYNTHESIZER, **{"return_value.synthesize.side_effect": answers_from(answers)}):
        return harness.run_detailed(golden)


def _without_counts(metrics: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if key not in _COUNTS}


@pytest.mark.parametrize(
    ("has_results", "answer", "reason"),
    [
        (False, "anything", "no_results"),
        (False, "INSUFFICIENT_EVIDENCE: x", "no_results"),
        (True, "INSUFFICIENT_EVIDENCE: no retrieved chunk mentions rate limiting", _NO_EVIDENCE),
        (True, "\n  INSUFFICIENT_EVIDENCE: x", _NO_EVIDENCE),
        (True, "Insufficient_evidence: x", None),
        (True, "The answer. INSUFFICIENT_EVIDENCE:", None),
        (True, "", None),
    ],
)
def test_abstain_reason_comes_from_the_retrieval_first_then_the_answer(
    has_results: bool, answer: str, reason: str | None
) -> None:
    assert abstain_reason_for(has_results, answer) == reason


@pytest.mark.parametrize(
    ("entry", "paths"),
    [
        ({}, frozenset()),
        ({"expected_citations": ["a.py", "a.py", "b.py"]}, frozenset({"a.py", "b.py"})),
        # Paths are kept as written: no normalisation, so a spelling the index never uses misses.
        (
            {"expected_citations": ["./a.py", "a.py", "s\\a.py"]},
            frozenset({"./a.py", "a.py", "s\\a.py"}),
        ),
        # Not stripped either: " a.py" is a different path from "a.py" and must not match it.
        ({"expected_citations": [" a.py"]}, frozenset({" a.py"})),
    ],
)
def test_expected_citation_paths_is_the_set_of_the_lines_paths(
    entry: dict, paths: frozenset[str]
) -> None:
    assert expected_citation_paths(entry) == paths


class TestTheAggregateOverHandBuiltRecords:
    def test_the_r5b_aggregate_over_hand_built_records(self) -> None:
        _assert_r5b(aggregate_synthesis_metrics(metrics_records()))

    def test_an_errored_record_changes_only_n_queries_and_unscoreable(self) -> None:
        placeholders = {
            "hallucination": 1.0,
            "completeness": 0.0,
            "faithfulness": 0.0,
            "overall": 0.0,
        }
        e7 = make_record(id="e7", answerable=False, error="boom", **placeholders)
        with_e7 = aggregate_synthesis_metrics([*metrics_records(), e7])
        assert (with_e7["n_queries"], with_e7["unscoreable"]) == (7.0, 1.0)
        assert _without_counts(with_e7) == _without_counts(
            aggregate_synthesis_metrics(metrics_records())
        )

    def test_no_records_gives_every_key_at_zero(self) -> None:
        assert aggregate_synthesis_metrics([]) == {key: 0.0 for key in AGGREGATE_KEYS}

    def test_an_unanswerable_record_that_answered_is_a_zero_numerator(self) -> None:
        metrics = aggregate_synthesis_metrics([make_record(answerable=False)])
        assert (metrics["n_unanswerable"], metrics["n_abstained"]) == (1.0, 0.0)
        assert (metrics["abstention_precision"], metrics["abstention_recall"]) == (0.0, 0.0)

    def test_an_answerable_abstention_alone_is_a_zero_denominator(self) -> None:
        metrics = aggregate_synthesis_metrics(
            [make_record(abstained=True, abstain_reason=_NO_EVIDENCE)]
        )
        assert (metrics["n_abstained"], metrics["n_unanswerable"]) == (1.0, 0.0)
        assert (metrics["abstention_precision"], metrics["abstention_recall"]) == (0.0, 0.0)

    def test_a_ratio_runs_over_its_own_denominator(self) -> None:
        records = [
            make_record(answerable=False, abstained=True, abstain_reason=_NO_EVIDENCE),
            make_record(),
            make_record(),
            make_record(expected_citations_hit=1, expected_citations_total=2),
            make_record(expected_citations_hit=1, expected_citations_total=1),
            make_record(expected_citations_hit=0, expected_citations_total=3),
        ]
        metrics = aggregate_synthesis_metrics(records)
        assert (metrics["n_unanswerable"], metrics["abstention_recall"]) == (1.0, 1.0)
        # A per-line mean: (1/2 + 1/1 + 0/3) / 3 is 0.5; the pooled 2 hits of 6 would read 1/3.
        assert (metrics["n_expected_citation_entries"], metrics["citation_recall"]) == (3.0, 0.5)

    def test_a_no_results_abstention_counts_like_an_insufficient_evidence_one(self) -> None:
        records = [
            make_record(answerable=False, abstained=True, abstain_reason="no_results"),
            make_record(answerable=False, abstained=True, abstain_reason=_NO_EVIDENCE),
            make_record(abstained=True, abstain_reason="no_results"),
        ]
        metrics = aggregate_synthesis_metrics(records)
        assert (metrics["n_abstained"], metrics["n_unanswerable"]) == (3.0, 2.0)
        assert metrics["abstention_precision"] == pytest.approx(2 / 3)
        assert metrics["abstention_recall"] == 1.0


class TestTheR5bExampleEndToEnd:
    def test_the_r5b_example_end_to_end(self, tmp_path: Path) -> None:
        retriever = StubRetriever(tagged=True)
        records = _run(tmp_path, G6_LINES, G6_ANSWERS, retriever)
        assert [r.id for r in records] == ["e1", "e2", "e3", "e4", "e5", "e6"]
        assert [r.error for r in records] == [None] * 6
        assert [r.answerable for r in records] == [True, True, False, False, True, False]
        assert {r.id: _facts(r) for r in records} == _R5B_FACTS
        e1, e2, _e3, _e4, e5, _e6 = records
        assert (e1.hallucination, e1.completeness) == (0.0, 1.0)
        assert (e2.completeness, e5.completeness) == (1.0, 0.0)
        _assert_r5b(aggregate_synthesis_metrics(records))
        statuses = sum((Counter(r.citation_statuses) for r in records), Counter())
        assert statuses == {"valid": 4, "file_missing": 1, "unknown": 1}
        assert retriever.calls == 6

    def test_an_untagged_retrieval_verifies_nothing(self, tmp_path: Path) -> None:
        records = _run(tmp_path, G6_LINES, G6_ANSWERS, StubRetriever())
        assert [(r.citations_total, r.citation_statuses) for r in records] == [(0, {})] * 6
        metrics = aggregate_synthesis_metrics(records)
        zero = ("citation_validity", "n_cited_entries", "citations_total", "citations_valid")
        assert [metrics[key] for key in (*zero, "citation_recall")] == [0.0] * 5
        assert metrics["n_expected_citation_entries"] == 2.0
        assert (metrics["n_abstained"], metrics["abstention_precision"]) == (2.0, 0.5)
        assert metrics["abstention_recall"] == pytest.approx(1 / 3)

    def test_a_stale_line_range_is_cited_but_not_valid_and_not_a_hit(self, tmp_path: Path) -> None:
        answers = {**G6_ANSWERS, "g8": _E8_ANSWER}
        records = _run(tmp_path, [*G6_LINES, _E8], answers, StubRetriever(tagged=True))
        e8 = records[6]
        assert (e8.id, _facts(e8)) == ("e8", (False, None, 1, 0, {"line_out_of_range": 1}, 0, 1))
        metrics = aggregate_synthesis_metrics(records)
        assert metrics["citation_validity"] == pytest.approx(0.5)
        assert metrics["citation_recall"] == pytest.approx(1 / 3)
        assert (metrics["n_cited_entries"], metrics["n_expected_citation_entries"]) == (5.0, 3.0)

    def test_an_errored_query_is_counted_and_in_nothing_else(self, tmp_path: Path) -> None:
        retriever = StubRetriever(tagged=True, raise_for=frozenset({"g7"}))
        records = _run(tmp_path, [*G6_LINES, _E7], G6_ANSWERS, retriever)
        e7 = records[6]
        assert (e7.id, e7.error, e7.answerable) == ("e7", "index is unreadable", False)
        assert _facts(e7) == _NOTHING
        assert (e7.hallucination, e7.completeness, e7.faithfulness, e7.overall) == (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        metrics = aggregate_synthesis_metrics(records)
        assert (metrics["n_queries"], metrics["unscoreable"]) == (7.0, 1.0)
        six = aggregate_synthesis_metrics(records[:6])
        _assert_r5b(six)
        assert _without_counts(metrics) == _without_counts(six)

    def test_an_abstaining_answerable_line_scores_completeness_one_without_fragments(
        self, tmp_path: Path
    ) -> None:
        bare = '{"id": "e9", "query": "g9"}'
        answers = {"g5": G6_ANSWERS["g5"], "g9": G6_ANSWERS["g5"]}
        lines = [G6_LINES[4], bare]
        e5, e9 = _run(tmp_path, lines, answers, StubRetriever(tagged=True))
        assert (e5.abstained, e5.completeness) == (True, 0.0)
        assert (e9.abstained, e9.completeness) == (True, 1.0)
        metrics = aggregate_synthesis_metrics([e5, e9])
        assert (metrics["completeness"], metrics["abstention_precision"]) == (0.5, 0.0)


class _EmptyRetriever:
    """A retrieval that found nothing (the replay change replaces this with `StubRetriever`)."""

    def retrieve(self, query: str) -> RetrievedContext:
        return RetrievedContext(
            query=query, results=[], context_text="No relevant code found.", total_tokens=0
        )


def test_an_empty_retrieval_is_a_no_results_abstention_whatever_the_synthesizer_says(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path, _EmptyRetriever())  # type: ignore[arg-type]
    golden = write_golden(tmp_path, [V1 % "q1"])
    configured = {"return_value.synthesize.return_value": ANSWER}
    configured["return_value.last_abstain_reason"] = _NO_EVIDENCE
    with patch(_SYNTHESIZER, **configured):
        (record,) = harness.run_detailed(golden)
    assert (record.abstained, record.abstain_reason) == (True, "no_results")
    assert (record.citations_total, record.citation_statuses, record.error) == (0, {}, None)


def test_failed_record_and_a_bare_scored_record_carry_no_trust_facts() -> None:
    assert _facts(failed_record({}, 1, RuntimeError("x"))) == _NOTHING
    record = scored_record({"query": "q"}, 1, _SCORES)
    assert _facts(record) == _NOTHING
    assert (record.id, record.query, record.answerable, record.error) == ("q0001", "q", True, None)
    repeated = scored_record({"query": "q", "expected_citations": ["a.py", "a.py"]}, 1, _SCORES)
    assert (repeated.expected_citations_total, repeated.expected_citations_hit) == (1, 0)
    # A line without a query reads as "" in both record kinds; a line with one keeps it.
    failed = failed_record({"query": "q"}, 1, RuntimeError("x"))
    assert (failed.query, failed_record({}, 1, RuntimeError("x")).query) == ("q", "")
    assert scored_record({}, 1, _SCORES).query == ""
