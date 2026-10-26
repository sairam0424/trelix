"""Recorded answers for `trelix eval-synthesis` (roadmap C-5, PR 6; unrelated to `eval-suite`'s
replay of frozen query plans): `load_replay`, `write_replay_file` and
`SynthesisEvalHarness.run_detailed(replay=...)`. The CLI is in test_cli_eval_synthesis_replay.py.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied alone, the restore proved by sha256; the
results are in the PR body)
 1. Drop the pre-flight (the `missing_replay_answers` check) -> `TestAQueryTheFileLacks`.
 2. Build the Synthesizer under replay -> `_live_then_replay` (`call_count == 0`).
 3. Drop `stream_to_stdout=False` -> `test_a_live_run_prints_nothing_and_keeps_the_answers`.
 4. Accept `schema_version` 2, 0 (`version > 1`), `true` or `1.0`, a non-string answer, or list
    the unknown keys in set order instead of sorted -> `_REFUSED` rows.
 5. `read_text()` for `read_text_capped` -> `test_a_file_over_the_cap_is_refused_unread`.
 6. Skip `abstain_reason_for` for a replayed answer -> `TestAnEmptyRetrievalAndAnAbstention...`.
 7. Drop the once-per-run warning -> `test_with_no_llm_configured_the_run_warns_once...`.
 8. `"1"` as the written `schema_version`, or `json.dump` for `write_outcome_file` ->
    `TestTheWriter`.
 9. Count a repeated missing query twice, name a sixth, or `>=` for `>` ->
    `TestAQueryTheFileLacks`.
10. Key `last_answers` by anything but the query, record `""` for a raised retrieval, record after
    scoring, or keep a previous run's answers -> `TestWhatIsRecorded`.
"""

from __future__ import annotations

import json
import logging
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.unit.ask_cli_harness import ScriptedChatClient
from tests.unit.synthesis_eval_fixtures import (
    ANSWER,
    G6_ANSWERS,
    G6_LINES,
    V1,
    StubRetriever,
    answers_from,
    make_harness,
    write_f2,
    write_golden,
)
from trelix.eval.synthesis_records import (
    ReplayError,
    SynthesisRecord,
    aggregate_synthesis_metrics,
    load_replay,
    replay_payload,
    write_replay_file,
)

_SYNTHESIZER = "trelix.eval.synthesis.Synthesizer"
_CHAT_CLIENT = "trelix.retrieval.synthesizer.build_chat_client"
_P = "replay file: "
_OK = {"schema_version": 1, "answers": {}}
_QUERIES = ("q1", "q2", "q3")
_THREE = dict.fromkeys(_QUERIES, ANSWER)


def _golden(tmp_path: Path, *queries: str) -> str:
    return write_golden(tmp_path, [V1 % query for query in queries or _QUERIES])


def _live_then_replay(
    tmp_path: Path,
    golden: str,
    answers: Mapping[str, str],
    stub: StubRetriever,
    extra: Mapping[str, str] | None = None,
) -> tuple[list[SynthesisRecord], list[SynthesisRecord], dict[str, str]]:
    """A live run (the Synthesizer patched to answer from `answers`), then a replay of what it
    recorded plus `extra` under a FRESH patch that must see no call: the live records, the
    replayed records and the live run's `last_answers`."""
    write_f2(tmp_path)
    harness = make_harness(tmp_path, stub)
    with patch(_SYNTHESIZER, **{"return_value.synthesize.side_effect": answers_from(answers)}):
        live = harness.run_detailed(golden)
    recorded = dict(harness.last_answers)
    with patch(_SYNTHESIZER) as replay_synthesizer:
        replayed = harness.run_detailed(golden, replay={**recorded, **(extra or {})})
    assert replay_synthesizer.call_count == 0
    return live, replayed, recorded


_SHAPE = _P + '"answers" must be an object mapping each golden query to its answer'
_REFUSED = [
    ([], [_P + "the top level is not an object"]),
    ({"answers": {}}, [_P + '"schema_version" is missing']),
    ({**_OK, "schema_version": 2}, [_P + '"schema_version" must be 1 (got 2)']),
    ({**_OK, "schema_version": 0}, [_P + '"schema_version" must be 1 (got 0)']),
    ({**_OK, "schema_version": True}, [_P + '"schema_version" must be 1 (got True)']),
    ({**_OK, "schema_version": 1.0}, [_P + '"schema_version" must be 1 (got 1.0)']),
    ({**_OK, "schema_version": "1"}, [_P + "\"schema_version\" must be 1 (got '1')"]),
    ({"schema_version": 1}, [_P + '"answers" is missing']),
    ({**_OK, "answers": []}, [_SHAPE]),
    ({**_OK, "answers": {"q": 3}}, [_P + "\"answers\"['q'] must be a string (got 3)"]),
    ({**_OK, "answers": {"q": None}}, [_P + "\"answers\"['q'] must be a string (got None)"]),
    ({**_OK, "extra": 1}, [_P + "unsupported key 'extra'"]),
    (
        # Six keys in no sorted order: a set's order would match the sorted one about 1 time in 720.
        {**_OK, "zulu": 1, "echo": 2, "yankee": 3, "alpha": 4, "mike": 5, "bravo": 6},
        [_P + f"unsupported key '{key}'" for key in "alpha bravo echo mike yankee zulu".split()],
    ),
    (
        {"schema_version": 2, "answers": []},
        [_P + '"schema_version" must be 1 (got 2)', _SHAPE],
    ),
    (
        # The fixed order is version, then answers, then any other key.
        {"schema_version": 2, "answers": [], "x": 1},
        [_P + '"schema_version" must be 1 (got 2)', _SHAPE, _P + "unsupported key 'x'"],
    ),
]


class TestTheFile:
    @pytest.mark.parametrize("answers", [{"q": "a"}, {}, {"q": ""}])
    def test_a_valid_file_gives_its_answers(self, tmp_path: Path, answers: dict[str, str]) -> None:
        path = tmp_path / "answers.json"
        path.write_text(json.dumps({"schema_version": 1, "answers": answers}), encoding="utf-8")
        assert load_replay(path) == answers

    @pytest.mark.parametrize(("document", "problems"), _REFUSED)
    def test_a_file_that_cannot_be_used_is_refused_with_every_problem(
        self, tmp_path: Path, document: object, problems: list[str]
    ) -> None:
        path = tmp_path / "answers.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ReplayError) as refused:
            load_replay(path)
        assert list(refused.value.problems) == problems
        assert str(refused.value) == "; ".join(problems)
        assert isinstance(refused.value, ValueError)

    def test_a_missing_or_unparsable_file_is_refused_not_raised(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_bytes(b"{")
        with pytest.raises(ReplayError) as unreadable:
            load_replay(tmp_path / "absent.json")
        with pytest.raises(ReplayError) as unparsable:
            load_replay(tmp_path / "bad.json")
        assert unreadable.value.problems[0].startswith(_P + "cannot read the file:")
        assert unparsable.value.problems[0].startswith(_P + "not valid JSON (")

    def test_a_file_over_the_cap_is_refused_unread(self, tmp_path: Path) -> None:
        path = tmp_path / "big.json"
        with path.open("wb") as handle:
            handle.truncate(20 * 1024 * 1024 + 1)
        with pytest.raises(ReplayError) as refused:
            load_replay(path)
        assert refused.value.problems == (
            _P + "cannot read the file: is larger than 20971520 bytes",
        )


class TestTheWriter:
    def test_a_written_file_reads_back_and_the_next_one_replaces_it(self, tmp_path: Path) -> None:
        target = tmp_path / "answers.json"
        target.write_text("old", encoding="utf-8")
        assert write_replay_file(str(target), {"q": "café [C1]\n"}) is None
        assert load_replay(target) == {"q": "café [C1]\n"}
        assert write_replay_file(str(target), {"p": "b"}) is None
        assert load_replay(target) == {"p": "b"}

    def test_the_file_is_ascii_sorted_private_and_ends_with_one_newline(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "answers.json"
        write_replay_file(str(target), {"q": "café"})
        raw = target.read_bytes()
        assert raw.isascii() and b"caf\\u00e9" in raw
        assert raw.endswith(b"}\n") and not raw.endswith(b"\n\n")
        assert raw.index(b'"answers"') < raw.index(b'"schema_version"')
        if sys.platform != "win32":
            assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_a_missing_directory_is_an_error_message_and_the_payload_has_the_documents_shape(
        self, tmp_path: Path
    ) -> None:
        error = write_replay_file(str(tmp_path / "absent" / "answers.json"), {"q": "a"})
        assert error is not None and error.startswith("FileNotFoundError:")
        assert replay_payload({"a": "b"}) == {"schema_version": 1, "answers": {"a": "b"}}


_R5B = {"n_queries": 6.0, "n_abstained": 2.0, "citations_total": 6.0, "citations_valid": 4.0}
_R5B |= {"completeness": 0.6667}
_V1_AGGREGATE = {"n_queries": 3.0, "n_abstained": 0.0, "citations_total": 0.0}
_V1_AGGREGATE |= {"hallucination_rate": 0.0, "completeness": 1.0}


class TestAReplayEqualsTheLiveRun:
    @pytest.mark.parametrize(
        ("lines", "answers", "tagged", "anchors"),
        [
            pytest.param(G6_LINES, G6_ANSWERS, True, _R5B, id="r5b-example"),
            pytest.param([V1 % q for q in _QUERIES], _THREE, False, _V1_AGGREGATE, id="v1-file"),
        ],
    )
    def test_the_records_and_the_aggregate_are_the_same(
        self,
        tmp_path: Path,
        lines: tuple[str, ...],
        answers: dict[str, str],
        tagged: bool,
        anchors: dict[str, float],
    ) -> None:
        golden = write_golden(tmp_path, lines)
        live, replayed, recorded = _live_then_replay(
            tmp_path, golden, answers, StubRetriever(tagged=tagged)
        )
        assert recorded == answers and len(live) == len(answers)
        assert replayed == live
        metrics = aggregate_synthesis_metrics(replayed)
        assert metrics == pytest.approx(aggregate_synthesis_metrics(live))
        assert {key: round(metrics[key], 4) for key in anchors} == anchors


_NONE = "replay file has no answer for"
_JWT = '{"query": "how is a jwt verified", "relevant_files": ["src/auth/middleware.py"]}'
_FIVE = "5 queries: 'q1', 'q2', 'q3', 'q4', 'q5'"
_SEVEN = "7 queries: 'q1', 'q2', 'q3', 'q4', 'q5', ... and 2 more"


class TestAQueryTheFileLacks:
    @pytest.mark.parametrize(
        ("lines", "replay", "message"),
        [
            ([_JWT], {}, f"{_NONE} 1 query: 'how is a jwt verified'"),
            ([V1 % f"q{n}" for n in range(1, 6)], {}, f"{_NONE} {_FIVE}"),
            ([V1 % f"q{n}" for n in range(1, 8)], {}, f"{_NONE} {_SEVEN}"),
            ([V1 % "q1", V1 % "q1"], {}, f"{_NONE} 1 query: 'q1'"),
            ([V1 % "q1", V1 % "q2"], {"q1": "a"}, f"{_NONE} 1 query: 'q2'"),
        ],
        ids=["one", "five", "seven", "one-on-two-lines", "the-file-covers-part"],
    )
    def test_a_query_the_file_lacks_stops_the_run_before_any_query(
        self, tmp_path: Path, lines: list[str], replay: dict[str, str], message: str
    ) -> None:
        stub = StubRetriever()
        harness = make_harness(tmp_path, stub)
        with patch(_SYNTHESIZER) as synthesizer, pytest.raises(ValueError) as refused:
            harness.run_detailed(write_golden(tmp_path, lines), replay=replay)
        assert str(refused.value) == message
        assert (stub.calls, synthesizer.call_count) == (0, 0)


class TestAnEmptyRetrievalAndAnAbstentionUnderReplay:
    @pytest.mark.parametrize(
        ("answer", "empty", "reason", "cited"),
        [
            ("anything", True, "no_results", 0),
            ("\n  INSUFFICIENT_EVIDENCE: x", False, "insufficient_evidence", 0),
            ("The answer [C1].", False, None, 1),
        ],
    )
    def test_they_read_as_they_do_in_a_live_run(
        self, tmp_path: Path, answer: str, empty: bool, reason: str | None, cited: int
    ) -> None:
        write_f2(tmp_path)
        stub = StubRetriever(tagged=True, empty_for=frozenset({"q1"} if empty else ()))
        with patch(_SYNTHESIZER):
            [record] = make_harness(tmp_path, stub).run_detailed(
                _golden(tmp_path, "q1"), replay={"q1": answer}
            )
        assert (record.abstained, record.abstain_reason) == (reason is not None, reason)
        assert record.citations_total == cited


class TestWhatIsRecorded:
    def test_a_query_whose_retrieval_raised_has_no_recorded_answer(self, tmp_path: Path) -> None:
        golden, stub = _golden(tmp_path), StubRetriever(raise_for=frozenset({"q2"}))
        live, replayed, recorded = _live_then_replay(
            tmp_path, golden, _THREE, stub, extra={"q2": "anything"}
        )
        assert sorted(recorded) == ["q1", "q3"]
        assert (live[1].id, live[1].error) == ("q0002", "index is unreadable")
        assert (live[1].hallucination, live[1].completeness) == (1.0, 0.0)
        assert replayed == live  # the hand-made q2 answer is never read: its retrieval still raises
        with pytest.raises(ValueError) as refused:
            make_harness(tmp_path, stub).run_detailed(golden, replay=recorded)
        assert str(refused.value) == f"{_NONE} 1 query: 'q2'"

    def test_a_query_whose_scoring_raised_still_records_its_answer(self, tmp_path: Path) -> None:
        harness = make_harness(tmp_path, StubRetriever())
        with (
            patch(_SYNTHESIZER, **{"return_value.synthesize.return_value": ANSWER}),
            patch("trelix.eval.synthesis.evaluate_synthesis", side_effect=ValueError("bad")),
        ):
            [record] = harness.run_detailed(_golden(tmp_path, "q1"))
        assert (record.error, harness.last_answers) == ("bad", {"q1": ANSWER})

    def test_a_query_whose_synthesis_raised_records_an_empty_answer(self, tmp_path: Path) -> None:
        harness = make_harness(tmp_path, StubRetriever())
        with patch(_SYNTHESIZER, **{"return_value.synthesize.side_effect": RuntimeError("down")}):
            [record] = harness.run_detailed(_golden(tmp_path, "q1"))
        assert (record.error, harness.last_answers) == (None, {"q1": ""})

    def test_each_run_starts_from_an_empty_record_of_answers(self, tmp_path: Path) -> None:
        harness = make_harness(tmp_path, StubRetriever())
        with patch(_SYNTHESIZER, **{"return_value.synthesize.side_effect": answers_from(_THREE)}):
            harness.run_detailed(_golden(tmp_path))
            harness.run_detailed(_golden(tmp_path, "q2"))
        assert harness.last_answers == {"q2": ANSWER}


_SENTINEL = ("ANSWER_SENTINEL_7f3a ", "verify [C1].")


class TestALiveRun:
    def test_a_live_run_prints_nothing_and_keeps_the_answers(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = make_harness(tmp_path, StubRetriever())
        client = ScriptedChatClient(_SENTINEL, _SENTINEL, _SENTINEL)
        with (
            patch(_CHAT_CLIENT, return_value=client),
            caplog.at_level(logging.WARNING, logger="trelix.eval.synthesis"),
        ):
            harness.run_detailed(_golden(tmp_path))
        assert capsys.readouterr().out == "" and client.calls == 3
        # A configured run has nothing to warn about (the warning is for the no-LLM case below).
        assert [r for r in caplog.records if r.name == "trelix.eval.synthesis"] == []
        assert harness.last_answers == dict.fromkeys(_QUERIES, "ANSWER_SENTINEL_7f3a verify [C1].")

    def test_with_no_llm_configured_the_run_warns_once_and_records_empty_answers(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = make_harness(tmp_path, StubRetriever())
        with (
            patch(_CHAT_CLIENT, return_value=ScriptedChatClient(configured=False)),
            caplog.at_level(logging.WARNING, logger="trelix.eval.synthesis"),
        ):
            harness.run_detailed(_golden(tmp_path))
        warnings = [r for r in caplog.records if r.name == "trelix.eval.synthesis"]
        assert [r.levelno for r in warnings] == [logging.WARNING]
        assert "No LLM is configured" in warnings[0].getMessage()
        assert capsys.readouterr().out == ""
        assert harness.last_answers == dict.fromkeys(_QUERIES, "")
