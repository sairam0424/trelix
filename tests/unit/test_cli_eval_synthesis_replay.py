"""`trelix eval-synthesis --record FILE` and `--replay FILE` (roadmap C-5, PR 6): refusals, round
trips and stdout, over the design's R5b example (`G6_LINES`, queries g1..g6) with a stub retriever.

Rich wraps `err_console` at 80 columns under `CliRunner` and the design's `Evaluation failed: replay
file has no answer for 1 query: 'how is a jwt verified'` line is 81 characters, so every test widens
the terminal (precedent: test_cli_audit.py) and asserts one literal substring of the output.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied alone, the restore proved by sha256; the
results are in the PR body)
 1. Write the record file before the run or the golden validation -> the `refused-golden` and
    `unwritable-record` rows.
 2. Key `last_answers` by `id` -> `test_a_recorded_run_replays_to_the_same_table`.
 3. Record `""` for a raised retrieval -> `test_a_query_whose_retrieval_raised...`.
 4. Drop the `--replay`/`--record` conflict -> `conflict`; let `FileNotFoundError` escape
    `load_replay` -> `missing-file`; narrow the terminal to 80 columns -> the `design-literal` row.
 5. Drop `stream_to_stdout=False` -> `test_a_recorded_run_replays_to_the_same_table` (sentinel).
 6. Skip `--per-query-out` when `--record` is given, or the reverse -> `TestOutputs`.
 7. Let the first failed write end the command (the old `raise` inside the first `if`) ->
    `test_a_per_query_path_that_cannot_be_written_does_not_lose_the_recorded_answers`.
 8. Report no error for the second failed write ->
    `test_each_file_that_cannot_be_written_is_reported`.
 9. Drop `--replay` or `--record` from the help, or the planner's plan-cache pointer from the
    `--replay` help -> `test_the_help_lists_both_flags_and_points_at_the_plan_cache`.
"""

from __future__ import annotations

import json
import re
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner, Result

from tests.unit.ask_cli_harness import ScriptedChatClient
from tests.unit.synthesis_eval_fixtures import (
    G6_ANSWERS,
    G6_LINES,
    StubRetriever,
    answers_from,
    install_stub_harness,
    write_f2,
    write_golden,
)
from trelix.cli.main import app
from trelix.eval.synthesis_records import write_replay_file

runner = CliRunner()

_SYNTHESIZER = "trelix.eval.synthesis.Synthesizer"
_CHAT_CLIENT = "trelix.retrieval.synthesizer.build_chat_client"
_TABLE = "Synthesis Quality Results"
_NO_G1 = "Evaluation failed: replay file has no answer for 1 query: 'g1'"
_NO_JWT = "Evaluation failed: replay file has no answer for 1 query: 'how is a jwt verified'"  # noqa: E501
_VERSION_2 = 'Evaluation failed: replay file: "schema_version" must be 1 (got 2)'
_UNWRITABLE = "Could not write the replay file: FileNotFoundError:"
# One scripted answer per golden line, in file order, each carrying the stdout sentinel.
_SENTINEL_ANSWERS = {f"g{n}": f"ANSWER_SENTINEL_7f3a_{n} verify [C1]." for n in range(1, 7)}
_SENTINEL_TUPLES = [(f"ANSWER_SENTINEL_7f3a_{n} ", "verify [C1].") for n in range(1, 7)]
_LIVE = {"return_value.synthesize.side_effect": answers_from(G6_ANSWERS)}


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **stub: object) -> tuple[str, list]:
    """G6 on disk with F2 beside it, and a stub-backed harness: the golden path and the configs
    the harness was built with."""
    constructed = install_stub_harness(monkeypatch, StubRetriever(tagged=True, **stub))  # type: ignore[arg-type]
    write_f2(tmp_path)
    return write_golden(tmp_path, G6_LINES), constructed


def _invoke(tmp_path: Path, golden: str, *extra: str) -> Result:
    return runner.invoke(app, ["eval-synthesis", str(tmp_path), "--golden", golden, *extra])


def _recorded(path: Path, answers: dict[str, str]) -> str:
    assert write_replay_file(str(path), answers) is None
    return str(path)


def _paths(tmp_path: Path) -> dict[str, str]:
    """The files the refusal rows name by placeholder (`{R}` and so on)."""
    texts = {
        "LONG": '{"query": "how is a jwt verified", "relevant_files": ["a.py"]}',
        "BAD_GOLDEN": '{"query": "q", "answerable": "no"}',
        "BAD": '{"schema_version": 2, "answers": {}}',
    }
    for name, text in texts.items():
        (tmp_path / name).write_text(text + "\n", encoding="utf-8")
    without_g1 = {query: answer for query, answer in G6_ANSWERS.items() if query != "g1"}
    return {
        **{name: str(tmp_path / name) for name in texts},
        "R": _recorded(tmp_path / "R", G6_ANSWERS),
        "R2": _recorded(tmp_path / "R2", without_g1),
        "EMPTY": _recorded(tmp_path / "EMPTY", {}),
        "OUT": str(tmp_path / "OUT"),
        "ABSENT": str(tmp_path / "ABSENT"),
    }


# name -> (extra args, exit code, must contain, must not contain, harness built?, OUT exists?)
_REFUSALS = {
    "a-query-the-file-lacks": (["--replay", "{R2}"], 1, [_NO_G1], ["Traceback"], True, None),
    "the-design-literal-on-one-line": (
        ["--golden", "{LONG}", "--replay", "{EMPTY}"], 1, [_NO_JWT], ["Traceback"], True, None
    ),
    "a-wrong-schema-version": (["--replay", "{BAD}"], 1, [_VERSION_2], ["Traceback"], False, None),
    "missing-file": (
        ["--replay", "/nonexistent/x.json"], 1, ["cannot read the file:"],
        ["Golden file not found", "Traceback"], False, None,
    ),
    "conflict": (
        ["--replay", "{R}", "--record", "{OUT}"], 2, ["--replay and --record cannot be combined"],
        [], False, False,
    ),
    "unwritable-record": (
        ["--record", "{ABSENT}/out.json"], 1, [_TABLE, _UNWRITABLE], ["Traceback"], True, None
    ),
    "refused-golden": (
        ["--golden", "{BAD_GOLDEN}", "--record", "{OUT}"], 1, ["line 1:"], [_TABLE], True, False
    ),
}  # fmt: skip


class TestRefusals:
    @pytest.mark.parametrize("case", _REFUSALS.values(), ids=_REFUSALS.keys())
    def test_a_refused_invocation_says_why_and_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple
    ) -> None:
        extra, code, must_contain, must_not_contain, built, out_exists = case
        golden, constructed = _setup(tmp_path, monkeypatch)
        paths = _paths(tmp_path)

        result = _invoke(tmp_path, golden, *[arg.format(**paths) for arg in extra])

        text = _plain(result.output)
        assert result.exit_code == code
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert all(needle in text for needle in must_contain)
        assert not any(needle in text for needle in must_not_contain)
        assert bool(constructed) is built
        if out_exists is not None:
            assert Path(paths["OUT"]).exists() is out_exists


class TestReplay:
    def test_a_replay_scores_the_recorded_answers_without_an_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, _ = _setup(tmp_path, monkeypatch)
        answers = _recorded(tmp_path / "r.json", G6_ANSWERS)
        with patch(_SYNTHESIZER) as synthesizer:
            result = _invoke(tmp_path, golden, "--replay", answers)
        assert result.exit_code == 0
        assert _TABLE in result.stdout and "0.6667" in result.stdout
        assert "of 2 abstained" in result.stdout and "of 4 cited answers" in result.stdout
        assert synthesizer.call_count == 0


class TestRecord:
    def test_a_recorded_run_replays_to_the_same_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, _ = _setup(tmp_path, monkeypatch)
        out = tmp_path / "out.json"
        with patch(_CHAT_CLIENT, return_value=ScriptedChatClient(*_SENTINEL_TUPLES)):
            live = _invoke(tmp_path, golden, "--record", str(out))
        assert live.exit_code == 0
        document = {"schema_version": 1, "answers": _SENTINEL_ANSWERS}
        assert json.loads(out.read_text(encoding="utf-8")) == document
        if sys.platform != "win32":
            assert stat.S_IMODE(out.stat().st_mode) == 0o600
        assert "ANSWER_SENTINEL" not in live.stdout and _TABLE in live.stdout

        with patch(_SYNTHESIZER) as synthesizer:
            replayed = _invoke(tmp_path, golden, "--replay", str(out))
        assert replayed.exit_code == 0 and synthesizer.call_count == 0
        assert replayed.stdout == live.stdout

    def test_a_query_whose_retrieval_raised_has_no_recorded_answer_and_blocks_its_replay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, _ = _setup(tmp_path, monkeypatch, raise_for=frozenset({"g1"}))
        out = tmp_path / "out.json"
        with patch(_SYNTHESIZER, **_LIVE):
            assert _invoke(tmp_path, golden, "--record", str(out)).exit_code == 0
        recorded = json.loads(out.read_text(encoding="utf-8"))["answers"]
        assert sorted(recorded) == ["g2", "g3", "g4", "g5", "g6"]

        install_stub_harness(monkeypatch, StubRetriever(tagged=True))
        replayed = _invoke(tmp_path, golden, "--replay", str(out))
        assert replayed.exit_code == 1
        assert _NO_G1 in _plain(replayed.output) and "Traceback" not in replayed.output


_PER_QUERY_FAILED = "Could not write per-query results: FileNotFoundError:"
_ALL_QUERIES = ["g1", "g2", "g3", "g4", "g5", "g6"]


class TestOutputs:
    """`--record` and `--per-query-out` together: each file is attempted whatever happens to the
    other, so a bad per-query path cannot discard the answers of a paid run."""

    def test_both_files_are_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        golden, _ = _setup(tmp_path, monkeypatch)
        out, per_query = tmp_path / "out.json", tmp_path / "pq.json"
        with patch(_SYNTHESIZER, **_LIVE):
            result = _invoke(
                tmp_path, golden, "--record", str(out), "--per-query-out", str(per_query)
            )
        assert result.exit_code == 0
        assert json.loads(per_query.read_text(encoding="utf-8"))["harness"] == "synthesis"
        assert sorted(json.loads(out.read_text(encoding="utf-8"))["answers"]) == _ALL_QUERIES

    def test_a_per_query_path_that_cannot_be_written_does_not_lose_the_recorded_answers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, _ = _setup(tmp_path, monkeypatch)
        out, absent = tmp_path / "out.json", tmp_path / "ABSENT"
        with patch(_SYNTHESIZER, **_LIVE):
            result = _invoke(
                tmp_path, golden, "--per-query-out", str(absent / "pq.json"), "--record", str(out)
            )
        assert result.exit_code == 1
        assert _PER_QUERY_FAILED in _plain(result.output)
        assert sorted(json.loads(out.read_text(encoding="utf-8"))["answers"]) == _ALL_QUERIES

    def test_each_file_that_cannot_be_written_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, _ = _setup(tmp_path, monkeypatch)
        absent = tmp_path / "ABSENT"
        with patch(_SYNTHESIZER, **_LIVE):
            result = _invoke(
                tmp_path,
                golden,
                *["--per-query-out", str(absent / "pq.json"), "--record", str(absent / "out.json")],
            )
        text = _plain(result.output)
        assert result.exit_code == 1
        assert _PER_QUERY_FAILED in text and _UNWRITABLE in text
        assert "Traceback" not in text


def test_the_help_lists_both_flags_and_points_at_the_plan_cache() -> None:
    result = runner.invoke(app, ["eval-synthesis", "--help"])
    text = _plain(result.output)
    assert result.exit_code == 0
    # Each option's own row: the flag, its metavar, then the start of its own help text. Not a
    # mention of the flag in another option's help, and not tied to the metavar word, which
    # click/typer render as "TEXT" up to click 8.1 and as "<str>" from click 8.5 on.
    assert re.search(r"--replay\s+\S+\s+Score the answers recorded", text)
    assert re.search(r"--record\s+\S+\s+After the run, write every answer", text)
    assert "TRELIX_RETRIEVAL_PLAN_CACHE_FILE" in text
