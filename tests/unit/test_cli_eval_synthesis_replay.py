"""`trelix eval-synthesis --record FILE` and `--replay FILE` (roadmap C-5, PR 6): refusals, round
trips and stdout, over the design's R5b example (`G6_LINES`, queries g1..g6) with a stub retriever.

Rich wraps `err_console` at 80 columns under `CliRunner` and the design's `Evaluation failed: replay
file has no answer for 1 query: 'how is a jwt verified'` line is 81 characters, so every test widens
the terminal (precedent: test_cli_audit.py) and asserts one literal substring of the output.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied alone, the restore proved by sha256; the
results are in the PR body)
 1. Write the record file before the run or the golden validation -> the `refused-golden` row and
    `test_a_record_directory_removed_during_the_run_is_reported_after_the_table`.
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
10. Output preflight (`TestOutputPreflight`): drop the parent-exists, parent-is-a-directory,
    writable (or only its W or X half) or target-is-a-directory check, or the same-file check ->
    the row or parameter named for it; compare lexically, with `.absolute()`, by file name only or
    by directory only -> the `one-file-...` rows, `test_two_files_of_one_name...` and
    `TestOutputs`; skip one flag, swap the two flags' order or drop `_safe_text` -> their rows;
    print the lines on stdout, or on both streams -> both refusal tests (stdout is empty, stderr
    holds the lines); leave Rich emoji translation on -> the `an-emoji-code-...` row;
    exit 0, or not at all -> every refusal; call the check after the harness is built -> the
    `len(constructed) == 0` assertions, after the replay file is read -> `a-bad-output-is-named-
    before-a-missing-replay-file-is-read`, before the conflict check -> `test_the_replay_and_
    record_conflict_is_reported_before_the_check`; drop `soft_wrap` or `highlight=False` from the
    print -> `test_a_refusal_is_one_unbroken_line_on_a_narrow_terminal`.
11. A missing directory is refused before the run, so the write after the run is reached through
    a directory that disappears during it (`_live_then_remove`).
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
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
from trelix.core.models import RetrievedContext
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


def _live_then_remove(directory: Path) -> dict[str, object]:
    """`_LIVE`, and the empty `directory` removed while the first answer is produced: the paid run
    whose output directory vanishes, which the check made before the run cannot foresee."""
    answer = answers_from(G6_ANSWERS)

    def synthesize(context: RetrievedContext) -> str:
        if directory.exists():
            directory.rmdir()
        return answer(context)

    return {"return_value.synthesize.side_effect": synthesize}


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
        out, gone = tmp_path / "out.json", tmp_path / "gone"
        gone.mkdir()
        with patch(_SYNTHESIZER, **_live_then_remove(gone)):
            result = _invoke(
                tmp_path, golden, "--per-query-out", str(gone / "pq.json"), "--record", str(out)
            )
        assert result.exit_code == 1
        assert _PER_QUERY_FAILED in _plain(result.output)
        assert sorted(json.loads(out.read_text(encoding="utf-8"))["answers"]) == _ALL_QUERIES

    def test_each_file_that_cannot_be_written_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, _ = _setup(tmp_path, monkeypatch)
        gone = tmp_path / "gone"
        gone.mkdir()
        with patch(_SYNTHESIZER, **_live_then_remove(gone)):
            result = _invoke(
                tmp_path,
                golden,
                *["--per-query-out", str(gone / "pq.json"), "--record", str(gone / "out.json")],
            )
        text = _plain(result.output)
        assert result.exit_code == 1
        assert _PER_QUERY_FAILED in text and _UNWRITABLE in text
        assert "Traceback" not in text

    def test_a_record_directory_removed_during_the_run_is_reported_after_the_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, constructed = _setup(tmp_path, monkeypatch)
        gone = tmp_path / "gone"
        gone.mkdir()
        with patch(_SYNTHESIZER, **_live_then_remove(gone)):
            result = _invoke(tmp_path, golden, "--record", str(gone / "out.json"))
        text = _plain(result.output)
        assert result.exit_code == 1
        assert _TABLE in text and _UNWRITABLE in text
        assert "Traceback" not in text
        assert len(constructed) == 1


_SAME = "--record and --per-query-out must name different files"
_CANNOT_DENY_ACCESS = sys.platform == "win32" or os.geteuid() == 0
# name -> (flags, the exact stderr lines); {T} is tmp_path (also the working directory), which
# holds the directory `adir`, the file `afile`, the directory `real` and the symlink `link` to it,
# and has no `gone`, `also`, `gone[bold]` or `:warning:`.
_PREFLIGHT_REFUSALS = {
    "record-directory-missing": (
        ["--record", "{T}/gone/out.json"], ["--record: directory {T}/gone does not exist"]
    ),
    "per-query-directory-missing": (
        ["--per-query-out", "{T}/gone/pq.json"],
        ["--per-query-out: directory {T}/gone does not exist"],
    ),
    "both-directories-missing": (
        ["--per-query-out", "{T}/also/pq.json", "--record", "{T}/gone/out.json"],
        ["--record: directory {T}/gone does not exist",
         "--per-query-out: directory {T}/also does not exist"],
    ),
    "markup-in-the-path-is-printed-literally": (
        ["--record", "{T}/gone[bold]/out.json"],
        ["--record: directory {T}/gone[bold] does not exist"],
    ),
    "an-emoji-code-in-the-path-is-printed-literally": (
        ["--record", "{T}/:warning:/out.json"],
        ["--record: directory {T}/:warning: does not exist"],
    ),
    "record-is-a-directory": (
        ["--record", "{T}/adir"], ["--record: {T}/adir is a directory; name the file itself"]
    ),
    "per-query-is-a-directory": (
        ["--per-query-out", "{T}/adir"],
        ["--per-query-out: {T}/adir is a directory; name the file itself"],
    ),
    "record-parent-is-a-file": (
        ["--record", "{T}/afile/out.json"], ["--record: {T}/afile is not a directory"]
    ),
    "per-query-parent-is-a-file": (
        ["--per-query-out", "{T}/afile/pq.json"], ["--per-query-out: {T}/afile is not a directory"]
    ),
    "one-path-twice": (["--record", "{T}/out.json", "--per-query-out", "{T}/out.json"], [_SAME]),
    "one-file-spelled-relative-and-absolute": (
        ["--record", "out.json", "--per-query-out", "{T}/out.json"], [_SAME]
    ),
    "a-bad-output-is-named-before-a-missing-replay-file-is-read": (
        ["--replay", "/nonexistent/x.json", "--per-query-out", "{T}/gone/pq.json"],
        ["--per-query-out: directory {T}/gone does not exist"],
    ),
    "one-file-through-a-symlinked-directory": (
        ["--record", "{T}/real/out.json", "--per-query-out", "{T}/link/out.json"], [_SAME]
    ),
}  # fmt: skip


def _stubbed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, list, StubRetriever]:
    """`_setup` with the retriever kept, so a test can count what it was asked."""
    retriever = StubRetriever(tagged=True)
    constructed = install_stub_harness(monkeypatch, retriever)
    write_f2(tmp_path)
    return write_golden(tmp_path, G6_LINES), constructed, retriever


def _tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


class TestOutputPreflight:
    """`--record` and `--per-query-out` are checked before the harness is built, so a path that
    cannot be written costs no query: exit 1, one stderr line per problem, nothing built, asked or
    created. (The write after the run is covered by `TestOutputs`.)"""

    @pytest.mark.parametrize("case", _PREFLIGHT_REFUSALS.values(), ids=_PREFLIGHT_REFUSALS.keys())
    def test_a_path_that_cannot_be_written_is_refused_before_anything_is_built(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple
    ) -> None:
        flags, lines = case
        if sys.platform == "win32" and "{T}/link/out.json" in flags:
            pytest.skip("symlinks need privileges on Windows")
        golden, constructed, retriever = _stubbed(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "adir").mkdir()
        (tmp_path / "afile").write_text("x", encoding="utf-8")
        (tmp_path / "real").mkdir()
        if sys.platform != "win32":
            (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
        before = _tree(tmp_path)
        fill = {"T": str(tmp_path)}

        with patch(_SYNTHESIZER) as synthesizer:
            result = _invoke(tmp_path, golden, *[flag.format(**fill) for flag in flags])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert _plain(result.stderr).splitlines() == [line.format(**fill) for line in lines]
        assert len(constructed) == 0 and retriever.calls == 0 and synthesizer.call_count == 0
        assert _tree(tmp_path) == before

    @pytest.mark.skipif(_CANNOT_DENY_ACCESS, reason="root and Windows ignore directory modes")
    @pytest.mark.parametrize("mode", [0o500, 0o600], ids=["cannot-write", "cannot-search"])
    @pytest.mark.parametrize("flag", ["--record", "--per-query-out"])
    def test_a_directory_that_cannot_be_written_to_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, mode: int
    ) -> None:
        golden, constructed, retriever = _stubbed(tmp_path, monkeypatch)
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(mode)
        try:
            result = _invoke(tmp_path, golden, flag, str(locked / "out.json"))
        finally:
            locked.chmod(0o700)

        assert result.exit_code == 1
        assert result.stdout == ""
        assert _plain(result.stderr).splitlines() == [f"{flag}: cannot write in {locked}"]
        assert len(constructed) == 0 and retriever.calls == 0

    def test_a_refusal_is_one_unbroken_line_on_a_narrow_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A forced-colour console 40 columns wide: Rich folds a longer line and, with its
        highlighter on, splits the red span, so the exact bytes pin `soft_wrap` and `highlight`."""
        golden, _, _ = _stubbed(tmp_path, monkeypatch)
        narrow = Console(
            stderr=True, force_terminal=True, color_system="standard", width=40, no_color=False
        )  # no_color=False: a NO_COLOR in the developer's environment must not strip the bytes
        monkeypatch.setattr("trelix.cli.main.err_console", narrow)
        missing = tmp_path / "a-quite-long-directory-name-here" / "and-one-more-level"

        result = _invoke(tmp_path, golden, "--record", str(missing / "out.json"))

        assert result.exit_code == 1
        assert result.stderr == f"\x1b[31m--record: directory {missing} does not exist\x1b[0m\n"

    def test_a_replay_file_that_is_missing_keeps_its_own_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, constructed, _ = _stubbed(tmp_path, monkeypatch)
        per_query = tmp_path / "pq.json"

        result = _invoke(
            tmp_path, golden, "--replay", "/nonexistent/x.json", "--per-query-out", str(per_query)
        )

        assert result.exit_code == 1
        assert "Evaluation failed: replay file: cannot read the file:" in _plain(result.output)
        assert len(constructed) == 0 and not per_query.exists()

    def test_the_replay_and_record_conflict_is_reported_before_the_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, constructed, _ = _stubbed(tmp_path, monkeypatch)
        answers = _recorded(tmp_path / "r.json", G6_ANSWERS)

        result = _invoke(
            tmp_path, golden, "--replay", answers, "--record", str(tmp_path / "gone" / "out.json")
        )

        assert result.exit_code == 2
        assert "--replay and --record cannot be combined" in _plain(result.output)
        assert "does not exist" not in _plain(result.output) and len(constructed) == 0

    def test_a_replay_run_with_a_writable_output_still_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, constructed, retriever = _stubbed(tmp_path, monkeypatch)
        answers, per_query = _recorded(tmp_path / "r.json", G6_ANSWERS), tmp_path / "pq.json"

        result = _invoke(tmp_path, golden, "--replay", answers, "--per-query-out", str(per_query))

        assert result.exit_code == 0 and _TABLE in result.stdout
        assert len(constructed) == 1 and retriever.calls == 6
        assert json.loads(per_query.read_text(encoding="utf-8"))["harness"] == "synthesis"

    def test_two_files_of_one_name_in_existing_directories_still_run_once_and_are_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden, constructed, retriever = _stubbed(tmp_path, monkeypatch)
        for name in ("records", "scores"):
            (tmp_path / name).mkdir()
        out, per_query = tmp_path / "records" / "out.json", tmp_path / "scores" / "out.json"

        with patch(_SYNTHESIZER, **_LIVE):
            result = _invoke(
                tmp_path, golden, "--record", str(out), "--per-query-out", str(per_query)
            )

        assert result.exit_code == 0
        assert len(constructed) == 1 and retriever.calls == 6
        assert sorted(json.loads(out.read_text(encoding="utf-8"))["answers"]) == _ALL_QUERIES
        assert json.loads(per_query.read_text(encoding="utf-8"))["harness"] == "synthesis"


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
