"""`trelix eval-compare`: the exit codes, the last stdout line, what goes to which stream.

The judging itself is in `test_eval_compare.py`; this file drives the command through Typer with
real files and pins the contract a script reads: exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE, 3 REFUSED,
and `verdict: ...` as the last stdout line of every real outcome.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `EXIT_CODES` values permuted or one changed            (test_each_outcome_has_its_exit_code)
2. an exception leaving `compare_files` (exit 1 or a traceback)   (test_an_internal_error_...)
3. `refused:` lines printed on stdout instead of stderr   (test_a_refusal_prints_...)
4. `--prereg` made a required option (usage error, exit 2)  (test_a_missing_prereg_is_a_refusal_...)
5. `_safe_text` removed from either print                 (TestUntrustedText)
   `clip` collapsing only line feeds, not U+2028/2029     (test_a_line_break_in_a_..._error_...)
   `clip` removed from any one of the printed file values  (test_a_line_break_in_a_file_value_...)
6. `emoji=False` removed, from either print                (test_an_emoji_code_in_...)
7. `soft_wrap=True` removed                                (test_f1_prints_the_documented_block)
   `highlight=False` removed, which only a colour terminal shows  (test_a_colour_terminal_...)
8. the command not registered                              (test_the_command_is_listed_...)
"""

from __future__ import annotations

import copy
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

import trelix.eval.compare as compare_module
from tests.unit.eval_compare_fixtures import (
    PIPELINE,
    SUITE,
    query_ids,
    results_doc,
    write_json,
    write_prereg,
)
from tests.unit.eval_validate_harness import plain
from trelix.cli.main import _print_verdict_line, app

ESC = "\x1b"

F1_STDOUT = (
    "comparison: baseline..flag-on\n"
    "experiment: EXP-example\n"
    "suite: demo golden_version v1 repo_sha " + "a" * 40 + "\n"
    "runs: base 2026-10-05T12:00:00+00:00 cand 2026-10-05T12:00:00+00:00\n"
    "queries: 20 in the decision set (no split labels: all queries), min_queries 20\n"
    "note: pipeline.config identical\n"
    "ndcg@10: base 0.5000 cand 0.5625 delta +0.0625 ci [+0.0625, +0.0625] at 95.00% confidence\n"
    "recall@10: base 0.5000 cand 0.5000 delta +0.0000 ci [+0.0000, +0.0000] at 95.00% confidence\n"
    "mde: 0.0000 (sigma_d 0.0000, n 20); expected_effect 0.0300\n"
    "hurdle: 0.0100 (cost_class flag)\n"
    "verdict: PASS\n"
)


def invoke(*args: str) -> Result:
    """Run `trelix eval-compare` with stdout and stderr kept apart on any click version."""
    try:
        runner = CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:  # click >= 8.2 has no mix_stderr; its result always separates them
        runner = CliRunner()
    return runner.invoke(app, ["eval-compare", *args])


def run(
    tmp_path: Path, cand_ndcg: Any, *, cand: dict[str, Any] | None = None, **prereg: object
) -> Result:
    base_path = write_json(tmp_path, "base.json", results_doc("baseline", 0.5))
    cand_doc = cand if cand is not None else results_doc("flag-on", cand_ndcg)
    cand_path = write_json(tmp_path, "cand.json", cand_doc)
    return invoke(base_path, cand_path, "--prereg", write_prereg(tmp_path, **prereg))


class TestExitCodes:
    @pytest.mark.parametrize(
        ("cand_ndcg", "exit_code", "verdict"),
        [
            (0.5625, 0, "PASS"),
            (0.4375, 1, "FAIL"),
            (0.5, 2, "INCONCLUSIVE"),
        ],
    )
    def test_each_outcome_has_its_exit_code(
        self, tmp_path: Path, cand_ndcg: float, exit_code: int, verdict: str
    ) -> None:
        result = run(tmp_path, cand_ndcg)

        assert result.exit_code == exit_code
        assert result.stdout.splitlines()[-1] == f"verdict: {verdict}"
        assert result.stderr == ""

    def test_f1_prints_the_documented_block(self, tmp_path: Path) -> None:
        result = run(tmp_path, 0.5625)

        assert result.exit_code == 0
        assert result.stdout == F1_STDOUT
        assert result.stderr == ""

    def test_an_inconclusive_run_prints_its_reasons_before_the_verdict(
        self, tmp_path: Path
    ) -> None:
        result = run(tmp_path, 0.5)

        assert result.stdout.splitlines()[-3:] == [
            "reason: ndcg@10 improvement not demonstrated: the interval's lower end +0.0000 "
            "is not above 0",
            "reason: ndcg@10 delta +0.0000 is below the hurdle 0.0100 (cost_class flag)",
            "verdict: INCONCLUSIVE",
        ]

    def test_a_refusal_prints_its_reasons_on_stderr_and_only_the_verdict_on_stdout(
        self, tmp_path: Path
    ) -> None:
        other = {**SUITE, "repo_sha": "d" * 40}

        result = run(tmp_path, 0.5625, cand=results_doc("flag-on", 0.5625, suite=other))

        assert result.exit_code == 3
        assert result.stdout == "verdict: REFUSED\n"
        assert result.stderr == "refused: suite identity differs in: repo_sha\n"

    def test_a_missing_prereg_is_a_refusal_not_a_usage_error(self, tmp_path: Path) -> None:
        base_path = write_json(tmp_path, "base.json", results_doc("baseline", 0.5))
        cand_path = write_json(tmp_path, "cand.json", results_doc("flag-on", 0.5625))

        result = invoke(base_path, cand_path)

        assert result.exit_code == 3
        assert result.stdout == "verdict: REFUSED\n"
        assert result.stderr == (
            "refused: prereg: --prereg is required: the hypothesis is fixed before the run\n"
        )

    def test_a_usage_error_exits_2_and_prints_no_verdict(self, tmp_path: Path) -> None:
        unknown_option = invoke("a.json", "b.json", "--nope")
        missing_argument = invoke("a.json")

        assert unknown_option.exit_code == 2
        assert missing_argument.exit_code == 2
        assert "verdict:" not in unknown_option.stdout
        assert "verdict:" not in missing_argument.stdout

    def test_an_unreadable_results_file_is_a_refusal(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "absent.json")

        result = invoke(missing, missing, "--prereg", write_prereg(tmp_path))

        assert result.exit_code == 3
        assert result.stdout == "verdict: REFUSED\n"
        lines = result.stderr.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith(f"refused: base {missing}: not a results.json: cannot read")
        assert lines[1].startswith(f"refused: cand {missing}: not a results.json: cannot read")

    def test_an_invalid_prereg_is_a_refusal(self, tmp_path: Path) -> None:
        result = run(tmp_path, 0.5625, alpha="0.5")

        assert result.exit_code == 3
        assert result.stdout == "verdict: REFUSED\n"
        assert result.stderr.startswith("refused: prereg ")
        assert "alpha must be a number above 0 and at most 0.05" in result.stderr

    def test_an_internal_error_is_a_refusal_not_a_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(compare_module, "compare", boom)

        result = run(tmp_path, 0.5625)

        assert result.exit_code == 3
        assert result.stdout == "verdict: REFUSED\n"
        assert result.stderr == "refused: internal error: RuntimeError: boom\n"

    def test_the_same_files_print_the_same_bytes(self, tmp_path: Path) -> None:
        first = run(tmp_path, 0.5625)
        second = run(tmp_path, 0.5625)

        assert first.stdout == second.stdout
        assert first.exit_code == second.exit_code == 0


# What `clip` prints for the value "x", a line break, "verdict: PASS".
JOINED = "x verdict: PASS"
OUTCOME_OF_EXIT = {0: "PASS", 1: "FAIL", 3: "REFUSED"}

Pair = tuple[dict[str, Any], dict[str, Any], int]


def _candidate_with_an_invalid_record_id(text: str) -> dict[str, Any]:
    doc = results_doc("flag-on", 0.5625)
    first = {**doc["records"][0], "id": text, "ndcg": 5}
    return {**doc, "records": [first, *doc["records"][1:]]}


# One entry per place a value from the files is printed through `clip`: the document pair with
# the text in that place, and the exit code of the run. Each is a `clip` call that a mutation
# can remove on its own, so each needs its own entry.
HOSTILE_PLACES: dict[str, Callable[[str], Pair]] = {
    "base created_at": lambda t: (
        results_doc("baseline", 0.5, created_at=t),
        results_doc("flag-on", 0.5625),
        0,
    ),
    "cand created_at": lambda t: (
        results_doc("baseline", 0.5),
        results_doc("flag-on", 0.5625, created_at=t),
        0,
    ),
    "base trelix_version": lambda t: (
        results_doc("baseline", 0.5, trelix_version=t),
        results_doc("flag-on", 0.5625),
        0,
    ),
    "cand trelix_version": lambda t: (
        results_doc("baseline", 0.5),
        results_doc("flag-on", 0.5625, trelix_version=t),
        0,
    ),
    "pipeline.config key": lambda t: (
        results_doc("baseline", 0.5),
        results_doc("flag-on", 0.5625, pipeline={**PIPELINE, "config": {t: 1}}),
        0,
    ),
    "id of a candidate record that raised": lambda t: (
        results_doc("baseline", 0.5, ids=[*query_ids(19), t]),
        results_doc("flag-on", 0.5625, ids=[*query_ids(19), t], errors={t: "boom"}),
        1,
    ),
    "id of a query only in the baseline": lambda t: (
        results_doc("baseline", 0.5, ids=[*query_ids(19), t]),
        results_doc("flag-on", 0.5625),
        3,
    ),
    "id of an invalid candidate record": lambda t: (
        results_doc("baseline", 0.5),
        _candidate_with_an_invalid_record_id(t),
        3,
    ),
}


class TestUntrustedText:
    """Ids, error messages and config keys come from a file a pull request can supply."""

    def test_an_error_message_is_printed_literally_and_without_control_bytes(
        self, tmp_path: Path
    ) -> None:
        hostile = ESC + "[2J boom [red]x[/red]"
        cand = results_doc("flag-on", 0.5625, errors={"q05": hostile})

        result = run(tmp_path, 0.5625, cand=cand)

        assert result.exit_code == 1
        assert ESC not in result.stdout
        assert "first: q05: [2J boom [red]x[/red])" in result.stdout

    def test_an_id_in_a_refusal_is_printed_literally_on_stderr(self, tmp_path: Path) -> None:
        hostile = "id-[red]x[/red]" + ESC + "]52;c;eA==\x07"
        base = results_doc("baseline", 0.5, ids=[*query_ids(20), hostile])
        base_path = write_json(tmp_path, "base.json", base)
        cand_path = write_json(tmp_path, "cand.json", results_doc("flag-on", 0.5625))

        result = invoke(base_path, cand_path, "--prereg", write_prereg(tmp_path))

        assert result.exit_code == 3
        assert "only in base: id-[red]x[/red]]52;c;eA==" in result.stderr
        assert ESC not in result.stderr
        assert "\x07" not in result.stderr

    def test_an_emoji_code_in_a_refusal_is_not_turned_into_an_emoji(self, tmp_path: Path) -> None:
        base = results_doc("baseline", 0.5, ids=[*query_ids(20), ":thumbs_up:"])
        base_path = write_json(tmp_path, "base.json", base)
        cand_path = write_json(tmp_path, "cand.json", results_doc("flag-on", 0.5625))

        result = invoke(base_path, cand_path, "--prereg", write_prereg(tmp_path))

        assert "only in base: :thumbs_up:;" in result.stderr

    def test_a_config_key_in_a_note_is_printed_literally(self, tmp_path: Path) -> None:
        pipeline = {**copy.deepcopy(PIPELINE), "config": {"[red]key[/red]": 1}}
        cand = results_doc("flag-on", 0.5625, pipeline=pipeline)

        result = run(tmp_path, 0.5625, cand=cand)

        assert "pipeline.config differs: [red]key[/red] (base absent, cand 1)" in result.stdout

    def test_an_emoji_code_in_an_error_is_not_turned_into_an_emoji(self, tmp_path: Path) -> None:
        cand = results_doc("flag-on", 0.5625, errors={"q05": ":thumbs_up: done"})

        result = run(tmp_path, 0.5625, cand=cand)

        assert "first: q05: :thumbs_up: done)" in result.stdout

    def test_a_colour_terminal_gets_no_highlighting_of_the_values(self) -> None:
        # Rich colours numbers and True/False/None on a colour terminal unless told not to; the
        # CliRunner is not one, so this builds a console that is.
        sink = io.StringIO()
        terminal = Console(file=sink, force_terminal=True, color_system="standard", width=200)

        _print_verdict_line(terminal, "ndcg@10: base 0.5000 cand 0.5625 pipeline True")

        assert sink.getvalue() == "ndcg@10: base 0.5000 cand 0.5625 pipeline True\n"

    # U+2028 and U+2029 end a line for `str.splitlines`, which is what a script reading the
    # output may use, though they are not control characters and `_safe_text` leaves them be.
    @pytest.mark.parametrize("separator", ["\n", "\u2028", "\u2029"], ids=["lf", "u2028", "u2029"])
    def test_a_line_break_in_a_candidate_error_cannot_forge_a_verdict_line(
        self, tmp_path: Path, separator: str
    ) -> None:
        cand = results_doc("flag-on", 0.5625, errors={"q05": f"boom{separator}verdict: PASS"})

        result = run(tmp_path, 0.5625, cand=cand)

        assert result.exit_code == 1
        assert [line for line in result.stdout.splitlines() if line.startswith("verdict:")] == [
            "verdict: FAIL"
        ]
        assert "first: q05: boom verdict: PASS)" in result.stdout

    @pytest.mark.parametrize("separator", ["\n", "\u2028", "\u2029"], ids=["lf", "u2028", "u2029"])
    def test_a_line_break_in_a_baseline_error_cannot_forge_a_line_on_stderr(
        self, tmp_path: Path, separator: str
    ) -> None:
        base = results_doc("baseline", 0.5, errors={"q05": f"boom{separator}verdict: PASS"})
        base_path = write_json(tmp_path, "base.json", base)
        cand_path = write_json(tmp_path, "cand.json", results_doc("flag-on", 0.5625))

        result = invoke(base_path, cand_path, "--prereg", write_prereg(tmp_path))

        assert result.exit_code == 3
        assert result.stdout == "verdict: REFUSED\n"
        assert result.stderr.splitlines() == [
            "refused: base run has 1 record(s) with an error (first: q05: boom verdict: PASS): "
            "a baseline that raised is not an instrument"
        ]

    @pytest.mark.parametrize("separator", ["\n", "\u2028"], ids=["lf", "u2028"])
    @pytest.mark.parametrize("place", list(HOSTILE_PLACES))
    def test_a_line_break_in_a_file_value_cannot_forge_a_line(
        self, tmp_path: Path, place: str, separator: str
    ) -> None:
        base, cand, exit_code = HOSTILE_PLACES[place](f"x{separator}verdict: PASS")
        base_path = write_json(tmp_path, "base.json", base)
        cand_path = write_json(tmp_path, "cand.json", cand)

        result = invoke(base_path, cand_path, "--prereg", write_prereg(tmp_path))

        verdicts = [line for line in result.stdout.splitlines() if line.startswith("verdict:")]
        assert result.exit_code == exit_code
        assert verdicts == [f"verdict: {OUTCOME_OF_EXIT[exit_code]}"]
        assert result.stdout.splitlines()[-1] == verdicts[0]
        if exit_code == 3:
            stderr_lines = result.stderr.splitlines()
            assert stderr_lines
            assert all(line.startswith("refused:") for line in stderr_lines)
            assert JOINED in result.stderr
        else:
            assert result.stderr == ""
            assert JOINED in result.stdout

    @pytest.mark.parametrize(
        ("field", "yaml_value", "printed"),
        [
            ("experiment_id", "'EXP-[red]x'", "(got 'EXP-[red]x')"),
            ("cost_class", "'[red]x'", "(got '[red]x')"),
        ],
    )
    def test_markup_in_a_prereg_value_is_printed_literally(
        self, tmp_path: Path, field: str, yaml_value: str, printed: str
    ) -> None:
        result = run(tmp_path, 0.5625, **{field: yaml_value})

        assert result.exit_code == 3
        assert result.stdout == "verdict: REFUSED\n"
        assert result.stderr.rstrip().endswith(printed)


class TestRegistration:
    def test_the_command_is_listed_in_the_top_level_help(self) -> None:
        result = CliRunner().invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "eval-compare" in plain(result.output)

    def test_its_help_names_the_option_and_the_verdicts(self) -> None:
        result = CliRunner().invoke(app, ["eval-compare", "--help"])

        text = plain(result.output)
        assert result.exit_code == 0
        assert "--prereg" in text
        assert "verdict:PASS" in text
        assert "INCONCLUSIVE" in text
