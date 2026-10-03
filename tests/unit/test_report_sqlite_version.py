"""Pins scripts/report_sqlite_version.py.

The script prints the SQLite version an interpreter links and annotates it (`::warning::`
inside the WAL-reset range, `::notice::` outside), so CI answers "which SQLite do our
interpreters, images and binaries really use?". This file pins what it says: the range
table, the annotation text, the escaping, and that none of it is super-linear. That it is
informational, meaning it exits 0 on every path (a Python without sqlite3, a range check
that raises, a broken stdout), is pinned in
tests/unit/test_report_sqlite_version_exit_status.py, and the CI steps that call it in
tests/unit/test_report_sqlite_version_workflows.py.

The range table is written out as literals, not derived from the script's own numbers:
deriving the expectation from the code under test passes for any boundary at all. The
script carries an embedded copy of `trelix.store.db.wal_reset_risk` for interpreters
without trelix; `test_embedded_copy_agrees_with_trelix_over_a_sweep` is what keeps the two
in step, and `test_range_sentence_names_the_boundaries_of_the_real_function` keeps the
sentence printed with every annotation in step with both.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from trelix.store.db import wal_reset_risk

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "report_sqlite_version.py"

# upstream version -> is it in the WAL-reset range
_VERSION_TABLE = {
    "3.6.23": False,  # WAL does not exist yet
    "3.7.0": True,  # first WAL release
    "3.44.5": True,
    "3.44.6": False,  # backported fix
    "3.44.9": False,
    "3.45.0": True,
    "3.46.1": True,  # Debian trixie's upstream number
    "3.50.6": True,
    "3.50.7": False,  # backported fix
    "3.50.9": False,
    "3.51.0": True,
    "3.51.2": True,
    "3.51.3": False,  # the upstream fix
    "3.52.0": False,
}

_INTERPRETER = "CPython 3.12.3 on Linux x86_64"

_WARNING_3_46_1 = (
    "::warning::SQLite 3.46.1 (CPython 3.12.3 on Linux x86_64) is in the range where SQLite "
    "can lose committed WAL writes (3.7.0 through 3.51.2; fixed in 3.51.3, 3.50.7 and "
    "3.44.6). A distro build may carry the fix under a lower number. Informational only; "
    "see docs/TROUBLESHOOTING.md."
)
_NOTICE_3_51_3 = (
    "::notice::SQLite 3.51.3 (CPython 3.12.3 on Linux x86_64) is outside the range where "
    "SQLite can lose committed WAL writes (3.7.0 through 3.51.2; fixed in 3.51.3, 3.50.7 "
    "and 3.44.6)."
)


def _load_script() -> ModuleType:
    """Import scripts/report_sqlite_version.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_trelix_report_sqlite_version", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _as_tuple(dotted: str) -> tuple[int, ...]:
    return tuple(int(part) for part in dotted.split("."))


# ---------------------------------------------------------------------------
# The range table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("dotted", "expected"), list(_VERSION_TABLE.items()))
def test_embedded_range_table_boundaries(dotted: str, expected: bool) -> None:
    assert script.embedded_wal_reset_risk(_as_tuple(dotted)) is expected


def test_embedded_copy_agrees_with_trelix_over_a_sweep() -> None:
    """The copy may not drift from the function the shipped warning uses."""
    sweep = [(3, minor, patch) for minor in range(0, 61) for patch in range(0, 16)]
    sweep += [(2, 99, 9), (4, 0, 0), (3, 51), (3, 50, 7, 0), (3, 44, 5, 9), ()]
    answers = {version: wal_reset_risk(version) for version in sweep}
    assert set(answers.values()) == {True, False}  # the sweep crosses every boundary
    disagreements = [
        version
        for version in sweep
        if script.embedded_wal_reset_risk(version) is not answers[version]
    ]
    assert disagreements == []


def test_trelix_function_is_preferred_when_it_can_be_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def marker(version_info: tuple[int, ...]) -> bool:
        return True

    monkeypatch.setattr("trelix.store.db.wal_reset_risk", marker)
    check, label = script.resolve_wal_reset_risk()
    assert check is marker
    assert label == "trelix.store.db.wal_reset_risk"


def test_embedded_copy_is_used_and_labelled_when_trelix_cannot_be_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "trelix.store.db", None)  # makes the import raise
    check, label = script.resolve_wal_reset_risk()
    assert check is script.embedded_wal_reset_risk
    assert label == "embedded copy (trelix not importable: ModuleNotFoundError)"


@pytest.mark.parametrize("error_type", [RuntimeError, SystemExit], ids=["exception", "sys-exit"])
def test_a_trelix_import_that_fails_in_any_way_falls_back(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    """Including a `sys.exit` raised while trelix is imported: it is not an `Exception`."""

    class _ExplodingModule:
        def __getattr__(self, name: str) -> Any:
            raise error_type("settings rejected")

    monkeypatch.setitem(sys.modules, "trelix.store.db", _ExplodingModule())
    check, label = script.resolve_wal_reset_risk()
    assert check is script.embedded_wal_reset_risk
    assert label == f"embedded copy (trelix not importable: {error_type.__name__})"


# ---------------------------------------------------------------------------
# The annotation strings
# ---------------------------------------------------------------------------


def test_affected_version_is_a_warning_with_the_exact_text() -> None:
    line = script.format_annotation(at_risk=True, version="3.46.1", interpreter=_INTERPRETER)
    assert line == _WARNING_3_46_1


def test_fixed_version_is_a_notice_with_the_exact_text() -> None:
    line = script.format_annotation(at_risk=False, version="3.51.3", interpreter=_INTERPRETER)
    assert line == _NOTICE_3_51_3


def test_summary_line_is_plain_and_names_the_table() -> None:
    line = script.format_summary_line(
        version="3.46.1", interpreter=_INTERPRETER, table_source="trelix.store.db.wal_reset_risk"
    )
    assert line == (
        "SQLite 3.46.1 linked by CPython 3.12.3 on Linux x86_64; "
        "range table: trelix.store.db.wal_reset_risk"
    )


def test_failure_annotation_is_a_warning_naming_the_exception() -> None:
    line = script.format_failure_annotation(RuntimeError("boom"))
    assert line == (
        "::warning::The SQLite version report could not run (RuntimeError: boom). "
        "Informational only; this does not fail the job."
    )


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("plain", "plain"),
        ("100%", "100%25"),
        ("a\nb", "a%0Ab"),
        ("a\r\nb", "a%0D%0Ab"),
        ("%0A", "%250A"),  # percent first: an existing escape is not read as one
    ],
)
def test_escape_workflow_data(raw: str, escaped: str) -> None:
    assert script.escape_workflow_data(raw) == escaped


# Every dynamic value is made safe by `sanitize_field` (clip, ASCII-escape, command-escape),
# and the next three tests run against EVERY place a value reaches a line. One site is not
# enough: a site that called `escape_workflow_data` instead (newlines and percent only)
# would still pass a test of another site, and would then echo a 5,000-character
# non-ASCII exception message on a console that cannot encode it. The seventh site, the
# stderr note, is pinned in test_report_sqlite_version_exit_status.py.
#
# site id -> (render a line with the value in that one field, how that line must start)
_FIELD_SITES: dict[str, tuple[Callable[[str], str], str]] = {
    "annotation-version": (
        lambda value: script.format_annotation(at_risk=True, version=value, interpreter="i"),
        "::warning::SQLite ",
    ),
    "annotation-interpreter": (
        lambda value: script.format_annotation(at_risk=True, version="v", interpreter=value),
        "::warning::SQLite ",
    ),
    "summary-version": (
        lambda value: script.format_summary_line(version=value, interpreter="i", table_source="t"),
        "SQLite ",
    ),
    "summary-interpreter": (
        lambda value: script.format_summary_line(version="v", interpreter=value, table_source="t"),
        "SQLite ",
    ),
    "summary-table-source": (
        lambda value: script.format_summary_line(version="v", interpreter="i", table_source=value),
        "SQLite ",
    ),
    "failure-annotation": (
        lambda value: script.format_failure_annotation(ValueError(value)),
        "::warning::The SQLite version report could not run (ValueError: ",
    ),
}


@pytest.mark.parametrize("site", list(_FIELD_SITES))
def test_a_hostile_value_cannot_start_a_second_workflow_command(site: str) -> None:
    render, start = _FIELD_SITES[site]
    line = render("3.46.1%\n::error::injected\r::set-output name=x::y")
    # One physical line that starts with our own text: a workflow command is only read at
    # the start of a line, so the hostile `::error::` survives as inert text.
    assert len(line.splitlines()) == 1
    assert line.startswith(start)
    assert "3.46.1%25%0A::error::injected%0D::set-output name=x::y" in line


@pytest.mark.parametrize("site", list(_FIELD_SITES))
def test_non_ascii_is_written_as_escapes_so_any_console_can_print_it(site: str) -> None:
    render, _ = _FIELD_SITES[site]
    line = render("3.51.3é")
    assert line.isascii()
    assert "3.51.3\\xe9" in line


@pytest.mark.parametrize("site", list(_FIELD_SITES))
def test_an_overlong_value_is_clipped_at_every_site(site: str) -> None:
    render, _ = _FIELD_SITES[site]
    line = render("Q" * 5_000)
    assert ("Q" * 81) not in line  # nothing past the clip survives
    assert ("Q" * 60 + "...") in line  # it was cut and marked, not dropped
    assert len(line) < 500


def test_the_clip_is_exactly_80_characters_plus_an_ellipsis() -> None:
    line = script.format_annotation(at_risk=False, version="9" * 5_000, interpreter="x")
    assert ("9" * 80 + "...") in line
    assert ("9" * 81) not in line
    assert len(line) < 400


# ---------------------------------------------------------------------------
# The version survives a failure elsewhere in the report
# ---------------------------------------------------------------------------
#
# The version is the datum this script exists to collect. A range check that raises, or a
# platform query that fails, costs the label of the table or the description of the
# interpreter, never the line that carries the version.


def _pin_sqlite_version(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    monkeypatch.setattr("sqlite3.sqlite_version", dotted)
    monkeypatch.setattr("sqlite3.sqlite_version_info", _as_tuple(dotted))


def _exploding_check(error_type: type[BaseException]) -> Any:
    def check(version_info: tuple[int, ...]) -> bool:
        raise error_type("odd version")

    return check


@pytest.mark.parametrize("error_type", [RuntimeError, SystemExit], ids=["exception", "sys-exit"])
@pytest.mark.parametrize(("dotted", "expected"), [("3.46.1", True), ("3.51.3", False)])
def test_a_range_check_that_raises_falls_back_to_the_embedded_answer(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException], dotted: str, expected: bool
) -> None:
    _pin_sqlite_version(monkeypatch, dotted)
    monkeypatch.setattr(script, "describe_interpreter", lambda: _INTERPRETER)
    monkeypatch.setattr(
        script, "resolve_wal_reset_risk", lambda: (_exploding_check(error_type), "the real table")
    )
    summary, annotation = script.collect_lines()
    assert summary == (
        f"SQLite {dotted} linked by CPython 3.12.3 on Linux x86_64; "
        f"range table: embedded copy (range check failed: {error_type.__name__})"
    )
    assert annotation.startswith(f"{'::warning::' if expected else '::notice::'}SQLite {dotted} (")


def test_main_prints_the_version_when_the_range_check_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _pin_sqlite_version(monkeypatch, "3.46.1")
    monkeypatch.setattr(script, "describe_interpreter", lambda: _INTERPRETER)
    monkeypatch.setattr(
        script, "resolve_wal_reset_risk", lambda: (_exploding_check(RuntimeError), "the real table")
    )
    assert script.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "SQLite 3.46.1 linked by CPython 3.12.3 on Linux x86_64; "
        "range table: embedded copy (range check failed: RuntimeError)",
        _WARNING_3_46_1,
    ]


def test_a_keyboard_interrupt_in_the_range_check_still_stops_the_script() -> None:
    """Ctrl-C is the one thing deliberately not contained, here as in `main`."""
    with pytest.raises(KeyboardInterrupt):
        script.evaluate_risk(_exploding_check(KeyboardInterrupt), "the real table", (3, 46, 1))


def test_a_keyboard_interrupt_in_a_platform_query_still_stops_the_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted() -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(script.platform, "machine", interrupted)
    with pytest.raises(KeyboardInterrupt):
        script.describe_interpreter()


def test_a_range_check_that_works_keeps_its_own_label(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_sqlite_version(monkeypatch, "3.46.1")
    monkeypatch.setattr(script, "describe_interpreter", lambda: _INTERPRETER)
    monkeypatch.setattr(script, "resolve_wal_reset_risk", lambda: (lambda info: False, "my table"))
    summary, annotation = script.collect_lines()
    assert summary.endswith("range table: my table")
    assert annotation.startswith("::notice::")  # the check's answer is used, not the copy's


def test_a_platform_query_that_raises_costs_the_description_not_the_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> str:
        raise OSError("uname failed")

    monkeypatch.setattr(script.platform, "system", broken)
    assert script.describe_interpreter() == "unknown interpreter (OSError)"

    _pin_sqlite_version(monkeypatch, "3.46.1")
    summary, annotation = script.collect_lines()
    assert summary.startswith(
        "SQLite 3.46.1 linked by unknown interpreter (OSError); range table: "
    )
    assert annotation.startswith("::warning::SQLite 3.46.1 (unknown interpreter (OSError)) is in")


def test_the_interpreter_description_names_implementation_version_system_and_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is the only thing telling a reader which runner or architecture a line is from."""
    monkeypatch.setattr(script.platform, "python_implementation", lambda: "PyPyX")
    monkeypatch.setattr(script.platform, "python_version", lambda: "9.8.7")
    monkeypatch.setattr(script.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(script.platform, "machine", lambda: "riscv64")
    assert script.describe_interpreter() == "PyPyX 9.8.7 on Plan9 riscv64"


def test_range_sentence_names_the_boundaries_of_the_real_function() -> None:
    """The sentence is text, so nothing else notices when the table moves.

    The versions it names are read off `trelix.store.db.wal_reset_risk` over a sweep: the
    first affected version, the last affected one, and every version where an affected run
    ends in a fix. The sentence must name exactly those, so a new fixed line or a moved
    boundary fails here until the sentence (and the embedded copy) are edited to match.
    """
    sweep = [(3, minor, patch) for minor in range(0, 61) for patch in range(0, 21)]
    at_risk = [(version, wal_reset_risk(version)) for version in sweep]
    affected = [version for version, risky in at_risk if risky]
    fix_releases = [
        version
        for (previous, was_risky), (version, risky) in zip(at_risk, at_risk[1:], strict=False)
        if was_risky and not risky
    ]
    assert fix_releases == [(3, 44, 6), (3, 50, 7), (3, 51, 3)]  # the sweep saw every fix
    derived = [affected[0], affected[-1], *sorted(fix_releases, reverse=True)]

    annotation = script.format_annotation(at_risk=True, version="v", interpreter="i")
    named = re.findall(r"\d+\.\d+\.\d+", annotation)

    assert named == ["3.7.0", "3.51.2", "3.51.3", "3.50.7", "3.44.6"]
    assert named == [".".join(str(part) for part in version) for version in derived]


# ---------------------------------------------------------------------------
# Linear time on hostile input (rule: anything that sanitises text)
# ---------------------------------------------------------------------------
#
# Only `escape_workflow_data` sees the whole text: every other path clips to 80 characters
# first, so it does constant work however long the input is, and a quadratic
# `escape_workflow_data` hides behind it at 50,000 characters (a slicing variant measured
# 0.55 s there, under the bound). A quadratic pass added at a call site, before the clip,
# can hide at 50,000 too (one with a small constant measured 1.3 s there and 21 s at
# 200,000). Hence two sizes, and every path runs at both: 50,000 is the baseline, 200,000
# is where a quadratic variant leaves the 2 s bound and this implementation still takes a
# few milliseconds.

_HOSTILE_LENGTH = 50_000
_ESCAPE_HOSTILE_LENGTH = 200_000
_TIME_LIMIT_SECONDS = 2.0


def _hostile_inputs(length: int) -> dict[str, str]:
    return {
        "percent": "%" * length,
        "newlines": "\n" * length,
        "carriage-returns": "\r\n" * (length // 2),
        "backticks": "`" * length,
        "tildes": "~" * length,
        "spaces": " " * length,
        "nested-brackets": "[" * (length // 2) + "]" * (length // 2),
        "unterminated-comment": "<!--" * (length // 4),
        "zero-width": "\u200b" * length,
        "combining": "e" + "\u0301" * (length - 1),
        "astral": "\U0001f600" * length,
        "command-prefix": "::error::x\n" * (length // 11 + 1),
    }


_HOSTILE_INPUTS = _hostile_inputs(_HOSTILE_LENGTH)
_LARGE_HOSTILE_INPUTS = _hostile_inputs(_ESCAPE_HOSTILE_LENGTH)


def _seconds(call: Any, *args: Any, **kwargs: Any) -> float:
    started = time.perf_counter()
    call(*args, **kwargs)
    return time.perf_counter() - started


def _assert_every_path_is_fast(hostile: str) -> None:
    assert _seconds(script.escape_workflow_data, hostile) < _TIME_LIMIT_SECONDS
    assert _seconds(script.sanitize_field, hostile) < _TIME_LIMIT_SECONDS
    assert (
        _seconds(script.format_annotation, at_risk=True, version=hostile, interpreter=hostile)
        < _TIME_LIMIT_SECONDS
    )
    assert (
        _seconds(
            script.format_summary_line,
            version=hostile,
            interpreter=hostile,
            table_source=hostile,
        )
        < _TIME_LIMIT_SECONDS
    )
    assert _seconds(script.format_failure_annotation, ValueError(hostile)) < _TIME_LIMIT_SECONDS


@pytest.mark.parametrize("name", list(_HOSTILE_INPUTS))
def test_escaping_and_formatting_are_linear_on_hostile_text(name: str) -> None:
    hostile = _HOSTILE_INPUTS[name]
    assert len(hostile) >= _HOSTILE_LENGTH
    _assert_every_path_is_fast(hostile)


@pytest.mark.parametrize("name", list(_LARGE_HOSTILE_INPUTS))
def test_escaping_and_formatting_are_linear_at_200000_characters(name: str) -> None:
    hostile = _LARGE_HOSTILE_INPUTS[name]
    assert len(hostile) >= _ESCAPE_HOSTILE_LENGTH
    _assert_every_path_is_fast(hostile)
