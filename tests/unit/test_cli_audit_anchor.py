"""``trelix audit verify`` at the command level: anchor faults, and no traceback.

Two things are pinned here that the store-level tests cannot see.

1. THE MESSAGE MUST NOT LIE ABOUT WHICH ROW IS WRONG. Every tamper verdict used
   the one sentence "first divergent entry id: <id>". For a row fault that is
   true. For an anchor fault it is not: the surviving rows chain perfectly, and
   the id reported is the first one whose *existence* can no longer be proven —
   there is no row there to inspect. Sending a responder to look at a row that is
   not in the table is a small lie with a real cost during an incident.

2. THE CHECKER MUST NOT BE CRASHABLE BY THE DATA IT CHECKS. A one-character
   ``UPDATE`` of ``audit_meta.count`` made this command exit on an unhandled
   ``ValueError`` — no verdict, no id, a Python traceback. That is a denial of
   the tooling an incident responder reaches for first, achievable by anyone who
   can write the file they are trying to hide activity in.

The exit-code contract (0 intact / 1 tamper / 2 could-not-check) and the
markup-safety pins live in tests/unit/test_cli_audit.py, which is deliberately
left untouched; store-level verdicts live in
tests/unit/test_audit_anchor_presence.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import AuditStore
from trelix.cli.main import app

runner = CliRunner()

#: stdlib ``logging.Handler.handleError`` writes this to stderr when a handler's
#: own ``emit()`` raises. See ``_combined_output`` in tests/unit/test_cli_audit.py
#: for the leaked-handler mechanism; the assertions below are substring checks
#: over captured text and a stray traceback would silently satisfy some of them.
_LOGGING_ERROR_MARKER = "--- Logging error ---"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the console width so Rich does not wrap a message mid-assertion."""
    monkeypatch.setenv("COLUMNS", "200")


def _event(i: int) -> AuditEvent:
    return AuditEvent(
        ts=f"2026-08-12T10:00:0{i}Z",
        principal=f"user-{i}@https://idp.example",
        action=ACTION_AUTH,
        resource=f"/search#{i}",
        outcome=OUTCOME_SUCCESS,
        status_code=200,
        client_ip="10.0.0.1",
        request_id=f"req-{i}",
        trace_id=None,
        duration_ms=i,
        detail=None,
    )


def _seed(db: Path, count: int = 4) -> None:
    store = AuditStore(db)
    for i in range(1, count + 1):
        assert store.append(_event(i)) is True
    store.close()


def _raw_exec(db: Path, *statements: str) -> None:
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        for sql in statements:
            conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def _crash(exception: BaseException | None) -> BaseException | None:
    """The exception, unless it is the clean ``typer.Exit`` a verdict raises.

    ``typer.Exit(1)`` reaches CliRunner as ``SystemExit(1)``, so "did this command
    crash" cannot be asked as ``exception is None`` — the tamper path always has
    one. Anything else is a real crash, which is the whole point of the
    malformed-anchor tests below.
    """
    if isinstance(exception, SystemExit):
        return None
    return exception


def _run_verify(db: Path) -> tuple[int, str, BaseException | None]:
    result = runner.invoke(app, ["audit", "verify", "--db", str(db)])
    output = result.output
    try:
        stderr = result.stderr
    except ValueError:  # click 8.1 mixes stderr into .output and raises here
        stderr = ""
    combined = f"{output}{stderr}"
    assert _LOGGING_ERROR_MARKER not in combined, (
        "an earlier test leaked a logging handler bound to a closed CliRunner buffer; "
        f"these substring assertions cannot be trusted. Captured:\n{combined}"
    )
    return result.exit_code, combined, _crash(result.exception)


# --- the anchor attack, end to end -----------------------------------------
def test_verify_reports_a_wiped_anchor_as_tamper(tmp_path: Path) -> None:
    """THE BUG at the command level: this printed "Audit chain intact." exit 0.

    A tail deletion plus ``DELETE FROM audit_meta`` — one extra statement, no
    hashing — and the command that a CI integrity gate calls returned success.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)
    _raw_exec(
        db,
        "DELETE FROM audit_log WHERE id = (SELECT MAX(id) FROM audit_log)",
        "DELETE FROM audit_meta",
    )

    exit_code, combined, exception = _run_verify(db)

    assert exception is None, f"verify crashed with {exception!r}"
    assert exit_code == 1
    assert "TAMPERED" in combined
    assert "anchor_missing" in combined
    assert "intact" not in combined.lower()


def test_verify_does_not_call_an_anchor_fault_a_divergent_entry(tmp_path: Path) -> None:
    """The message names what is actually unknown, not a row that is fine.

    All four surviving rows verify; what the wiped anchor destroyed is any proof
    that a fifth entry never existed. Calling id 5 "the first divergent entry"
    would be a claim about a row that is not in the table.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)
    _raw_exec(db, "DELETE FROM audit_meta")

    exit_code, combined, _ = _run_verify(db)

    assert exit_code == 1
    assert "divergent" not in combined
    assert "first unprovable entry id: 5" in combined


def test_verify_on_a_malformed_count_exits_1_without_a_traceback(tmp_path: Path) -> None:
    """THE CRASH: ``UPDATE audit_meta SET value='x' WHERE key='count'``.

    ``int()`` in the read path meant this exited on ``ValueError`` — the command
    an operator runs to find out whether the log is trustworthy could be made to
    produce no verdict at all. Now it is a verdict.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)
    _raw_exec(db, "UPDATE audit_meta SET value = 'x' WHERE key = 'count'")

    exit_code, combined, exception = _run_verify(db)

    assert exception is None, f"verify crashed with {exception!r}"
    assert exit_code == 1
    assert "TAMPERED" in combined
    assert "anchor_corrupt" in combined


def test_verify_on_an_absurdly_long_count_exits_1_without_a_traceback(tmp_path: Path) -> None:
    """The same crash by a different route: CPython refuses ``int()`` above 4300
    digits, so the value is rejected on shape before it is ever parsed."""
    db = tmp_path / "audit.db"
    _seed(db, 4)
    _raw_exec(db, "UPDATE audit_meta SET value = printf('%.5000d', 9) WHERE key = 'count'")

    exit_code, combined, exception = _run_verify(db)

    assert exception is None, f"verify crashed with {exception!r}"
    assert exit_code == 1
    assert "anchor_corrupt" in combined


# --- CONTROLS: the sentences that were already true stay true ---------------
def test_control_row_fault_still_names_the_first_divergent_entry_id(tmp_path: Path) -> None:
    """CONTROL — passes before and after.

    Row 2 is present and its content no longer matches its stored hash, so
    "first divergent entry id: 2" is a true statement and the wording that
    tests/unit/test_cli_audit.py pins must not drift.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)
    _raw_exec(db, "UPDATE audit_log SET principal='attacker' WHERE id=2")

    exit_code, combined, _ = _run_verify(db)

    assert exit_code == 1
    assert "first divergent entry id: 2" in combined


def test_row_fault_also_carries_its_reason_code(tmp_path: Path) -> None:
    """The reason is printed for every verdict, not only the anchor ones.

    An alert that has to distinguish "a row was edited" from "the anchor is gone"
    needs a stable token to match on; the free-text sentence is for humans.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)
    _raw_exec(db, "UPDATE audit_log SET principal='attacker' WHERE id=2")

    _, combined, _ = _run_verify(db)

    assert "row_mutated" in combined


def test_control_clean_chain_still_exits_0(tmp_path: Path) -> None:
    """CONTROL — widening what counts as tamper must not flag a healthy log."""
    db = tmp_path / "audit.db"
    _seed(db, 4)

    exit_code, combined, _ = _run_verify(db)

    assert exit_code == 0
    assert "Audit chain intact." in combined


def test_control_empty_but_openable_log_still_exits_0(tmp_path: Path) -> None:
    """CONTROL — a brand-new database has no anchor rows yet, and that is normal.

    This is the state the presence check has to leave alone, and the reason it is
    gated on there being at least one entry.
    """
    db = tmp_path / "audit.db"
    AuditStore(db).close()

    exit_code, combined, _ = _run_verify(db)

    assert exit_code == 0
    assert "Audit chain intact." in combined
