"""Regression tests for the ``trelix audit`` sub-app (trelix.cli.main).

Two audit-tooling defects are pinned here.

1. FALSE INTEGRITY ASSURANCE. ``AuditStore.__init__`` deliberately never
   raises — a broken audit sink must not take down the host — so an unopenable
   path yields a *disabled* store whose ``verify_chain()`` returns ``None``.
   The CLI printed that as "Audit chain intact." and exited 0, for a database
   it had never opened: ``None`` ambiguously meant both "verified clean" and
   "never checked". ``_open_audit_store`` now gates on ``AuditStore.is_open``,
   making the exit-code contract explicit:

     0 = chain intact
     1 = tamper detected (the message names the first divergent id)
     2 = could not check

2. REQUEST-SIDE DoS OF THE AUDIT LOG. Audit rows record attacker-controlled
   data (a request path, a JWT ``sub``). An unauthenticated ``GET /%5B/red%5D``
   stored ``resource="/[/red]"``, and ``trelix audit list`` then died with
   ``rich.errors.MarkupError`` (exit 1) having rendered ZERO rows — destroying
   exactly the record a responder needs mid-incident. Every cell is escaped
   now, so markup-shaped values render literally.

Style mirrors tests/unit/test_cli_smoke.py: real CliRunner invocations against
the real ``app``, with a real (temp) audit.db. See tests/unit/test_audit_store.py
for the store-level ``is_open`` tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trelix.audit.events import ACTION_AUTH, OUTCOME_DENIED, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import AuditStore
from trelix.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the Rich console width so cell contents are never ellipsized.

    Rich falls back to 80 columns when stdout is not a terminal (as under
    CliRunner), and an 80-column, 7-column table shrinks the ``principal``
    cell to "[bold]evil[…" — which would make the literal-payload assertions
    below fail for a reason that has nothing to do with escaping. Rich reads
    ``COLUMNS`` live from ``os.environ``, so setenv works even though the
    module-level Console was built at import time.
    """
    monkeypatch.setenv("COLUMNS", "200")


def _event(i: int, **overrides: object) -> AuditEvent:
    base: dict[str, object] = {
        "ts": f"2026-08-12T10:00:0{i}Z",
        "principal": f"user-{i}@https://idp.example",
        "action": ACTION_AUTH,
        "resource": f"/search#{i}",
        "outcome": OUTCOME_SUCCESS,
        "status_code": 200,
        "client_ip": "10.0.0.1",
        "request_id": f"req-{i}",
        "trace_id": f"trace-{i}",
        "duration_ms": i,
        "detail": None,
    }
    base.update(overrides)
    return AuditEvent(**base)  # type: ignore[arg-type]


def _seed(db_path: Path, count: int = 3, **overrides: object) -> None:
    store = AuditStore(db_path)
    for i in range(1, count + 1):
        assert store.append(_event(i, **overrides)) is True
    store.close()


def _combined_output(result: object) -> str:
    """stdout plus stderr, across click versions.

    click 8.1's CliRunner mixes stderr into ``.output`` and raises ValueError
    on ``.stderr``; 8.2+ captures them separately. The audit commands print
    failures to ``err_console`` (stderr) and successes to stdout, so both
    streams matter here.
    """
    output = result.output  # type: ignore[attr-defined]
    try:
        stderr = result.stderr  # type: ignore[attr-defined]
    except ValueError:
        stderr = ""
    return f"{output}{stderr}"


# --- exit code 2: could not check -----------------------------------------
def test_verify_on_unopenable_db_exits_2_and_never_claims_intact(tmp_path: Path) -> None:
    """THE BUG: this printed "Audit chain intact." and exited 0.

    ``tmp_path`` is a directory, so sqlite cannot open it as a database — the
    store is disabled and nothing is ever read. Reporting that as a clean
    chain is a false integrity assurance, so it must be exit 2 instead.
    """
    result = runner.invoke(app, ["audit", "verify", "--db", str(tmp_path)])
    combined = _combined_output(result)

    assert result.exit_code == 2
    assert "intact" not in combined.lower()
    assert "Could not open audit database" in combined


def test_list_on_unopenable_db_exits_2_not_a_silent_empty_result(tmp_path: Path) -> None:
    """An unreadable log must not look like an empty log."""
    result = runner.invoke(app, ["audit", "list", "--db", str(tmp_path)])
    combined = _combined_output(result)

    assert result.exit_code == 2
    assert "No audit entries" not in combined
    assert "Could not open audit database" in combined


def test_export_on_unopenable_db_exits_2_not_a_silent_empty_result(tmp_path: Path) -> None:
    """A pipeline consuming NDJSON must be able to tell "no rows" from
    "never opened" — otherwise it exports an empty file and exits 0."""
    result = runner.invoke(app, ["audit", "export", "--db", str(tmp_path)])
    combined = _combined_output(result)

    assert result.exit_code == 2
    assert "Could not open audit database" in combined


# --- exit code 0 / 1: intact vs tampered ----------------------------------
def test_verify_clean_chain_exits_0_and_says_intact(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    _seed(db, 3)

    result = runner.invoke(app, ["audit", "verify", "--db", str(db)])

    assert result.exit_code == 0
    assert "Audit chain intact." in _combined_output(result)


def test_verify_tampered_chain_exits_1_and_names_the_divergent_id(tmp_path: Path) -> None:
    """Exit 1 is reserved for detected tamper, and the operator needs the id."""
    db = tmp_path / "audit.db"
    _seed(db, 3)

    # Mutate row 2's content through a separate connection, leaving its stored
    # hashes untouched — the same tampering test_audit_store.py simulates.
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        conn.execute("UPDATE audit_log SET principal='attacker' WHERE id=2")
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(app, ["audit", "verify", "--db", str(db)])
    combined = _combined_output(result)

    assert result.exit_code == 1
    assert "TAMPERED" in combined
    assert "first divergent entry id: 2" in combined
    assert "intact" not in combined.lower()


# --- rich markup in attacker-controlled columns ----------------------------
def test_list_renders_markup_shaped_values_literally(tmp_path: Path) -> None:
    """THE BUG: ``GET /%5B/red%5D`` stored "/[/red]", and rendering it raised
    rich.errors.MarkupError — exit 1, zero rows, audit log unreadable.

    Asserting only "no exception" would be weak (a command that renders
    nothing also raises nothing), so every payload's literal text must appear
    in the table too.
    """
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    assert (
        store.append(
            _event(
                1,
                principal="[bold]evil[/bold]@https://idp.example",
                action="[not a tag]",
                resource="/[/red]",
                outcome=OUTCOME_DENIED,
                status_code=401,
            )
        )
        is True
    )
    # A second, ordinary row proves the table did not stop at the hostile one.
    assert store.append(_event(2)) is True
    store.close()

    result = runner.invoke(app, ["audit", "list", "--db", str(db)])
    combined = _combined_output(result)

    assert result.exception is None, f"rendering raised {result.exception!r}"
    assert result.exit_code == 0
    for payload in ("[/red]", "[bold]evil[/bold]", "[not a tag]"):
        assert payload in combined, f"{payload!r} was swallowed by Rich markup parsing"
    # The ordinary row rendered too — a MarkupError aborted the whole table.
    assert "/search#2" in combined


def test_list_renders_closing_tag_only_resource_without_markup_error(tmp_path: Path) -> None:
    """The minimal reproducer: a lone closing tag is unparseable markup, so it
    is what actually raised MarkupError (an opening tag merely swallows text)."""
    from rich.errors import MarkupError

    db = tmp_path / "audit.db"
    store = AuditStore(db)
    assert store.append(_event(1, resource="/[/red]")) is True
    store.close()

    result = runner.invoke(app, ["audit", "list", "--db", str(db)])

    assert not isinstance(result.exception, MarkupError)
    assert result.exit_code == 0
    assert "/[/red]" in _combined_output(result)
