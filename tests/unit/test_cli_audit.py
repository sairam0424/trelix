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

Beyond those two pins, the remaining branches of the sub-app are covered here
directly: ``audit list``'s empty-result branch (an *openable* log with nothing
to show, which must not look like the exit-2 unopenable case), and ``audit
export``'s format gate plus its NDJSON happy path (one machine-readable JSON
object per line, oldest first, chain hashes included, markup-shaped values
byte-preserved because export uses builtin ``print`` and not the Rich console).

Style mirrors tests/unit/test_cli_smoke.py: real CliRunner invocations against
the real ``app``, with a real (temp) audit.db. See tests/unit/test_audit_store.py
for the store-level ``is_open`` tests.
"""

from __future__ import annotations

import json
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


#: stdlib ``logging.Handler.handleError`` writes this to ``sys.stderr`` when a
#: handler's own ``emit()`` raises. See ``_combined_output``.
_LOGGING_ERROR_MARKER = "--- Logging error ---"


def _combined_output(result: object) -> str:
    """stdout plus stderr, across click versions — and a guard on the contents.

    click 8.1's CliRunner mixes stderr into ``.output`` and raises ValueError
    on ``.stderr``; 8.2+ captures them separately. The audit commands print
    failures to ``err_console`` (stderr) and successes to stdout, so both
    streams matter here.

    The guard exists because this module's assertions are substring checks over
    that captured text, and a leaked log handler contaminates it with a Python
    traceback. Unlike every other command in cli/main.py, none of the ``audit``
    subcommands call ``_setup_logging()``, so they run against whatever root
    handler the session happens to have. ``_setup_logging()`` builds a bare
    ``logging.StreamHandler()`` bound to the ``sys.stderr`` of its moment — under
    CliRunner, a buffer that is closed when that invocation ends — and leaves it
    on the root logger. The next invocation that logs anything (``audit verify``
    on an unopenable path logs a WARNING from audit/store.py) hits the closed
    buffer, and ``handleError`` dumps a full traceback into THIS invocation's
    captured output.

    That traceback carries the pytest frame stack, i.e. the running test's own
    name — which is how ``test_verify_on_unopenable_db_exits_2_and_never_claims_intact``
    failed its own ``"intact" not in combined`` check by matching the tail of its
    own function name, with the actual cause 128 truncated lines away. The other
    two unopenable-path tests were worse: their assertions happened to still hold
    over the contaminated text, so they passed while asserting over a traceback.

    Nine modules leak the handler, each measured by running it before this one
    (``pytest tests/unit/test_cli_audit.py tests/unit/<mod>.py`` with collection
    reversed → 3 failures here): test_cli_migrate_vectors, test_cli_smoke,
    test_cli_watch_all_signals, test_dimension_guard, test_dry_run,
    test_git_linker, test_graph_api, test_prune, test_review_pr_json.
    test_cli_markup_safety is the only CLI-invoking module that snapshots and
    restores ``root.handlers`` (its ``_no_leaked_log_handler`` fixture), and it
    measures 0. Asserting here turns the contamination into one line naming the
    leak; it does not prevent the leak, which belongs in a shared fixture.
    """
    output = result.output  # type: ignore[attr-defined]
    try:
        stderr = result.stderr  # type: ignore[attr-defined]
    except ValueError:
        stderr = ""
    combined = f"{output}{stderr}"
    assert _LOGGING_ERROR_MARKER not in combined, (
        "an earlier test leaked a logging.StreamHandler bound to a now-closed CliRunner "
        "buffer onto the root logger, so this invocation's log records dumped a traceback "
        "into its own captured output. The audit assertions below are substring checks "
        f"over that text and cannot be trusted. Captured:\n{combined}"
    )
    return combined


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


def test_verify_on_NONEXISTENT_db_exits_2_and_creates_nothing(tmp_path: Path) -> None:
    """The gap the test above did not cover, and the reason the guard never fired.

    ``tmp_path`` is a DIRECTORY, so sqlite cannot open it, ``is_open`` is False,
    and the original guard fired correctly — that test passed. An ABSENT path
    behaves oppositely: ``AuditStore(path)`` creates the file and its schema, so
    ``is_open`` is True, the guard was skipped, and `audit verify` printed
    "Audit chain intact." with exit 0 for a path it had never read, leaving a
    fresh ~28 KB SQLite file behind.

    That is a false all-clear from a read-only integrity command: a CI gate
    pointed at a typo'd or not-yet-created --db passed green. The existence check
    now runs BEFORE construction.
    """
    missing = tmp_path / "does_not_exist.db"
    assert not missing.exists()

    result = runner.invoke(app, ["audit", "verify", "--db", str(missing)])
    combined = _combined_output(result)

    assert result.exit_code == 2, "could not check must be 2, not 0"
    assert "intact" not in combined.lower(), "claimed a clean chain it never read"
    assert not missing.exists(), "a read-only verify created the database"


def test_list_and_export_on_nonexistent_db_also_exit_2(tmp_path: Path) -> None:
    """The same absent-path hole applied to every audit subcommand, not just verify."""
    missing = tmp_path / "absent.db"

    for argv in (["audit", "list"], ["audit", "export"]):
        result = runner.invoke(app, [*argv, "--db", str(missing)])
        assert result.exit_code == 2, f"{argv} exited {result.exit_code}"
        assert not missing.exists(), f"{argv} created the database"


def test_verify_on_a_real_db_still_reports_intact_and_exits_0(tmp_path: Path) -> None:
    """Control: the existence check must not break the working path."""
    db = tmp_path / "audit.db"
    _seed(db, count=3)

    result = runner.invoke(app, ["audit", "verify", "--db", str(db)])

    assert result.exit_code == 0
    assert "intact" in _combined_output(result).lower()


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


# --- `audit list`: the empty-but-openable log ------------------------------
def test_list_on_openable_empty_db_says_no_entries_and_exits_0(tmp_path: Path) -> None:
    """An openable log with zero rows is a *success*, not a failure.

    This is the other half of the exit-code contract: exit 2 means "could not
    check", so a log that opened fine and simply has nothing in it must exit 0
    and say so — and must NOT render a table header, which would suggest rows
    exist. Constructing the store creates the schema, so the file is a valid,
    empty audit.db rather than an unopenable path.
    """
    db = tmp_path / "audit.db"
    AuditStore(db).close()
    assert db.exists()

    result = runner.invoke(app, ["audit", "list", "--db", str(db)])
    combined = _combined_output(result)

    assert result.exit_code == 0
    assert "No audit entries." in combined
    assert "Could not open audit database" not in combined
    # The table (title + header row) must not be printed for an empty log.
    assert "Audit Log" not in combined
    assert "principal" not in combined


def test_list_with_nonpositive_limit_shows_nothing_instead_of_the_whole_log(
    tmp_path: Path,
) -> None:
    """``--limit -1`` must show no rows, not every row.

    ``AuditStore.recent`` guards ``n <= 0`` and returns ``[]``. Without that
    guard the value reaches sqlite as ``LIMIT -1``, which sqlite defines as *no
    limit* — so a negative ``--limit`` would silently dump the entire audit log
    to the terminal. Pinned here because the guard is invisible at this layer.
    """
    db = tmp_path / "audit.db"
    _seed(db, 3)

    result = runner.invoke(app, ["audit", "list", "--limit", "-1", "--db", str(db)])
    combined = _combined_output(result)

    assert result.exit_code == 0
    assert "No audit entries." in combined
    assert "/search#1" not in combined
    assert "/search#3" not in combined


# --- `audit export`: format gate ------------------------------------------
def test_export_with_unsupported_format_exits_1_and_names_the_format(tmp_path: Path) -> None:
    """A typo'd ``--format`` must fail loudly and emit no rows.

    Silently falling back to NDJSON (or emitting a partial stream before
    complaining) would let a pipeline believe it received the format it asked
    for. The message has to name the offending value so the operator can see
    the typo, and nothing from the log may reach stdout.
    """
    db = tmp_path / "audit.db"
    _seed(db, 3)

    result = runner.invoke(app, ["audit", "export", "--format", "csv", "--db", str(db)])
    combined = _combined_output(result)

    assert result.exit_code == 1
    assert "Unsupported format" in combined
    assert "'csv'" in combined
    assert "only 'ndjson' is supported" in combined
    # Not one exported row escaped the gate.
    assert "user-1@https://idp.example" not in combined
    assert "entry_hash" not in combined


def test_export_validates_format_before_touching_the_database(tmp_path: Path) -> None:
    """Exit 1 (bad format), not exit 2 (could not open), when both are wrong.

    The format gate runs before ``_open_audit_store``, so a bad ``--format``
    reports the bad format even if ``--db`` is also unusable (``tmp_path`` is a
    directory). Ordering matters: telling the user their database is broken
    when the real mistake was the format sends them down the wrong path.
    """
    result = runner.invoke(app, ["audit", "export", "--format", "csv", "--db", str(tmp_path)])
    combined = _combined_output(result)

    assert result.exit_code == 1
    assert "Unsupported format" in combined
    assert "Could not open audit database" not in combined


# --- `audit export`: the NDJSON happy path ---------------------------------
@pytest.mark.parametrize("format_args", [[], ["--format", "ndjson"]])
def test_export_emits_one_json_object_per_line_oldest_first(
    tmp_path: Path, format_args: list[str]
) -> None:
    """The actual export: valid NDJSON, oldest first, chain hashes included.

    Parametrized over the default and the explicit ``--format ndjson`` so the
    documented default cannot drift away from the named format. ``iter_for_export``
    yields append order (oldest first) — the opposite of ``audit list``'s
    newest-first ``recent()`` — because an offline verifier has to replay the
    hash chain forwards, which is also why ``prev_hash``/``entry_hash`` must be
    part of every exported row.
    """
    db = tmp_path / "audit.db"
    _seed(db, 3)

    result = runner.invoke(app, ["audit", "export", *format_args, "--db", str(db)])

    assert result.exception is None, f"export raised {result.exception!r}"
    assert result.exit_code == 0

    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 3, f"expected 3 NDJSON lines, got {lines!r}"

    rows = []
    for line in lines:
        row = json.loads(line)  # must be valid JSON on its own — this is NDJSON
        assert isinstance(row, dict), f"expected a JSON object per line, got {row!r}"
        rows.append(row)

    required = {
        "id",
        "ts",
        "principal",
        "action",
        "resource",
        "outcome",
        "status_code",
        "client_ip",
        "request_id",
        "trace_id",
        "duration_ms",
        "detail",
        "prev_hash",
        "entry_hash",
    }
    for row in rows:
        assert required <= set(row), f"missing keys: {sorted(required - set(row))}"

    assert [row["id"] for row in rows] == [1, 2, 3]
    assert [row["resource"] for row in rows] == ["/search#1", "/search#2", "/search#3"]
    assert [row["principal"] for row in rows] == [
        f"user-{i}@https://idp.example" for i in (1, 2, 3)
    ]
    assert [row["status_code"] for row in rows] == [200, 200, 200]
    assert rows[0]["prev_hash"] == "0" * 64  # genesis
    # The chain links forward, which is what an offline verifier replays.
    assert [row["prev_hash"] for row in rows[1:]] == [row["entry_hash"] for row in rows[:-1]]
    assert rows[0]["detail"] is None  # SQL NULL round-trips as JSON null, not ""


def test_export_preserves_markup_shaped_values_byte_for_byte(tmp_path: Path) -> None:
    """Export is machine-readable, so it must NOT go through the Rich console.

    ``audit list`` escapes markup for display; export must do the opposite and
    reproduce the stored bytes exactly. Routing this through ``console.print``
    instead of builtin ``print`` would strip "[/bold]" as a style tag and wrap
    long lines, corrupting the JSON — so the payload is re-read from the parsed
    object, not merely searched for in the output.
    """
    db = tmp_path / "audit.db"
    hostile = "/[/red] [bold]x[/bold]"
    store = AuditStore(db)
    assert (
        store.append(_event(1, resource=hostile, outcome=OUTCOME_DENIED, status_code=401)) is True
    )
    store.close()

    result = runner.invoke(app, ["audit", "export", "--db", str(db)])

    assert result.exit_code == 0
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected 1 NDJSON line, got {lines!r}"
    row = json.loads(lines[0])
    assert row["resource"] == hostile
    assert row["outcome"] == OUTCOME_DENIED
    assert row["status_code"] == 401
