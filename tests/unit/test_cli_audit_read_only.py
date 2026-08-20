"""``trelix audit`` at the command level: no traceback, and no write, ever.

Four things are pinned here that the store-level tests cannot see.

1. NO STORED VALUE MAY END A COMMAND IN A TRACEBACK. Measured before the fix:
   61 tracebacks across 648 real-CLI invocations, one root cause — a TEXT cell
   holding undecodable UTF-8 raised inside ``fetchall()``, upstream of the check
   written to judge it.

2. ``export`` MUST NOT DIE ON A BLOB. ``json.dumps`` raised "Object of type bytes
   is not JSON serializable", and export is the command an offline verifier and
   every SIEM shipper depend on.

3. THE READ COMMANDS MUST NOT WRITE THE FILE THEY READ. Byte-for-byte, plus
   ``user_version``, ``sqlite_master`` and the absence of sidecar files. The
   published claim was false: pointed at a foreign database, ``verify`` grew it
   from 8 KB to 32 KB, added five tables, and reported an intact chain.

4. A DAMAGED FILE IS NOT A TAMPER VERDICT. A corrupted b-tree page must exit 2
   (could not check), not 1 (tamper detected) on a traceback.

The exit-code contract and markup-safety pins live in tests/unit/test_cli_audit.py,
which is deliberately left untouched; anchor-fault messages live in
tests/unit/test_cli_audit_anchor.py; store-level verdicts in
tests/unit/test_audit_read_hardening.py.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import AuditStore
from trelix.cli.main import app

runner = CliRunner()

#: See ``_combined_output`` in tests/unit/test_cli_audit.py — a leaked logging
#: handler bound to a closed buffer writes this and can satisfy substring checks.
_LOGGING_ERROR_MARKER = "--- Logging error ---"

#: A TEXT cell holding invalid UTF-8; ``typeof(value)`` is still ``'text'``.
_UNDECODABLE = "CAST(x'FFFE41' AS TEXT)"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the console width so Rich does not wrap a message mid-assertion."""
    monkeypatch.setenv("COLUMNS", "200")


def _event(i: int) -> AuditEvent:
    return AuditEvent(
        ts=f"2026-08-12T10:00:{i:02d}Z",
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


def _seed(db: Path, count: int = 5) -> Path:
    store = AuditStore(db)
    for i in range(1, count + 1):
        assert store.append(_event(i)) is True
    store.close()
    return db


def _raw_exec(db: Path, *statements: str) -> None:
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        for sql in statements:
            conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def _run(*argv: str) -> tuple[int, str, BaseException | None]:
    """Invoke the CLI and return (exit code, combined output, real crash or None).

    ``typer.Exit`` reaches CliRunner as ``SystemExit``, so "did this crash" cannot
    be ``exception is None`` — every nonzero verdict has one. Anything that is not
    a ``SystemExit`` is a real crash, which is the point of most of this file.
    """
    result = runner.invoke(app, list(argv))
    try:
        stderr = result.stderr
    except ValueError:  # click 8.1 mixes stderr into .output and raises here
        stderr = ""
    combined = f"{result.output}{stderr}"
    assert _LOGGING_ERROR_MARKER not in combined, (
        "an earlier test leaked a logging handler bound to a closed CliRunner buffer; "
        f"these substring assertions cannot be trusted. Captured:\n{combined}"
    )
    crash = None if isinstance(result.exception, SystemExit) else result.exception
    return result.exit_code, combined, crash


def _fingerprint(db: Path) -> tuple[str, int, tuple[str, ...]]:
    """(sha256 of the file, user_version, every sqlite_master name)."""
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        names = tuple(sorted(str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master")))
    finally:
        conn.close()
    return digest, version, names


def _foreign_db(path: Path) -> Path:
    """A perfectly ordinary SQLite file that is not an audit log."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany("INSERT INTO customers(name) VALUES(?)", [("ada",), ("grace",)])
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    finally:
        conn.close()
    return path


def _damage_a_btree_page(db: Path) -> int:
    """Overwrite 256 bytes of one b-tree page, leaving the FILE openable.

    Returns the offset used. Asserts the precondition the F6 tests need and cannot
    get any other way: ``sqlite_master`` must still read, so the open guard does
    NOT fire and only the read guard can produce exit 2. If the page choice ever
    stops satisfying that, these tests must fail loudly rather than pass for the
    wrong reason.
    """
    original = db.read_bytes()
    page_size = int.from_bytes(original[16:18], "big") or 65536
    for page in range(2, len(original) // page_size + 1):
        offset = (page - 1) * page_size + 8
        if offset + 256 > len(original):
            break
        db.write_bytes(original)
        with db.open("r+b") as handle:
            handle.seek(offset)
            handle.write(b"\xde\xad\xbe\xef" * 64)
        if _opens_but_cannot_read_the_log(db):
            return offset
    db.write_bytes(original)
    pytest.fail("no page produced a damaged-but-openable database")


def _opens_but_cannot_read_the_log(db: Path) -> bool:
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.DatabaseError:
        return False
    else:
        try:
            conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        except sqlite3.DatabaseError:
            return True
        return False
    finally:
        conn.close()


# ===========================================================================
# 1. No stored value may end a command in a traceback
# ===========================================================================
@pytest.mark.parametrize("key", ["count", "head_hash"])
def test_verify_on_an_undecodable_anchor_exits_1_without_a_traceback(
    tmp_path: Path, key: str
) -> None:
    """THE BUG: an unhandled ``sqlite3.OperationalError`` — no verdict, no id.

    The plain-BLOB case was already handled (exit 1, ``anchor_corrupt``, no
    traceback), which is what makes this a placement bug and not a logic bug: the
    value never reached the check. A responder gets a reason code, or an alert has
    nothing to match on.
    """
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, f"UPDATE audit_meta SET value = {_UNDECODABLE} WHERE key = '{key}'")

    exit_code, combined, crash = _run("audit", "verify", "--db", str(db))

    assert crash is None, f"the data it checks crashed the checker: {crash!r}"
    assert exit_code == 1
    assert "TAMPERED" in combined
    assert "(anchor_corrupt)" in combined


@pytest.mark.parametrize("column", ["principal", "resource", "detail", "prev_hash", "entry_hash"])
def test_verify_on_an_undecodable_row_column_exits_1_without_a_traceback(
    tmp_path: Path, column: str
) -> None:
    """The same fault in ``audit_log`` — verify reads every column of every row."""
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, f"UPDATE audit_log SET {column} = {_UNDECODABLE} WHERE id = 3")

    exit_code, combined, crash = _run("audit", "verify", "--db", str(db))

    assert crash is None, f"crashed on column {column}: {crash!r}"
    assert exit_code == 1
    assert "first divergent entry id: 3" in combined


@pytest.mark.parametrize("column", ["principal", "resource", "detail"])
def test_list_and_export_survive_an_undecodable_row_column(tmp_path: Path, column: str) -> None:
    """``list`` and ``export`` are not verifiers: they show what is stored.

    Both used ``SELECT *``, so a single planted cell killed them too — and these
    are the commands that hand an incident responder the log itself.
    """
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, f"UPDATE audit_log SET {column} = {_UNDECODABLE} WHERE id = 3")

    for argv in (["audit", "list"], ["audit", "export"]):
        exit_code, _, crash = _run(*argv, "--db", str(db))
        assert crash is None, f"{argv} crashed on column {column}: {crash!r}"
        assert exit_code == 0, f"{argv} exited {exit_code}"


# ===========================================================================
# 2. export must not die on a BLOB
# ===========================================================================
def test_export_emits_valid_ndjson_for_a_blob_column(tmp_path: Path) -> None:
    """THE BUG: ``TypeError: Object of type bytes is not JSON serializable``.

    A column's declared type does not constrain what SQLite stores, so any BLOB in
    any column killed the whole export — including the rows before it, because the
    exception arrives mid-stream. ``default=str`` mirrors ``_canonical_hash``: the
    value is rendered rather than dropped, so an operator can see that the cell is
    not text.
    """
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, "UPDATE audit_log SET detail = x'DEADBEEF' WHERE id = 3")

    exit_code, combined, crash = _run("audit", "export", "--db", str(db))

    assert crash is None, f"a single BLOB killed export: {crash!r}"
    assert exit_code == 0
    rows = [json.loads(line) for line in combined.splitlines() if line.startswith("{")]
    assert [r["id"] for r in rows] == [1, 2, 3, 4, 5], "rows were lost mid-stream"
    assert rows[2]["detail"] == "b'\\xde\\xad\\xbe\\xef'"


# ===========================================================================
# 3. The read commands must not write the file they read
# ===========================================================================
@pytest.mark.parametrize("command", ["verify", "list", "export"])
def test_a_foreign_database_exits_2_and_is_not_changed_by_one_byte(
    tmp_path: Path, command: str
) -> None:
    """THE BUG, and the reason the old pin was blind to it.

    Measured before: ``user_version=7 size=8192 tables=customers`` in,
    ``size=32768 tables=customers,audit_log,sqlite_sequence,indexes,audit_meta``
    out, and "Audit chain intact." at exit 0 — a green integrity verdict on a
    database that is not an audit log. The existing pin started from an
    already-initialised ``audit.db``, where the DDL is a no-op, so it structurally
    could not see this. All three commands, because all three opened it the same way.
    """
    db = _foreign_db(tmp_path / "customers.db")
    before = _fingerprint(db)

    exit_code, combined, crash = _run("audit", command, "--db", str(db))

    assert crash is None
    assert exit_code == 2, "could not check must be 2, not 0 and not 1"
    assert "intact" not in combined.lower(), "claimed a clean chain about a foreign database"
    assert "Not an audit database" in combined
    assert _fingerprint(db) == before, "a read-only command modified the file"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["customers.db"]


def test_a_zero_byte_file_exits_2_and_stays_zero_bytes(tmp_path: Path) -> None:
    """Measured before: 0 -> 28672 bytes, five tables created, "intact", exit 0.

    A zero-byte file is a valid empty SQLite database as far as the driver is
    concerned, so the existence check passed and the DDL then built a whole audit
    schema inside it.
    """
    db = tmp_path / "empty.db"
    db.write_bytes(b"")

    exit_code, combined, crash = _run("audit", "verify", "--db", str(db))

    assert crash is None
    assert exit_code == 2
    assert "intact" not in combined.lower()
    assert db.stat().st_size == 0


@pytest.mark.parametrize("command", ["verify", "list", "export"])
def test_a_real_audit_db_is_not_changed_by_one_byte(tmp_path: Path, command: str) -> None:
    """The positive half of the claim, on the file the commands are FOR.

    The sidecar assertion matters as much as the digest: a ``-journal`` or ``-wal``
    file appearing next to ``audit.db`` is writing, even if the main file's bytes
    are unchanged when the process exits.
    """
    db = _seed(tmp_path / "audit.db", 40)
    before = _fingerprint(db)

    exit_code, _, crash = _run("audit", command, "--db", str(db))

    assert crash is None
    assert exit_code == 0
    assert _fingerprint(db) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audit.db"]


def test_verifying_a_TAMPERED_db_still_writes_nothing(tmp_path: Path) -> None:
    """The path most likely to be forgotten: the one that reports a finding.

    A verdict of tamper must not modify the evidence — an operator's next step is
    to hand the file to someone else, and a changed digest destroys that.
    """
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, "UPDATE audit_log SET principal = 'evil' WHERE id = 3")
    before = _fingerprint(db)

    exit_code, combined, crash = _run("audit", "verify", "--db", str(db))

    assert crash is None
    assert exit_code == 1
    assert "(row_mutated)" in combined
    assert _fingerprint(db) == before


# ===========================================================================
# 4. A damaged file is not a tamper verdict
# ===========================================================================
@pytest.mark.parametrize("command", ["verify", "list", "export"])
def test_a_corrupted_page_exits_2_without_a_traceback(tmp_path: Path, command: str) -> None:
    """THE BUG: a traceback and exit 1 — the tamper exit code — on a damaged disk.

    256 bytes of garbage over a b-tree page leaves the file openable and makes the
    first row read raise ``sqlite3.DatabaseError``. Exit 1 with no reason code
    tells an alert "tampered" and tells the operator nothing; restoring from backup
    and hunting an intruder are different first moves.
    """
    db = _seed(tmp_path / "audit.db", 60)
    _damage_a_btree_page(db)

    exit_code, combined, crash = _run("audit", command, "--db", str(db))

    assert crash is None, f"{command} tracebacked on a damaged page: {crash!r}"
    assert exit_code == 2, "a damaged file is 'could not check', not 'tampered'"
    assert "TAMPERED" not in combined
    assert "intact" not in combined.lower()
    assert "Could not read audit database" in combined


# ===========================================================================
# The twelfth detection, at the command level
# ===========================================================================
def test_a_total_wipe_is_reported_as_tamper_with_its_own_reason_code(tmp_path: Path) -> None:
    """Previously "Audit chain intact." at exit 0.

    ``log_emptied`` is an unprovable-id reason: every surviving row verified —
    there are none — so the message must not send a responder to inspect entry 1.
    """
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, "DELETE FROM audit_log", "DELETE FROM audit_meta")

    exit_code, combined, crash = _run("audit", "verify", "--db", str(db))

    assert crash is None
    assert exit_code == 1
    assert "(log_emptied)" in combined
    assert "no surviving entry diverged" in combined
    assert "first unprovable entry id: 1" in combined


def test_an_empty_but_never_appended_db_still_exits_0(tmp_path: Path) -> None:
    """Control for the test above: the state that must stay clean.

    A brand-new ``audit.db`` has zero rows, no anchor and no ``sqlite_sequence``
    row. If the wipe detection ever fired here, enabling auditing would report
    tamper before the first request.
    """
    db = tmp_path / "audit.db"
    AuditStore(db).close()

    exit_code, combined, crash = _run("audit", "verify", "--db", str(db))

    assert crash is None
    assert exit_code == 0
    assert "Audit chain intact." in combined
