"""Store-level pins for two claims the audit reader now makes.

1. **No stored value can crash a read.** ``CAST(x'FFFE41' AS TEXT)`` keeps TEXT
   storage class while holding bytes sqlite3 cannot decode, so ``fetchall()``
   raised ``sqlite3.OperationalError`` *before* the anchor shape validation could
   run. The validation was right and in the wrong place. Every such value is now
   a finding: ``anchor_corrupt`` in ``audit_meta``, a row fault in ``audit_log``.

2. **The reading paths never write the database they read.** ``read_only=True``
   opens ``mode=ro`` and runs no DDL, and it refuses to say anything at all about
   a file that does not already contain both audit tables. The read-write
   constructor still creates the file and its schema — that is the serving path,
   and the claim is deliberately narrow.

The third claim added at the same time — that a *sloppy* total wipe is now
detected, and exactly which four defeats still work — lives in
tests/unit/test_audit_wipe_detection.py. CLI-level pins (exit codes, tracebacks,
byte-identity through the real command) live in
tests/unit/test_cli_audit_read_only.py. The detected/undetected shapes that predate
all of this live in tests/unit/test_audit_anchor_presence.py and
tests/unit/test_audit_undetectable.py.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import (
    REASON_ANCHOR_CORRUPT,
    REASON_ANCHOR_MISSING,
    REASON_ROW_MISLINKED,
    REASON_ROW_MUTATED,
    AuditStore,
)

#: A TEXT cell holding invalid UTF-8. ``typeof(value)`` is still ``'text'``.
_UNDECODABLE = "CAST(x'FFFE41' AS TEXT)"


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
        detail=f"detail-{i}",
    )


def _seed(db: Path, count: int = 5) -> Path:
    store = AuditStore(db)
    for i in range(1, count + 1):
        assert store.append(_event(i)) is True
    assert store.verify_chain() is None
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


def _verify(db: Path) -> tuple[int | None, str | None]:
    """Verify through the READ-ONLY path — the one the CLI uses."""
    store = AuditStore(db, read_only=True)
    try:
        assert store.is_open, f"read-only open failed: missing={store.missing_tables}"
        result = store.verify()
        return result.tampered_id, result.reason
    finally:
        store.close()


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


# ===========================================================================
# 1. No stored value can crash a read
# ===========================================================================
@pytest.mark.parametrize("key", ["count", "head_hash"])
def test_an_undecodable_anchor_value_is_reported_not_raised(tmp_path: Path, key: str) -> None:
    """THE BUG: this raised ``sqlite3.OperationalError`` out of ``fetchall()``.

    ``_anchor_value_is_wellformed`` already rejected a BLOB on ``isinstance`` —
    that case exits 1 with ``anchor_corrupt`` and always did. A TEXT cell holding
    undecodable bytes never reached it: sqlite3's default ``text_factory`` decodes
    strict UTF-8 while building the row, so the exception fired one layer below the
    validation. Right check, wrong place.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, f"UPDATE audit_meta SET value = {_UNDECODABLE} WHERE key = '{key}'")

    assert _verify(db) == (6, REASON_ANCHOR_CORRUPT)


def test_undecodable_values_in_both_anchor_rows_are_reported(tmp_path: Path) -> None:
    """Both at once is still one verdict, not a crash."""
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, f"UPDATE audit_meta SET value = {_UNDECODABLE}")

    assert _verify(db) == (6, REASON_ANCHOR_CORRUPT)


def test_an_undecodable_anchor_KEY_reads_as_a_missing_anchor(tmp_path: Path) -> None:
    """A mangled key is the anchor row being gone, which is already a finding.

    ``_read_meta_locked`` keys its dict on ``audit_meta.key``; bytes there simply
    never match ``'count'``, so the presence gate reports ``anchor_missing``. Worth
    pinning because the alternative — a ``bytes`` key raising during dict
    construction — was the pre-fix behaviour.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, f"UPDATE audit_meta SET key = {_UNDECODABLE} WHERE key = 'count'")

    assert _verify(db) == (6, REASON_ANCHOR_MISSING)


@pytest.mark.parametrize(
    ("column", "expected_reason"),
    [
        ("ts", REASON_ROW_MUTATED),
        ("principal", REASON_ROW_MUTATED),
        ("action", REASON_ROW_MUTATED),
        ("resource", REASON_ROW_MUTATED),
        ("outcome", REASON_ROW_MUTATED),
        ("detail", REASON_ROW_MUTATED),
        ("prev_hash", REASON_ROW_MISLINKED),
        ("entry_hash", REASON_ROW_MUTATED),
    ],
)
def test_an_undecodable_audit_log_column_is_reported_as_a_row_fault(
    tmp_path: Path, column: str, expected_reason: str
) -> None:
    """Every TEXT column of ``audit_log``, not just the anchor.

    Measured before the fix: 15 of 15 (five columns x verify/list/export) ended in
    a traceback. A row holding bytes cannot be the row ``append`` wrote, so the
    honest verdict is a row fault — and a row fault is where the named id really is
    a row you can go and look at.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, f"UPDATE audit_log SET {column} = {_UNDECODABLE} WHERE id = 3")

    assert _verify(db) == (3, expected_reason)


def test_recent_and_export_hand_back_raw_bytes_rather_than_raising(tmp_path: Path) -> None:
    """``list`` and ``export`` read every column, so they crashed too.

    They are not verifiers and must not invent a verdict; they hand the operator
    the bytes that are actually stored and let them see it.
    """
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, f"UPDATE audit_log SET principal = {_UNDECODABLE} WHERE id = 3")

    store = AuditStore(db, read_only=True)
    try:
        recent = store.recent(10)
        exported = list(store.iter_for_export())
    finally:
        store.close()

    assert [r["principal"] for r in recent if r["id"] == 3] == [b"\xff\xfeA"]
    assert [r["principal"] for r in exported if r["id"] == 3] == [b"\xff\xfeA"]


def test_the_text_factory_does_not_change_legitimate_values(tmp_path: Path) -> None:
    """Control: decodable text still arrives as ``str``, and the chain still verifies.

    The fix replaces sqlite3's ``text_factory``, which is the decoder every read in
    this module goes through. If it altered ordinary values, every ``entry_hash``
    would recompute differently and a clean log would report ``row_mutated``.
    """
    db = _seed(tmp_path / "audit.db")

    store = AuditStore(db, read_only=True)
    try:
        rows = list(store.iter_for_export())
        assert store.verify().tampered_id is None
    finally:
        store.close()

    assert [type(r["principal"]) for r in rows] == [str] * 5
    assert rows[2]["principal"] == "user-3@https://idp.example"
    assert rows[2]["detail"] == "detail-3"


# ===========================================================================
# 2. The reading paths never write
# ===========================================================================
def test_read_only_open_of_a_foreign_database_gives_no_verdict(tmp_path: Path) -> None:
    """THE BUG: this reported an intact chain about a customers database.

    The read commands used the writer's constructor, so opening any SQLite file
    ran ``CREATE TABLE IF NOT EXISTS`` against it: an 8 KB one-table file became a
    32 KB five-table one, and ``verify`` then walked the zero rows it had just
    created and called them intact. Nothing here may be a verdict about a chain,
    because there is no chain.
    """
    db = tmp_path / "customers.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO customers(name) VALUES(?)", [("a",), ("b",)])
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()
    before = _fingerprint(db)

    store = AuditStore(db, read_only=True)
    try:
        assert store.is_open is False
        assert store.missing_tables == ("audit_log", "audit_meta")
    finally:
        store.close()

    assert _fingerprint(db) == before, "a read-only open modified the file"
    assert before[2] == ("customers",)


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        ("audit_log", "audit_meta"),
        ("audit_meta", "audit_log"),
    ],
)
def test_read_only_open_refuses_a_file_holding_only_one_audit_table(
    tmp_path: Path, present: str, missing: str
) -> None:
    """Both tables are required, because either one alone proves nothing.

    ``audit_meta`` alone cannot be walked. ``audit_log`` alone is worse: the
    presence gate reads a missing anchor as tamper, so a half-shaped file would
    have produced ``anchor_missing`` — a confident tamper verdict about a file that
    was never an audit log. "Could not check" is the only true answer.
    """
    db = tmp_path / "half.db"
    conn = sqlite3.connect(str(db))
    try:
        if present == "audit_log":
            conn.execute("CREATE TABLE audit_log(id INTEGER PRIMARY KEY, ts TEXT)")
        else:
            conn.execute("CREATE TABLE audit_meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()
    finally:
        conn.close()

    store = AuditStore(db, read_only=True)
    try:
        assert store.is_open is False
        assert store.missing_tables == (missing,)
    finally:
        store.close()


def test_read_only_open_of_an_absent_path_creates_nothing(tmp_path: Path) -> None:
    """``mode=ro`` cannot create a database, so the file stays absent."""
    missing = tmp_path / "nope.db"

    store = AuditStore(missing, read_only=True)
    try:
        assert store.is_open is False
    finally:
        store.close()

    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_read_only_open_creates_no_parent_directory(tmp_path: Path) -> None:
    """The read-write constructor mkdirs the parent; the reader must not."""
    nested = tmp_path / "a" / "b" / "audit.db"

    AuditStore(nested, read_only=True).close()

    assert not (tmp_path / "a").exists()


def test_a_read_only_store_cannot_append(tmp_path: Path) -> None:
    """The failure contract applies: ``append`` returns ``False``, never writes."""
    db = _seed(tmp_path / "audit.db")
    before = _fingerprint(db)

    store = AuditStore(db, read_only=True)
    try:
        assert store.append(_event(99)) is False
    finally:
        store.close()

    assert _fingerprint(db) == before


def test_a_full_read_of_a_real_chain_changes_not_one_byte(tmp_path: Path) -> None:
    """verify + recent + export over a real chain, then a byte-for-byte compare.

    ``verify`` opens a DEFERRED read transaction (the snapshot fix), which is the
    one place a read could have taken a write lock or left a journal behind. The
    sidecar assertion is part of the claim: a ``-journal``/``-wal`` file appearing
    next to the database IS writing to the database's directory.
    """
    db = _seed(tmp_path / "audit.db", 40)
    before = _fingerprint(db)

    store = AuditStore(db, read_only=True)
    try:
        assert store.verify().tampered_id is None
        assert len(store.recent(100)) == 40
        assert len(list(store.iter_for_export())) == 40
    finally:
        store.close()

    assert _fingerprint(db) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audit.db"]


def test_the_read_write_constructor_still_creates_the_schema(tmp_path: Path) -> None:
    """The narrow claim, stated as a test: only the READ paths are read-only.

    The serving path has to create ``audit.db``, run its DDL and take a write lock
    to append. Saying "trelix never writes an audit database it opens" would be the
    same kind of overclaim this change exists to remove.
    """
    db = tmp_path / "fresh.db"

    store = AuditStore(db)
    try:
        assert store.is_open is True
    finally:
        store.close()

    assert db.exists()
    assert "audit_log" in _fingerprint(db)[2]
