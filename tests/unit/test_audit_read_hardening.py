"""Store-level pins for three claims the audit reader now makes.

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

3. **A sloppy total wipe is detected**, via SQLite's own ``sqlite_sequence``
   high-water mark, which ``DELETE`` does not reset. Four ways to defeat that are
   pinned here as still working, because a limitation published in docs/AUDIT.md
   is a claim like any other.

CLI-level pins (exit codes, tracebacks, byte-identity through the real command)
live in tests/unit/test_cli_audit_read_only.py. The detected/undetected shapes
that predate this file live in tests/unit/test_audit_anchor_presence.py and
tests/unit/test_audit_undetectable.py.
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import (
    REASON_ANCHOR_CORRUPT,
    REASON_ANCHOR_MISSING,
    REASON_COUNT_MISMATCH,
    REASON_LOG_EMPTIED,
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


def _seq_rows(db: Path) -> list[tuple[str, int]]:
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        rows = conn.execute("SELECT name, seq FROM sqlite_sequence").fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _non_docstring_literals(source: str) -> list[str]:
    """Every ``str`` literal in *source* except module/class/function docstrings.

    SQL is always a string literal handed to ``execute``. Excluding docstrings is
    what stops the SQL-grep pin below from matching the prose that describes the
    statement it is looking for. ``ast`` drops ``#`` comments for free.
    """
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


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


# ===========================================================================
# 3. A sloppy total wipe is detected — and exactly which defeats still work
# ===========================================================================
def test_a_total_wipe_of_both_tables_is_detected(tmp_path: Path) -> None:
    """Previously NOT detected, and the rationale for that was wrong.

    Four places called an emptied ``audit.db`` "byte-indistinguishable" from a new
    one. It is not: ``audit_log`` is AUTOINCREMENT, so SQLite keeps a
    ``sqlite_sequence`` high-water mark and ``DELETE`` does not reset it. Zero rows
    with ``seq = 5`` is a state normal operation cannot produce.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, "DELETE FROM audit_log", "DELETE FROM audit_meta")

    assert _seq_rows(db) == [("audit_log", 5)]
    assert _verify(db) == (1, REASON_LOG_EMPTIED)


def test_a_fresh_never_appended_database_has_no_sequence_row_and_verifies_clean(
    tmp_path: Path,
) -> None:
    """The state that must stay clean, and the reason absence cannot be a finding."""
    db = tmp_path / "audit.db"
    AuditStore(db).close()

    assert _seq_rows(db) == []
    assert _verify(db) == (None, None)


def test_a_rolled_back_append_does_not_arm_the_check(tmp_path: Path) -> None:
    """A failed write must not look like a wipe.

    AUTOINCREMENT's high-water mark is maintained inside the transaction, so a
    rolled-back insert could plausibly have left ``seq`` behind and turned every
    failed append on an empty log into a tamper report. Measured: it does not.
    """
    db = tmp_path / "audit.db"
    AuditStore(db).close()

    conn = sqlite3.connect(str(db), timeout=5)
    try:
        conn.execute(
            "INSERT INTO audit_log(ts, principal, action, outcome, prev_hash, entry_hash) "
            "VALUES('t', 'p', 'a', 'success', 'x', 'y')"
        )
        conn.rollback()
    finally:
        conn.close()

    assert _seq_rows(db) == []
    assert _verify(db) == (None, None)


def test_a_restored_sqlite3_dump_with_a_DUPLICATE_sequence_row_verifies_clean(
    tmp_path: Path,
) -> None:
    """TRAP: ``.dump`` + restore legitimately produces two rows for one table.

    The ``sqlite3`` CLI emits ``INSERT INTO sqlite_sequence`` *and* AUTOINCREMENT
    recreates the row on the first insert, so a restored dump carries
    ``[('audit_log', 5), ('audit_log', 5)]``. A check written as "exactly one row,
    and seq == COUNT(*)" would call every restored dump tamper. All matching rows
    are reduced with ``max`` for this reason.
    """
    source = _seed(tmp_path / "source.db")
    restored = tmp_path / "restored.db"

    dump = subprocess.run(
        ["sqlite3", str(source), ".dump"], capture_output=True, text=True, check=True
    ).stdout
    subprocess.run(
        ["sqlite3", str(restored)], input=dump, capture_output=True, text=True, check=True
    )

    assert _seq_rows(restored) == [("audit_log", 5), ("audit_log", 5)]
    assert _verify(restored) == (None, None)


@pytest.mark.parametrize("how", ["vacuum", "vacuum_into", "backup"])
def test_the_copying_operations_preserve_the_sequence_and_verify_clean(
    tmp_path: Path, how: str
) -> None:
    """``VACUUM``, ``VACUUM INTO`` and ``.backup()`` all keep ``seq`` exactly."""
    source = _seed(tmp_path / "source.db")
    target = tmp_path / "target.db"

    if how == "vacuum":
        _raw_exec(source, "VACUUM")
        target = source
    elif how == "vacuum_into":
        _raw_exec(source, f"VACUUM INTO '{target}'")
    else:
        src = sqlite3.connect(str(source))
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

    assert _seq_rows(target) == [("audit_log", 5)]
    assert _verify(target) == (None, None)


@pytest.mark.parametrize(
    ("label", "extra"),
    [
        ("seq cleared", "DELETE FROM sqlite_sequence WHERE name = 'audit_log'"),
        ("seq zeroed", "UPDATE sqlite_sequence SET seq = 0 WHERE name = 'audit_log'"),
        ("seq negative", "UPDATE sqlite_sequence SET seq = -1 WHERE name = 'audit_log'"),
        ("seq blobbed", "UPDATE sqlite_sequence SET seq = x'00' WHERE name = 'audit_log'"),
    ],
)
def test_a_wipe_that_also_neutralises_the_sequence_is_STILL_not_detected(
    tmp_path: Path, label: str, extra: str
) -> None:
    """DELIBERATELY NOT DETECTED, and documented as such in docs/AUDIT.md.

    Absence, zero and a forged type are all indistinguishable from a database that
    was never appended to — the state that must verify clean. This check closes the
    sloppy wipe and nothing more; the docs say so, and this test is what keeps that
    sentence honest.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, "DELETE FROM audit_log", "DELETE FROM audit_meta", extra)

    assert _verify(db) == (None, None), f"{label} became detectable — update docs/AUDIT.md"


def test_removing_the_sequence_table_outright_is_STILL_not_detected(tmp_path: Path) -> None:
    """DELIBERATELY NOT DETECTED: ``writable_schema`` needs no extra privilege.

    ``DROP TABLE sqlite_sequence`` is refused by SQLite ("may not be dropped"), but
    ``PRAGMA writable_schema=ON`` + a ``DELETE`` from ``sqlite_master`` removes it
    for anyone who can already write the file — which is everyone this whole
    module is defending against.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(
        db,
        "DELETE FROM audit_log",
        "DELETE FROM audit_meta",
        "PRAGMA writable_schema = ON",
        "DELETE FROM sqlite_master WHERE name = 'sqlite_sequence'",
        "PRAGMA writable_schema = OFF",
    )

    assert _seq_rows(db) == []
    assert _verify(db) == (None, None)


def test_a_seq_forged_down_to_the_surviving_row_count_is_STILL_not_detected(
    tmp_path: Path,
) -> None:
    """DELIBERATELY NOT DETECTED: the check only fires on an EMPTY log.

    A partial truncation with both anchor values realigned was already undetectable
    (tests/unit/test_audit_undetectable.py); forging ``seq`` down to match adds
    nothing to detect. Pinned so nobody reads the wipe detection as coverage of
    truncation.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(
        db,
        "DELETE FROM audit_log WHERE id > 2",
        "UPDATE sqlite_sequence SET seq = 2 WHERE name = 'audit_log'",
        "UPDATE audit_meta SET value = (SELECT COUNT(*) FROM audit_log) WHERE key = 'count'",
        "UPDATE audit_meta SET value = "
        "(SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1) WHERE key = 'head_hash'",
    )

    assert _seq_rows(db) == [("audit_log", 2)]
    assert _verify(db) == (None, None)


def test_a_wipe_that_leaves_the_anchor_behind_is_still_a_count_mismatch(tmp_path: Path) -> None:
    """Precedence: the more specific verdict wins.

    ``DELETE FROM audit_log`` alone leaves ``count = 5`` in place, which names the
    fault better than ``log_emptied`` does — so the new check is last and only
    fires on an empty log that passed every other gate.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, "DELETE FROM audit_log")

    assert _verify(db) == (1, REASON_COUNT_MISMATCH)


def test_deleting_rows_would_report_log_emptied_which_is_why_pruning_must_update_this(
    tmp_path: Path,
) -> None:
    """The documented future false positive, pinned so it cannot be a surprise.

    ``TRELIX_AUDIT_RETENTION_DAYS`` is accepted but unimplemented. The moment a
    pruning job issues ``DELETE FROM audit_log``, a fully-pruned log will report
    ``log_emptied`` — exactly what this test does. Whoever implements pruning will
    have to change this test, and that is the point: the caveat in docs/AUDIT.md
    and in ``_max_audit_log_seq_locked`` has a failing test attached to it.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(
        db,
        "DELETE FROM audit_log WHERE ts < '2027-01-01'",
        "UPDATE audit_meta SET value = '0' WHERE key = 'count'",
        f"UPDATE audit_meta SET value = '{'0' * 64}' WHERE key = 'head_hash'",
    )

    assert _verify(db) == (1, REASON_LOG_EMPTIED)


def test_no_shipped_code_path_deletes_from_audit_log_today() -> None:
    """The premise the check rests on, checked instead of asserted in prose.

    "A log that reaches zero rows was wiped" only holds while no shipped code path
    deletes audit rows. This walks every string literal in ``src/`` that is not a
    docstring — so the prose in ``_max_audit_log_seq_locked`` describing the future
    false positive does not match itself — and fails the moment someone adds
    pruning without revisiting the check.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "trelix"
    offenders = sorted(
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if any(
            "delete from audit_log" in literal.lower()
            for literal in _non_docstring_literals(path.read_text(encoding="utf-8"))
        )
    )

    assert offenders == [], (
        "src/ now deletes audit rows, so an empty log is no longer proof of a wipe — "
        "update AuditStore._max_audit_log_seq_locked and docs/AUDIT.md"
    )


def test_a_wipe_survives_a_copy_of_the_file(tmp_path: Path) -> None:
    """``cp`` of a wiped database is still a wiped database.

    ``sqlite_sequence`` is a real table in the file, so the evidence travels with
    it. Worth pinning because the discriminator would be worthless if it lived in
    memory or in a sidecar file.
    """
    db = _seed(tmp_path / "audit.db")
    _raw_exec(db, "DELETE FROM audit_log", "DELETE FROM audit_meta")
    copied = tmp_path / "copy.db"
    shutil.copyfile(db, copied)

    assert _verify(copied) == (1, REASON_LOG_EMPTIED)
