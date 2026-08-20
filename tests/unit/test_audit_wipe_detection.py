"""Store-level pins for the twelfth detected shape: a total wipe of the log.

Four places used to call an emptied ``audit.db`` "byte-indistinguishable" from a
legitimately new one. It is not. ``audit_log.id`` is ``INTEGER PRIMARY KEY
AUTOINCREMENT``, so SQLite keeps the highest id it ever handed out in its internal
``sqlite_sequence`` table and ``DELETE`` does not reset it. Zero rows with
``seq > 0`` is a state normal operation cannot produce, which is the branch's own
principle applied to a discriminator that was already in the file.

**It closes the sloppy wipe and nothing else.** Four defeats are pinned below as
still working, because a limitation published in docs/AUDIT.md is a claim like any
other, and two measured look-alikes are pinned as clean: a rolled-back append (no
``seq`` row at all) and a restored ``sqlite3 .dump`` (a DUPLICATE ``seq`` row).

The other two claims from the same change — crash-proof reads and read-only reads —
are in tests/unit/test_audit_read_hardening.py.
"""

from __future__ import annotations

import ast
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import (
    REASON_COUNT_MISMATCH,
    REASON_LOG_EMPTIED,
    AuditStore,
)


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


def test_a_seq_set_BELOW_the_real_row_count_is_STILL_not_detected(tmp_path: Path) -> None:
    """DELIBERATELY NOT DETECTED, and it is the one defeat that could be closed.

    ``seq`` below ``COUNT(*)`` is impossible for a legitimate writer — the
    high-water mark only ever moves up, and every copying operation preserves it.
    So this *could* be a finding. It is not, because the shipped check is scoped to
    an empty log and nothing more: widening it is a separate change with its own
    false-positive surface (pruning, restored partial dumps) to establish. Named in
    docs/AUDIT.md as a defeat that works, and pinned here so that stays true.
    """
    db = _seed(tmp_path / "audit.db")

    _raw_exec(db, "UPDATE sqlite_sequence SET seq = 1 WHERE name = 'audit_log'")

    assert _seq_rows(db) == [("audit_log", 1)]
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
