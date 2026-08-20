"""The shapes an in-DB anchor provably CANNOT detect, pinned as such.

Every one of these asserts ``verify_chain() is None`` — that is, it asserts the
absence of a capability, deliberately. They exist because
docs/AUDIT.md and the :mod:`trelix.audit.store` docstring now enumerate these
shapes by name, and a published limitation is a claim like any other: if one of
them ever becomes detectable, the test fails and whoever made it detectable has
to go and update the documentation that says it is not.

The shared root cause is that the chain and the anchor it is checked against live
in the same file. Verification can only ever catch *incomplete* tampering — a
writer who updates the anchor consistently passes, and three of the shapes below
cost no hash computation at all -- for the truncation case because the value the
anchor then needs is already sitting in a surviving row.

What would close these is an anchor the attacker cannot write: export
``(count, head_hash)`` off-box — to a CI artifact, a SIEM, an object-lock (WORM)
bucket, another host — and compare on every run. None of that ships in trelix, and
this file is not a plan to build it; it is the honest boundary of what does ship.

Detected shapes are pinned in tests/unit/test_audit_anchor_presence.py and
tests/unit/test_audit_store.py.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import _CONTENT_COLUMNS, GENESIS_HASH, AuditStore, _canonical_hash


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
    assert store.verify_chain() is None
    store.close()


def _raw_exec(db: Path, *statements: str) -> None:
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        for sql in statements:
            conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def _verify(db: Path) -> tuple[int | None, str | None]:
    store = AuditStore(db)
    try:
        result = store.verify()
        return result.tampered_id, result.reason
    finally:
        store.close()


def _rows(db: Path) -> int:
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])
    finally:
        conn.close()


def test_a_total_wipe_of_both_tables_is_not_detected(tmp_path: Path) -> None:
    """NOT DETECTED: two DELETEs, no hashing.

    An emptied database is byte-indistinguishable from a legitimately new one:
    zero entries and no anchor is exactly the state ``AuditStore(path)`` creates,
    which is also why that state has to verify clean. Nothing inside the file
    records that a chain ever began.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, "DELETE FROM audit_log", "DELETE FROM audit_meta")

    assert _rows(db) == 0
    assert _verify(db) == (None, None)


def test_a_truncated_tail_with_both_anchor_values_realigned_is_not_detected(
    tmp_path: Path,
) -> None:
    """NOT DETECTED, and it needs NO hashing whatsoever.

    This is the shape that matters most — erasing recent activity — and it is the
    cheapest. The new head is *copied out of a surviving row*, and the new count
    is ``COUNT(*)``; neither value has to be computed. The claim that an attack
    requires recomputing every subsequent ``entry_hash`` was true only of a
    content rewrite, and materially overstated the cost of this one.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(
        db,
        "DELETE FROM audit_log WHERE id = (SELECT MAX(id) FROM audit_log)",
        "UPDATE audit_meta SET value = (SELECT COUNT(*) FROM audit_log) WHERE key = 'count'",
        "UPDATE audit_meta SET value = "
        "(SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1) "
        "WHERE key = 'head_hash'",
    )

    assert _rows(db) == 3
    assert _verify(db) == (None, None)


def test_a_wholly_fabricated_chain_is_not_detected(tmp_path: Path) -> None:
    """NOT DETECTED: rows invented, chain computed forwards, anchor set to match.

    ``verify`` checks that the rows are self-consistent, not that they are the
    rows that were written. Anyone who knows the hashing scheme can produce a log
    that verifies and says whatever they want it to say — here, that a single
    benign request happened and nothing else ever did.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    forged = [
        {
            "ts": "2026-08-12T09:00:00Z",
            "principal": "static-token",
            "action": ACTION_AUTH,
            "resource": "/health",
            "outcome": OUTCOME_SUCCESS,
            "status_code": 200,
            "client_ip": None,
            "request_id": "forged",
            "trace_id": None,
            "duration_ms": 1,
            "detail": None,
        }
    ]
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        conn.execute("DELETE FROM audit_log")
        prev = GENESIS_HASH
        for i, content in enumerate(forged, 1):
            entry_hash = _canonical_hash(prev, content)
            conn.execute(
                "INSERT INTO audit_log (id, ts, principal, action, resource, outcome, "
                "status_code, client_ip, request_id, trace_id, duration_ms, detail, "
                "prev_hash, entry_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (i, *[content[col] for col in _CONTENT_COLUMNS], prev, entry_hash),
            )
            prev = entry_hash
        conn.execute("UPDATE audit_meta SET value = ? WHERE key = 'count'", (str(len(forged)),))
        conn.execute("UPDATE audit_meta SET value = ? WHERE key = 'head_hash'", (prev,))
        conn.commit()
    finally:
        conn.close()

    assert _rows(db) == 1
    assert _verify(db) == (None, None)


def test_one_forged_row_appended_onto_the_real_head_is_not_detected(tmp_path: Path) -> None:
    """NOT DETECTED: one sha256, no rehashing of history.

    Adding an entry that never happened does not require touching anything that
    came before it — its ``prev_hash`` is the genuine head. Only the anchor has to
    be advanced, and the anchor is in the same file. The cost is one hash, not one
    per existing row.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    conn = sqlite3.connect(str(db), timeout=5)
    try:
        head_row = conn.execute(
            "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        head = head_row[0]
        content = {
            "ts": "2026-08-12T23:59:59Z",
            "principal": "someone-else@https://idp.example",
            "action": ACTION_AUTH,
            "resource": "/search#planted",
            "outcome": OUTCOME_SUCCESS,
            "status_code": 200,
            "client_ip": "10.0.0.9",
            "request_id": "planted",
            "trace_id": None,
            "duration_ms": 7,
            "detail": None,
        }
        entry_hash = _canonical_hash(head, content)
        conn.execute(
            "INSERT INTO audit_log (ts, principal, action, resource, outcome, status_code, "
            "client_ip, request_id, trace_id, duration_ms, detail, prev_hash, entry_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*[content[col] for col in _CONTENT_COLUMNS], head, entry_hash),
        )
        conn.execute("UPDATE audit_meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'count'")
        conn.execute("UPDATE audit_meta SET value = ? WHERE key = 'head_hash'", (entry_hash,))
        conn.commit()
    finally:
        conn.close()

    assert _rows(db) == 5
    assert _verify(db) == (None, None)


def test_deleting_the_whole_file_is_not_detected(tmp_path: Path) -> None:
    """NOT DETECTED at the store level: the next open creates a fresh chain.

    ``rm audit.db`` leaves nothing behind to contradict. The CLI is the only layer
    that can say anything here, and only about the window before something
    recreates the file: ``trelix audit verify`` on an absent path exits 2 without
    creating it (pinned in tests/unit/test_cli_audit.py). Once the serving process
    reopens the path, even that signal is gone.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    db.unlink()
    assert not db.exists()

    assert _verify(db) == (None, None)  # AuditStore(path) creates a new, empty chain
    assert _rows(db) == 0


def test_replacing_the_file_with_another_valid_chain_is_not_detected(tmp_path: Path) -> None:
    """NOT DETECTED: a different chain in the same place verifies perfectly.

    Nothing in the file identifies *which* chain it is meant to be, so a
    substituted audit.db — an older backup, or one written elsewhere — is
    indistinguishable from the original.
    """
    original = tmp_path / "audit.db"
    _seed(original, 4)
    substitute = tmp_path / "elsewhere.db"
    _seed(substitute, 1)

    shutil.copyfile(substitute, original)

    assert _rows(original) == 1  # three entries gone
    assert _verify(original) == (None, None)
