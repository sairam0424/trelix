"""Unit tests for the tamper-evident audit store (trelix.audit.store).

Covers: hash-chain integrity, the genesis hash, mutation detection, tail /
middle deletion detection, query-hashing privacy, resilient write failure,
SQL-injection safety, and the guarantee that a secret never lands raw in any
column.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from trelix.audit.events import (
    ACTION_AUTH,
    ACTION_SEARCH,
    OUTCOME_DENIED,
    OUTCOME_SUCCESS,
    AuditEvent,
)
from trelix.audit.store import (
    _CONTENT_COLUMNS,
    GENESIS_HASH,
    AuditStore,
    _canonical_hash,
)


def _event(i: int, **overrides) -> AuditEvent:  # type: ignore[no-untyped-def]
    base = {
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


def _raw_exec(db_path: Path, sql: str) -> None:
    """Mutate the DB file through a separate connection (bypassing the store's
    meta-anchor bookkeeping), simulating tampering."""
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


# --- happy path ------------------------------------------------------------
def test_chain_integrity_over_n_appends(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    for i in range(1, 8):
        assert store.append(_event(i)) is True

    assert store.verify_chain() is None

    # Independently recompute the whole chain from the exported rows.
    prev = GENESIS_HASH
    rows = list(store.iter_for_export())
    assert len(rows) == 7
    for row in rows:
        content = {col: row[col] for col in _CONTENT_COLUMNS}
        assert row["prev_hash"] == prev
        assert row["entry_hash"] == _canonical_hash(prev, content)
        prev = row["entry_hash"]


def test_genesis_hash_is_sixty_four_zeros(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    assert store.append(_event(1)) is True

    rows = list(store.iter_for_export())
    first = rows[0]
    assert first["prev_hash"] == "0" * 64 == GENESIS_HASH
    content = {col: first[col] for col in _CONTENT_COLUMNS}
    assert first["entry_hash"] == _canonical_hash(GENESIS_HASH, content)


def test_recent_returns_newest_first(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    for i in range(1, 6):
        store.append(_event(i))
    recent = store.recent(3)
    assert [r["request_id"] for r in recent] == ["req-5", "req-4", "req-3"]


# --- tamper detection ------------------------------------------------------
def test_verify_chain_names_mutated_rows_id(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 5):
        store.append(_event(i))
    assert store.verify_chain() is None

    # Mutate the content of row id=2 without touching its stored hashes.
    _raw_exec(db, "UPDATE audit_log SET principal='attacker' WHERE id=2")

    assert store.verify_chain() == 2


def test_verify_chain_detects_truncated_tail(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 5):  # ids 1..4
        store.append(_event(i))
    assert store.verify_chain() is None

    # Delete the tail row directly — the surviving 1..3 still form a valid
    # chain, so only the count/head anchor can catch this.
    _raw_exec(db, "DELETE FROM audit_log WHERE id = (SELECT MAX(id) FROM audit_log)")

    result = store.verify_chain()
    assert result is not None
    assert result == 4  # first missing id


def test_verify_chain_detects_deleted_middle_row(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 5):
        store.append(_event(i))

    _raw_exec(db, "DELETE FROM audit_log WHERE id = 2")

    # Gapless check fires at the missing id.
    assert store.verify_chain() == 2


# --- privacy: query hashing ------------------------------------------------
def test_log_queries_false_stores_sha256_not_raw(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    query = "how does password reset work"
    store.append(_event(1, action=ACTION_SEARCH, detail=query))  # default log_queries=False

    row = list(store.iter_for_export())[0]
    expected = "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert row["detail"] == expected
    assert query not in (row["detail"] or "")
    # Chain must still verify with the hashed detail baked in.
    assert store.verify_chain() is None


def test_log_queries_true_stores_raw(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    query = "select * from users"
    store.append(_event(1, action=ACTION_SEARCH, detail=query), log_queries=True)
    row = list(store.iter_for_export())[0]
    assert row["detail"] == query


def test_secret_never_appears_raw_in_any_column(tmp_path: Path) -> None:
    """A token-looking value fed in as a search query is hashed, and never
    lands raw in any column of any row."""
    store = AuditStore(tmp_path / "audit.db")
    # Built at runtime (not a literal) so it reads as a token but is obviously
    # synthetic and never trips secret scanners.
    secret = "-".join(["sk", "live", "DEADBEEF0123456789ABCDEF"])

    # Auth-denied event carries no token; search query happens to be the secret.
    store.append(_event(1, action=ACTION_AUTH, outcome=OUTCOME_DENIED, detail="invalid api key"))
    store.append(_event(2, action=ACTION_SEARCH, detail=secret))  # log_queries=False -> hashed

    seen_hash = False
    expected_hash = "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()
    for row in store.iter_for_export():
        for col in row:
            value = "" if row[col] is None else str(row[col])
            assert secret not in value, f"raw secret leaked in column {col!r}"
            if value == expected_hash:
                seen_hash = True
    assert seen_hash, "query should have been stored as its sha256 hash"


# --- SQL-injection safety --------------------------------------------------
def test_sql_injection_resource_stored_literally(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    payload = "'; DROP TABLE audit_log;--"

    assert store.append(_event(1, resource=payload)) is True
    # Table must survive and further appends must work.
    assert store.append(_event(2)) is True

    rows = list(store.iter_for_export())
    assert rows[0]["resource"] == payload  # stored verbatim, not executed
    assert store.verify_chain() is None

    # Table still exists and is queryable through a fresh connection.
    conn = sqlite3.connect(str(db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    finally:
        conn.close()
    assert count == 2


# --- resilient write failure ----------------------------------------------
def test_unwritable_path_returns_false_no_raise(tmp_path: Path) -> None:
    # Pointing the store at a directory makes sqlite unable to open a DB file;
    # construction must not raise and append must return False.
    store = AuditStore(tmp_path)  # tmp_path is a directory
    result = store.append(_event(1))
    assert result is False


def test_unwritable_path_fail_closed_raises(tmp_path: Path) -> None:
    store = AuditStore(tmp_path)  # directory -> disabled store
    with pytest.raises(Exception):
        store.append(_event(1), fail_closed=True)


# --- is_open: "never checked" is not "verified clean" ----------------------
def test_is_open_true_for_a_usable_path(tmp_path: Path) -> None:
    assert AuditStore(tmp_path / "audit.db").is_open is True


def test_is_open_false_for_unopenable_path(tmp_path: Path) -> None:
    """Regression: init swallows open failures by design (a broken audit sink
    must never crash the caller), so a disabled store is indistinguishable
    from a healthy one by its read results alone — ``verify_chain()`` returns
    ``None``, which reads exactly like "chain intact". Any caller that REPORTS
    integrity has to gate on ``is_open`` first; see
    tests/unit/test_cli_audit.py for the CLI-level guard that used to print
    "Audit chain intact." and exit 0 for a database it never opened."""
    store = AuditStore(tmp_path)  # a directory — sqlite cannot open it

    assert store.is_open is False
    # The ambiguity itself: no divergence reported, yet nothing was checked.
    assert store.verify_chain() is None


def test_is_open_false_after_close(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    assert store.append(_event(1)) is True
    store.close()

    assert store.is_open is False
