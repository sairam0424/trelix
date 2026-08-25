"""Per-column and per-digest pins for the audit hash chain (trelix.audit.store).

Round-3 mutation testing found nine surviving mutants in ``src/trelix/audit/``.
Five of them dropped a single field from the hashed payload — ``status_code``,
``client_ip``, ``request_id``, ``trace_id``, ``duration_ms`` — and the whole
existing audit suite (167 tests) still passed, which means editing any of those
five columns in a stored row was **undetected by verification**. The existing
per-column test, ``test_audit_read_hardening.py::
test_an_undecodable_audit_log_column_is_reported_as_a_row_fault``, covers only the
six TEXT columns it lists by hand; and the two happy-path tests in
``test_audit_store.py`` recompute their expectation by iterating
``_CONTENT_COLUMNS`` through ``_canonical_hash``, so they hold for *any* payload
recipe the module happens to use.

WHY THIS FILE IMPORTS ``_CONTENT_COLUMNS`` (the usual rule is: never import the
expected value from the module under test). Nothing here is *pinned* to the
import. The parametrization is driven by it precisely so that coverage tracks the
real column set: add a hashed column and it gets a case automatically instead of
being silently untested, and ``test_content_columns_cover_every_hashable_schema_
column`` cross-checks that tuple against the ``audit_log`` schema in both
directions so a column added to the table but *not* to the hashed payload fails
loudly. The one place a value IS pinned —
``test_golden_entry_hash_of_a_fully_populated_event`` — writes the digest as a
literal computed outside this process and imports nothing from the module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from trelix.audit.events import ACTION_ADMIN, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import (
    _CONTENT_COLUMNS,
    REASON_LOG_EMPTIED,
    REASON_ROW_MUTATED,
    AuditStore,
)

# Columns of ``audit_log`` that are deliberately NOT part of the hashed payload,
# written as a literal: ``id`` is DB-assigned and only exists after the insert,
# and the two hash columns are the output of the hash, not an input to it.
_UNHASHED_SCHEMA_COLUMNS = frozenset({"id", "prev_hash", "entry_hash"})

# One replacement value per hashed column, each guaranteed different from what
# ``_event`` writes. Bound as a parameter, never formatted into the SQL.
_TAMPERED_VALUE: dict[str, Any] = {
    "ts": "2099-12-31T23:59:59Z",
    "principal": "attacker@https://evil.example",
    "action": "search",
    "resource": "/admin/wipe",
    "outcome": "denied",
    "status_code": 403,
    "client_ip": "198.51.100.66",
    "request_id": "req-forged",
    "trace_id": "trace-forged",
    "duration_ms": 999_999,
    "detail": "forged detail",
}

# Collection-time guard, on purpose: a hashed column with no replacement value
# here (or a stale entry here for a column that no longer exists) must be an
# ERROR while pytest is collecting, not a quietly missing parametrization.
_MISSING_VALUES = set(_CONTENT_COLUMNS) - set(_TAMPERED_VALUE)
_STALE_VALUES = set(_TAMPERED_VALUE) - set(_CONTENT_COLUMNS)
if _MISSING_VALUES or _STALE_VALUES:  # pragma: no cover - collection error path
    raise RuntimeError(
        "_TAMPERED_VALUE must have exactly one entry per hashed column; "
        f"missing={sorted(_MISSING_VALUES)} stale={sorted(_STALE_VALUES)}"
    )


def _event(i: int) -> AuditEvent:
    """An event with EVERY hashed field non-null.

    Every field has to be non-null or "change this column to something else"
    would not be a change for the null ones, and the parametrized test below
    would pass for a payload recipe that omits them. ``ACTION_ADMIN`` is not in
    ``QUERY_ACTIONS``, so ``detail`` is stored verbatim and is a real column to
    tamper with rather than a digest of one.
    """
    return AuditEvent(
        ts=f"2026-08-23T12:00:{i:02d}Z",
        principal=f"sub-{i}@https://idp.example",
        action=ACTION_ADMIN,
        resource=f"/admin/reindex#{i}",
        outcome=OUTCOME_SUCCESS,
        status_code=200,
        client_ip="203.0.113.7",
        request_id=f"req-{i}",
        trace_id=f"trace-{i}",
        duration_ms=40 + i,
        detail=f"manual reindex {i}",
    )


def _seed(db: Path, count: int = 5) -> Path:
    store = AuditStore(db)
    try:
        for i in range(1, count + 1):
            assert store.append(_event(i)) is True
        assert store.verify_chain() is None
    finally:
        store.close()
    return db


def _raw_exec(db: Path, sql: str, params: tuple[Any, ...] = ()) -> None:
    """Tamper with the file through a separate connection, bypassing the store."""
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _scalar(db: Path, sql: str, params: tuple[Any, ...] = ()) -> Any:
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        row = conn.execute(sql, params).fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def _verify(db: Path) -> tuple[int | None, str | None]:
    """Verify through the READ-ONLY path — the one ``trelix audit verify`` uses."""
    store = AuditStore(db, read_only=True)
    try:
        assert store.is_open, f"read-only open failed: missing={store.missing_tables}"
        result = store.verify()
        return result.tampered_id, result.reason
    finally:
        store.close()


# --- every hashed column, one at a time ------------------------------------
@pytest.mark.parametrize("column", _CONTENT_COLUMNS)
def test_changing_any_single_hashed_column_is_detected(tmp_path: Path, column: str) -> None:
    """MUTATION that must make this fail: dropping ``column`` from the hashed
    payload, e.g. ``content = {k: v for k, v in content.items() if k != "client_ip"}``
    at the top of ``_canonical_hash``. Applied symmetrically that mutation keeps
    append and verify self-consistent, so the whole pre-existing audit suite
    passes while an edit to that column verifies INTACT. Measured: five of the
    eleven columns — status_code, client_ip, request_id, trace_id, duration_ms —
    survived exactly that.
    """
    db = _seed(tmp_path / "audit.db")
    before = _scalar(db, "SELECT " + column + " FROM audit_log WHERE id = 3")  # noqa: S608
    new_value = _TAMPERED_VALUE[column]

    # Precondition: the UPDATE has to be a real change. A no-op UPDATE would
    # leave a valid chain, and this test would then pass for any recipe.
    assert before != new_value, (
        f"_event() no longer discriminates for column {column!r}: it already "
        f"stores {before!r}, which is the value this test tampers with"
    )

    _raw_exec(db, "UPDATE audit_log SET " + column + " = ? WHERE id = 3", (new_value,))  # noqa: S608
    assert _scalar(db, "SELECT " + column + " FROM audit_log WHERE id = 3") == new_value  # noqa: S608

    assert _verify(db) == (3, REASON_ROW_MUTATED)


def test_content_columns_cover_every_hashable_schema_column(tmp_path: Path) -> None:
    """MUTATION that must make this fail: adding a column to the ``audit_log``
    DDL without adding it to ``_CONTENT_COLUMNS`` (or removing one from
    ``_CONTENT_COLUMNS``). Two independent declarations in the module — the
    storage contract and the hashing contract — have to agree, and set equality
    is asserted in BOTH directions so an addition on either side fails.
    """
    db = _seed(tmp_path / "audit.db", count=1)
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        schema_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(audit_log)")}
    finally:
        conn.close()

    # The three names excluded from hashing are pinned as a literal, so renaming
    # or dropping one of them is a failure rather than a silent re-derivation.
    assert _UNHASHED_SCHEMA_COLUMNS <= schema_columns
    hashable = schema_columns - _UNHASHED_SCHEMA_COLUMNS

    assert hashable == set(_CONTENT_COLUMNS)
    assert set(_CONTENT_COLUMNS) == hashable
    assert len(_CONTENT_COLUMNS) == len(set(_CONTENT_COLUMNS))  # no duplicate entries


# --- the digest itself ------------------------------------------------------
def test_golden_entry_hash_of_a_fully_populated_event(tmp_path: Path) -> None:
    """MUTATIONS that must make this fail: any change to the digest recipe —
    dropping a field from the hashed payload, ``sort_keys=True`` -> ``False``,
    dropping ``prev_hash`` from the pre-image, truncating the digest, or changing
    the JSON separators.

    Both literals below were produced OUTSIDE this process and nothing here is
    imported from ``trelix.audit.store``. ``entry_hash`` for the first entry is
    ``sha256(("0" * 64) + canonical_json).hexdigest()`` where ``canonical_json``
    is the string spelled out in ``expected_canonical``.

    A cross-version claim, not just a format one: this digest is what the
    shipped writer puts on disk, so a change to the recipe silently reports
    ``row_mutated`` for every row of every audit.db written by an earlier build.
    Regenerating these literals is therefore a migration decision, not a test fix.
    """
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    try:
        assert (
            store.append(
                AuditEvent(
                    ts="2026-08-23T12:00:00Z",
                    principal="sub-1@https://idp.example",
                    action=ACTION_ADMIN,
                    resource="/admin/reindex",
                    outcome=OUTCOME_SUCCESS,
                    status_code=200,
                    client_ip="203.0.113.7",
                    request_id="req-abc",
                    trace_id="trace-xyz",
                    duration_ms=42,
                    detail="manual reindex",
                )
            )
            is True
        )
        rows = list(store.iter_for_export())
    finally:
        store.close()

    expected_canonical = (
        '{"action":"admin","client_ip":"203.0.113.7","detail":"manual reindex",'
        '"duration_ms":42,"outcome":"success",'
        '"principal":"sub-1@https://idp.example","request_id":"req-abc",'
        '"resource":"/admin/reindex","status_code":200,"trace_id":"trace-xyz",'
        '"ts":"2026-08-23T12:00:00Z"}'
    )
    assert len(expected_canonical) == 267  # guards a silently edited literal above

    assert len(rows) == 1
    assert rows[0]["prev_hash"] == (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    assert rows[0]["entry_hash"] == (
        "32d1d07e707daea147a01d079efda250cadc7f6d11912dcb7604a774f1e0ba5e"
    )
    assert _verify(db) == (None, None)


def test_two_identical_events_get_different_entry_hashes(tmp_path: Path) -> None:
    """MUTATION that must make this fail: dropping ``prev_hash`` from the digest
    pre-image, i.e. ``sha256((prev_hash + canonical)...)`` ->
    ``sha256((canonical)...)``. Applied symmetrically that survives the whole
    existing audit suite, yet it stops the digest from committing to the
    predecessor: an edit to row k then needs only row k's own ``entry_hash``
    recomputed and row k+1's ``prev_hash`` relinked, instead of cascading
    through every later row, so partial tampering that verify catches today
    becomes clean.

    Deliberately forges nothing and recomputes no digest, so it cannot be fooled
    by a mutated hash function: two entries whose hashed content is byte-for-byte
    identical must still hash differently, because their predecessors differ.
    """
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    try:
        for _ in range(2):
            assert store.append(_event(1)) is True
        rows = list(store.iter_for_export())
    finally:
        store.close()

    assert len(rows) == 2
    first, second = rows

    # Precondition: the ONLY differing input must be the predecessor. If _event
    # ever grows a per-call varying field the digests would differ for a content
    # reason and this test would stop discriminating.
    content_first = {col: first[col] for col in _CONTENT_COLUMNS}
    content_second = {col: second[col] for col in _CONTENT_COLUMNS}
    assert content_first == content_second, (
        "_event(1) no longer produces two byte-identical payloads, so this test "
        "no longer isolates the predecessor as the only differing hash input"
    )
    assert first["prev_hash"] != second["prev_hash"]

    assert first["entry_hash"] != second["entry_hash"]
    assert _verify(db) == (None, None)


# --- the emptied-log discriminator -----------------------------------------
def test_a_wiped_single_entry_log_is_detected(tmp_path: Path) -> None:
    """MUTATION that must make this fail: ``seq_high > 0`` -> ``seq_high > 1`` in
    ``verify``. The existing wipe tests all seed five entries, so the boundary at
    exactly one entry ever appended was untested and that off-by-one made a
    one-entry log wipe verify INTACT.
    """
    db = _seed(tmp_path / "audit.db", count=1)
    assert _scalar(db, "SELECT seq FROM sqlite_sequence WHERE name = 'audit_log'") == 1

    _raw_exec(db, "DELETE FROM audit_log")
    _raw_exec(db, "DELETE FROM audit_meta")

    assert _verify(db) == (1, REASON_LOG_EMPTIED)


def test_a_lower_duplicate_sqlite_sequence_row_does_not_hide_a_wipe(tmp_path: Path) -> None:
    """MUTATION that must make this fail: ``max(integers)`` -> ``min(integers)``
    in ``_max_audit_log_seq_locked``. ``sqlite_sequence`` has no uniqueness
    constraint, so an extra row for ``audit_log`` can be inserted; reducing with
    ``min`` lets one row carrying ``seq = 0`` mask the real high-water mark and
    a total wipe then verifies INTACT. The module's docstring says all matching
    rows are reduced with ``max`` on purpose; nothing tested it.
    """
    db = _seed(tmp_path / "audit.db", count=5)

    _raw_exec(db, "DELETE FROM audit_log")
    _raw_exec(db, "DELETE FROM audit_meta")
    _raw_exec(db, "INSERT INTO sqlite_sequence(name, seq) VALUES ('audit_log', 0)")

    # Precondition: two rows really are present, one of them the planted low one.
    seqs = sorted(
        row[0]
        for row in sqlite3.connect(str(db)).execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'audit_log'"
        )
    )
    assert seqs == [0, 5], f"planting the duplicate sqlite_sequence row did not take: {seqs}"

    assert _verify(db) == (1, REASON_LOG_EMPTIED)
