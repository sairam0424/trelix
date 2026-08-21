"""The integrity anchor must be PRESENT and well-formed, not merely consistent.

``verify_chain`` gated both of its anchor comparisons on the anchor being there::

    if meta_count is not None and actual_count != meta_count: ...
    if meta_head  is not None and prev != meta_head: ...
    return None    # = intact

so an absent anchor read as "no information available" rather than as "the thing
I check by is gone". ``append`` writes the entry and BOTH ``audit_meta`` rows in
one transaction, which makes "entries present, an anchor row absent" a state no
writer can produce — and it verified clean, exit 0. Removing the anchor cost one
extra ``DELETE`` and no hash computation.

Why the presence check cannot produce a false positive: ``git log`` on
``src/trelix/audit/store.py`` is a single commit, and ``audit_meta`` plus both
``_META_*`` keys are in that first version. No released version has ever written
an ``audit_log`` row without both anchor rows, so "entries with a missing or
malformed anchor" has no legitimate producer, legacy databases included.
``test_append_always_writes_both_anchor_rows`` pins that premise from the writer
side; ``test_fresh_empty_database_verifies_intact`` and
``test_no_false_positive_over_a_long_legitimate_chain`` pin the two legitimate
states that must stay at "no fault".

Malformed values are reported, never raised. ``_read_meta_locked`` used to
``int()`` the count as it read it, so a one-character ``UPDATE`` ended
``trelix audit verify`` in an unhandled ``ValueError`` — and an integrity checker
that the data it checks can crash is a denial of the tooling an incident
responder reaches for first. See tests/unit/test_cli_audit_anchor.py for that at
the command level, and tests/unit/test_audit_undetectable.py for the shapes an
in-DB anchor provably cannot see.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import (
    _CONTENT_COLUMNS,
    _META_COUNT,
    _META_HEAD,
    REASON_ANCHOR_CORRUPT,
    REASON_ANCHOR_MISSING,
    REASON_COUNT_MISMATCH,
    REASON_HEAD_MISMATCH,
    REASON_ID_GAP,
    REASON_ROW_MISLINKED,
    REASON_ROW_MUTATED,
    AuditStore,
    _canonical_hash,
)


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


def _seed(db: Path, count: int = 4) -> None:
    """Build a legitimate chain of *count* entries and close the writer."""
    store = AuditStore(db)
    for i in range(1, count + 1):
        assert store.append(_event(i)) is True
    assert store.verify_chain() is None
    store.close()


def _raw_exec(db: Path, *statements: str) -> None:
    """Tamper through a separate connection, bypassing the store's bookkeeping."""
    conn = sqlite3.connect(str(db), timeout=5)
    try:
        for sql in statements:
            conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def _verify(db: Path) -> tuple[int | None, str | None]:
    """Open a fresh store on *db* and return ``(tampered_id, reason)``."""
    store = AuditStore(db)
    try:
        result = store.verify()
        # verify_chain() is the long-standing surface; it must never disagree.
        assert store.verify_chain() == result.tampered_id
        return result.tampered_id, result.reason
    finally:
        store.close()


_DROP_COUNT = f"DELETE FROM audit_meta WHERE key='{_META_COUNT}'"
_DROP_HEAD = f"DELETE FROM audit_meta WHERE key='{_META_HEAD}'"
_DELETE_TAIL = "DELETE FROM audit_log WHERE id = (SELECT MAX(id) FROM audit_log)"
_REALIGN_COUNT = (
    f"UPDATE audit_meta SET value = (SELECT COUNT(*) FROM audit_log) WHERE key='{_META_COUNT}'"
)
_REALIGN_HEAD = (
    "UPDATE audit_meta SET value = "
    "(SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1) "
    f"WHERE key='{_META_HEAD}'"
)


# --- CONTROL: what already worked must keep working ------------------------
def test_control_tail_delete_with_the_anchor_intact_is_still_detected(tmp_path: Path) -> None:
    """CONTROL — passes before and after this change.

    A deleted tail with the anchor untouched is the case the anchor was added
    for: rows 1..3 still chain perfectly, and only the count anchor knows a
    fourth entry existed. Widening what counts as tamper must not have changed
    this verdict or its id.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, _DELETE_TAIL)

    assert _verify(db) == (4, REASON_COUNT_MISMATCH)


def test_control_mutated_row_is_still_named_by_its_own_id(tmp_path: Path) -> None:
    """CONTROL — a present, wrong row is still reported at its own id."""
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, "UPDATE audit_log SET principal='attacker' WHERE id=2")

    assert _verify(db) == (2, REASON_ROW_MUTATED)


def test_control_deleted_middle_row_is_still_detected(tmp_path: Path) -> None:
    """CONTROL — the gapless-id check still fires at the missing id."""
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, "DELETE FROM audit_log WHERE id = 2")

    assert _verify(db) == (2, REASON_ID_GAP)


def test_control_two_swapped_rows_are_detected_as_a_mislinked_row(tmp_path: Path) -> None:
    """CONTROL — reordering breaks the ``prev_hash`` linkage, not any row's own hash.

    Pinned because the module docstring lists reordering as detected, and this is
    the only shape that reaches the ``row_mislinked`` branch: each row's stored
    hashes still match its own content, so only the link to the running head is
    wrong.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    # Swap the whole payload of rows 2 and 3, hashes included, so both rows remain
    # internally consistent and only their order is wrong.
    _raw_exec(
        db,
        "CREATE TEMP TABLE swap AS SELECT * FROM audit_log WHERE id IN (2, 3)",
        "UPDATE audit_log SET (ts, principal, action, resource, outcome, status_code, "
        "client_ip, request_id, trace_id, duration_ms, detail, prev_hash, entry_hash) = "
        "(SELECT ts, principal, action, resource, outcome, status_code, client_ip, "
        "request_id, trace_id, duration_ms, detail, prev_hash, entry_hash FROM swap "
        "WHERE id = 3) WHERE id = 2",
        "UPDATE audit_log SET (ts, principal, action, resource, outcome, status_code, "
        "client_ip, request_id, trace_id, duration_ms, detail, prev_hash, entry_hash) = "
        "(SELECT ts, principal, action, resource, outcome, status_code, client_ip, "
        "request_id, trace_id, duration_ms, detail, prev_hash, entry_hash FROM swap "
        "WHERE id = 2) WHERE id = 3",
    )

    assert _verify(db) == (2, REASON_ROW_MISLINKED)


def test_control_a_row_appended_without_advancing_the_anchor_is_detected(
    tmp_path: Path,
) -> None:
    """CONTROL — the count anchor catches growth as well as truncation.

    Pinned because the module docstring lists "a row appended without the anchor
    being advanced" as detected, and every other count-anchor test moves the row
    count *down*. The planted row is internally valid — its ``prev_hash`` is the
    genuine head and its ``entry_hash`` is correctly computed — so the chain walk
    passes it and only the anchor knows the log grew. Advancing the anchor as well
    makes this shape invisible; that is pinned as a limitation in
    tests/unit/test_audit_undetectable.py.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    conn = sqlite3.connect(str(db), timeout=5)
    try:
        head = conn.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()[
            0
        ]
        content = {col: _event(9).__getattribute__(col) for col in _CONTENT_COLUMNS}
        conn.execute(
            "INSERT INTO audit_log (ts, principal, action, resource, outcome, status_code, "
            "client_ip, request_id, trace_id, duration_ms, detail, prev_hash, entry_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *[content[col] for col in _CONTENT_COLUMNS],
                head,
                _canonical_hash(head, content),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    tampered_id, reason = _verify(db)
    assert reason == REASON_COUNT_MISMATCH
    assert tampered_id == 5


# --- THE BUG: an absent anchor is a fault, not an absence of information ----
def test_tail_delete_plus_anchor_wipe_is_detected(tmp_path: Path) -> None:
    """THE HEADLINE BUG: this reported INTACT (``verify_chain() is None``).

    One extra ``DELETE FROM audit_meta`` on top of the tail deletion above — no
    hash computation at all — turned a detected attack into a clean bill of
    health, because both anchor comparisons were skipped when the anchor was
    absent.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, _DELETE_TAIL, "DELETE FROM audit_meta")

    tampered_id, reason = _verify(db)
    assert reason == REASON_ANCHOR_MISSING
    # 3 rows survive, all verifying; entry 4 is the first whose existence can no
    # longer be proven.
    assert tampered_id == 4


def test_anchor_wipe_with_the_log_untouched_is_detected(tmp_path: Path) -> None:
    """Deleting only the anchor is itself tamper: it disarms gate 2 for later.

    Nothing is wrong with the entries yet, so this is the staging move — after it
    a tail deletion is invisible. Reported as a fault at the point it happens
    rather than after the damage it enables.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, "DELETE FROM audit_meta")

    assert _verify(db) == (5, REASON_ANCHOR_MISSING)


@pytest.mark.parametrize(
    ("label", "drop"),
    [("count", _DROP_COUNT), ("head_hash", _DROP_HEAD)],
)
def test_dropping_only_one_anchor_row_is_detected(tmp_path: Path, label: str, drop: str) -> None:
    """BOTH rows are required, not either.

    With the log untouched, dropping one anchor row left the other one agreeing,
    so ``verify_chain()`` returned ``None`` for each of these individually. That
    is why the check is "count absent OR head absent", not "both absent".
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, drop)

    assert _verify(db) == (5, REASON_ANCHOR_MISSING), f"dropping {label!r} alone went unnoticed"


@pytest.mark.parametrize(
    ("label", "statements"),
    [
        ("drop count, realign head", (_DELETE_TAIL, _DROP_COUNT, _REALIGN_HEAD)),
        ("drop head, realign count", (_DELETE_TAIL, _DROP_HEAD, _REALIGN_COUNT)),
    ],
)
def test_tail_delete_with_one_anchor_row_dropped_and_the_other_realigned_is_detected(
    tmp_path: Path, label: str, statements: tuple[str, ...]
) -> None:
    """Why requiring BOTH is load-bearing, in the shape that actually erases data.

    Drop one anchor row and copy the surviving one's new value straight out of a
    row that is still in the table. No hashing, and before this change every one
    of these verified clean.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, *statements)

    assert _verify(db) == (4, REASON_ANCHOR_MISSING), f"{label} went unnoticed"


# --- malformed anchor values: a finding, never a traceback ------------------
@pytest.mark.parametrize(
    ("label", "value_sql"),
    [
        ("not a number", "'x'"),
        ("negative", "'-1'"),
        ("explicit plus", "'+4'"),
        ("leading space", "' 4'"),
        ("trailing newline", "'4' || char(10)"),
        ("underscore separator", "'4_0'"),
        ("leading zero", "'04'"),
        ("empty string", "''"),
        ("hex literal", "'0x4'"),
        ("float", "'4.0'"),
        ("20 digits (wider than a SQLite INTEGER)", "'1' || '0000000000000000000'"),
        ("5000 digits", "printf('%.5000d', 9)"),
        ("a BLOB, not text", "X'0500'"),
    ],
)
def test_malformed_count_is_reported_not_raised(tmp_path: Path, label: str, value_sql: str) -> None:
    """A value the verifier exists to judge must not be able to abort it.

    ``'x'`` and the 5000-digit case each raised ``ValueError`` out of
    ``verify_chain`` before this change (CPython refuses ``int()`` above 4300
    digits), which is why the count is bounded on SHAPE before it is parsed.
    ``'-1'`` did not raise — it silently produced a *negative* verdict id, 0.
    ``X'0500'`` is stored as a BLOB even though the column is declared TEXT, so
    it arrives as ``bytes`` and needs the isinstance guard.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, f"UPDATE audit_meta SET value = {value_sql} WHERE key='{_META_COUNT}'")

    tampered_id, reason = _verify(db)  # must return, not raise
    assert reason == REASON_ANCHOR_CORRUPT, f"{label} was not reported as a corrupt anchor"
    assert tampered_id == 5


@pytest.mark.parametrize(
    ("label", "value_sql"),
    [
        ("not hex at all", "'x'"),
        ("empty string", "''"),
        ("uppercased digest", f"upper((SELECT value FROM audit_meta WHERE key='{_META_HEAD}'))"),
        ("63 hex digits", "substr((SELECT value FROM audit_meta WHERE key='head_hash'), 1, 63)"),
        ("65 chars", "(SELECT value FROM audit_meta WHERE key='head_hash') || 'a'"),
        ("trailing newline", "(SELECT value FROM audit_meta WHERE key='head_hash') || char(10)"),
        ("a BLOB, not text", "X'0500'"),
    ],
)
def test_malformed_head_hash_is_reported_as_a_corrupt_anchor(
    tmp_path: Path, label: str, value_sql: str
) -> None:
    """A head that is not a digest cannot be *compared*, so "mismatch" understates it.

    These were reported as ``head_mismatch`` at the last row's id, which reads as
    "the chain diverged at entry 4". Nothing diverged: the anchor is unusable, and
    the honest report is that entry 5's existence can no longer be proven.
    ``append`` writes ``hexdigest()``, which is always lowercase, so an uppercased
    digest is a value this writer cannot produce.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, f"UPDATE audit_meta SET value = {value_sql} WHERE key='{_META_HEAD}'")

    tampered_id, reason = _verify(db)  # must return, not raise
    assert reason == REASON_ANCHOR_CORRUPT, f"{label} was not reported as a corrupt anchor"
    assert tampered_id == 5


def test_a_valid_but_wrong_head_is_still_a_mismatch_not_a_corruption(tmp_path: Path) -> None:
    """The shape gate must not swallow the comparison it precedes.

    A well-formed digest that simply is not this chain's head is exactly what
    gate 2 was built to catch, and it must keep its own reason rather than being
    absorbed into ``anchor_corrupt``.
    """
    db = tmp_path / "audit.db"
    _seed(db, 4)

    _raw_exec(db, f"UPDATE audit_meta SET value = '{'a' * 64}' WHERE key='{_META_HEAD}'")

    assert _verify(db) == (4, REASON_HEAD_MISMATCH)


# --- no false positives ----------------------------------------------------
def test_append_always_writes_both_anchor_rows(tmp_path: Path) -> None:
    """The premise that makes the presence check false-positive-free.

    If ``append`` could ever leave an entry without both anchor rows, the check
    above would flag ordinary operation. It writes the row and both anchor rows
    in one transaction, so this holds after the first append and after every one
    after it.
    """
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    try:
        for i in range(1, 6):
            assert store.append(_event(i)) is True
            conn = sqlite3.connect(str(db), timeout=5)
            try:
                keys = {row[0] for row in conn.execute("SELECT key FROM audit_meta").fetchall()}
                count = conn.execute("SELECT value FROM audit_meta WHERE key='count'").fetchone()
            finally:
                conn.close()
            assert {_META_COUNT, _META_HEAD} <= keys, f"anchor incomplete after append {i}"
            assert count[0] == str(i)
    finally:
        store.close()


def test_fresh_empty_database_verifies_intact(tmp_path: Path) -> None:
    """The one legitimate state with no anchor at all: a brand-new database.

    ``AuditStore(path)`` creates the schema and writes no ``audit_meta`` rows
    until the first append, so "no entries and no anchor" is normal and must stay
    exit 0. This is why the presence check is gated on ``rows > 0``. It is also
    why a total wipe of both tables is NOT detectable — see
    tests/unit/test_audit_undetectable.py.
    """
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    try:
        result = store.verify()
        assert (result.tampered_id, result.reason) == (None, None)
    finally:
        store.close()


def test_no_false_positive_over_a_long_legitimate_chain(tmp_path: Path) -> None:
    """Verify after every append: normal operation is never reported as tamper."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    try:
        for i in range(1, 61):
            assert store.append(_event(i % 10)) is True
            result = store.verify()
            assert result.tampered_id is None, f"append {i} was reported as {result.reason}"
    finally:
        store.close()


def test_reopening_a_chain_written_by_another_store_verifies_intact(tmp_path: Path) -> None:
    """A closed-and-reopened database is the ordinary CLI case, and must be clean."""
    db = tmp_path / "audit.db"
    _seed(db, 4)

    assert _verify(db) == (None, None)


def test_disabled_store_still_reports_no_fault(tmp_path: Path) -> None:
    """Unchanged on purpose: a store that never opened cannot report tamper.

    ``tmp_path`` is a directory, so sqlite cannot open it. Turning that into a
    tamper verdict would make an unreadable path indistinguishable from an
    attack; ``is_open`` is the discriminator, and the CLI gates on it (exit 2).
    """
    store = AuditStore(tmp_path)

    assert store.is_open is False
    result = store.verify()
    assert (result.tampered_id, result.reason) == (None, None)
