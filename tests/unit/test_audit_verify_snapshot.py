"""``verify`` reads the rows and the anchor as ONE snapshot, not two.

``verify`` compares the live ``audit_log`` rows against the ``audit_meta`` count
and head. Those were two statements outside any transaction, so they were two
different snapshots, and ``AuditStore._lock`` cannot close that gap: it is a
per-INSTANCE ``threading.Lock``, so it serializes nothing against the other
connection or the other process doing the appending. One legitimate append
landing between the two reads made the anchor describe one more entry than the
rows did, and ``verify`` reported that as a truncated tail.

Measured on the shape ``trelix.api.app`` builds — one legitimate writer doing
2,870 appends while 30 CLI-shaped verify runs went past — 24 of the 30 reported
TAMPERED on an undamaged database. An independent post-hoc walk of the finished
file found 2,870 rows, a matching count, a matching head and no broken link:
nothing had been damaged. That is the worst failure mode a tamper detector has,
because it teaches an operator that exit 1 means "run it again".

The first test below is deterministic rather than probabilistic: it drives a
concurrent append into the exact window between the two reads by wrapping the
seam between them, and then asserts the verdict. The reads themselves, the
transaction and the verdict logic are the real ones. If someone splits the reads
back apart, that test fails — which is the property the claim needs, and the
thing the abandoned attempt at this change lacked.

The second is the operational shape end to end: a real writer thread against a
real second store instance, asserting that a healthy database is never once
reported as tampered.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import AuditStore

#: How long the interference hook gives the other connection to commit. Its only
#: job is to remove "the append had not got there yet" as an explanation for a
#: passing test; it is a bound, not a wait for success.
_COMMIT_WINDOW_S = 1.0


def _event(i: int) -> AuditEvent:
    return AuditEvent(
        ts=f"2026-08-12T10:00:{i % 60:02d}Z",
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


def test_an_append_landing_between_the_two_reads_is_not_reported_as_tamper(
    tmp_path: Path,
) -> None:
    """THE BUG, deterministically: this returned ``count_mismatch`` at id 5.

    The writer here is a legitimate append from a *different* ``AuditStore``
    instance — the same relationship a serving process has with
    ``trelix audit verify`` — driven into the window between the row read and the
    anchor read. Nothing about the database is damaged at any point.

    The hook wraps ``_read_meta_locked``, i.e. the boundary between the two reads,
    and does not touch either read or the verdict. Splitting the reads back into
    separate transactions makes this test fail.
    """
    db = tmp_path / "audit.db"
    writer = AuditStore(db)
    verifier = AuditStore(db)  # a second instance, therefore a second connection
    try:
        for i in range(1, 5):
            assert writer.append(_event(i)) is True
        assert verifier.verify_chain() is None

        interfered = threading.Event()
        committed = threading.Event()
        append_result: list[bool] = []
        threads: list[threading.Thread] = []

        def _append_from_the_other_connection() -> None:
            append_result.append(writer.append(_event(5)))
            committed.set()

        original_read_meta = verifier._read_meta_locked

        def _interfering_read_meta() -> dict[str, object]:
            interfered.set()
            thread = threading.Thread(target=_append_from_the_other_connection)
            thread.start()
            threads.append(thread)
            # Bounded on purpose. Before the fix the row read had already released
            # its lock, so this commits at once and the anchor read below sees an
            # entry the rows did not. After the fix the open read transaction holds
            # the other connection off, so this times out and the append lands
            # once the snapshot is released.
            committed.wait(timeout=_COMMIT_WINDOW_S)
            return original_read_meta()

        verifier._read_meta_locked = _interfering_read_meta  # type: ignore[method-assign]
        try:
            result = verifier.verify()
        finally:
            verifier._read_meta_locked = original_read_meta  # type: ignore[method-assign]
            for thread in threads:
                thread.join(timeout=30)

        assert interfered.is_set(), (
            "the hook never ran, so nothing was interleaved and this test proved "
            "nothing — the seam it wraps must still be called between the two reads"
        )
        assert result.tampered_id is None, (
            f"a legitimate concurrent append was reported as {result.reason} at id "
            f"{result.tampered_id}: the two reads did not come from one snapshot"
        )

        # The append is not lost, merely serialized behind the snapshot, and the
        # database it produced verifies clean afterwards.
        assert append_result == [True]
        assert verifier.verify_chain() is None
    finally:
        verifier.close()
        writer.close()


def test_a_legitimate_writer_never_makes_verify_report_tamper(tmp_path: Path) -> None:
    """The operational shape: a real writer thread, a real second store, no false alarm.

    Deliberately does not assert on how many appends land — under contention an
    append may hit its busy timeout, and that is the documented single-writer
    caveat, not this test's subject. The claim is only that a healthy database is
    never reported as tampered.
    """
    db = tmp_path / "audit.db"
    writer = AuditStore(db)
    verifier = AuditStore(db)
    try:
        # Seed enough rows that the row scan is not instantaneous — the window
        # this test is about is proportional to how long that scan takes.
        for i in range(200):
            assert writer.append(_event(i)) is True

        stop = threading.Event()
        appended = [0]

        def _keep_appending() -> None:
            for i in range(200, 800):
                if stop.is_set():
                    return
                if writer.append(_event(i)):
                    appended[0] += 1

        thread = threading.Thread(target=_keep_appending)
        thread.start()
        verdicts = []
        # A verify that could not acquire its read lock is NOT a tamper report, and this
        # test's claim is only about verdicts. Letting the exception escape made the leg
        # red for a reason the test does not speak to: it failed once on CI's Python 3.12
        # with "database is locked" while 3,451 other tests passed, and 8 local rounds of
        # the identical shape produced 240 verdicts with zero raises — so the mechanism is
        # runner contention, not logic. Counted rather than swallowed, and the count is
        # asserted below, so "every verify was locked out" cannot masquerade as success.
        locked_out = 0
        try:
            while thread.is_alive() and len(verdicts) + locked_out < 30:
                try:
                    verdicts.append(verifier.verify())
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) and "busy" not in str(exc).lower():
                        raise
                    locked_out += 1
        finally:
            stop.set()
            thread.join(timeout=60)

        assert verdicts, (
            "the writer finished before a single verify produced a verdict, or every "
            f"verify was locked out ({locked_out}) — no overlap was actually tested"
        )
        false_alarms = [(v.tampered_id, v.reason) for v in verdicts if v.tampered_id is not None]
        assert not false_alarms, (
            f"{len(false_alarms)} of {len(verdicts)} verify runs reported tamper on an "
            f"undamaged database: {false_alarms[:5]}"
        )
        assert appended[0] > 0, "no concurrent append landed, so nothing overlapped"
        assert verifier.verify_chain() is None, "the database really was fine all along"
    finally:
        verifier.close()
        writer.close()


def test_verify_leaves_no_transaction_open_and_writes_nothing(tmp_path: Path) -> None:
    """The snapshot must not turn a read-only command into a writer or a blocker.

    A transaction left open would hold the file against the serving process
    indefinitely, and a snapshot that ended in COMMIT rather than ROLLBACK would
    make an integrity *check* a write to the artifact it is checking.
    """
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    try:
        for i in range(1, 4):
            assert store.append(_event(i)) is True
        before = db.stat().st_mtime_ns

        assert store.verify_chain() is None

        assert store._conn is not None
        assert store._conn.in_transaction is False, "verify left a transaction open"
        assert db.stat().st_mtime_ns == before, "verify modified the file it audits"

        # A separate connection can still take the write lock immediately.
        other = AuditStore(db)
        try:
            assert other.append(_event(4)) is True
        finally:
            other.close()
    finally:
        store.close()
