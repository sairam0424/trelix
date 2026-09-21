"""Unit tests for AuditStore.prune() (TRELIX_AUDIT_RETENTION_DAYS's backing job).

Covers: partial pruning leaves a verifiable chain, batching across multiple
pages, the older_than_days<0 guard, a no-op prune when nothing qualifies, and
that repeated prune() calls are idempotent and keep verifying clean.

Full-emptying and the "hand-rolled DELETE without the watermark is still
detected" cases live in tests/unit/test_audit_wipe_detection.py, alongside the
rest of that module's wipe-detection suite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import AuditStore
from trelix.core.config import AuditConfig

_OLD_TS = "2020-01-01T00:00:00+00:00"


def _event(i: int, *, ts: str) -> AuditEvent:
    return AuditEvent(
        ts=ts,
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


def _now_ts() -> str:
    return datetime.now(UTC).isoformat()


def test_prune_removes_only_rows_older_than_the_cutoff(tmp_path: Path) -> None:
    """The core contract: old rows go, recent rows stay, and the survivors
    still verify clean."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 4):
        assert store.append(_event(i, ts=_OLD_TS)) is True
    for i in range(4, 7):
        assert store.append(_event(i, ts=_now_ts())) is True

    removed = store.prune(older_than_days=1)

    assert removed == 3
    assert [r["id"] for r in store.recent(10)] == [6, 5, 4]
    assert store.verify_chain() is None
    store.close()


def test_prune_across_multiple_batches_removes_everything_eligible(tmp_path: Path) -> None:
    """batch_size smaller than the eligible row count must still finish the job
    in one prune() call, not leave a partial prefix behind."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 11):
        assert store.append(_event(i, ts=_OLD_TS)) is True

    removed = store.prune(older_than_days=1, batch_size=3)

    assert removed == 10
    assert store.recent(10) == []
    assert store.verify_chain() is None
    store.close()


def test_prune_with_negative_older_than_days_is_a_no_op(tmp_path: Path) -> None:
    """Mirrors recent()'s n<=0 guard: a caller mistake, not a database finding."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 4):
        assert store.append(_event(i, ts=_OLD_TS)) is True

    removed = store.prune(older_than_days=-1)

    assert removed == 0
    assert len(store.recent(10)) == 3
    assert store.verify_chain() is None
    store.close()


def test_prune_when_nothing_qualifies_is_a_true_no_op(tmp_path: Path) -> None:
    """Every row is recent -> zero removed, and the log is left byte-for-byte
    verifiable exactly as if prune() had never been called."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 4):
        assert store.append(_event(i, ts=_now_ts())) is True

    removed = store.prune(older_than_days=365)

    assert removed == 0
    assert len(store.recent(10)) == 3
    assert store.verify_chain() is None
    store.close()


def test_repeated_prune_calls_are_idempotent(tmp_path: Path) -> None:
    """A second prune() with nothing new to remove must not disturb the
    watermark left by the first."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 4):
        assert store.append(_event(i, ts=_OLD_TS)) is True
    for i in range(4, 6):
        assert store.append(_event(i, ts=_now_ts())) is True

    first = store.prune(older_than_days=1)
    second = store.prune(older_than_days=1)

    assert first == 3
    assert second == 0
    assert store.verify_chain() is None
    store.close()


def test_prune_then_append_then_prune_again_stays_verifiable(tmp_path: Path) -> None:
    """A prune -> live append -> prune sequence exercises the watermark moving
    forward twice, not just once."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 4):
        assert store.append(_event(i, ts=_OLD_TS)) is True

    assert store.prune(older_than_days=1) == 3

    for i in range(4, 7):
        assert store.append(_event(i, ts=_OLD_TS)) is True
    assert store.verify_chain() is None

    assert store.prune(older_than_days=1) == 3
    assert store.recent(10) == []
    assert store.verify_chain() is None
    store.close()


def test_prune_reopened_store_still_verifies_clean(tmp_path: Path) -> None:
    """The watermark must be durable across a close/reopen, not merely an
    in-memory bookkeeping value."""
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    for i in range(1, 4):
        assert store.append(_event(i, ts=_OLD_TS)) is True
    for i in range(4, 6):
        assert store.append(_event(i, ts=_now_ts())) is True
    assert store.prune(older_than_days=1) == 3
    store.close()

    reopened = AuditStore(db)
    try:
        assert reopened.verify_chain() is None
        assert [r["id"] for r in reopened.recent(10)] == [5, 4]
    finally:
        reopened.close()


def test_audit_config_rejects_negative_retention_days() -> None:
    """A negative value used to be accepted silently — now prune() reads this
    field, so a nonsense value should fail fast at config load, not surface as
    a confusing prune() no-op later."""
    assert AuditConfig(retention_days=0).retention_days == 0
    assert AuditConfig().retention_days == 365
    with pytest.raises(ValidationError):
        AuditConfig(retention_days=-1)


def test_prune_retrofits_incremental_auto_vacuum(tmp_path: Path) -> None:
    """prune()'s incremental_vacuum calls only reclaim space if auto_vacuum is
    actually INCREMENTAL -- confirms the retrofit in _open_read_write ran."""
    import sqlite3

    db = tmp_path / "audit.db"
    store = AuditStore(db)
    store.close()

    conn = sqlite3.connect(str(db))
    try:
        (mode,) = conn.execute("PRAGMA auto_vacuum").fetchone()
    finally:
        conn.close()
    assert mode == 2  # INCREMENTAL
