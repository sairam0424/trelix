"""
Tamper-evident audit log storage (hash-chained, append-only).

AuditStore owns a SEPARATE ``audit.db`` — deliberately NOT the disposable
index DB (``.trelix/index.db``), because the index is rebuilt at will while
the audit trail must survive re-indexing. It uses the same idempotent-DDL
style as :mod:`trelix.store.db` (``CREATE TABLE IF NOT EXISTS`` +
``CREATE INDEX IF NOT EXISTS``) so opening an existing DB is a safe no-op.

Integrity model — tamper-EVIDENT, not tamper-PROOF. This detects naive or
accidental corruption (a stray ``UPDATE``, a truncated file, a dropped tail).
It does NOT stop a determined attacker with write access to ``audit.db``, who
could rewrite a row *and* recompute every subsequent ``entry_hash`` plus the
``audit_meta`` anchor in the same transaction — ``verify_chain`` would then
pass. For a stronger guarantee, sign the head hash with a key held OUTSIDE the
DB (HMAC/asymmetric) and/or anchor it to an append-only/WORM sink. Two gates:
  1. **Hash chain.** Each row stores ``prev_hash`` (the previous row's
     ``entry_hash``; genesis = 64 zeros) and ``entry_hash =
     sha256(prev_hash || canonical_json(content))``. Mutating a row's content
     breaks its own recomputed hash; deleting/reordering a middle row breaks
     the next row's ``prev_hash`` linkage. Either is caught by walking the
     chain.
  2. **In-DB count/head anchor** (same ``audit.db`` — bookkeeping, not an
     external root of trust). The chain alone cannot detect a deleted *tail*
     (the survivors still form a valid chain), so an ``audit_meta`` row records
     the running entry ``count`` and current ``head_hash``, updated atomically
     with every append. ``verify_chain`` compares the live row count / head
     against this anchor and also checks ids are gapless.

Failure contract (mirrors retrieval/telemetry.py's swallow-and-log, but at
WARNING): a write failure never raises by default — ``append`` logs a WARNING
and returns ``False`` — so a broken audit sink can never take down a request.
With ``fail_closed=True`` the failure is re-raised for callers that require a
durable trail. All SQL is parameterized; no value is ever string-formatted
into a statement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from trelix.audit.events import QUERY_ACTIONS, AuditEvent

logger = logging.getLogger("trelix.audit")

# Genesis prev_hash for the very first entry.
GENESIS_HASH = "0" * 64

# Logical (hashed) content columns, in a fixed order. Excludes the DB-assigned
# `id` (not known until after insert) and the two hash columns themselves.
_CONTENT_COLUMNS = (
    "ts",
    "principal",
    "action",
    "resource",
    "outcome",
    "status_code",
    "client_ip",
    "request_id",
    "trace_id",
    "duration_ms",
    "detail",
)

_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    principal   TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    resource    TEXT,
    outcome     TEXT    NOT NULL,
    status_code INTEGER,
    client_ip   TEXT,
    request_id  TEXT,
    trace_id    TEXT,
    duration_ms INTEGER,
    detail      TEXT,
    prev_hash   TEXT    NOT NULL,
    entry_hash  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_audit_log_principal_ts ON audit_log(principal, ts);
CREATE TABLE IF NOT EXISTS audit_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_META_COUNT = "count"
_META_HEAD = "head_hash"


def _canonical_hash(prev_hash: str, content: dict[str, Any]) -> str:
    """entry_hash = sha256(prev_hash || canonical_json(content))."""
    canonical = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


class AuditStore:
    """Append-only, hash-chained audit log backed by its own SQLite file.

    Construction never raises: if the DB cannot be opened (bad/unwritable
    path), it logs a WARNING and enters a disabled state where writes return
    ``False`` (or raise under ``fail_closed``) — auditing must never crash the
    host application.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_DDL)
            conn.commit()
            self._conn = conn
        except Exception as exc:  # noqa: BLE001 — audit init must never crash the caller
            logger.warning(
                "AuditStore init failed for %s; auditing disabled: %s", self._db_path, exc
            )
            self._conn = None

    @property
    def is_open(self) -> bool:
        """Whether the store actually opened a database.

        Init deliberately never raises (a broken audit sink must not take down the
        caller), so every read returns an empty/None result when the open failed.
        For ``verify_chain`` that is dangerous: ``None`` means "no divergence
        found", which a caller cannot distinguish from "nothing was ever checked".
        Callers that report integrity MUST gate on this first — pointing
        ``trelix audit verify`` at a wrong path used to print "chain intact" and
        exit 0 without having opened anything.
        """
        return self._conn is not None

    # -- write ---------------------------------------------------------------
    def append(
        self,
        event: AuditEvent,
        *,
        log_queries: bool = False,
        fail_closed: bool = False,
    ) -> bool:
        """Append one event to the chain.

        Returns ``True`` on success. On any write failure logs a WARNING and
        returns ``False`` — unless ``fail_closed`` is set, in which case the
        error is re-raised. ``log_queries=False`` (the default) stores
        ``sha256(detail)`` for query actions instead of the raw query text.
        """
        try:
            if self._conn is None:
                raise RuntimeError("AuditStore is disabled (DB unavailable)")
            content = self._event_to_content(event, log_queries=log_queries)
            with self._lock:
                prev_hash = self._head_hash_locked()
                entry_hash = _canonical_hash(prev_hash, content)
                params = tuple(content[col] for col in _CONTENT_COLUMNS)
                with self._conn:  # atomic: row insert + meta anchor update
                    self._conn.execute(
                        "INSERT INTO audit_log ("
                        "ts, principal, action, resource, outcome, status_code, "
                        "client_ip, request_id, trace_id, duration_ms, detail, "
                        "prev_hash, entry_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (*params, prev_hash, entry_hash),
                    )
                    self._conn.execute(
                        "INSERT INTO audit_meta(key, value) VALUES(?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
                        (_META_COUNT,),
                    )
                    self._conn.execute(
                        "INSERT INTO audit_meta(key, value) VALUES(?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (_META_HEAD, entry_hash),
                    )
            return True
        except Exception as exc:  # noqa: BLE001 — swallow-and-log unless fail_closed
            logger.warning("Audit append failed (non-fatal): %s", exc)
            if fail_closed:
                raise
            return False

    def _event_to_content(self, event: AuditEvent, *, log_queries: bool) -> dict[str, Any]:
        """Map an event to its stored content dict, hashing query text when
        query logging is disabled."""
        detail = event.detail
        if detail is not None and not log_queries and event.action in QUERY_ACTIONS:
            detail = "sha256:" + hashlib.sha256(detail.encode("utf-8")).hexdigest()
        return {
            "ts": event.ts,
            "principal": event.principal,
            "action": event.action,
            "resource": event.resource,
            "outcome": event.outcome,
            "status_code": event.status_code,
            "client_ip": event.client_ip,
            "request_id": event.request_id,
            "trace_id": event.trace_id,
            "duration_ms": event.duration_ms,
            "detail": detail,
        }

    def _head_hash_locked(self) -> str:
        """Current chain head (last entry_hash), or GENESIS_HASH if empty.
        Caller must hold ``self._lock``."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["entry_hash"] if row is not None else GENESIS_HASH

    # -- verify --------------------------------------------------------------
    def verify_chain(self) -> int | None:
        """Verify the whole chain. Returns ``None`` if intact, otherwise the
        id of the first divergent entry (a mutated/mislinked row), or the id
        of the first missing entry when the tail was truncated/deleted.

        Detection:
          - recomputed ``entry_hash`` != stored  -> mutated content.
          - stored ``prev_hash`` != running head  -> deleted/reordered row.
          - id gap                                 -> deleted middle row.
          - live count/head != meta anchor         -> truncated/deleted tail.
        """
        if self._conn is None:
            return None
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, principal, action, resource, outcome, status_code, "
                "client_ip, request_id, trace_id, duration_ms, detail, prev_hash, entry_hash "
                "FROM audit_log ORDER BY id ASC"
            ).fetchall()
            meta = self._read_meta_locked()

        prev = GENESIS_HASH
        expected_id = 1
        for row in rows:
            if row["id"] != expected_id:  # gapless check catches middle deletes
                return expected_id
            content = {col: row[col] for col in _CONTENT_COLUMNS}
            if row["prev_hash"] != prev:
                return int(row["id"])
            if row["entry_hash"] != _canonical_hash(prev, content):
                return int(row["id"])
            prev = row["entry_hash"]
            expected_id += 1

        # External anchor: detect a truncated/deleted tail the chain can't see.
        actual_count = len(rows)
        meta_count = meta.get(_META_COUNT)
        if meta_count is not None and actual_count != meta_count:
            # first missing id (ids are gapless 1..actual_count at this point)
            return int(min(actual_count, meta_count)) + 1
        meta_head = meta.get(_META_HEAD)
        if meta_head is not None and prev != meta_head:
            return actual_count if actual_count > 0 else 1
        return None

    def _read_meta_locked(self) -> dict[str, Any]:
        assert self._conn is not None
        out: dict[str, Any] = {}
        for row in self._conn.execute("SELECT key, value FROM audit_meta").fetchall():
            if row["key"] == _META_COUNT:
                out[_META_COUNT] = int(row["value"])
            else:
                out[row["key"]] = row["value"]
        return out

    # -- read ----------------------------------------------------------------
    def recent(self, n: int = 100) -> list[dict[str, Any]]:
        """Return up to ``n`` most-recent entries, newest first."""
        if self._conn is None or n <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(row) for row in rows]

    def iter_for_export(self) -> Iterator[dict[str, Any]]:
        """Yield every entry in append order (oldest first) for export.

        Rows are materialized under the lock, then yielded outside it, so the
        generator never holds the connection lock while a consumer is slow.
        """
        if self._conn is None:
            return
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
        for row in rows:
            yield dict(row)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
