"""Just-in-time principal provisioning, persisted inside ``audit.db``.

The ``principals`` table lives in the SAME SQLite file the audit trail uses
(:mod:`trelix.audit.store`) — a verified caller's identity and their audit
records belong together and must survive re-indexing of the disposable code
index. It uses the same idempotent-DDL style (``CREATE TABLE IF NOT EXISTS``)
so opening an existing DB is a safe no-op.

``jit_upsert`` implements *just-in-time* provisioning: the first time an
identity ``(subject, issuer)`` is seen it is inserted with ``first_seen ==
last_seen``; on every subsequent sight ONLY ``last_seen`` advances —
``first_seen`` is immutable, and profile fields are not overwritten. Identity
is the ``(subject, issuer)`` pair (email is never part of the key), mirroring
:class:`trelix.auth.principal.Principal`.

Resilience mirrors :class:`trelix.audit.store.AuditStore`: construction never
raises, and a write failure logs a WARNING and returns ``False`` rather than
taking down the request.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trelix.auth.principal import Principal

logger = logging.getLogger("trelix.auth")

_DDL = """
CREATE TABLE IF NOT EXISTS principals (
    subject      TEXT NOT NULL,
    issuer       TEXT NOT NULL,
    email        TEXT,
    display_name TEXT,
    groups_json  TEXT,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    PRIMARY KEY (subject, issuer)
);
"""


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class PrincipalStore:
    """Persist verified principals (JIT-provisioned) inside ``audit.db``.

    Construction never raises: if the DB cannot be opened it logs a WARNING and
    enters a disabled state where writes return ``False`` and reads return
    empty.
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
        except Exception as exc:  # noqa: BLE001 — provisioning must never crash the caller
            logger.warning(
                "PrincipalStore init failed for %s; JIT provisioning disabled: %s",
                self._db_path,
                exc,
            )
            self._conn = None

    def jit_upsert(self, principal: Principal, *, now: str | None = None) -> bool:
        """Provision or touch a principal.

        First sight: INSERT with ``first_seen == last_seen``. Subsequent sight:
        advance ``last_seen`` ONLY — ``first_seen`` and profile fields are left
        untouched. Returns ``True`` on success, ``False`` on a (logged) write
        failure. ``now`` is injectable for deterministic tests.
        """
        try:
            if self._conn is None:
                raise RuntimeError("PrincipalStore is disabled (DB unavailable)")
            ts = now or _utcnow_iso()
            groups_json = json.dumps(list(principal.groups))
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO principals "
                    "(subject, issuer, email, display_name, groups_json, "
                    "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(subject, issuer) DO UPDATE SET last_seen = excluded.last_seen",
                    (
                        principal.subject,
                        principal.issuer,
                        principal.email,
                        principal.display_name,
                        groups_json,
                        ts,
                        ts,
                    ),
                )
            return True
        except Exception as exc:  # noqa: BLE001 — swallow-and-log, never fatal
            logger.warning("PrincipalStore jit_upsert failed (non-fatal): %s", exc)
            return False

    def get(self, subject: str, issuer: str) -> dict[str, Any] | None:
        """Return the stored row for ``(subject, issuer)`` or ``None``."""
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM principals WHERE subject = ? AND issuer = ?",
                (subject, issuer),
            ).fetchone()
        return dict(row) if row is not None else None

    def count(self) -> int:
        """Total number of provisioned principals."""
        if self._conn is None:
            return 0
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM principals").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
