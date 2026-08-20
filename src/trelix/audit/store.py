"""
Tamper-evident audit log storage (hash-chained, append-only).

AuditStore owns a SEPARATE ``audit.db`` — deliberately NOT the disposable
index DB (``.trelix/index.db``), because the index is rebuilt at will while
the audit trail must survive re-indexing. The *writing* path uses the same
idempotent-DDL style as :mod:`trelix.store.db` (``CREATE TABLE IF NOT EXISTS`` +
``CREATE INDEX IF NOT EXISTS``) so reopening an existing DB is a safe no-op. The
*reading* path runs no DDL at all — see ``read_only`` below; idempotent DDL is a
no-op only on a file that already has the schema, and on any other file it is a
schema change.

Integrity model — tamper-EVIDENT, not tamper-PROOF. Three gates, and all three
live inside the same file as the thing they check:
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
     with every append. ``verify_chain`` requires both anchor values to be
     **present and well-formed** and then compares the live row count / head
     against them; it also checks ids are gapless. Presence is part of the
     check, not a precondition for it: an absent anchor is a fault, because
     ``append`` writes the entry and both anchor rows in one transaction and so
     cannot produce a non-empty log without them.
  3. **SQLite's own ``sqlite_sequence`` high-water mark**, read only, for the one
     shape neither gate above can see: a log emptied completely. See
     :meth:`AuditStore._max_audit_log_seq_locked` — including the four ways to
     defeat it, all of which still work, and the false positive it will become
     when retention pruning lands.

What is and is not detected. Every shape below is pinned by a test —
tests/unit/test_audit_anchor_presence.py, tests/unit/test_audit_store.py and
tests/unit/test_audit_wipe_detection.py for the detected ones,
tests/unit/test_audit_undetectable.py for the rest. The
attacker's hashing cost is stated for each because cheapness is *not* what
separates the two lists — most of the detected shapes are free too:

DETECTED:
  - a row's content edited — none;
  - a middle row deleted, or two rows swapped — none;
  - the tail truncated with the anchor left untouched — none;
  - either ``audit_meta`` row deleted, or both — none;
  - an anchor value malformed, out of range, the wrong storage type, or not valid
    UTF-8 — none;
  - a row appended with a valid ``prev_hash`` and ``entry_hash``, without the
    anchor being advanced — one sha256;
  - a *sloppy* total wipe — both tables emptied and nothing else — none. See
    :meth:`AuditStore._max_audit_log_seq_locked`: ``audit_log`` is AUTOINCREMENT,
    so SQLite keeps a ``sqlite_sequence`` row for it and ``DELETE`` does not reset
    it. "No rows, but ids were once handed out here" is a state normal operation
    cannot produce. This closes the sloppy wipe and nothing more — four ways to
    defeat it are listed below and all of them still work.

NOT DETECTED:
  - a total wipe that ALSO clears ``sqlite_sequence``, or forges its ``seq`` to
    match a truncated log, or sets ``seq`` below the real row count, or removes
    the row/table via ``PRAGMA writable_schema=ON`` (plain ``DROP TABLE
    sqlite_sequence`` is refused by SQLite; that route needs no extra privilege,
    so absence of the row is deliberately NOT treated as a finding — it is
    indistinguishable from a database that was never appended to) — no hashing;
  - the tail truncated with BOTH anchor values realigned — no hashing, because
    the new head is copied out of a surviving row and the new count is
    ``COUNT(*)``;
  - a row appended with ``prev_hash`` set to the real head and the anchor
    advanced — one sha256, and no rehashing of anything already in the table;
  - the whole log rebuilt with a recomputed chain — one sha256 per forged row;
  - the file deleted, or replaced by another valid chain — no hashing; nothing
    inside the file says which chain it is supposed to be.

**An in-DB anchor can only ever detect *incomplete* tampering.** A writer who
updates the anchor consistently passes, and for the truncation case that costs no
hashing at all, because the value the anchor then needs is already sitting in a
surviving row. The claim this paragraph replaced — that an attack must "recompute
every subsequent ``entry_hash``" — was true only of a content rewrite and
materially overstated the cost of erasing recent activity.

What would close the undetected shapes is an anchor the attacker cannot write:
export ``(count, head_hash)`` off-box — a CI artifact, syslog/SIEM, an object-lock
(WORM) bucket, another host, a transparency log — and compare it on every run,
and/or sign the head with a key held outside the DB. **None of that ships today**,
and nothing inside a single SQLite file can substitute for it. This module makes
destruction loud when it is sloppy; it does not make the trail trustworthy against
a determined writer. See docs/AUDIT.md.

**The reading paths never write.** ``AuditStore(path, read_only=True)`` — what
every ``trelix audit`` subcommand uses — opens ``file:<path>?mode=ro``, runs no
DDL, and refuses to report anything at all unless ``audit_log`` AND ``audit_meta``
already exist (:attr:`AuditStore.missing_tables`). Before that, the read commands
ran the writer's ``CREATE TABLE IF NOT EXISTS`` script, so ``trelix audit verify``
pointed at an unrelated SQLite file *added the audit schema to it* — 8 KB to 32 KB,
five tables where there had been one — and then reported "Audit chain intact." at
exit 0: a green integrity verdict on a database that is not an audit log. The
default read-write constructor still creates the file, its schema and a write lock;
that is the serving path and it has to.

**No stored value can crash a read.** ``audit_meta.value`` and every TEXT column
of ``audit_log`` come back through a ``text_factory`` that hands over raw ``bytes``
when the cell is not decodable UTF-8. ``CAST(x'FFFE41' AS TEXT)`` keeps TEXT
storage class while holding bytes the driver cannot decode, and that raised inside
``fetchall()`` — BEFORE the shape validation below could judge it, so the right
check never ran. A ``bytes`` value fails :func:`_anchor_value_is_wellformed` on
``isinstance`` and is reported as ``anchor_corrupt``; in an ``audit_log`` column it
fails the link or hash comparison and is reported as a row fault. A value this
module exists to judge must be reported, never allowed to abort the one command an
incident responder reaches for first.

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
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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

#: Tables a file must already contain before this module will call it an audit log.
#: A read-only open that cannot find both reports nothing rather than a verdict —
#: see :attr:`AuditStore.missing_tables`.
_REQUIRED_TABLES = ("audit_log", "audit_meta")

# -- verdict vocabulary ------------------------------------------------------
# `verify` names WHICH fault it found, because two families of fault ask an
# operator for different first moves and the id alone cannot tell them apart: a
# row fault names a row that is present and wrong, while an anchor/count fault
# names an id whose EXISTENCE can no longer be proven, where there is nothing to
# go and look at.
REASON_ID_GAP = "id_gap"
REASON_ROW_MISLINKED = "row_mislinked"
REASON_ROW_MUTATED = "row_mutated"
REASON_COUNT_MISMATCH = "count_mismatch"
REASON_HEAD_MISMATCH = "head_mismatch"
REASON_ANCHOR_MISSING = "anchor_missing"
REASON_ANCHOR_CORRUPT = "anchor_corrupt"
REASON_LOG_EMPTIED = "log_emptied"

#: Reasons whose reported id is the first entry whose existence can no longer be
#: proven, rather than a divergent row. Reporting one of these as "the first
#: divergent entry" would send an investigator to inspect a row that is not in
#: the table.
UNPROVABLE_ID_REASONS = frozenset(
    {
        REASON_ID_GAP,
        REASON_COUNT_MISMATCH,
        REASON_HEAD_MISMATCH,
        REASON_ANCHOR_MISSING,
        REASON_ANCHOR_CORRUPT,
        REASON_LOG_EMPTIED,
    }
)

# Anchor value shapes, checked BEFORE any parsing. Both patterns are fully
# anchored with ``\A``/``\Z`` — not ``$``, which would accept a trailing newline
# — because the question asked is "is this byte-for-byte what append() writes",
# and a value with anything appended is not.
#
# `count` is bounded to 19 digits, the widest a SQLite INTEGER can be, so an
# absurd value is rejected on shape and never reaches ``int()``: CPython raises
# ValueError above 4300 digits, which turned a one-cell UPDATE into an unhandled
# traceback out of `trelix audit verify`. Leading '+'/'-'/whitespace, leading
# zeros and underscores are all refused — ``int()`` accepts several of them and
# this writer produces none of them. Refusing '-1' here is also what makes a
# negative count a finding.
_COUNT_RE = re.compile(r"\A(?:0|[1-9][0-9]{0,18})\Z")

# `head_hash` is always ``hashlib.sha256(...).hexdigest()`` — exactly 64
# LOWERCASE hex digits. An uppercased or wrong-length digest is a value no
# released version of ``append`` has ever written.
_HEAD_RE = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """One verification verdict: the id to report, and which fault was found.

    ``tampered_id is None`` means no fault was found. That is NOT proof of
    integrity — the chain is checked against an anchor stored in the same file,
    so a writer who updates both consistently passes. See the module docstring
    for the shapes that are and are not detected.
    """

    tampered_id: int | None
    reason: str | None = None


def _text_or_bytes(raw: bytes) -> str | bytes:
    """``text_factory``: decode a TEXT cell, or hand back the raw bytes.

    sqlite3's default ``text_factory`` is ``str``, which decodes strict UTF-8 and
    raises ``sqlite3.OperationalError`` from inside ``fetchall()`` when a cell will
    not decode. TEXT storage class does not guarantee decodable bytes —
    ``CAST(x'FFFE41' AS TEXT)`` produces a TEXT cell holding invalid UTF-8 — so
    that exception fired on the read, *before* any validation could look at the
    value, and every ``trelix audit`` subcommand ended in a traceback.

    Returning ``bytes`` keeps the value in play so the checks that exist can judge
    it: :func:`_anchor_value_is_wellformed` fails it on ``isinstance``, and an
    ``audit_log`` cell fails its link/hash comparison. Legitimate data is
    unaffected — decodable bytes still come back as ``str``, byte-for-byte what
    ``str`` would have produced.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw


def _anchor_value_is_wellformed(value: Any, pattern: re.Pattern[str]) -> bool:
    """Whether an ``audit_meta`` value is exactly what ``append`` would write.

    ``isinstance`` comes first because ``audit_meta.value`` is declared TEXT but
    SQLite is dynamically typed: TEXT affinity converts an assigned INTEGER to
    text, yet an assigned BLOB is stored as a BLOB and arrives here as ``bytes``,
    which used to reach ``int()`` and raise out of the verifier. It is also what
    rejects a TEXT cell holding undecodable UTF-8, which :func:`_text_or_bytes`
    delivers as ``bytes`` for exactly this reason.
    """
    return isinstance(value, str) and pattern.match(value) is not None


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

    ``read_only=True`` is the mode every reader must use: no directory is created,
    no DDL runs, the connection is ``mode=ro``, and a file that does not already
    have both audit tables is refused (:attr:`missing_tables`) instead of being
    given a verdict. The default is read-write because the serving path has to
    create the file and hold a write lock to append.
    """

    def __init__(self, db_path: Path | str, *, read_only: bool = False) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        #: Audit tables absent from a ``read_only`` open, in :data:`_REQUIRED_TABLES`
        #: order; empty when the open succeeded or failed for any other reason.
        #: Non-empty means "this file is not an audit log", which a caller must
        #: report as *could not check* rather than as a verdict about a chain.
        self.missing_tables: tuple[str, ...] = ()
        try:
            self._conn = self._open_read_only() if read_only else self._open_read_write()
        except Exception as exc:  # noqa: BLE001 — audit init must never crash the caller
            logger.warning(
                "AuditStore init failed for %s; auditing disabled: %s", self._db_path, exc
            )
            self._conn = None

    def _open_read_write(self) -> sqlite3.Connection:
        """Open for appending: create the parent, the file and the schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.text_factory = _text_or_bytes
        conn.executescript(_DDL)
        conn.commit()
        return conn

    def _open_read_only(self) -> sqlite3.Connection | None:
        """Open for reading only, or return ``None`` with :attr:`missing_tables` set.

        ``mode=ro`` is what makes "verify does not write the file it audits" a fact
        rather than an intention: SQLite refuses every write on this handle, so no
        DDL, no journal and no ``user_version`` can be applied even by mistake.
        ``Path.as_uri()`` builds the URI because a path containing ``?`` or ``#``
        would otherwise be parsed as URI query/fragment syntax and silently open
        the wrong file.

        The table check is the second half, and it is the half that matters for a
        *foreign* database: ``mode=ro`` alone would leave a customer database
        unmodified and still walk zero rows of a nonexistent chain and call it
        intact. Requiring both tables turns that into "could not check".
        """
        uri = self._db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.text_factory = _text_or_bytes
        try:
            present = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        except Exception:
            conn.close()
            raise
        absent = tuple(name for name in _REQUIRED_TABLES if name not in present)
        if absent:
            conn.close()
            self.missing_tables = absent
            return None
        return conn

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

    @contextmanager
    def _read_snapshot_locked(self) -> Iterator[sqlite3.Connection]:
        """Run several reads as ONE SQLite read transaction — one snapshot.

        ``verify`` compares the ``audit_log`` rows against the ``audit_meta``
        anchor. Read outside a transaction those are two separate snapshots, and
        ``self._lock`` cannot close the gap: it is a per-INSTANCE
        ``threading.Lock``, so it serializes nothing against the other connection
        or the other process that is actually appending. A legitimate append
        landing between the two reads made the anchor describe one more entry
        than the rows did, and ``verify`` reported that as a truncated tail.

        Measured on the shape api/app.py builds — one legitimate writer doing
        2,870 appends, 30 CLI-shaped verify runs — 24 of 30 reported TAMPERED on
        an undamaged database (independent post-hoc walk: rows=2870, count and
        head both match, no broken link). An operational condition reported as an
        attack is the worst failure a tamper detector has: it teaches operators
        that exit 1 means "try again".

        DEFERRED, not IMMEDIATE: this must not take a write lock on a file it is
        only reading. The extra contention is negligible because SQLite already
        holds the SHARED lock for the whole of the long ``audit_log`` scan; the
        transaction only extends it across the tiny ``audit_meta`` read. And it
        ends in ROLLBACK, on the success path too — a read transaction has
        nothing to commit, and ``verify`` must never write the file it audits.
        """
        assert self._conn is not None
        conn = self._conn
        conn.execute("BEGIN DEFERRED")
        try:
            yield conn
        finally:
            conn.rollback()

    # -- verify --------------------------------------------------------------
    def verify_chain(self) -> int | None:
        """Verify the whole chain, returning only the id :meth:`verify` reports.

        ``None`` means no fault was found — which is not proof of integrity; see
        :meth:`verify` for what that verdict does and does not cover, and for the
        reason code that says which fault was found.
        """
        return self.verify().tampered_id

    def verify(self) -> VerifyResult:
        """Verify the whole chain and name the fault found, if any.

        Detection:
          - recomputed ``entry_hash`` != stored   -> mutated content.
          - stored ``prev_hash`` != running head  -> deleted/reordered row.
          - id gap                                -> deleted middle row.
          - an anchor row absent, with entries    -> the anchor itself is gone.
          - an anchor value malformed             -> the anchor is unusable.
          - live count/head != meta anchor        -> truncated/deleted tail.
          - no rows, but ids were handed out      -> the log was emptied.

        Never raises on the *content* of ``audit_meta`` or ``audit_log``: a value
        this method exists to judge must be reported as a finding, not allowed to
        abort the one command an incident responder reaches for first. A value that
        is not decodable UTF-8 arrives as ``bytes`` (see :func:`_text_or_bytes`) and
        fails the relevant check rather than raising out of ``fetchall()``.

        Does still raise ``sqlite3.DatabaseError`` for a *damaged file* — a
        corrupted b-tree page is not a tamper verdict and must not be reported as
        one, so it is left for the caller to render as "could not check".
        """
        if self._conn is None:
            # Disabled store: nothing was read, and this deliberately still
            # reads as "no fault". See `is_open` — callers that REPORT integrity
            # have to gate on it, because this method cannot tell them apart.
            return VerifyResult(None)
        # Both reads inside ONE transaction: the rows and the anchor they are
        # compared against have to come from the same committed state, or a
        # concurrent legitimate append is indistinguishable from a deleted tail.
        with self._lock, self._read_snapshot_locked() as conn:
            rows = conn.execute(
                "SELECT id, ts, principal, action, resource, outcome, status_code, "
                "client_ip, request_id, trace_id, duration_ms, detail, prev_hash, entry_hash "
                "FROM audit_log ORDER BY id ASC"
            ).fetchall()
            meta = self._read_meta_locked()
            seq_high = self._max_audit_log_seq_locked(conn)

        prev = GENESIS_HASH
        expected_id = 1
        for row in rows:
            if row["id"] != expected_id:  # gapless check catches middle deletes
                return VerifyResult(expected_id, REASON_ID_GAP)
            content = {col: row[col] for col in _CONTENT_COLUMNS}
            if row["prev_hash"] != prev:
                return VerifyResult(int(row["id"]), REASON_ROW_MISLINKED)
            if row["entry_hash"] != _canonical_hash(prev, content):
                return VerifyResult(int(row["id"]), REASON_ROW_MUTATED)
            prev = row["entry_hash"]
            expected_id += 1

        actual_count = len(rows)
        raw_count = meta.get(_META_COUNT)
        raw_head = meta.get(_META_HEAD)

        # The id every anchor fault reports. No surviving row is divergent in
        # these shapes — the walk above just passed on all of them — so naming
        # row 1, or any present row, would be a false statement about a row that
        # is fine. What is no longer provable is whether an entry n+1 ever
        # existed, which is the id the count-mismatch branch below has always
        # returned for the same reason.
        first_unprovable = actual_count + 1

        # PRESENCE, not merely consistency. `append` writes the entry and BOTH
        # anchor rows in one transaction, so "entries present, an anchor row
        # absent" is a state no writer can reach: this file has a single commit
        # in its history and `audit_meta` plus both keys are in that first
        # version, so no released version has ever written an entry without them
        # — legacy databases included. Reading an absent anchor as "no
        # information available" therefore made deleting it a total bypass of
        # gate 2 (one extra DELETE, no hashing), and a log whose tail had been
        # cut verified INTACT.
        #
        # BOTH must be present, not either: with one row dropped, the surviving
        # one can be realigned to a value already in the file — also with no
        # hashing — and that combination verified clean too.
        if actual_count > 0 and (raw_count is None or raw_head is None):
            return VerifyResult(first_unprovable, REASON_ANCHOR_MISSING)

        # Shape before parse, and a bad shape is a finding rather than an
        # exception. Parsing used to live in the read path, so one edited
        # character ended `trelix audit verify` in an unhandled ValueError: an
        # integrity checker that the data it checks can crash is a denial of the
        # tooling, not merely a rough edge.
        if raw_count is not None and not _anchor_value_is_wellformed(raw_count, _COUNT_RE):
            return VerifyResult(first_unprovable, REASON_ANCHOR_CORRUPT)
        if raw_head is not None and not _anchor_value_is_wellformed(raw_head, _HEAD_RE):
            return VerifyResult(first_unprovable, REASON_ANCHOR_CORRUPT)

        # External anchor: detect a truncated/deleted tail the chain can't see.
        # `raw_count` matched _COUNT_RE above, so int() here cannot raise.
        if raw_count is not None:
            meta_count = int(raw_count)
            if actual_count != meta_count:
                # first missing id (ids are gapless 1..actual_count at this point)
                return VerifyResult(min(actual_count, meta_count) + 1, REASON_COUNT_MISMATCH)
        if raw_head is not None and prev != raw_head:
            return VerifyResult(actual_count if actual_count > 0 else 1, REASON_HEAD_MISMATCH)

        # Last, and only for an empty log, because every gate above is more
        # specific: a wipe that leaves the anchor behind is already a
        # count_mismatch, and that names the fault better than this does.
        if actual_count == 0 and seq_high is not None and seq_high > 0:
            return VerifyResult(first_unprovable, REASON_LOG_EMPTIED)
        return VerifyResult(None)

    def _read_meta_locked(self) -> dict[str, Any]:
        """Return the ``audit_meta`` rows as RAW, unparsed values.

        Parsing deliberately does not happen here. This used to ``int()`` the
        count while reading it, which let a value the verifier exists to judge
        raise out of the verifier instead of being reported by it.
        """
        assert self._conn is not None
        return {
            row["key"]: row["value"]
            for row in self._conn.execute("SELECT key, value FROM audit_meta").fetchall()
        }

    def _max_audit_log_seq_locked(self, conn: sqlite3.Connection) -> int | None:
        """Highest id SQLite records as ever handed out to ``audit_log``, if any.

        ``audit_log.id`` is ``INTEGER PRIMARY KEY AUTOINCREMENT``, so SQLite keeps
        the high-water mark in its internal ``sqlite_sequence`` table and ``DELETE``
        does **not** reset it. That makes "zero rows, but ``seq`` > 0" a state
        normal operation cannot produce, and it is the only in-file discriminator
        between an emptied log and a genuinely new one. Measured: fresh ``[]``; after
        a rolled-back append ``[]`` (so a failed append is not a false positive);
        after 5 appends ``[('audit_log', 5)]``; after emptying BOTH tables still
        ``[('audit_log', 5)]``.

        This is not a genesis marker. Nothing is written, no schema is added, no
        ``user_version`` is touched, and nothing new appears in the file — the row
        is SQLite's own bookkeeping, already there, and this only reads it.

        ``None`` (no finding) is returned for every shape that cannot be told apart
        from a database that was never appended to, and each of these is a defeat
        that still works:

        * the table is absent. ``DROP TABLE sqlite_sequence`` is refused by SQLite
          ("may not be dropped"), but ``PRAGMA writable_schema=ON`` +
          ``DELETE FROM sqlite_master WHERE name='sqlite_sequence'`` removes it with
          no extra privilege. An absent table is also what a database with no
          AUTOINCREMENT table looks like, so absence cannot be a finding.
        * the row is absent or ``seq`` was zeroed — same argument.
        * ``seq`` is not an INTEGER (a BLOB, a float, undecodable TEXT). Judging a
          forged type as tamper is tempting, but the check below only fires on an
          EMPTY log, and reporting on a type instead of on a value would make the
          verdict depend on how the wipe was done rather than on what it destroyed.
        * ``seq`` forged to match a truncated log, or set below the real row count.

        ALL matching rows are reduced with ``max`` on purpose. ``sqlite3`` CLI
        ``.dump`` emits ``INSERT INTO sqlite_sequence`` *and* AUTOINCREMENT
        recreates the row on the first insert, so a restored dump legitimately
        carries a DUPLICATE row — measured ``[('audit_log', 5), ('audit_log', 5)]``.
        A check written as "exactly one row" would call every restored dump tamper.
        (``VACUUM``, ``VACUUM INTO`` and ``.backup()`` all preserve ``seq`` exactly.)

        **This check WILL become a false positive when retention pruning lands.**
        ``TRELIX_AUDIT_RETENTION_DAYS`` is accepted but unimplemented and nothing in
        ``src/`` issues ``DELETE FROM audit_log`` today, which is the only reason a
        pruned-to-empty log cannot occur. Whoever implements pruning must update
        this check (and docs/AUDIT.md) or every fully-pruned log will report
        ``log_emptied``.

        Caller must hold ``self._lock``.
        """
        try:
            rows = conn.execute(
                "SELECT seq FROM sqlite_sequence WHERE name = ?", ("audit_log",)
            ).fetchall()
        except sqlite3.OperationalError:
            return None  # no such table: sqlite_sequence
        seqs = [row["seq"] for row in rows]
        integers = [value for value in seqs if isinstance(value, int)]
        return max(integers) if integers else None

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
