"""Three edges of the audit read path that review found unpinned or unhandled.

Each of these was reachable without any tampering at all, which is what makes them
worth their own file: a tamper detector that misreports an *operational* condition
teaches an operator to distrust its verdicts, and then it protects nothing.

1. **A hot rollback journal.** ``audit.db`` runs SQLite's default rollback journal,
   so a writer killed between the journal fsync and its unlink leaves a file that
   must be rolled back before it can be read — and a ``mode=ro`` handle cannot
   perform a rollback. Measured: SIGKILLing the real append loop left a journal in
   33 of 40 trials, 6 of which were unreadable. Before the read path opened
   read-only this "worked", because a read-WRITE open silently performed the
   rollback — the reader recovered the database by writing to the artifact it was
   auditing. Refusing is the better trade, but only if the operator is told which
   problem they actually have, so it reports its own cause rather than the generic
   "check that it is readable" about a file that is readable.

2. **``Path.as_uri()`` is load-bearing.** The whole read-only guarantee rests on the
   ``mode=ro`` URI, and building that URI by interpolation lets a path containing
   ``?`` or ``#`` be parsed as query/fragment syntax — silently opening a different
   file, or none. Reverting that one line to an f-string passed the entire suite, so
   the guarantee was resting on an untested detail.

3. **A caller-supplied limit larger than SQLite can bind.** ``--limit 10**21``
   raised ``OverflowError`` out of ``audit list``: a traceback at exit 1, the code
   reserved for *the chain is damaged*, on a healthy database.

Store-level pins for the read-hardening and wipe-detection claims live in
tests/unit/test_audit_read_hardening.py and tests/unit/test_audit_wipe_detection.py;
CLI exit codes and byte-identity live in tests/unit/test_cli_audit_read_only.py.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from trelix.audit.events import ACTION_AUTH, OUTCOME_SUCCESS, AuditEvent
from trelix.audit.store import AuditStore


def _event(i: int) -> AuditEvent:
    return AuditEvent(
        ts=f"2026-08-21T00:0{i % 10}:00+00:00",
        principal="svc@local",
        action=ACTION_AUTH,
        resource="/verify",
        outcome=OUTCOME_SUCCESS,
        detail=f"entry {i}",
    )


def _chain(path: Path, n: int = 4) -> None:
    """Write a genuine chain through the shipped writer, then close it cleanly."""
    store = AuditStore(path)
    for i in range(n):
        assert store.append(_event(i)) is True
    store.close()


class TestAHotRollbackJournalIsItsOwnDiagnosis:
    """An interrupted writer is an operational state, not a verdict about a chain."""

    @staticmethod
    def _with_hot_journal(tmp_path: Path) -> Path:
        """A db plus the ``-journal`` of a transaction that never committed.

        Copied out mid-transaction rather than by killing a process, so the test is
        deterministic where a SIGKILL is not — measured, a killed writer leaves a
        journal in roughly 4 of 5 attempts and an unreadable one in far fewer.

        The tiny page cache is what makes the journal *hot* rather than merely
        present. A journal only has to be replayed once uncommitted pages have
        actually reached the database file; with the default cache the whole
        transaction sits in memory until commit, so the copied pair is self-
        consistent and opens cleanly. Verified both ways: one insert under the
        default cache gives ``is_open=True``, and the spill below gives
        ``is_open=False`` with ``needs_journal_recovery`` set.
        """
        source = tmp_path / "live" / "audit.db"
        source.parent.mkdir()
        _chain(source)

        conn = sqlite3.connect(source)
        conn.execute("PRAGMA cache_size = 1")
        conn.execute("BEGIN IMMEDIATE")
        for i in range(4000):
            conn.execute(
                "INSERT INTO audit_log (ts, principal, action, resource, outcome, "
                "prev_hash, entry_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"spill-{i}", "p", "a", "/r", "success", "0" * 64, "1" * 64),
            )
        journal = source.with_name(source.name + "-journal")
        assert journal.exists(), "expected a rollback journal while a write txn is open"

        victim = tmp_path / "victim"
        victim.mkdir()
        target = victim / "audit.db"
        shutil.copy2(source, target)
        shutil.copy2(journal, target.with_name(target.name + "-journal"))

        conn.rollback()
        conn.close()
        return target

    def test_it_is_reported_as_needing_recovery_not_as_an_unreadable_path(
        self, tmp_path: Path
    ) -> None:
        db = self._with_hot_journal(tmp_path)
        store = AuditStore(db, read_only=True)
        assert store.is_open is False
        assert store.needs_journal_recovery is True
        # Not conflated with "this file is not an audit log", whose remedy differs.
        assert store.missing_tables == ()

    def test_the_read_only_open_leaves_the_pair_untouched(self, tmp_path: Path) -> None:
        """The point of refusing: recovery must not be performed by the reader."""
        db = self._with_hot_journal(tmp_path)
        journal = db.with_name(db.name + "-journal")
        before = (db.read_bytes(), journal.read_bytes())

        AuditStore(db, read_only=True)

        assert journal.exists(), "the reader must not have replayed and removed it"
        assert (db.read_bytes(), journal.read_bytes()) == before

    def test_one_read_write_open_recovers_it_as_the_message_promises(self, tmp_path: Path) -> None:
        """The remedy the CLI prints has to actually work, or it is worse than none."""
        db = self._with_hot_journal(tmp_path)
        assert AuditStore(db, read_only=True).is_open is False

        recovered = AuditStore(db)  # read-write: performs the rollback
        recovered.close()

        after = AuditStore(db, read_only=True)
        assert after.is_open is True
        assert after.needs_journal_recovery is False
        assert after.verify_chain() is None

    def test_a_healthy_database_never_claims_to_need_recovery(self, tmp_path: Path) -> None:
        """Control. The flag must not be reachable by an ordinary open."""
        db = tmp_path / "audit.db"
        _chain(db)
        store = AuditStore(db, read_only=True)
        assert store.is_open is True
        assert store.needs_journal_recovery is False


class TestTheReadOnlyUriSurvivesAwkwardPaths:
    """Pins ``Path.as_uri()``: interpolation would parse these as URI syntax."""

    def test_a_directory_containing_a_question_mark_still_opens_read_only(
        self, tmp_path: Path
    ) -> None:
        odd = tmp_path / "repo?v=2"
        odd.mkdir()
        db = odd / "audit.db"
        _chain(db)

        store = AuditStore(db, read_only=True)
        assert store.is_open is True, "as_uri() must escape '?' rather than start a query"
        assert store.verify_chain() is None
        assert len(store.recent(10)) == 4

    def test_a_directory_containing_a_hash_still_opens_read_only(self, tmp_path: Path) -> None:
        odd = tmp_path / "repo#main"
        odd.mkdir()
        db = odd / "audit.db"
        _chain(db)

        store = AuditStore(db, read_only=True)
        assert store.is_open is True, "as_uri() must escape '#' rather than start a fragment"
        assert store.verify_chain() is None

    def test_the_handle_is_still_genuinely_read_only_on_such_a_path(self, tmp_path: Path) -> None:
        """Escaping the path must not have been achieved by dropping mode=ro."""
        odd = tmp_path / "repo?x#y"
        odd.mkdir()
        db = odd / "audit.db"
        _chain(db)

        store = AuditStore(db, read_only=True)
        assert store.is_open is True
        assert store._conn is not None
        try:
            store._conn.execute("INSERT INTO audit_meta (key, value) VALUES ('k', 'v')")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc)
        else:  # pragma: no cover - would mean the guarantee is gone
            raise AssertionError("mode=ro was lost while fixing the path escaping")


class TestALimitLargerThanSqliteCanBindIsNotAFinding:
    """A caller mistake must not surface as the exit code meaning 'damaged'."""

    def test_a_limit_past_the_sqlite_ceiling_returns_rows_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "audit.db"
        _chain(db)
        store = AuditStore(db, read_only=True)

        rows = store.recent(10**21)  # unclamped this raised OverflowError at bind time

        assert len(rows) == 4

    def test_the_clamp_does_not_change_an_ordinary_limit(self, tmp_path: Path) -> None:
        db = tmp_path / "audit.db"
        _chain(db, n=6)
        store = AuditStore(db, read_only=True)

        assert len(store.recent(2)) == 2
        assert len(store.recent(6)) == 6
        assert store.recent(0) == []
        assert store.recent(-1) == []
