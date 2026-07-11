"""Tests for ReadOnlyConnectionPool — a small pool of read-only SQLite
connections against a WAL-mode database, enabling parallel BM25 reads
without serializing on the single shared writer connection."""

from __future__ import annotations

import sqlite3
import threading


class TestReadOnlyConnectionPool:
    def _make_wal_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t (v) VALUES ('hello')")
        conn.commit()
        conn.close()
        return db_path

    def test_acquire_returns_a_working_connection(self, tmp_path):
        from trelix.store.read_pool import ReadOnlyConnectionPool

        db_path = self._make_wal_db(tmp_path)
        pool = ReadOnlyConnectionPool(db_path, pool_size=2)
        with pool.acquire() as conn:
            row = conn.execute("SELECT v FROM t WHERE id = 1").fetchone()
            assert row[0] == "hello"
        pool.close_all()

    def test_acquired_connection_is_read_only(self, tmp_path):
        from trelix.store.read_pool import ReadOnlyConnectionPool

        db_path = self._make_wal_db(tmp_path)
        pool = ReadOnlyConnectionPool(db_path, pool_size=2)
        with pool.acquire() as conn:
            try:
                conn.execute("INSERT INTO t (v) VALUES ('should fail')")
                conn.commit()
                assert False, "write on a read-only connection must raise"
            except sqlite3.OperationalError:
                pass  # expected — read-only connection rejects writes
        pool.close_all()

    def test_concurrent_acquires_do_not_block_each_other(self, tmp_path):
        """N threads acquiring+querying simultaneously must all succeed —
        proves the pool actually permits parallel reads, not serialized
        access through one shared connection."""
        from trelix.store.read_pool import ReadOnlyConnectionPool

        db_path = self._make_wal_db(tmp_path)
        pool = ReadOnlyConnectionPool(db_path, pool_size=4)
        results = []
        errors = []

        def worker():
            try:
                with pool.acquire() as conn:
                    row = conn.execute("SELECT v FROM t WHERE id = 1").fetchone()
                    results.append(row[0])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"unexpected errors under concurrent access: {errors}"
        assert results == ["hello"] * 20
        pool.close_all()

    def test_pool_size_bounds_the_number_of_connections(self, tmp_path):
        from trelix.store.read_pool import ReadOnlyConnectionPool

        db_path = self._make_wal_db(tmp_path)
        pool = ReadOnlyConnectionPool(db_path, pool_size=2)
        assert len(pool._connections) == 2
        pool.close_all()
