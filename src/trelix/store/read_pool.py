"""
Read-only SQLite connection pool for parallel BM25 reads.

Database opens exactly one writer connection with check_same_thread=False,
shared across the whole process — every bm25_search() call funnels through
that single connection object regardless of how many application threads
call it, so concurrent FTS5 MATCH queries cannot truly execute in parallel
even though the database is already in WAL mode (which permits many
concurrent readers at the SQLite-engine level).

This pool opens a small fixed number of SEPARATE connections, each using
SQLite's read-only URI mode (file:{path}?mode=ro) plus PRAGMA query_only,
so read-heavy deployments can draw an independent connection per concurrent
BM25 query instead of serializing on the shared writer connection.

Opt-in via StoreConfig.bm25_read_pool_size (default 0 = disabled, meaning
Database.bm25_search() uses the existing single-connection path unchanged).
"""

from __future__ import annotations

import queue
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ReadOnlyConnectionPool:
    """A fixed-size pool of read-only SQLite connections against one WAL-mode
    database file. acquire() blocks until a connection is available if the
    pool is momentarily exhausted (pool_size concurrent readers max)."""

    def __init__(self, db_path: Path, pool_size: int = 4) -> None:
        self._db_path = db_path
        self._pool_size = pool_size
        self._available: queue.Queue[sqlite3.Connection] = queue.Queue()
        self._connections: list[sqlite3.Connection] = []
        for _ in range(pool_size):
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
            conn.execute("PRAGMA query_only = ON")
            conn.row_factory = sqlite3.Row
            self._connections.append(conn)
            self._available.put(conn)

    @contextmanager
    def acquire(self) -> Iterator[sqlite3.Connection]:
        conn = self._available.get()
        try:
            yield conn
        finally:
            self._available.put(conn)

    def close_all(self) -> None:
        for conn in self._connections:
            conn.close()
        self._connections = []
