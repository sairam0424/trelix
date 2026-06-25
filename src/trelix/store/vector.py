"""
Vector store: stores and searches chunk embeddings using sqlite-vec.

sqlite-vec is a SQLite extension that adds fast vector similarity search
with no external infrastructure — perfect for local/dev use.
Swap out for Qdrant by changing this file only (same interface).

HNSW support:
    sqlite-vec >= 0.1.6 ships an HNSW index via the +hnsw() auxiliary
    column syntax.  VectorStore tries to create the table with HNSW
    enabled and falls back to a plain flat vec0 scan when the installed
    version does not support it.  The active mode is exposed via
    ``hnsw_active`` and ``info()``.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading
from pathlib import Path

import sqlite_vec

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Stores chunk embeddings in a SQLite database using sqlite-vec.

    Usage:
        store = VectorStore(db_path, dimension=1536)
        store.upsert(chunk_id=1, embedding=[0.1, 0.2, ...])
        results = store.search(query_embedding, k=20)  # -> list of (chunk_id, score)

    HNSW parameters:
        hnsw            -- enable HNSW index (default True)
        hnsw_m          -- max connections per layer, default 16
        hnsw_ef_construction -- build-time beam width, default 200
    """

    def __init__(
        self,
        db_path: Path,
        dimension: int = 1536,
        *,
        hnsw: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
    ) -> None:
        self._dim = dimension
        self._hnsw_requested = hnsw
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction

        # check_same_thread=False allows use from worker threads (retrieval is read-only).
        # _lock serialises all execute() calls so the connection's internal state stays consistent.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._hnsw_active: bool = self._setup_table()

        if self._hnsw_active:
            logger.info(
                "Vector store: HNSW (m=%d, ef_construction=%d)",
                hnsw_m,
                hnsw_ef_construction,
            )
        else:
            logger.info("Vector store: flat scan")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_table(self) -> bool:
        """
        Create the vec0 virtual table.

        Returns True when HNSW was successfully activated, False when the
        installed sqlite-vec version does not support the +hnsw() syntax
        and we fell back to flat vec0.
        """
        if self._hnsw_requested:
            hnsw_active = self._try_create_hnsw_table()
            if hnsw_active:
                return True
            logger.warning(
                "sqlite-vec HNSW not supported by installed version — "
                "falling back to flat vec0 scan"
            )

        # Plain flat vec0
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings
            USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self._dim}]
            )
            """
        )
        self._conn.commit()
        return False

    def _try_create_hnsw_table(self) -> bool:
        """
        Attempt to create the chunk_embeddings table with an HNSW index.

        sqlite-vec >= 0.1.6 supports the ``+hnsw(m=N, ef_construction=N)``
        auxiliary column syntax.  If the table already exists (re-open) this
        is a no-op and we infer HNSW is active by inspecting the table schema.

        Returns True on success, False if the version does not support HNSW.
        """
        # If the table already exists, check whether it was created with HNSW.
        existing = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
        ).fetchone()
        if existing is not None:
            return "+hnsw" in (existing[0] or "").lower()

        try:
            self._conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings
                USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self._dim}],
                    +hnsw(m={self._hnsw_m}, ef_construction={self._hnsw_ef_construction})
                )
                """
            )
            self._conn.commit()
            return True
        except sqlite3.OperationalError:
            # Either the syntax is unsupported or another error — fall back to flat.
            return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def hnsw_active(self) -> bool:
        """True when the HNSW index is in use for this store."""
        return self._hnsw_active

    def upsert(self, chunk_id: int, embedding: list[float]) -> None:
        packed = self._pack(embedding)
        # sqlite-vec virtual tables do not support INSERT OR REPLACE semantics —
        # delete first, then insert to achieve a true upsert.
        self._conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.execute(
            "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, packed),
        )
        self._conn.commit()

    def upsert_batch(self, items: list[tuple[int, list[float]]]) -> None:
        """Batch upsert for efficiency during indexing."""
        try:
            for chunk_id, emb in items:
                packed = self._pack(emb)
                self._conn.execute(
                    "DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,)
                )
                self._conn.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, packed),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def search(self, query_embedding: list[float], k: int = 20) -> list[tuple[int, float]]:
        """
        Return top-k (chunk_id, distance) pairs.  Lower distance = more similar.
        sqlite-vec uses L2 distance by default.
        Thread-safe: guarded by _lock so concurrent worker threads don't interleave.

        When HNSW is active, sqlite-vec automatically routes the MATCH query
        through the HNSW index for O(log n) approximate nearest-neighbour search.
        """
        packed = self._pack(query_embedding)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT chunk_id, distance
                FROM chunk_embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (packed, k),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def delete(self, chunk_id: int) -> None:
        self._conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.commit()

    def delete_batch(self, chunk_ids: list[int]) -> None:
        """Delete multiple embeddings by chunk_id. Used to clean stale vectors on re-index."""
        if not chunk_ids:
            return
        for chunk_id in chunk_ids:
            self._conn.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,)
            )
        self._conn.commit()

    def info(self) -> dict:
        """
        Return a summary dict suitable for ``trelix stats``.

        Returns:
            {
                "backend": "sqlite-vec",
                "hnsw": bool,
                "dimension": int,
                "count": int,
            }
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunk_embeddings"
        ).fetchone()
        count = row[0] if row else 0
        return {
            "backend": "sqlite-vec",
            "hnsw": self._hnsw_active,
            "dimension": self._dim,
            "count": count,
        }

    def _pack(self, embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    def close(self) -> None:
        self._conn.close()
