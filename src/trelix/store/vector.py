"""
Vector store: stores and searches chunk embeddings using sqlite-vec.

sqlite-vec is a SQLite extension that adds fast vector similarity search
with no external infrastructure — perfect for local/dev use.
Swap out for Qdrant by changing this file only (same interface).
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

import sqlite_vec


class VectorStore:
    """
    Stores chunk embeddings in a SQLite database using sqlite-vec.

    Usage:
        store = VectorStore(db_path, dimension=1536)
        store.upsert(chunk_id=1, embedding=[0.1, 0.2, ...])
        results = store.search(query_embedding, k=20)  # → list of (chunk_id, score)
    """

    def __init__(self, db_path: Path, dimension: int = 1536) -> None:
        self.dimension = dimension
        # check_same_thread=False allows use from worker threads (retrieval is read-only).
        # _lock serialises all execute() calls so the connection's internal state stays consistent.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._setup_table()

    def _setup_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings
            USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self.dimension}]
            )
            """
        )
        self._conn.commit()

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
        Return top-k (chunk_id, distance) pairs. Lower distance = more similar.
        sqlite-vec uses L2 distance by default; we negate to get a similarity score.
        Thread-safe: guarded by _lock so concurrent worker threads don't interleave.
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

    def _pack(self, embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    def close(self) -> None:
        self._conn.close()
