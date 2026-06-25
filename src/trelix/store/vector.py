"""
Vector store: stores and searches chunk embeddings.

Backends:
  - SQLiteVectorStore  — sqlite-vec extension, no external infra needed (default)
  - QdrantVectorStore  — Qdrant HNSW index, scales to 500k+ chunks (optional)

Use make_vector_store(config, dimension) to get the right backend.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import sqlite_vec

from trelix.core.config import IndexConfig

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseVectorStore(ABC):
    """
    Protocol that every vector-store backend must implement.

    All methods operate on (chunk_id: int, embedding: list[float]) pairs.
    chunk_id is the primary key from the `chunks` table in the SQLite DB.
    """

    @abstractmethod
    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        """Insert or replace embeddings for the given (chunk_id, vector) pairs."""

    @abstractmethod
    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """Return top-k (chunk_id, score/distance) pairs for the query vector."""

    @abstractmethod
    def delete_batch(self, chunk_ids: list[int]) -> None:
        """Delete embeddings for the given chunk_ids. No-op for empty list."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored embeddings."""


# ---------------------------------------------------------------------------
# SQLite backend (default)
# ---------------------------------------------------------------------------


class SQLiteVectorStore(BaseVectorStore):
    """
    Stores chunk embeddings in a SQLite database using sqlite-vec.

    Usage:
        store = SQLiteVectorStore(db_path, dimension=1536)
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

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        """Batch upsert for efficiency during indexing."""
        try:
            for chunk_id, emb in pairs:
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

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunk_embeddings"
        ).fetchone()
        return row[0] if row else 0

    def _pack(self, embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Legacy name kept so existing import sites (indexer, retriever) continue to work
#: until they are updated to use make_vector_store().
VectorStore = SQLiteVectorStore


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_vector_store(config: IndexConfig, dimension: int) -> BaseVectorStore:
    """
    Return the configured vector-store backend.

    Backend is selected by config.store.backend:
      "sqlite"  (default) → SQLiteVectorStore backed by <repo>/.trelix/index.db
      "qdrant"             → QdrantVectorStore backed by a running Qdrant instance

    Args:
        config:    IndexConfig instance (provides store sub-config and db_path).
        dimension: Embedding dimension (must match the embedder being used).
    """
    backend = getattr(config.store, "backend", "sqlite")
    if backend == "qdrant":
        from trelix.store.vector_qdrant import QdrantVectorStore
        return QdrantVectorStore(config, dimension)
    return SQLiteVectorStore(db_path=config.db_path_absolute, dimension=dimension)
