"""
Vector store: stores and searches chunk embeddings.

Backends:
  - SQLiteVectorStore  — sqlite-vec extension, no external infra needed (default)
  - QdrantVectorStore  — Qdrant HNSW index, scales to 500k+ chunks (optional)

Use make_vector_store(config, dimension) to get the right backend.

HNSW support (SQLite backend):
    sqlite-vec >= 0.1.6 ships an HNSW index via the +hnsw() auxiliary
    column syntax.  SQLiteVectorStore tries to create the table with HNSW
    enabled and falls back to a plain flat vec0 scan when the installed
    version does not support it.  The active mode is exposed via
    ``hnsw_active`` and ``info()``.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import sqlite_vec

from trelix.core.config import IndexConfig

logger = logging.getLogger(__name__)

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
        self._conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
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
                self._conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
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
        self._conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
        self._conn.commit()

    def delete_batch(self, chunk_ids: list[int]) -> None:
        """Delete multiple embeddings by chunk_id. Used to clean stale vectors on re-index."""
        if not chunk_ids:
            return
        for chunk_id in chunk_ids:
            self._conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()
        return row[0] if row else 0

    def info(self) -> dict[str, Any]:
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
        return {
            "backend": "sqlite-vec",
            "hnsw": self._hnsw_active,
            "dimension": self._dim,
            "count": self.count(),
        }

    def _pack(self, embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    def upsert_file_summary_embedding(self, file_id: int, embedding: list[float]) -> None:
        """
        Insert or replace a file-level summary embedding.

        Uses the same vec0 virtual table as symbol chunks but stores the
        file_id as a *negative* chunk_id sentinel so the retriever can
        distinguish file-summary entries from symbol-chunk entries.

        Convention: chunk_id = -(file_id) for file-summary rows.
        This avoids a separate virtual table while keeping the search
        interface identical.
        """
        sentinel_id = -file_id
        packed = self._pack(embedding)
        with self._lock:
            self._conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (sentinel_id,))
            self._conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                (sentinel_id, packed),
            )
            self._conn.commit()

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
    if backend == "lance":
        from trelix.store.vector_lance import LanceVectorStore

        uri = config.store.lance_uri
        if not Path(uri).is_absolute():
            uri = str(Path(config.repo_path) / uri)
        return LanceVectorStore(
            uri=uri,
            table_name=config.store.lance_table,
            dimension=dimension,
        )
    if backend == "qdrant":
        from trelix.store.vector_qdrant import QdrantVectorStore

        return QdrantVectorStore(config, dimension)
    return SQLiteVectorStore(db_path=config.db_path_absolute, dimension=dimension)
