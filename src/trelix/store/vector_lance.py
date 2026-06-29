"""
LanceDB vector store backend.

LanceDB 0.6+ provides ARM-native HNSW with zero SQLite dependency.
Validated 3-5x faster insert at >100k vectors vs sqlite-vec (vecdb-bench).

Best for:
    - Repos > 500k chunks (where sqlite-vec HNSW becomes memory-constrained)
    - Apple Silicon and ARM servers (native SIMD)
    - Multi-repo deployments sharing a vector store

Usage:
    TRELIX_STORE_BACKEND=lance LANCE_URI=.trelix/lance trelix index ./my-repo
    pip install trelix[lance]
"""

from __future__ import annotations

import logging
from typing import Any

from trelix.store.vector import BaseVectorStore

logger = logging.getLogger("trelix.store.lance")

_lancedb: Any | None
try:
    import lancedb as _lancedb_module

    _lancedb = _lancedb_module
except ImportError:
    _lancedb = None

lancedb = _lancedb


class LanceVectorStore(BaseVectorStore):
    """
    Vector store backed by LanceDB.

    Each trelix index gets one LanceDB table named `chunks` (configurable).
    Vectors are stored as fixed-size float32 arrays in a `vector` column.
    chunk_id (from trelix's SQLite `chunks` table) is stored for lookup.
    """

    def __init__(
        self,
        uri: str,
        table_name: str = "chunks",
        dimension: int = 1024,
    ) -> None:
        if lancedb is None:
            raise ImportError(
                "lancedb is required for the lance store backend. "
                "Install with: pip install 'trelix[lance]'"
            )
        self._uri = uri
        self._table_name = table_name
        self._dimension = dimension
        self._db = lancedb.connect(uri)
        self._table = self._get_or_create_table()

    def _get_or_create_table(self) -> Any:
        try:
            return self._db.open_table(self._table_name)
        except Exception:
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("chunk_id", pa.int64()),
                    pa.field("vector", pa.list_(pa.float32(), self._dimension)),
                ]
            )
            return self._db.create_table(
                self._table_name,
                schema=schema,
                mode="create",
            )

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        """Insert or replace embeddings — delete-then-add pattern for upserts."""
        import pyarrow as pa

        if not pairs:
            return
        ids = [p[0] for p in pairs]
        vecs = [p[1] for p in pairs]
        data = pa.table(
            {
                "chunk_id": pa.array(ids, type=pa.int64()),
                "vector": pa.array(vecs, type=pa.list_(pa.float32(), self._dimension)),
            }
        )
        # Delete existing rows for these chunk_ids then add fresh
        id_list = ", ".join(str(i) for i in ids)
        try:
            self._table.delete(f"chunk_id IN ({id_list})")
        except Exception:
            pass
        self._table.add(data)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """Return top-k (chunk_id, distance) pairs for the query vector."""
        rows = self._table.search(query).limit(k).to_list()
        return [(row["chunk_id"], row.get("_distance", 0.0)) for row in rows]

    def delete_batch(self, chunk_ids: list[int]) -> None:
        """Delete embeddings for the given chunk_ids. No-op for empty list."""
        if not chunk_ids:
            return
        id_list = ", ".join(str(i) for i in chunk_ids)
        try:
            self._table.delete(f"chunk_id IN ({id_list})")
        except Exception as exc:
            logger.warning("LanceDB delete_batch failed: %s", exc)

    def count(self) -> int:
        """Return the total number of stored embeddings."""
        try:
            return int(self._table.count_rows())
        except Exception:
            return 0

    def upsert_file_summary_embedding(self, file_id: int, embedding: list[float]) -> None:
        """No-op stub — LanceDB backend does not store file-summary embeddings."""
