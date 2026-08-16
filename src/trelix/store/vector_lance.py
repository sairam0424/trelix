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
        # Delete existing rows for these chunk_ids then add fresh.
        #
        # The delete failing is NOT recoverable by adding anyway: LanceDB has no
        # chunk_id uniqueness constraint, so the add turns "replace" into "append"
        # and one chunk_id gains a row per reindex (measured 1 -> 2 -> 3 -> 4 rows
        # for a single chunk across three failed-delete upserts). Every later
        # search() then returns that chunk_id N times, spending N of the k result
        # slots on one chunk, and count() drifts above the SQLite chunks table —
        # damage no subsequent upsert can undo, only a full reindex.
        #
        # So abort instead of adding: the table keeps the previous single row for
        # these ids (stale vector, still searchable) and the caller learns the
        # batch did not land rather than shipping a corrupted index. This is
        # deliberately louder than delete_batch below, where a swallowed failure
        # only leaves rows behind that a later upsert can still replace.
        id_list = ", ".join(str(i) for i in ids)
        try:
            self._table.delete(f"chunk_id IN ({id_list})")
        except Exception as exc:
            logger.error(
                "LanceDB upsert_batch aborted: delete of %d chunk_id(s) failed (%s) — "
                "adding anyway would create duplicate rows for those ids",
                len(ids),
                exc,
            )
            raise
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
        """
        Insert or replace a file-level summary embedding.

        Uses chunk_id = -(file_id) as a negative sentinel to distinguish
        file-summary entries from regular chunk entries — same convention as
        SQLiteVectorStore.
        """
        self.upsert_batch([(-(file_id), embedding)])

    def search_file_summaries(
        self, query_embedding: list[float], k: int
    ) -> list[tuple[int, float]]:
        """Search file-summary rows (negative chunk_ids). Returns (file_id, score) pairs."""
        results = self.search(query_embedding, k=k * 5)
        return [(-cid, score) for cid, score in results if cid < 0][:k]

    _SUB_CHUNK_OFFSET = 10_000_000

    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        """Store sub-chunk embedding using chunk_id = sub_chunk_id + _SUB_CHUNK_OFFSET."""
        self.upsert_batch([(sub_chunk_id + self._SUB_CHUNK_OFFSET, embedding)])

    def search_sub_chunks(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        """Search sub-chunk embeddings only. Returns (sub_chunk_id, score) pairs."""
        results = self.search(query_embedding, k=k * 5)
        return [
            (cid - self._SUB_CHUNK_OFFSET, score)
            for cid, score in results
            if cid >= self._SUB_CHUNK_OFFSET
        ][:k]
