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

Concurrency:
    A LanceDB table handle pins the version it opened at, and upsert_batch
    replaces rows with two separate commits (delete then add). Both facts are
    load-bearing here — see _checkout_latest and _WRITE_LOCKS. Writes from a
    second OS process on the same URI are still outside this module's reach;
    upsert_batch reports the duplicate rows they leave rather than hiding them.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from trelix.store.vector import BaseVectorStore

logger = logging.getLogger("trelix.store.lance")

# One write lock per LanceDB table, shared by every LanceVectorStore in this
# process that points at it.
#
# upsert_batch replaces rows with a delete followed by an add — two separate
# LanceDB commits — and LanceDB enforces no uniqueness on chunk_id. Two writers
# that both delete a chunk_id before either adds leave TWO rows for it, and both
# calls return normally: measured on lancedb 0.33.0, a forced interleave on one
# shared table handle gives 2 rows for one chunk_id with zero exceptions raised.
#
# This lock covers only that interleave. It is necessary but not sufficient —
# _checkout_latest fixes the other half, and the cross-handle case needs both.
# Table.merge_insert, LanceDB's own upsert primitive, was measured worse rather
# than better: 4 threads over 50 chunk_ids left 53-62 rows via merge_insert
# against 50-53 via delete+add, equally silently, so it is not a way out of
# locking.
#
# Keyed per table rather than one global lock so unrelated tables still write in
# parallel. Costs roughly a third of two-writer upsert throughput (median 135 ->
# 87 batches of 32 per second, 9 runs) by reducing two writers to one; the
# indexer caps embedding concurrency at 4 API calls, so the embedder, not this,
# is the pipeline's limit unless the embedder is local and very fast.
_WRITE_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def _write_lock(uri: str, table_name: str) -> threading.RLock:
    """Return the process-wide write lock for one (uri, table_name) pair."""
    key = (uri if "://" in uri else os.path.realpath(uri), table_name)
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = _WRITE_LOCKS[key] = threading.RLock()
        return lock


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
        self._write_lock = _write_lock(uri, table_name)

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
        predicate = f"chunk_id IN ({id_list})"
        # The lock makes delete+add one write step for this table (see
        # _WRITE_LOCKS): without it a concurrent writer on the same chunk_id
        # duplicates the row and nothing raises.
        with self._write_lock:
            try:
                # Refresh before deleting: on a pinned snapshot the delete
                # removes nothing and the add below appends a second row for a
                # chunk_id that already had one. See _checkout_latest.
                self._checkout_latest()
                self._table.delete(predicate)
            except Exception as exc:
                logger.error(
                    "LanceDB upsert_batch aborted: refresh/delete of %d chunk_id(s) failed "
                    "(%s) — adding anyway would create duplicate rows for those ids",
                    len(ids),
                    exc,
                )
                raise
            self._table.add(data)
            self._report_duplicate_rows(ids, predicate)

    def _checkout_latest(self) -> None:
        """
        Point this handle at the table's newest version.

        A handle pins the version it opened at and never advances on another
        handle's or process's writes. Measured on lancedb 0.33.0, two handles on
        one URI: after handle B upserts one row, A.count() returns 0 against a
        true 1, and A.delete("chunk_id IN (3)") commits a new table version yet
        removes nothing — the predicate ran against A's old snapshot. That is how
        upsert_batch duplicates a row without any call failing, and it is why the
        write lock alone is not enough. Costs 0.21 ms on a 200k-row table,
        against 9.2 ms for one k=10 search.
        """
        self._table.checkout_latest()

    def _report_duplicate_rows(self, ids: list[int], predicate: str) -> None:
        """
        Count the rows the upsert just wrote and log if there is more than one
        per chunk_id.

        The lock above is process-local, so a second OS process on the same URI
        — the "multi-repo deployments sharing a vector store" case in this
        module's docstring — can still interleave with delete+add. Duplicates
        only disappear on a later clean upsert of the same chunk_id (verified:
        2 rows -> 1 after one), and the indexer embeds each chunk once per run,
        so in practice they survive the run: search() then spends N of its k
        slots on one chunk and count() drifts above the SQLite chunks table.
        Silence there is what made the earlier delete-swallowing bug invisible,
        so report it instead.

        Costs one filtered count_rows — measured 2.3 ms for 32 ids over 200k
        rows, against an embedding-bound pipeline.

        Only an over-count is reported. Fewer rows than written means a
        concurrent delete_batch removed some, which is a legitimate caller
        action, not damage.
        """
        expected = len(set(ids))
        try:
            written = int(self._table.count_rows(predicate))
        except Exception as exc:
            logger.warning(
                "LanceDB upsert_batch could not verify %d chunk_id(s): %s", expected, exc
            )
            return
        if written <= expected:
            return
        repeats_in_batch = len(ids) - expected
        cause = (
            f"This batch repeated {repeats_in_batch} chunk_id(s)."
            if repeats_in_batch
            else (
                "A writer outside this process interleaved with the delete+add "
                "(the per-table lock cannot span processes)."
            )
        )
        logger.error(
            "LanceDB upsert_batch left %d rows for %d chunk_id(s) — %d duplicate row(s). %s "
            "search() will return those chunk_ids more than once and count() will "
            "over-report until they are reindexed. ids: %s",
            written,
            expected,
            written - expected,
            cause,
            predicate,
        )

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """Return top-k (chunk_id, distance) pairs for the query vector."""
        # Same pinned-snapshot problem as the write path: without this a handle
        # opened before another process indexed answers from the version it
        # opened at and silently misses every vector written since. 0.21 ms
        # against a 9.2 ms search on 200k rows.
        self._checkout_latest()
        rows = self._table.search(query).limit(k).to_list()
        return [(row["chunk_id"], row.get("_distance", 0.0)) for row in rows]

    def delete_batch(self, chunk_ids: list[int]) -> None:
        """Delete embeddings for the given chunk_ids. No-op for empty list."""
        if not chunk_ids:
            return
        id_list = ", ".join(str(i) for i in chunk_ids)
        # Same lock as upsert_batch: landing between a concurrent upsert's delete
        # and its add would let that add put the row back, silently undoing this
        # delete. And the same refresh — on a pinned snapshot this delete commits
        # a new version but removes nothing, so the row survives with no error.
        try:
            with self._write_lock:
                self._checkout_latest()
                self._table.delete(f"chunk_id IN ({id_list})")
        except Exception as exc:
            logger.warning("LanceDB delete_batch failed: %s", exc)

    def count(self) -> int:
        """Rows in the table, sentinels included — see `BaseVectorStore.count`.

        NOT the figure to compare against the SQLite `chunks` table: it counts the
        negative file-summary and offset sub-chunk rows too, so the difference is holes
        minus sentinels. `stored_chunk_ids()` answers that question. What this DOES answer,
        and a set of ids cannot, is how many rows were written — which is why
        `_report_duplicate_rows` above uses a filtered count and not an id set (verified:
        4,501 rows collapse to 4,500 distinct ids).
        """
        try:
            # Refresh first — a pinned handle under-reports (measured 0 against a true 1
            # after another handle's upsert).
            self._checkout_latest()
            return int(self._table.count_rows())
        except Exception:
            return 0

    def stored_chunk_ids(self) -> set[int]:
        """Every real chunk_id with a vector in this table. See the base method.

        `_checkout_latest()` first, and it is mandatory rather than defensive: reproduced
        on lancedb 0.33.0, a handle opened before another handle's write enumerated 4,500
        ids against a true 4,501 and missed the new one entirely — the same pinned-snapshot
        failure `_checkout_latest` documents. Under-reporting HERE is the expensive
        direction: every id it misses reads as a hole and is re-embedded for money.

        Projects one column instead of materialising vectors, and filters in the query
        rather than in Python so the vector payload never crosses the boundary. `to_lance()`
        would be the other way to scan and is not available — it raises ImportError asking
        for pylance, which is not a trelix dependency. `.limit(None)` is passed explicitly
        even though the default measured unlimited at fixture size: lancedb is un-pinned,
        so relying on that default is relying on someone else's choice not to change.
        """
        self._checkout_latest()
        arrow = (
            self._table.search()
            .where(f"chunk_id > 0 AND chunk_id < {self._SUB_CHUNK_OFFSET}")
            .select(["chunk_id"])
            .limit(None)
            .to_arrow()
        )
        return {int(cid) for cid in arrow.column("chunk_id").to_pylist()}

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

    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        """Store sub-chunk embedding using chunk_id = sub_chunk_id + _SUB_CHUNK_OFFSET.

        Offset inherited from `BaseVectorStore` — the identical copy that shadowed it here
        is gone, because `stored_chunk_ids()` above now keys its sentinel filter off it.
        """
        self.upsert_batch([(sub_chunk_id + self._SUB_CHUNK_OFFSET, embedding)])

    def search_sub_chunks(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        """Search sub-chunk embeddings only. Returns (sub_chunk_id, score) pairs."""
        results = self.search(query_embedding, k=k * 5)
        return [
            (cid - self._SUB_CHUNK_OFFSET, score)
            for cid, score in results
            if cid >= self._SUB_CHUNK_OFFSET
        ][:k]
