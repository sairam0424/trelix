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
        """Return the total number of stored embeddings.

        SENTINEL-INCLUSIVE: file-summary rows (at `-file_id`) and sub-chunk rows (at
        `sub_chunk_id + _SUB_CHUNK_OFFSET`) live in the same store and are counted here.
        So this must NOT be compared against `Database.count_chunks()` to find chunks
        missing a vector — the difference is holes minus sentinels, which understates the
        holes and inverts sign once sentinels outnumber them. Measured on a fixture with
        5,000 chunks, 500 deliberate holes and 95 sentinels: 405 against a true 500. Use
        `stored_chunk_ids()` and take the set difference.

        Still the right primitive for the question it does answer: rows written. That is
        what `LanceVectorStore._report_duplicate_rows` needs, because a set of ids cannot
        see a duplicate row (verified: 4,501 Lance rows collapse to 4,500 distinct ids).
        """

    # Sub-chunk vectors are stored at `sub_chunk_id + _SUB_CHUNK_OFFSET` by every
    # backend. Deleting them is defined here, concretely, so callers never have to know
    # the offset — `Database` in particular has no business doing that arithmetic, and
    # a foreign key cannot reach a vector store at all.
    _SUB_CHUNK_OFFSET = 10_000_000

    @classmethod
    def _is_chunk_id(cls, chunk_id: int) -> bool:
        """True for a real chunk vector, false for a summary or sub-chunk sentinel.

        Defined here beside `_SUB_CHUNK_OFFSET` rather than per backend: all three now
        key their sentinel filter off it, and three copies of "what counts as a real
        chunk" are three places for it to drift.
        """
        return 0 < chunk_id < cls._SUB_CHUNK_OFFSET

    def delete_sub_chunk_embeddings(self, sub_chunk_ids: list[int]) -> None:
        """Delete the vectors belonging to the given sub_chunk row ids."""
        if not sub_chunk_ids:
            return
        self.delete_batch([sid + self._SUB_CHUNK_OFFSET for sid in sub_chunk_ids])

    def stored_chunk_ids(self) -> set[int]:
        """Every real chunk_id this store holds a vector for (sentinels excluded).

        Exists so a caller can diff it against the `chunks` table and find the chunks a
        crashed indexing run left with no vector. Those holes are otherwise permanent:
        `Indexer._insert_one` commits a file's content hash and its chunk rows in Phase 2,
        BEFORE Phase 3 embeds them, so any interruption in between — SIGKILL, laptop
        sleep, CI timeout, OOM, quota exhaustion — leaves rows that every later
        incremental run declares up to date. Reproduced on a real 68,880-chunk index
        holding 61,652 vectors: 7,228 chunks (10.5%) unretrievable, and a re-index
        reported "Files to embed 0".

        Returns ids, not a count, deliberately — see `count()` for why a count cannot
        answer this. One direction only: chunk rows lacking a vector. The reverse
        (vectors whose chunk row is gone) is the caller's set difference the other way
        and is reported, never acted on; deleting is `--prune`'s job.

        Default implementation raises, following `recreate()` above, so a backend that
        cannot enumerate says so rather than returning an empty set — which a caller
        would read as "every chunk is a hole" and act on by re-embedding the entire
        repository. Deliberately not an `@abstractmethod`: that turns a missing
        capability into a construction-time TypeError, i.e. trelix does not start.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot enumerate its stored chunk_ids, so trelix "
            "cannot tell which chunks are missing a vector on this backend"
        )

    def recreate(self) -> None:
        """Discard every stored vector and rebuild the store at this store's dimension.

        Needed to recover from an embedding-provider change. Deleting rows is not
        sufficient for backends whose vector width is fixed at creation time, which is
        why this is a distinct operation rather than a `clear()`.

        Default implementation raises, so a backend that cannot support it says so
        instead of silently appearing to succeed — the failure mode this exists to
        remove.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot recreate its vector store; "
            "delete the index and re-index instead"
        )

    @abstractmethod
    def upsert_file_summary_embedding(self, file_id: int, embedding: list[float]) -> None:
        """Insert or replace a file-level summary embedding."""

    @abstractmethod
    def search_file_summaries(
        self, query_embedding: list[float], k: int
    ) -> list[tuple[int, float]]:
        """Search file-summary embeddings only. Returns (file_id, score) pairs.

        Convention: summary embeddings are stored with chunk_id = -(file_id).
        This method filters to negative chunk_ids and maps back to file_id.
        """

    @abstractmethod
    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        """Store embedding for a sub-symbol chunk.

        Uses chunk_id = sub_chunk_id + _SUB_CHUNK_OFFSET sentinel to avoid collision
        with regular chunk IDs (positive) and file-summary IDs (negative).
        """

    @abstractmethod
    def search_sub_chunks(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        """Search sub-chunk embeddings only. Returns (sub_chunk_id, score) pairs."""


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

    def recreate(self) -> None:
        """Drop the vec0 table and rebuild it at `self._dim`.

        Deleting rows cannot change the width: both creation paths use
        `CREATE VIRTUAL TABLE IF NOT EXISTS`, so re-issuing the DDL at a new dimension
        is a no-op while the old table exists. Verified — after DELETEing every row and
        re-issuing the CREATE at FLOAT[8], the schema still declared FLOAT[4] and an
        8-dim insert failed with "Expected 4 dimensions but received 8".

        Dropping is safe on a vec0 table: it removes all of its shadow tables
        (`_auxiliary`, `_chunks`, `_info`, `_rowids`, `_vector_chunks00`) with none left
        behind, and it is transactional, so a failure part-way leaves the old table in
        place rather than no table at all.
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute("DROP TABLE IF EXISTS chunk_embeddings")
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        # Rebuilt through the normal path, so HNSW is re-applied exactly as it would be
        # on a fresh index rather than being reimplemented here.
        self._hnsw_active = self._setup_table()

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
        rows = self._knn(packed, k)
        real = [(cid, dist) for cid, dist in rows if self._is_chunk_id(cid)]

        # Sentinel rows (file summaries at -file_id, sub-chunks at +_SUB_CHUNK_OFFSET)
        # live in this same table and compete for the ANN top-k. Anything they take is a
        # real result the caller never sees — Retriever._vector_search fetches exactly k
        # rows and has nothing to backfill from — so re-ask for k plus the exact number
        # of sentinels stored, which is provably enough even if every one of them
        # outranks the k-th real chunk.
        #
        # Filtering in SQL is not available: sqlite-vec rejects a `chunk_id > 0`
        # predicate alongside `embedding MATCH ? ... LIMIT ?`, and its `k = ?` form
        # applies the predicate after the ANN cut, silently returning fewer than k.
        if len(real) < k:
            sentinels = self._count_sentinels()
            if sentinels:
                rows = self._knn(packed, k + sentinels)
                real = [(cid, dist) for cid, dist in rows if self._is_chunk_id(cid)]

        return real[:k]

    def stored_chunk_ids(self) -> set[int]:
        """Every real chunk_id with a vector in this store. See the base method.

        A plain projection is safe here even though `search()` above records that vec0
        rejects a `chunk_id > 0` predicate: that restriction applies only ALONGSIDE
        `embedding MATCH`, and `_count_sentinels()` already runs this predicate shape in
        production. Measured 2.6 ms over 4,595 rows on a 4-dim fixture.

        A missing `chunk_embeddings` table raises `sqlite3.OperationalError` rather than
        returning an empty set, for the reason given on the base method.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM chunk_embeddings WHERE chunk_id > 0 AND chunk_id < ?",
                (self._SUB_CHUNK_OFFSET,),
            ).fetchall()
        return {int(r[0]) for r in rows}

    def _knn(self, packed: bytes, k: int) -> list[tuple[int, float]]:
        """Raw ANN query — returns sentinel rows as well as real chunks."""
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
        return [(int(r[0]), float(r[1])) for r in rows]

    def _count_sentinels(self) -> int:
        """How many non-chunk rows share the table. 0 on any of the usual indexes."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id <= 0 OR chunk_id >= ?",
                (self._SUB_CHUNK_OFFSET,),
            ).fetchone()
        return int(row[0]) if row else 0

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
        """Rows in the store, sentinels included — see `BaseVectorStore.count`.

        Takes `_lock` like every other read on this connection (`_knn`,
        `_count_sentinels`, `search_file_summaries`, `search_sub_chunks`,
        `stored_chunk_ids`). It was the one that did not, against a connection opened
        `check_same_thread=False` whose own comment says the lock "serialises all
        execute() calls so the connection's internal state stays consistent".
        """
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()
        return row[0] if row else 0

    def info(self) -> dict[str, Any]:
        """
        Return a summary dict for ``trelix stats``.

        `count` here is sentinel-inclusive (see `count()`), so it is a "rows written"
        figure and NOT a chunk-coverage figure. `trelix stats` reports coverage from
        `stored_chunk_ids()` against the `chunks` table instead, in both directions.

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

    def search_file_summaries(
        self, query_embedding: list[float], k: int
    ) -> list[tuple[int, float]]:
        """Search file-summary embeddings only. Returns (file_id, score) pairs.

        Summary rows are stored with chunk_id = -(file_id). This method fetches
        all summary-row embeddings, computes L2 distance in Python, and returns
        the top-k pairs as (file_id, score) where score is (1 / (1 + distance)).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, embedding FROM chunk_embeddings WHERE chunk_id < 0"
            ).fetchall()
        if not rows:
            return []
        q = query_embedding
        scored: list[tuple[int, float]] = []
        for chunk_id, emb_blob in rows:
            if isinstance(emb_blob, bytes):
                n = len(emb_blob) // 4
                emb = list(struct.unpack(f"{n}f", emb_blob))
            else:
                emb = list(emb_blob)
            dist = sum((a - b) ** 2 for a, b in zip(q, emb)) ** 0.5
            score = 1.0 / (1.0 + dist)
            scored.append((-int(chunk_id), score))  # chunk_id = -file_id, so file_id = -chunk_id
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        """Store sub-chunk embedding using chunk_id = sub_chunk_id + _SUB_CHUNK_OFFSET.

        The offset is inherited from `BaseVectorStore`; the identical copy that used to
        shadow it here is gone, along with the two in the Lance and Qdrant backends. All
        three now filter sentinels off the one definition, which is exactly the kind of
        shared constant a local copy drifts from.
        """
        self.upsert(chunk_id=sub_chunk_id + self._SUB_CHUNK_OFFSET, embedding=embedding)

    def search_sub_chunks(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        """Search sub-chunk embeddings only. Returns (sub_chunk_id, score) pairs."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, embedding FROM chunk_embeddings WHERE chunk_id >= ?",
                (self._SUB_CHUNK_OFFSET,),
            ).fetchall()
        if not rows:
            return []
        q = query_embedding
        scored: list[tuple[int, float]] = []
        for row in rows:
            chunk_id, emb_blob = row
            if isinstance(emb_blob, bytes):
                n = len(emb_blob) // 4
                emb = list(struct.unpack(f"{n}f", emb_blob))
            else:
                emb = list(emb_blob)
            dist = sum((a - b) ** 2 for a, b in zip(q, emb)) ** 0.5
            score = 1.0 / (1.0 + dist)
            scored.append((int(chunk_id) - self._SUB_CHUNK_OFFSET, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

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
      "lance"              → LanceVectorStore backed by a LanceDB directory at
                             config.store.lance_uri (default: <repo>/.trelix/lance).
                             Requires ``pip install 'trelix[lance]'``.

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
