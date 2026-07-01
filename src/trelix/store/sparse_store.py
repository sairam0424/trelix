"""
SparseStore — SQLite inverted index for SPLADE sparse embeddings.

Stores (chunk_id, token_id, weight) rows. Search computes dot-product
similarity by joining query tokens against the index.

Performance note: at 10k chunks × 128 tokens = 1.28M rows. The
idx_sparse_token index makes token lookups O(log n). Full dot-product
over 1.28M rows takes ~50ms on SQLite; acceptable for a 6th RRF leg.
For >100k chunks, consider moving to Qdrant's native sparse support.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class SparseStore:
    """
    SQLite-backed inverted index for sparse embeddings.

    Each chunk_id → {token_id: weight} mapping is stored as individual rows,
    enabling efficient dot-product search via SQL aggregation.
    """

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sparse_embeddings (
                    chunk_id INTEGER NOT NULL,
                    token_id INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (chunk_id, token_id)
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sparse_token ON sparse_embeddings(token_id)"
            )
            self._conn.commit()

    def upsert(self, chunk_id: int, sparse_vec: dict[int, float]) -> None:
        """Insert or replace the sparse vector for a chunk."""
        with self._lock:
            # Delete existing rows for this chunk first (clean overwrite)
            self._conn.execute(
                "DELETE FROM sparse_embeddings WHERE chunk_id = ?", (chunk_id,)
            )
            self._conn.executemany(
                "INSERT INTO sparse_embeddings (chunk_id, token_id, weight) VALUES (?, ?, ?)",
                [(chunk_id, tok_id, weight) for tok_id, weight in sparse_vec.items() if weight > 0],
            )
            self._conn.commit()

    def upsert_batch(self, pairs: list[tuple[int, dict[int, float]]]) -> None:
        """Bulk upsert a list of (chunk_id, sparse_vec) pairs."""
        with self._lock:
            chunk_ids = [p[0] for p in pairs]
            placeholders = ",".join("?" * len(chunk_ids))
            self._conn.execute(
                f"DELETE FROM sparse_embeddings WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            rows = [
                (chunk_id, tok_id, weight)
                for chunk_id, sparse_vec in pairs
                for tok_id, weight in sparse_vec.items()
                if weight > 0
            ]
            self._conn.executemany(
                "INSERT INTO sparse_embeddings (chunk_id, token_id, weight) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def search(self, query_sparse: dict[int, float], k: int = 20) -> list[tuple[int, float]]:
        """
        Compute dot-product similarity between query sparse vector and all indexed chunks.

        Returns list of (chunk_id, score) sorted by score descending.
        Only chunks with at least one overlapping token are returned.
        """
        if not query_sparse:
            return []

        token_ids = list(query_sparse.keys())
        weights = [query_sparse[t] for t in token_ids]

        # Build parameterized query: SUM(doc_weight * query_weight) per chunk
        placeholders = ",".join("?" * len(token_ids))
        # Create a CTE mapping token_id -> query_weight for the join
        case_exprs = " ".join(
            f"WHEN {tok_id} THEN {weight}" for tok_id, weight in zip(token_ids, weights)
        )

        sql = f"""
            SELECT chunk_id, SUM(weight * CASE token_id {case_exprs} ELSE 0 END) AS score
            FROM sparse_embeddings
            WHERE token_id IN ({placeholders})
            GROUP BY chunk_id
            ORDER BY score DESC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, token_ids + [k]).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows if row[1] > 0]
