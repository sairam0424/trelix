"""
SparseStore — SQLite inverted index for SPLADE sparse embeddings.

Stores (chunk_id, token_id, weight) rows. Search computes dot-product
similarity by joining query tokens against the index.

Performance note: at 10k chunks × 128 tokens = 1.28M rows. The
idx_sparse_token index makes token lookups O(log n). Full dot-product
over 1.28M rows takes ~50ms on SQLite; acceptable for a 7th RRF leg.
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
        """Insert or replace the sparse vector for a chunk (clean overwrite)."""
        with self._lock:
            self._conn.execute("DELETE FROM sparse_embeddings WHERE chunk_id = ?", (chunk_id,))
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
        Compute dot-product similarity between the query sparse vector and all indexed chunks.

        Weights are passed entirely as bound parameters via a VALUES CTE — no raw
        float strings are interpolated into the SQL text, satisfying the project's
        parameterized-query rule.

        Returns list of (chunk_id, score) sorted by score descending.
        Only chunks with at least one overlapping token are returned.
        """
        if not query_sparse:
            return []

        token_ids = list(query_sparse.keys())
        weights = [query_sparse[t] for t in token_ids]

        # Build a VALUES CTE that maps token_id → query_weight as bound params.
        # This keeps all float values out of the SQL string itself.
        row_placeholders = ",".join("(?,?)" for _ in token_ids)
        # Flatten pairs for binding: (tok0, w0, tok1, w1, ...)
        params: list[object] = []
        for tid, w in zip(token_ids, weights):
            params.extend([tid, w])

        # token_id IN (...) filter is still needed for the index seek on idx_sparse_token.
        in_placeholders = ",".join("?" * len(token_ids))
        params.extend(token_ids)
        params.append(k)

        sql = f"""
            WITH query_weights(token_id, qw) AS (
                VALUES {row_placeholders}
            )
            SELECT se.chunk_id, SUM(se.weight * qw.qw) AS score
            FROM sparse_embeddings se
            JOIN query_weights qw ON se.token_id = qw.token_id
            WHERE se.token_id IN ({in_placeholders})
            GROUP BY se.chunk_id
            ORDER BY score DESC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows if row[1] > 0]
