"""
Sparse retrieval leg using learned sparse embeddings (SPLADE-style).

Bridges SparseStore (inverted index) → SearchResult objects by hydrating
chunk/symbol/file from the Database.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.store.db import Database
    from trelix.store.sparse_store import SparseStore

from trelix.core.models import Chunk, SearchResult

logger = logging.getLogger("trelix.retrieval.sparse_search")


def sparse_search(
    store: SparseStore,
    db: Database,
    query_sparse: dict[int, float],
    k: int = 20,
) -> list[SearchResult]:
    """
    Search the sparse index and return hydrated SearchResult objects.

    Args:
        store: SparseStore instance
        db: Database for hydrating symbols and files
        query_sparse: {token_id: weight} sparse query vector
        k: max results

    Returns:
        list[SearchResult] with source="sparse", empty list on any failure
    """
    if not query_sparse:
        return []

    try:
        pairs = store.search(query_sparse, k=k)
    except Exception as exc:
        logger.debug("SparseStore.search failed: %s", exc)
        return []

    results: list[SearchResult] = []
    for chunk_id, score in pairs:
        try:
            row = db._conn.execute(
                "SELECT id, symbol_id, chunk_text, token_count FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue

            chunk = Chunk(
                id=int(row[0]),
                symbol_id=int(row[1]),
                chunk_text=row[2],
                token_count=int(row[3]),
            )
            sym_file = db.get_symbol_with_file(chunk.symbol_id)
            if sym_file is None:
                continue
            symbol, file_obj = sym_file
            results.append(
                SearchResult(
                    chunk=chunk,
                    symbol=symbol,
                    file=file_obj,
                    score=score,
                    rank=len(results) + 1,
                    source="sparse",
                )
            )
        except Exception as exc:
            logger.debug("sparse_search hydration failed for chunk %d: %s", chunk_id, exc)

    return results
