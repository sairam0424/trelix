"""Tests for SparseStore inverted index."""
from __future__ import annotations

from pathlib import Path

import pytest

from trelix.store.sparse_store import SparseStore


class TestSparseStore:
    def test_upsert_and_search(self, tmp_path: Path) -> None:
        store = SparseStore(tmp_path / "index.db")
        store.upsert(chunk_id=1, sparse_vec={100: 2.5, 200: 1.8})
        store.upsert(chunk_id=2, sparse_vec={150: 3.0, 200: 0.5})

        # Query matching token 200 — both chunks have it
        results = store.search({200: 1.0}, k=10)
        chunk_ids = [r[0] for r in results]
        assert 1 in chunk_ids
        assert 2 in chunk_ids

    def test_search_ranks_by_dot_product(self, tmp_path: Path) -> None:
        store = SparseStore(tmp_path / "index.db")
        store.upsert(chunk_id=1, sparse_vec={100: 5.0})  # high weight on token 100
        store.upsert(chunk_id=2, sparse_vec={100: 1.0})  # low weight on token 100

        results = store.search({100: 1.0}, k=2)
        # Chunk 1 should rank higher (5.0 > 1.0)
        assert results[0][0] == 1
        assert results[0][1] > results[1][1]

    def test_search_with_no_overlap_returns_empty(self, tmp_path: Path) -> None:
        store = SparseStore(tmp_path / "index.db")
        store.upsert(chunk_id=1, sparse_vec={100: 2.5})
        results = store.search({999: 1.0}, k=10)  # no overlap
        assert results == []

    def test_upsert_batch(self, tmp_path: Path) -> None:
        store = SparseStore(tmp_path / "index.db")
        pairs = [(i, {i * 10: float(i)}) for i in range(1, 6)]
        store.upsert_batch(pairs)
        # Search for token 20 (chunk_id=2)
        results = store.search({20: 1.0}, k=5)
        assert any(r[0] == 2 for r in results)

    def test_overwrite_existing_chunk(self, tmp_path: Path) -> None:
        store = SparseStore(tmp_path / "index.db")
        store.upsert(chunk_id=1, sparse_vec={100: 2.5, 200: 1.0})
        store.upsert(chunk_id=1, sparse_vec={300: 5.0})  # overwrite
        results = store.search({100: 1.0}, k=5)
        # After overwrite, token 100 should no longer be present for chunk 1
        assert not any(r[0] == 1 for r in results)
