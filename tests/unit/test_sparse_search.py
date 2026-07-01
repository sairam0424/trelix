"""Tests for sparse_search retrieval function."""
from __future__ import annotations

from pathlib import Path

from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.retrieval.sparse_search import sparse_search
from trelix.store.db import Database
from trelix.store.sparse_store import SparseStore


def _build_fixture(tmp_path: Path) -> tuple[Database, SparseStore, int]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10
        )
    )
    sid = db.insert_symbol(Symbol(
        file_id=fid, name="login", qualified_name="login",
        kind=SymbolKind.FUNCTION, line_start=1, line_end=5,
        signature="def login()", body="def login(): pass"
    ))
    # Store a chunk
    chunk_id = db.insert_chunk_for_symbol(sid, "def login(): pass", 5)
    store = SparseStore(tmp_path / "index.db")
    store.upsert(chunk_id=chunk_id, sparse_vec={100: 2.5, 200: 1.8})
    return db, store, chunk_id


class TestSparseSearch:
    def test_returns_search_results(self, tmp_path: Path) -> None:
        db, store, _ = _build_fixture(tmp_path)
        results = sparse_search(store, db, query_sparse={100: 1.0}, k=5)
        assert len(results) >= 1
        assert results[0].source == "sparse"

    def test_returns_empty_on_no_overlap(self, tmp_path: Path) -> None:
        db, store, _ = _build_fixture(tmp_path)
        results = sparse_search(store, db, query_sparse={999: 1.0}, k=5)
        assert results == []

    def test_scores_are_positive(self, tmp_path: Path) -> None:
        db, store, _ = _build_fixture(tmp_path)
        results = sparse_search(store, db, query_sparse={100: 1.0, 200: 0.5}, k=5)
        assert all(r.score > 0 for r in results)
