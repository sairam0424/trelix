"""Tests for the file-summary 5th retrieval leg."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trelix.core.config import IndexConfig, RetrievalConfig
from trelix.core.models import IndexedFile, Language, SearchResult, Symbol, SymbolKind
from trelix.store.db import Database
from trelix.store.vector import SQLiteVectorStore


def _build_db_with_summary(tmp_path: Path) -> tuple[Database, int, int]:
    """Return (db, file_id, summary_chunk_id) with one file and a stored summary."""
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path=str(tmp_path / "auth.py"),
            rel_path="auth.py",
            language=Language.PYTHON,
            hash="abc",
            size_bytes=100,
        )
    )
    db.upsert_file_summary(fid, "Handles user authentication and JWT token lifecycle.")
    return db, fid, -(fid)  # convention: chunk_id = -file_id for summary rows


class TestSearchFileSummaries:
    def test_search_file_summaries_returns_file_id_score_pairs(self, tmp_path: Path) -> None:
        db, fid, neg_fid = _build_db_with_summary(tmp_path)
        store = SQLiteVectorStore(tmp_path / "index.db", dimension=4)
        # Insert a fake summary embedding using the -(file_id) convention
        store.upsert_file_summary_embedding(fid, [0.1, 0.2, 0.3, 0.4])
        results = store.search_file_summaries([0.1, 0.2, 0.3, 0.4], k=5)
        assert len(results) >= 1
        returned_file_ids = [r[0] for r in results]
        assert fid in returned_file_ids

    def test_search_file_summaries_excludes_symbol_chunks(self, tmp_path: Path) -> None:
        """Regular chunk rows (positive chunk_id) must NOT appear in summary search."""
        db, fid, _ = _build_db_with_summary(tmp_path)
        store = SQLiteVectorStore(tmp_path / "index.db", dimension=4)
        # Insert a regular chunk embedding (positive id)
        store.upsert(chunk_id=42, embedding=[0.1, 0.2, 0.3, 0.4])
        store.upsert_file_summary_embedding(fid, [0.9, 0.9, 0.9, 0.9])
        summary_results = store.search_file_summaries([0.1, 0.2, 0.3, 0.4], k=10)
        returned_ids = [r[0] for r in summary_results]
        assert 42 not in returned_ids  # regular chunks excluded

    def test_summary_leg_disabled_by_default(self, tmp_path: Path) -> None:
        config = IndexConfig(repo_path=str(tmp_path))
        assert config.retrieval.file_summary_leg_enabled is False

    def test_summary_leg_config_fields(self, tmp_path: Path) -> None:
        config = IndexConfig(repo_path=str(tmp_path))
        # Fields exist and have sensible defaults
        assert config.retrieval.top_k_file_summary == 5
