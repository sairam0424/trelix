"""Tests for the file-summary 5th retrieval leg."""

from __future__ import annotations

from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.core.models import IndexedFile, Language
from trelix.store.db import Database
from trelix.store.vector import SQLiteVectorStore


def _seed_summary(db: Database, tmp_path: Path) -> int:
    """Insert one file with a stored summary into `db`; return its file_id.
    `-(file_id)` is the existing convention for a summary row's chunk_id."""
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
    return fid


class TestSearchFileSummaries:
    def test_search_file_summaries_returns_file_id_score_pairs(
        self, tmp_db: Database, tmp_path: Path
    ) -> None:
        fid = _seed_summary(tmp_db, tmp_path)
        store = SQLiteVectorStore(tmp_path / "index.db", dimension=4)
        # Insert a fake summary embedding using the -(file_id) convention
        store.upsert_file_summary_embedding(fid, [0.1, 0.2, 0.3, 0.4])
        results = store.search_file_summaries([0.1, 0.2, 0.3, 0.4], k=5)
        assert len(results) >= 1
        returned_file_ids = [r[0] for r in results]
        assert fid in returned_file_ids

    def test_search_file_summaries_excludes_symbol_chunks(
        self, tmp_db: Database, tmp_path: Path
    ) -> None:
        """Regular chunk rows (positive chunk_id) must NOT appear in summary search."""
        fid = _seed_summary(tmp_db, tmp_path)
        store = SQLiteVectorStore(tmp_path / "index.db", dimension=4)
        # Insert a regular chunk embedding (positive id)
        store.upsert(chunk_id=42, embedding=[0.1, 0.2, 0.3, 0.4])
        store.upsert_file_summary_embedding(fid, [0.9, 0.9, 0.9, 0.9])
        summary_results = store.search_file_summaries([0.1, 0.2, 0.3, 0.4], k=10)
        returned_ids = [r[0] for r in summary_results]
        assert 42 not in returned_ids  # regular chunks excluded

    def test_summary_leg_disabled_by_default(self, index_config: IndexConfig) -> None:
        assert index_config.retrieval.file_summary_leg_enabled is False

    def test_summary_leg_config_fields(self, index_config: IndexConfig) -> None:
        # Fields exist and have sensible defaults
        assert index_config.retrieval.top_k_file_summary == 5
