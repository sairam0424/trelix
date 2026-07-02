"""Tests for embedding dimension mismatch guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.store.db import Database
from trelix.store.dimension_guard import DimensionGuard, DimensionMismatchError


def _make_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "index.db")


class TestIndexMetadataDB:
    def test_get_dimension_returns_none_when_not_set(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        assert db.get_embedding_dimension() is None

    def test_set_and_get_dimension(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.set_embedding_dimension(3072)
        assert db.get_embedding_dimension() == 3072

    def test_set_dimension_overwrites_previous(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.set_embedding_dimension(384)
        db.set_embedding_dimension(1024)
        assert db.get_embedding_dimension() == 1024


class TestDimensionMismatchError:
    def test_error_message_contains_dimensions(self) -> None:
        err = DimensionMismatchError(stored=3072, current=384, provider="local")
        assert "3072" in str(err)
        assert "384" in str(err)

    def test_error_message_contains_migration_hint(self) -> None:
        err = DimensionMismatchError(stored=3072, current=384, provider="local")
        assert "migrate-vectors" in str(err)

    def test_is_exception(self) -> None:
        err = DimensionMismatchError(stored=1, current=2, provider="test")
        assert isinstance(err, Exception)


class TestDimensionGuard:
    def test_check_passes_when_dimensions_match(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.set_embedding_dimension(384)
        # Should not raise
        DimensionGuard.check(db, current_dimension=384, provider="local")

    def test_check_passes_when_no_stored_dimension(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        # No dimension stored yet — first run, no error
        DimensionGuard.check(db, current_dimension=384, provider="local")

    def test_check_raises_on_mismatch(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.set_embedding_dimension(3072)
        with pytest.raises(DimensionMismatchError) as exc_info:
            DimensionGuard.check(db, current_dimension=384, provider="local")
        assert "3072" in str(exc_info.value)
        assert "384" in str(exc_info.value)

    def test_record_stores_dimension(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        DimensionGuard.record(db, dimension=1024, provider="voyage")
        assert db.get_embedding_dimension() == 1024

    def test_reset_clears_dimension(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        db.set_embedding_dimension(3072)
        DimensionGuard.reset(db)
        assert db.get_embedding_dimension() is None


class TestIndexerDimensionGuard:
    """Verify DimensionGuard.check() fires at Indexer.__init__, not only at Retriever startup."""

    def test_indexer_raises_on_dimension_mismatch(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.indexing.indexer import Indexer
        from trelix.store.db import Database
        from trelix.store.dimension_guard import DimensionMismatchError

        cfg = IndexConfig(repo_path=str(tmp_path))
        db_path = cfg.db_path_absolute
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Seed the DB with a stored dimension of 3072
        db = Database(db_path)
        db.set_embedding_dimension(3072)
        db.close()

        # Mock the embedder to report a different dimension (384)
        mock_embedder = MagicMock()
        mock_embedder.dimension = 384

        mock_vector_store = MagicMock()

        with (
            patch("trelix.indexing.indexer.make_embedder", return_value=mock_embedder),
            patch("trelix.indexing.indexer.make_vector_store", return_value=mock_vector_store),
        ):
            with pytest.raises(DimensionMismatchError) as exc_info:
                Indexer(cfg)

        assert "3072" in str(exc_info.value)
        assert "384" in str(exc_info.value)

    def test_indexer_passes_when_dimensions_match(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.indexing.indexer import Indexer
        from trelix.store.db import Database

        cfg = IndexConfig(repo_path=str(tmp_path))
        db_path = cfg.db_path_absolute
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Seed the DB with a matching dimension
        db = Database(db_path)
        db.set_embedding_dimension(384)
        db.close()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384

        mock_vector_store = MagicMock()
        mock_chunker = MagicMock()
        mock_walker = MagicMock()

        with (
            patch("trelix.indexing.indexer.make_embedder", return_value=mock_embedder),
            patch("trelix.indexing.indexer.make_vector_store", return_value=mock_vector_store),
            patch("trelix.indexing.indexer.Chunker", return_value=mock_chunker),
            patch("trelix.indexing.indexer.FileWalker", return_value=mock_walker),
        ):
            # Should not raise
            indexer = Indexer(cfg)
            assert indexer is not None

    def test_indexer_passes_on_first_run_no_stored_dimension(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.indexing.indexer import Indexer

        cfg = IndexConfig(repo_path=str(tmp_path))
        db_path = cfg.db_path_absolute
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # No dimension stored — first run, guard is a no-op
        mock_embedder = MagicMock()
        mock_embedder.dimension = 384

        mock_vector_store = MagicMock()
        mock_chunker = MagicMock()
        mock_walker = MagicMock()

        with (
            patch("trelix.indexing.indexer.make_embedder", return_value=mock_embedder),
            patch("trelix.indexing.indexer.make_vector_store", return_value=mock_vector_store),
            patch("trelix.indexing.indexer.Chunker", return_value=mock_chunker),
            patch("trelix.indexing.indexer.FileWalker", return_value=mock_walker),
        ):
            # Should not raise
            indexer = Indexer(cfg)
            assert indexer is not None


class TestMigrateVectorsReset:
    def test_migrate_vectors_reset_clears_dimension(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from trelix.cli.main import app
        from trelix.core.config import IndexConfig
        from trelix.store.db import Database

        # Resolve the DB path the same way the CLI does
        cfg = IndexConfig(repo_path=str(tmp_path))
        db_path = cfg.db_path_absolute
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a DB with a stored dimension at the expected location
        db = Database(db_path)
        db.set_embedding_dimension(3072)
        assert db.get_embedding_dimension() == 3072
        db.close()

        runner = CliRunner()
        result = runner.invoke(app, ["migrate-vectors", str(tmp_path), "--reset"])
        assert result.exit_code == 0, f"exit_code={result.exit_code}, output={result.output!r}"
        assert "cleared" in result.output.lower()

        # After reset, dimension is gone
        db2 = Database(db_path)
        assert db2.get_embedding_dimension() is None
