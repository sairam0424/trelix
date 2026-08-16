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
    """`--reset` must leave an index a re-index can actually repair.

    The previous version of this test recorded a dimension, created NO vec0 table, and
    asserted only that the command exited 0 and printed the word "cleared" with the
    dimension record gone afterwards. Every one of those held while the command was a
    complete no-op: `db.clear_all_embeddings()` ran `DELETE FROM chunk_embeddings` on a
    connection with no sqlite-vec extension loaded, raising
    `OperationalError: no such module: vec0` into a bare `except: pass`. So the test
    passed against a command that deleted nothing, and would have kept passing through a
    naive fix.

    These assertions are on the resulting STATE instead: the vector table exists at the
    requested dimension, the hashes that gate re-parsing and re-embedding are blanked,
    and the recorded dimension is gone.
    """

    @staticmethod
    def _seed_index(tmp_path: Path):  # type: ignore[no-untyped-def]
        """An index with one file, one symbol, and a 4-dim vector table."""
        from trelix.core.config import IndexConfig
        from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
        from trelix.store.db import Database
        from trelix.store.vector import SQLiteVectorStore

        cfg = IndexConfig(repo_path=str(tmp_path))
        db_path = cfg.db_path_absolute
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = Database(db_path)
        file_id = db.upsert_file(IndexedFile(
            path=str(tmp_path / "a.py"), rel_path="a.py", language=Language.PYTHON,
            hash="originalhash", size_bytes=10,
        ))
        db.insert_symbol(Symbol(
            file_id=file_id, name="f", qualified_name="f", kind=SymbolKind.FUNCTION,
            line_start=1, line_end=2, signature="def f():", body="def f(): pass",
        ))
        db._conn.commit()
        db.set_embedding_dimension(4)
        # A real vec0 table at the OLD dimension, which is the thing a row delete
        # cannot change and the absence of which made the old test vacuous.
        SQLiteVectorStore(db_path, dimension=4).upsert(chunk_id=1, embedding=[1.0, 0, 0, 0])
        db.close()
        return db_path

    def _run_reset(self, tmp_path: Path, provider: str = "local"):  # type: ignore[no-untyped-def]
        from typer.testing import CliRunner

        from trelix.cli.main import app

        return CliRunner().invoke(
            app, ["migrate-vectors", str(tmp_path), "--reset", "--provider", provider]
        )

    def test_reset_succeeds_and_says_what_it_did(self, tmp_path: Path) -> None:
        self._seed_index(tmp_path)
        result = self._run_reset(tmp_path)
        assert result.exit_code == 0, f"exit={result.exit_code} output={result.output!r}"
        assert "rebuilt" in result.output.lower()

    def test_the_vector_table_is_rebuilt_at_the_new_dimension(self, tmp_path: Path) -> None:
        """The assertion the old test could not make, because it built no table."""
        import sqlite3

        db_path = self._seed_index(tmp_path)
        self._run_reset(tmp_path)

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'chunk_embeddings'"
            ).fetchone()[0]
        finally:
            conn.close()

        # The local provider is 384-dim; the seeded table was 4-dim.
        assert "FLOAT[384]" in ddl, f"table was not rebuilt: {ddl}"
        assert "FLOAT[4]" not in ddl

    def test_hashes_are_invalidated_so_a_reindex_re_embeds(self, tmp_path: Path) -> None:
        """Both levels: file hashes gate re-parsing, symbol hashes gate re-embedding."""
        from trelix.store.db import Database

        db_path = self._seed_index(tmp_path)
        self._run_reset(tmp_path)

        db = Database(db_path)
        try:
            assert db.get_file_hash("a.py") == "", "file hash survived the reset"
            symbol_hash = db._conn.execute("SELECT content_hash FROM symbols").fetchone()[0]
            assert symbol_hash == "", (
                "symbol content hash survived, so a re-index would re-parse and still "
                "embed nothing"
            )
        finally:
            db.close()

    def test_the_recorded_dimension_is_cleared(self, tmp_path: Path) -> None:
        from trelix.store.db import Database

        db_path = self._seed_index(tmp_path)
        self._run_reset(tmp_path)

        db = Database(db_path)
        try:
            assert db.get_embedding_dimension() is None
        finally:
            db.close()
