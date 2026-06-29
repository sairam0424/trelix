"""Tests for LanceDB vector store backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestLanceVectorStore:
    def test_importable(self) -> None:
        from trelix.store.vector_lance import LanceVectorStore

        assert LanceVectorStore is not None

    def test_is_base_vector_store(self) -> None:
        from trelix.store.vector import BaseVectorStore
        from trelix.store.vector_lance import LanceVectorStore

        assert issubclass(LanceVectorStore, BaseVectorStore)

    def test_upsert_batch_calls_lance(self, tmp_path) -> None:
        import sys

        from trelix.store.vector_lance import LanceVectorStore

        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.create_table.return_value = mock_table
        mock_db.open_table.return_value = mock_table

        mock_pa = MagicMock()
        mock_pa.int64.return_value = "int64"
        mock_pa.float32.return_value = "float32"
        mock_pa.list_.return_value = "list_float"
        mock_pa.schema.return_value = MagicMock()
        mock_pa.table.return_value = MagicMock()
        mock_pa.array.return_value = MagicMock()

        with (
            patch("trelix.store.vector_lance.lancedb") as mock_lance,
            patch.dict(sys.modules, {"pyarrow": mock_pa}),
        ):
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            store.upsert_batch([(1, [0.1, 0.2, 0.3, 0.4]), (2, [0.5, 0.6, 0.7, 0.8])])
            assert mock_table.add.called or mock_db.create_table.called

    def test_search_returns_list_of_tuples(self, tmp_path) -> None:
        from trelix.store.vector_lance import LanceVectorStore

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {"chunk_id": 1, "_distance": 0.1},
            {"chunk_id": 2, "_distance": 0.3},
        ]
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            results = store.search([0.1, 0.2, 0.3, 0.4], k=2)
            assert isinstance(results, list)
            assert len(results) == 2
            # Verify each element is a (chunk_id: int, distance: float) tuple
            for item in results:
                assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
                assert len(item) == 2, f"Expected 2-tuple, got length {len(item)}"
                chunk_id, distance = item
                assert isinstance(chunk_id, int), f"chunk_id should be int, got {type(chunk_id)}"
                assert isinstance(distance, float), (
                    f"distance should be float, got {type(distance)}"
                )
            # Verify exact values extracted from mock rows
            assert results[0] == (1, 0.1)
            assert results[1] == (2, 0.3)

    def test_count_returns_int(self, tmp_path) -> None:
        from trelix.store.vector_lance import LanceVectorStore

        mock_table = MagicMock()
        mock_table.count_rows.return_value = 42
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            assert store.count() == 42

    def test_delete_batch_calls_table_delete(self, tmp_path) -> None:
        """delete_batch must call table.delete with the correct SQL predicate."""
        from trelix.store.vector_lance import LanceVectorStore

        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            store.delete_batch([10, 20, 30])
            mock_table.delete.assert_called_once()
            call_arg = mock_table.delete.call_args[0][0]
            assert "10" in call_arg
            assert "20" in call_arg
            assert "30" in call_arg

    def test_delete_batch_empty_list_is_noop(self, tmp_path) -> None:
        """delete_batch with an empty list must not call table.delete."""
        from trelix.store.vector_lance import LanceVectorStore

        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            store.delete_batch([])
            mock_table.delete.assert_not_called()

    def test_import_error_raised_when_lancedb_missing(self, tmp_path) -> None:
        """__init__ must raise ImportError with install hint when lancedb is absent."""
        with patch("trelix.store.vector_lance.lancedb", None):
            from trelix.store.vector_lance import LanceVectorStore

            with pytest.raises(ImportError, match="pip install 'trelix\\[lance\\]'"):
                LanceVectorStore(
                    uri=str(tmp_path / "lance"),
                    table_name="chunks",
                    dimension=4,
                )
