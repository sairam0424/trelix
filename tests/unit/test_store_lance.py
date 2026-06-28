"""Tests for LanceDB vector store backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


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
        mock_result = MagicMock()
        mock_result.to_list.return_value = [
            {"chunk_id": 1, "_distance": 0.1},
            {"chunk_id": 2, "_distance": 0.3},
        ]
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
