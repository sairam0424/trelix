"""
Regression tests for LanceVectorStore.upsert_batch delete-then-add integrity.

upsert_batch used to swallow its delete (`except Exception: pass`) and add
anyway, so a failed delete silently turned "replace" into "append": one
chunk_id accumulated one extra row per reindex (measured: 1 -> 2 -> 3 -> 4
rows for a single chunk after three failed-delete upserts). Every future
search then returns that chunk_id N times, burning N of the k result slots,
and count() over-reports against the SQLite chunks table. Nothing was logged,
so the corruption was invisible until retrieval quality regressed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

lancedb = pytest.importorskip("lancedb", reason="lance extra not installed")

VEC = [0.1, 0.2, 0.3, 0.4]


def _store(tmp_path):
    from trelix.store.vector_lance import LanceVectorStore

    return LanceVectorStore(uri=str(tmp_path / "lance"), table_name="chunks", dimension=4)


def _break_delete(store) -> None:
    """Make table.delete fail the way a real backend does (predicate/IO/commit error)."""

    def boom(predicate: str) -> None:
        raise RuntimeError(f"simulated LanceDB delete failure: {predicate}")

    store._table.delete = boom


class TestUpsertBatchFailedDelete:
    def test_failed_delete_does_not_append_duplicate_rows(self, tmp_path) -> None:
        """A failed delete must leave the row count unchanged, not grow the table."""
        store = _store(tmp_path)
        store.upsert_batch([(1, VEC)])
        assert store.count() == 1

        _break_delete(store)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                store.upsert_batch([(1, [0.9, 0.9, 0.9, 0.9])])

        # Pre-fix this was 4: three swallowed deletes, three unconditional adds.
        assert store.count() == 1

    def test_failed_delete_raises_to_caller(self, tmp_path) -> None:
        """The indexer must learn the upsert did not happen; silence ships a bad index."""
        store = _store(tmp_path)
        _break_delete(store)
        with pytest.raises(RuntimeError, match="simulated LanceDB delete failure"):
            store.upsert_batch([(7, VEC)])

    def test_failed_delete_is_logged(self, tmp_path, caplog) -> None:
        """delete_batch already logs its failures; this site must too."""
        store = _store(tmp_path)
        _break_delete(store)
        with caplog.at_level(logging.ERROR, logger="trelix.store.lance"):
            with pytest.raises(RuntimeError):
                store.upsert_batch([(7, VEC)])
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert messages, "failed delete inside upsert_batch produced no ERROR log"
        assert any("upsert_batch" in m for m in messages)

    def test_failed_delete_does_not_add_via_mocked_table(self, tmp_path) -> None:
        """Same contract at the call level: no add() once the delete has failed."""
        from trelix.store.vector_lance import LanceVectorStore

        mock_table = MagicMock()
        mock_table.delete.side_effect = RuntimeError("delete rejected")
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(uri=str(tmp_path / "lance"), table_name="chunks", dimension=4)
            with pytest.raises(RuntimeError):
                store.upsert_batch([(1, VEC)])
        mock_table.add.assert_not_called()


class TestUpsertBatchHappyPath:
    def test_repeated_upsert_replaces_rather_than_appends(self, tmp_path) -> None:
        """Guard against over-correcting: a working delete must still be followed by add."""
        store = _store(tmp_path)
        for _ in range(3):
            store.upsert_batch([(1, VEC)])
        assert store.count() == 1

        rows = store._table.search(VEC).limit(10).to_list()
        assert [r["chunk_id"] for r in rows] == [1]

    def test_upsert_stores_the_latest_vector(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.upsert_batch([(1, VEC)])
        store.upsert_batch([(1, [1.0, 0.0, 0.0, 0.0])])
        rows = store._table.search([1.0, 0.0, 0.0, 0.0]).limit(10).to_list()
        assert len(rows) == 1
        assert pytest.approx(list(rows[0]["vector"]), abs=1e-6) == [1.0, 0.0, 0.0, 0.0]

    def test_empty_pairs_is_noop(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.upsert_batch([])
        assert store.count() == 0
