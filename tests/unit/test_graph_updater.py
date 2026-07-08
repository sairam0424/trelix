"""Tests for incremental graph updates after file changes."""

from __future__ import annotations

from pathlib import Path

from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.updater import GraphUpdater
from trelix.store.db import Database


def _make_db(tmp_path: Path) -> tuple[Database, int, int]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="h1", size_bytes=50
        )
    )
    sid = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="my_func",
            qualified_name="my_func",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=10,
            signature="def my_func()",
            body="def my_func(): pass",
        )
    )
    return db, fid, sid


class TestGraphUpdater:
    def test_update_file_does_not_raise_on_valid_file(self, tmp_path: Path) -> None:
        db, fid, sid = _make_db(tmp_path)
        updater = GraphUpdater(db)
        # Should complete without error
        updater.update_file("a.py")

    def test_update_file_on_unknown_file_is_noop(self, tmp_path: Path) -> None:
        db, _, _ = _make_db(tmp_path)
        updater = GraphUpdater(db)
        # Should not raise even if file not found
        updater.update_file("nonexistent.py")

    def test_graph_metadata_refreshed_after_update(self, tmp_path: Path) -> None:
        db, fid, sid = _make_db(tmp_path)
        updater = GraphUpdater(db)
        updater.update_file("a.py")
        # graph_metadata should have an entry for the symbol
        row = db._conn.execute(
            "SELECT symbol_id FROM graph_metadata WHERE symbol_id = ?", (sid,)
        ).fetchone()
        assert row is not None


class TestGraphUpdaterIncremental:
    def _make_updater(self, tmp_path):
        from unittest.mock import MagicMock

        from trelix.graph.updater import GraphUpdater

        db = MagicMock()
        updater = GraphUpdater(db)
        return updater

    def test_updater_initializes_empty_prev_partition(self, tmp_path):
        updater = self._make_updater(tmp_path)
        assert updater._prev_partition == {}

    def test_prev_partition_updated_after_update_file(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from trelix.graph.updater import GraphUpdater

        db = MagicMock()
        updater = GraphUpdater(db)

        fake_partition = {1: 0, 2: 0, 3: 1}
        fake_cg = MagicMock()
        fake_cg.node_count = 3

        with (
            patch("trelix.graph.updater.CodeGraph", return_value=fake_cg),
            patch(
                "trelix.graph.updater.detect_communities_incremental", return_value=fake_partition
            ),
            patch("trelix.graph.updater.assign_communities"),
            patch("trelix.graph.updater.compute_pagerank", return_value={}),
            patch("trelix.graph.updater.save_graph_metadata"),
        ):
            updater.update_file("src/auth.py")

        assert updater._prev_partition == fake_partition

    def test_incremental_called_with_prev_partition_on_second_update(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from trelix.graph.updater import GraphUpdater

        db = MagicMock()
        updater = GraphUpdater(db)
        updater._prev_partition = {1: 0, 2: 1}  # simulate prior state

        fake_cg = MagicMock()
        fake_cg.node_count = 2

        with (
            patch("trelix.graph.updater.CodeGraph", return_value=fake_cg),
            patch(
                "trelix.graph.updater.detect_communities_incremental", return_value={1: 0, 2: 1}
            ) as mock_inc,
            patch("trelix.graph.updater.assign_communities"),
            patch("trelix.graph.updater.compute_pagerank", return_value={}),
            patch("trelix.graph.updater.save_graph_metadata"),
        ):
            updater.update_file("src/auth.py")

        # prev_partition must be passed to incremental detection
        call_kwargs = mock_inc.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        passed_prev = kwargs.get("prev_partition") or call_kwargs[0][2]
        assert passed_prev == {1: 0, 2: 1}

    def test_update_file_non_fatal_on_failure(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from trelix.graph.updater import GraphUpdater

        db = MagicMock()
        updater = GraphUpdater(db)

        with patch("trelix.graph.updater.CodeGraph", side_effect=RuntimeError("db gone")):
            # Must not raise — non-fatal
            updater.update_file("src/auth.py")

        # prev_partition unchanged on failure
        assert updater._prev_partition == {}
