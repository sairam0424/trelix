"""Tests for MultiRepoWatcher (v2.4 watch-all)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.federation.registry import RepoEntry, RepoRegistry


def _registry(*paths: str) -> RepoRegistry:
    reg = RepoRegistry.__new__(RepoRegistry)
    reg._config_path = "/tmp/fake.json"
    reg._entries = [RepoEntry(alias=f"r{i}", path=p, weight=1.0) for i, p in enumerate(paths)]
    return reg


class TestMultiRepoWatcherInit:
    def test_importable(self) -> None:
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        assert MultiRepoWatcher is not None

    def test_stats_initial(self, tmp_path: Path) -> None:
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        reg = _registry(str(tmp_path))
        watcher = MultiRepoWatcher(reg)
        stats = watcher.stats()
        assert stats["repos_watched"] == 1
        assert stats["files_reindexed"] == 0
        assert stats["files_skipped_unchanged"] == 0

    def test_empty_registry_stats(self) -> None:
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        reg = _registry()  # no repos
        watcher = MultiRepoWatcher(reg)
        assert watcher.stats()["repos_watched"] == 0


watchfiles = pytest.importorskip("watchfiles", reason="watchfiles optional dep not installed")


class TestMultiRepoWatcherRun:
    @pytest.mark.asyncio
    async def test_run_stops_on_event(self, tmp_path: Path) -> None:
        """run() exits cleanly when stop_event is set."""
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        reg = _registry(str(tmp_path))
        watcher = MultiRepoWatcher(reg)

        stop_event = asyncio.Event()

        async def fake_awatch(*paths, stop_event=None, **kwargs):
            # Simulate no file changes, just yield once then wait for stop
            stop_event.set()  # trigger stop immediately
            return
            yield  # make it an async generator

        with patch("trelix.indexing.multi_watcher.awatch", new=fake_awatch):
            with patch("trelix.indexing.multi_watcher.Indexer"):
                await asyncio.wait_for(watcher.run(stop_event), timeout=2.0)

    @pytest.mark.asyncio
    async def test_run_skips_unchanged_files(self, tmp_path: Path) -> None:
        """Files with same hash are skipped (no re-index)."""
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        Change = watchfiles.Change

        test_file = tmp_path / "auth.py"
        test_file.write_text("def login(): pass")

        reg = _registry(str(tmp_path))
        watcher = MultiRepoWatcher(reg)

        # Pre-populate hash cache to simulate "already indexed"
        import hashlib

        content = test_file.read_bytes()
        watcher._file_hashes[str(test_file)] = hashlib.md5(content).hexdigest()

        stop_event = asyncio.Event()

        async def fake_awatch(*paths, stop_event=None, **kwargs):
            yield {(Change.modified, str(test_file))}
            stop_event.set()

        mock_indexer = MagicMock()

        with patch("trelix.indexing.multi_watcher.awatch", new=fake_awatch):
            with patch("trelix.indexing.multi_watcher.Indexer", return_value=mock_indexer):
                await asyncio.wait_for(watcher.run(stop_event), timeout=2.0)

        assert watcher.stats()["files_skipped_unchanged"] == 1
        assert watcher.stats()["files_reindexed"] == 0
        mock_indexer.index_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_reindexes_changed_files(self, tmp_path: Path) -> None:
        """Files with new content trigger re-index."""
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        Change = watchfiles.Change

        test_file = tmp_path / "auth.py"
        test_file.write_text("def login(): return True")  # NEW content

        reg = _registry(str(tmp_path))
        watcher = MultiRepoWatcher(reg)

        # Old hash (different from current file content)
        watcher._file_hashes[str(test_file)] = "old_hash_that_wont_match"

        stop_event = asyncio.Event()

        async def fake_awatch(*paths, stop_event=None, **kwargs):
            yield {(Change.modified, str(test_file))}
            stop_event.set()

        mock_indexer = MagicMock()

        with patch("trelix.indexing.multi_watcher.awatch", new=fake_awatch):
            with patch("trelix.indexing.multi_watcher.Indexer", return_value=mock_indexer):
                await asyncio.wait_for(watcher.run(stop_event), timeout=2.0)

        assert watcher.stats()["files_reindexed"] == 1

    @pytest.mark.asyncio
    async def test_run_noop_on_empty_registry(self) -> None:
        """run() returns immediately if no repos registered."""
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        reg = _registry()
        watcher = MultiRepoWatcher(reg)
        stop_event = asyncio.Event()
        # Should return instantly, no hang
        await asyncio.wait_for(watcher.run(stop_event), timeout=1.0)


class TestRequireWatchfiles:
    def test_import_error_message_is_helpful(self) -> None:
        """ImportError when watchfiles missing includes install hint."""
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"watchfiles": None}):
            with pytest.raises(ImportError, match="trelix\\[watch\\]|watchfiles"):
                import importlib

                from trelix.indexing import multi_watcher

                importlib.reload(multi_watcher)
                from trelix.indexing.multi_watcher import MultiRepoWatcher

                reg = _registry("/fake")
                MultiRepoWatcher(reg)._require_watchfiles()
