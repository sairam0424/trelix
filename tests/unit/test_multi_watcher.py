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
        """ImportError when watchfiles missing includes install hint.

        The reload has to be undone, and this test did not undo it. `multi_watcher` binds
        `Change` and `awatch` at module scope inside a `try/except ImportError`, so
        reloading it while `sys.modules["watchfiles"]` is None leaves BOTH set to None —
        and `patch.dict` restores sys.modules without restoring the module object it
        poisoned. Every later test in the session then saw `Change is None`, which
        silently disables the deletion branch of `MultiRepoWatcher.run` (its guard reads
        `if Change is not None and change_type == Change.deleted`), so a delete event fell
        through to the ignore filter instead.

        Nothing caught it for as long as no test depended on deletions. The first one that
        did — `test_multi_watcher_filtering.py::test_deleted_ignored_path_still_purges_index`
        — failed only in a full-suite run and passed in isolation, with the deleted path
        logged as "ignored (walker filters)".

        Restored in a `finally` so an assertion failure inside the block cannot leave the
        module broken either.
        """
        import importlib
        import sys
        from unittest.mock import patch

        from trelix.indexing import multi_watcher

        try:
            with patch.dict(sys.modules, {"watchfiles": None}):
                with pytest.raises(ImportError, match="trelix\\[watch\\]|watchfiles"):
                    importlib.reload(multi_watcher)
                    from trelix.indexing.multi_watcher import MultiRepoWatcher

                    reg = _registry("/fake")
                    MultiRepoWatcher(reg)._require_watchfiles()
        finally:
            # sys.modules["watchfiles"] is real again by here, so this rebinds Change and
            # awatch to the genuine objects.
            importlib.reload(multi_watcher)

        assert multi_watcher.Change is not None, (
            "reload did not restore watchfiles.Change — the deletion branch of "
            "MultiRepoWatcher.run is now silently disabled for the rest of this session"
        )
        assert multi_watcher.awatch is not None
