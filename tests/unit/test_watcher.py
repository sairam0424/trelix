"""
Unit tests for trelix.indexing.watcher.FileWatcher.

All watchdog imports and filesystem access are mocked so the tests run
without the watchdog package being installed and without touching the disk.
"""

from __future__ import annotations

import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers to build lightweight mock watchdog events
# ---------------------------------------------------------------------------


def _make_event(event_type: str, src_path: str, is_directory: bool = False) -> MagicMock:
    event = MagicMock()
    event.event_type = event_type
    event.src_path = src_path
    event.is_directory = is_directory
    return event


def _make_file_event(event_type: str, src_path: str) -> MagicMock:
    return _make_event(event_type, src_path, is_directory=False)


def _make_dir_event(event_type: str, src_path: str) -> MagicMock:
    return _make_event(event_type, src_path, is_directory=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(repo_path: str = "/repo") -> MagicMock:
    config = MagicMock()
    config.repo_path = repo_path
    config.walker.extra_ignore_extensions = []
    config.walker.extra_ignore_filenames = []
    config.walker.max_file_size_bytes = 10 * 1024 * 1024  # 10 MB
    config.walker.languages = ["python", "typescript", "javascript"]
    return config


def _make_indexer(repo_path: str = "/repo") -> MagicMock:
    indexer = MagicMock()
    indexer.config = _make_config(repo_path)
    indexer.db = MagicMock()
    indexer.vector_store = MagicMock()
    indexer.index_file.return_value = {
        "status": "ok",
        "symbols_updated": 5,
        "chunks_updated": 7,
        "ms": 120,
    }
    return indexer


def _make_walker(repo_path: str = "/repo") -> MagicMock:
    walker = MagicMock()
    walker.config = _make_config(repo_path)
    walker._is_ignored_file.return_value = False  # not ignored by default
    return walker


# ---------------------------------------------------------------------------
# Inject a minimal fake 'watchdog' package so tests run without it installed
# ---------------------------------------------------------------------------


def _inject_fake_watchdog() -> None:
    """Register a minimal watchdog stub in sys.modules (idempotent)."""
    if "watchdog" in sys.modules:
        return

    watchdog_pkg = types.ModuleType("watchdog")
    observers_mod = types.ModuleType("watchdog.observers")
    events_mod = types.ModuleType("watchdog.events")

    class FakeObserver:
        def schedule(self, handler: object, path: str, recursive: bool = True) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FakeFileSystemEventHandler:
        pass

    observers_mod.Observer = FakeObserver
    events_mod.FileSystemEventHandler = FakeFileSystemEventHandler

    watchdog_pkg.observers = observers_mod  # type: ignore[attr-defined]
    watchdog_pkg.events = events_mod  # type: ignore[attr-defined]

    sys.modules["watchdog"] = watchdog_pkg
    sys.modules["watchdog.observers"] = observers_mod
    sys.modules["watchdog.events"] = events_mod


_inject_fake_watchdog()

# Now we can safely import the module under test
from trelix.indexing.watcher import FileWatcher, _TrelixEventHandler  # noqa: E402

# ---------------------------------------------------------------------------
# Test: ImportError when watchdog not installed
# ---------------------------------------------------------------------------


class TestWatchdogImportError(unittest.TestCase):
    """Verify a friendly ImportError is raised when watchdog is absent."""

    def test_helpful_import_error_when_watchdog_missing(self) -> None:
        # Temporarily hide watchdog from sys.modules
        saved = {k: v for k, v in sys.modules.items() if k.startswith("watchdog")}
        for k in saved:
            del sys.modules[k]
        try:
            from trelix.indexing.watcher import _require_watchdog

            with self.assertRaises(ImportError) as ctx:
                _require_watchdog()

            msg = str(ctx.exception)
            self.assertIn("watchdog", msg.lower())
            self.assertIn("pip install", msg)
        finally:
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Test: modified / created events trigger debounced index_file call
# ---------------------------------------------------------------------------


class TestModifiedCreatedEvents(unittest.TestCase):
    def _make_watcher(self, debounce_ms: int = 50) -> FileWatcher:
        indexer = _make_indexer()
        walker = _make_walker()
        watcher = FileWatcher(indexer, walker, debounce_ms=debounce_ms)
        return watcher

    def test_modified_event_triggers_index_file(self) -> None:
        """on_modified for a .py file should call indexer.index_file after debounce."""
        watcher = self._make_watcher(debounce_ms=50)

        with patch.object(watcher, "_should_index", return_value=True):
            handler = _TrelixEventHandler(watcher)
            event = _make_file_event("modified", "/repo/src/auth.py")
            handler.dispatch(event)

            # Wait for debounce
            time.sleep(0.15)

        watcher._indexer.index_file.assert_called_once_with("/repo/src/auth.py")

    def test_created_event_triggers_index_file(self) -> None:
        """on_created for a .py file should call indexer.index_file after debounce."""
        watcher = self._make_watcher(debounce_ms=50)

        with patch.object(watcher, "_should_index", return_value=True):
            handler = _TrelixEventHandler(watcher)
            event = _make_file_event("created", "/repo/src/new_module.py")
            handler.dispatch(event)

            time.sleep(0.15)

        watcher._indexer.index_file.assert_called_once_with("/repo/src/new_module.py")

    def test_directory_event_is_ignored(self) -> None:
        """Directory events must not trigger re-indexing."""
        watcher = self._make_watcher(debounce_ms=50)

        handler = _TrelixEventHandler(watcher)
        event = _make_dir_event("modified", "/repo/src/")
        handler.dispatch(event)

        time.sleep(0.15)
        watcher._indexer.index_file.assert_not_called()


# ---------------------------------------------------------------------------
# Test: deleted event → delete_file_by_path called
# ---------------------------------------------------------------------------


class TestDeletedEvent(unittest.TestCase):
    def test_deleted_event_removes_index_data(self) -> None:
        """on_deleted should call db.delete_file_by_path with the correct paths."""
        indexer = _make_indexer(repo_path="/repo")
        walker = _make_walker(repo_path="/repo")
        watcher = FileWatcher(indexer, walker, debounce_ms=50)

        handler = _TrelixEventHandler(watcher)
        event = _make_file_event("deleted", "/repo/src/auth.py")
        handler.dispatch(event)

        # delete_file_by_path is synchronous (no debounce for deletes)
        indexer.db.delete_file_by_path.assert_called_once_with(
            "/repo/src/auth.py",
            "src/auth.py",
            indexer.vector_store,
        )

    def test_deleted_event_for_file_outside_repo(self) -> None:
        """Deletion outside repo root should still call delete_file_by_path (abs==rel)."""
        indexer = _make_indexer(repo_path="/repo")
        walker = _make_walker(repo_path="/repo")
        watcher = FileWatcher(indexer, walker, debounce_ms=50)

        handler = _TrelixEventHandler(watcher)
        event = _make_file_event("deleted", "/other/place/file.py")
        handler.dispatch(event)

        # rel_path fallback equals abs_path when outside repo root
        indexer.db.delete_file_by_path.assert_called_once_with(
            "/other/place/file.py",
            "/other/place/file.py",
            indexer.vector_store,
        )


# ---------------------------------------------------------------------------
# Test: gitignore-filtered files are NOT indexed
# ---------------------------------------------------------------------------


class TestGitignoreFilter(unittest.TestCase):
    def test_gitignore_filtered_file_not_indexed(self) -> None:
        """Files matching .gitignore must not be re-indexed on modification."""
        indexer = _make_indexer()
        walker = _make_walker()
        # Simulate walker returning True for .pyc files → ignored
        walker._is_ignored_file.return_value = True
        watcher = FileWatcher(indexer, walker, debounce_ms=50)

        handler = _TrelixEventHandler(watcher)
        event = _make_file_event("modified", "/repo/src/__pycache__/auth.cpython-311.pyc")
        handler.dispatch(event)

        time.sleep(0.15)
        indexer.index_file.assert_not_called()

    def test_unknown_extension_not_indexed(self) -> None:
        """.pyc / .so / .png extensions are not in EXTENSION_MAP → not indexed."""
        indexer = _make_indexer()
        walker = _make_walker()
        watcher = FileWatcher(indexer, walker, debounce_ms=50)

        for ext in (".pyc", ".so", ".png", ".DS_Store", ".log"):
            indexer.reset_mock()
            handler = _TrelixEventHandler(watcher)
            with patch("pathlib.Path.is_file", return_value=True):
                event = _make_file_event("modified", f"/repo/src/artifact{ext}")
                handler.dispatch(event)

        time.sleep(0.15)
        indexer.index_file.assert_not_called()


# ---------------------------------------------------------------------------
# Test: rapid edits debounce to a single index_file call
# ---------------------------------------------------------------------------


class TestDebouncing(unittest.TestCase):
    def test_rapid_edits_collapsed_to_single_call(self) -> None:
        """Five rapid edits to the same file must result in exactly one index_file call.

        Use a large debounce window (500 ms) and a very short inter-event sleep
        (5 ms) so that even under heavy CI scheduling jitter all events stay
        comfortably within the debounce window.
        """
        indexer = _make_indexer()
        walker = _make_walker()
        watcher = FileWatcher(indexer, walker, debounce_ms=500)

        with patch.object(watcher, "_should_index", return_value=True):
            handler = _TrelixEventHandler(watcher)
            for _ in range(5):
                event = _make_file_event("modified", "/repo/src/auth.py")
                handler.dispatch(event)
                time.sleep(0.005)  # 5 ms between saves — well within 500 ms window

            # Wait past the debounce window
            time.sleep(0.8)

        # Despite 5 events, index_file must be called exactly once
        self.assertEqual(indexer.index_file.call_count, 1)
        indexer.index_file.assert_called_with("/repo/src/auth.py")

    def test_two_different_files_each_get_one_call(self) -> None:
        """Rapid edits to two different files each produce exactly one call."""
        indexer = _make_indexer()
        walker = _make_walker()
        watcher = FileWatcher(indexer, walker, debounce_ms=500)

        with patch.object(watcher, "_should_index", return_value=True):
            handler = _TrelixEventHandler(watcher)
            for _ in range(3):
                handler.dispatch(_make_file_event("modified", "/repo/src/a.py"))
                handler.dispatch(_make_file_event("modified", "/repo/src/b.py"))
                time.sleep(0.005)

            time.sleep(0.8)

        self.assertEqual(indexer.index_file.call_count, 2)
        calls = {c.args[0] for c in indexer.index_file.call_args_list}
        self.assertIn("/repo/src/a.py", calls)
        self.assertIn("/repo/src/b.py", calls)

    def test_pending_timer_cancelled_on_stop(self) -> None:
        """Pending debounce timers must be cancelled when watcher.stop() is called."""
        indexer = _make_indexer()
        walker = _make_walker()
        watcher = FileWatcher(indexer, walker, debounce_ms=500)

        with patch.object(watcher, "_should_index", return_value=True):
            handler = _TrelixEventHandler(watcher)
            handler.dispatch(_make_file_event("modified", "/repo/src/auth.py"))

        # Stop before the 500 ms debounce fires
        watcher.stop()
        time.sleep(0.6)

        # index_file must NOT have been called
        indexer.index_file.assert_not_called()


# ---------------------------------------------------------------------------
# Test: start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle(unittest.TestCase):
    def test_start_stop_calls_observer(self) -> None:
        """start() and stop() must call Observer.start/stop/join in order."""
        indexer = _make_indexer()
        walker = _make_walker()
        watcher = FileWatcher(indexer, walker)

        mock_observer = MagicMock()
        # Observer is imported inside start() from watchdog.observers, so patch there
        with patch("watchdog.observers.Observer", return_value=mock_observer):
            # Reload the Observer import inside watcher.start() by patching the module attr
            import watchdog.observers as _obs_mod

            original_observer = _obs_mod.Observer
            _obs_mod.Observer = MagicMock(return_value=mock_observer)
            try:
                watcher.start()
                watcher.stop()
            finally:
                _obs_mod.Observer = original_observer

        mock_observer.start.assert_called_once()
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


# ---------------------------------------------------------------------------
# Test: db.delete_file_by_path (unit — no watchdog required)
# ---------------------------------------------------------------------------


class TestDeleteFileByPath(unittest.TestCase):
    """
    White-box tests for Database.delete_file_by_path().
    Uses an in-memory SQLite database to avoid any filesystem coupling.
    """

    def _make_db(self) -> object:
        """Create a Database backed by in-memory SQLite."""
        import tempfile
        from pathlib import Path as P

        from trelix.store.db import Database

        # Use a temporary file so WAL mode works (WAL requires a real path)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = Database(P(tmp.name))
        return db

    def test_delete_file_by_path_returns_false_when_not_found(self) -> None:
        db = self._make_db()
        result = db.delete_file_by_path("/nonexistent/file.py", "file.py", None)
        self.assertFalse(result)
        db.close()

    def test_delete_file_by_path_removes_file_row(self) -> None:
        from trelix.core.models import IndexedFile, Language

        db = self._make_db()

        # Insert a file
        f = IndexedFile(
            path="/repo/src/auth.py",
            rel_path="src/auth.py",
            language=Language.PYTHON,
            hash="abc123",
            size_bytes=100,
        )
        db.upsert_file(f)

        # Verify it exists
        row = db._conn.execute(
            "SELECT id FROM files WHERE rel_path = ?", ("src/auth.py",)
        ).fetchone()
        self.assertIsNotNone(row)

        # Delete it
        result = db.delete_file_by_path("/repo/src/auth.py", "src/auth.py", None)
        self.assertTrue(result)

        # Verify it's gone
        row = db._conn.execute(
            "SELECT id FROM files WHERE rel_path = ?", ("src/auth.py",)
        ).fetchone()
        self.assertIsNone(row)
        db.close()

    def test_delete_file_by_path_deletes_vectors(self) -> None:
        from trelix.core.models import IndexedFile, Language

        db = self._make_db()

        f = IndexedFile(
            path="/repo/src/utils.py",
            rel_path="src/utils.py",
            language=Language.PYTHON,
            hash="def456",
            size_bytes=200,
        )
        db.upsert_file(f)

        mock_vs = MagicMock()
        # Patch get_chunk_ids_for_file to return fake chunk ids
        with patch.object(db, "get_chunk_ids_for_file", return_value=[10, 20, 30]):
            result = db.delete_file_by_path("/repo/src/utils.py", "src/utils.py", mock_vs)

        self.assertTrue(result)
        mock_vs.delete_batch.assert_called_once_with([10, 20, 30])
        db.close()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Test: DimensionGuard called at FileWatcher.__init__ time
# ---------------------------------------------------------------------------


import pytest  # noqa: E402


class TestFileWatcherDimensionGuard:
    """FileWatcher must call DimensionGuard at __init__ time."""

    def _make_mock_indexer(self, dimension: int = 384, provider: str = "local"):
        from unittest.mock import MagicMock

        indexer = MagicMock()
        indexer.config.repo_path = "/tmp/repo"
        indexer.config.embedder.provider = provider
        # Indexer exposes .db and .embedder (no underscore prefix)
        indexer.embedder = MagicMock()
        indexer.embedder.dimension = dimension
        indexer.db = MagicMock()
        return indexer

    def test_raises_dimension_mismatch_on_init(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from trelix.store.dimension_guard import DimensionMismatchError

        indexer = self._make_mock_indexer(dimension=384, provider="local")
        walker = MagicMock()

        with patch(
            "trelix.indexing.watcher.DimensionGuard.check",
            side_effect=DimensionMismatchError(stored=3072, current=384, provider="local"),
        ):
            with pytest.raises(DimensionMismatchError) as exc_info:
                FileWatcher(indexer, walker)

        assert "3072" in str(exc_info.value)
        assert "384" in str(exc_info.value)

    def test_starts_normally_when_dimensions_match(self, tmp_path):
        from unittest.mock import MagicMock, patch

        indexer = self._make_mock_indexer(dimension=384, provider="local")
        walker = MagicMock()

        with patch("trelix.indexing.watcher.DimensionGuard.check") as mock_check:
            fw = FileWatcher(indexer, walker)

        mock_check.assert_called_once()

    def test_dimension_guard_not_called_when_no_db(self, tmp_path):
        """Gracefully skips guard when indexer.db is None (fresh indexer before first index run)."""
        from unittest.mock import MagicMock, patch

        # Simulate an indexer where db is explicitly None (before first index run)
        indexer = MagicMock()
        indexer.config.repo_path = "/tmp/repo"
        indexer.db = None  # no database yet
        indexer.embedder = MagicMock()
        indexer.embedder.dimension = 384
        walker = MagicMock()

        with patch("trelix.indexing.watcher.DimensionGuard.check") as mock_check:
            fw = FileWatcher(indexer, walker)

        # Guard must NOT be called when db is None
        mock_check.assert_not_called()
        assert fw is not None
