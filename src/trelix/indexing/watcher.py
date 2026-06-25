"""
FileWatcher: incremental re-indexing on filesystem changes.

Uses watchdog to monitor a repository directory and automatically re-indexes
files when they are created or modified, and removes stale index data when
files are deleted.

Debouncing: rapid edits to the same file are collapsed into a single index
call (default 500 ms window) so that saving mid-edit doesn't trigger partial
re-indexes.

Usage:
    from trelix.indexing.watcher import FileWatcher
    from trelix.indexing.indexer import Indexer
    from trelix.indexing.walker import FileWalker

    indexer = Indexer(config)
    walker  = FileWalker(config)
    watcher = FileWatcher(indexer, walker)
    watcher.start()
    # ... block until KeyboardInterrupt ...
    watcher.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.indexing.indexer import Indexer
    from trelix.indexing.walker import FileWalker

logger = logging.getLogger("trelix.watcher")


def _require_watchdog() -> None:
    """Raise a helpful ImportError if watchdog is not installed."""
    try:
        import watchdog  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'watch' command requires the 'watchdog' package.\n"
            "Install it with:  pip install 'trelix[watch]'\n"
            "  or:             pip install watchdog>=4.0.0"
        ) from exc


class FileWatcher:
    """
    Watch a repository directory and incrementally re-index changed files.

    Args:
        indexer:      Indexer instance already configured for the repo.
        walker:       FileWalker instance — used for gitignore / extension
                      filtering before deciding whether to re-index a path.
        debounce_ms:  Milliseconds to wait after the last edit to a file
                      before calling index_file().  Rapid saves within this
                      window are collapsed into a single re-index.
    """

    def __init__(
        self,
        indexer: Indexer,
        walker: FileWalker,
        debounce_ms: int = 500,
    ) -> None:
        _require_watchdog()
        self._indexer = indexer
        self._walker = walker
        self._debounce_s = debounce_ms / 1000.0
        self._observer: object | None = None
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the filesystem observer (non-blocking)."""
        from watchdog.observers import Observer  # type: ignore[import-untyped]

        handler = _TrelixEventHandler(self)
        observer = Observer()
        repo_path = str(self._indexer.config.repo_path)
        observer.schedule(handler, path=repo_path, recursive=True)
        observer.start()
        self._observer = observer
        logger.info("FileWatcher started — watching %s", repo_path)

    def stop(self) -> None:
        """Stop the filesystem observer and cancel pending debounce timers."""
        # Cancel all pending debounce timers
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

        if self._observer is not None:
            self._observer.stop()  # type: ignore[attr-defined]
            self._observer.join()  # type: ignore[attr-defined]
            self._observer = None
        logger.info("FileWatcher stopped")

    # ------------------------------------------------------------------
    # Internal: debounce + dispatch
    # ------------------------------------------------------------------

    def _schedule_reindex(self, abs_path: str) -> None:
        """Schedule a debounced re-index for abs_path."""
        with self._lock:
            existing = self._timers.get(abs_path)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(
                self._debounce_s,
                self._do_reindex,
                args=(abs_path,),
            )
            self._timers[abs_path] = timer
        timer.start()

    def _do_reindex(self, abs_path: str) -> None:
        """Called after the debounce window — actually re-indexes the file."""
        with self._lock:
            self._timers.pop(abs_path, None)

        if not self._should_index(abs_path):
            logger.debug("Skipping non-indexable path: %s", abs_path)
            return

        t0 = time.perf_counter()
        try:
            result = self._indexer.index_file(abs_path)
            elapsed = time.perf_counter() - t0

            repo_root = Path(self._indexer.config.repo_path).resolve()
            try:
                rel = str(Path(abs_path).relative_to(repo_root))
            except ValueError:
                rel = abs_path

            if result.get("status") == "ok":
                symbols = result.get("symbols_updated", 0)
                chunks = result.get("chunks_updated", 0)
                skipped = result.get("skipped", False)
                if not skipped:
                    print(
                        f"[trelix] Updated: {rel} "
                        f"({symbols} symbols, {chunks} chunks, {elapsed:.1f}s)"
                    )
                else:
                    logger.debug("No change in %s — skipped", rel)
            else:
                print(f"[trelix] Error indexing {rel}: {result.get('error', 'unknown')}")

        except Exception as exc:
            logger.error("Unexpected error re-indexing %s: %s", abs_path, exc)
            print(f"[trelix] Error indexing {abs_path}: {exc}")

    def handle_deleted(self, abs_path: str) -> None:
        """Remove index data for a deleted file."""
        # Cancel any pending debounce for this path
        with self._lock:
            timer = self._timers.pop(abs_path, None)
            if timer is not None:
                timer.cancel()

        repo_root = Path(self._indexer.config.repo_path).resolve()
        try:
            rel_path = str(Path(abs_path).relative_to(repo_root))
        except ValueError:
            rel_path = abs_path

        try:
            self._indexer.db.delete_file_by_path(
                abs_path,
                rel_path,
                self._indexer.vector_store,
            )
            logger.info("Removed index data for deleted file: %s", rel_path)
        except Exception as exc:
            logger.error("Error removing index data for %s: %s", rel_path, exc)

    # ------------------------------------------------------------------
    # Internal: filtering
    # ------------------------------------------------------------------

    def _should_index(self, abs_path: str) -> bool:
        """
        Return True if this path is indexable — delegates to walker filters.

        Mirrors the logic in FileWalker.walk():
          - Must have a known extension (in EXTENSION_MAP)
          - Must not be ignored by gitignore
          - Must not match extra_ignore_extensions / extra_ignore_filenames
          - Must not exceed max_file_size_bytes
          - Language must be in the allowed languages list
        """
        from trelix.indexing.walker import EXTENSION_MAP

        path = Path(abs_path)

        if not path.is_file():
            return False

        # Extension must be known
        if path.suffix.lower() not in EXTENSION_MAP:
            return False

        # Gitignore / directory ignore
        try:
            if self._walker._is_ignored_file(path):
                return False
        except ValueError:
            # Path is outside repo root — ignore
            return False

        # Extra ignore extensions
        ignore_extensions = set(self._walker.config.walker.extra_ignore_extensions)
        if any(path.name.endswith(ext) for ext in ignore_extensions):
            return False

        # Extra ignore filenames
        ignore_filenames = set(self._walker.config.walker.extra_ignore_filenames)
        if path.name in ignore_filenames:
            return False

        # Language must be in allowed set
        language = EXTENSION_MAP.get(path.suffix.lower())
        if language not in set(self._walker.config.walker.languages):
            return False

        # Size filter
        try:
            size = path.stat().st_size
            if size > self._walker.config.walker.max_file_size_bytes:
                return False
        except OSError:
            return False

        return True


# ---------------------------------------------------------------------------
# watchdog event handler
# ---------------------------------------------------------------------------


class _TrelixEventHandler:
    """
    Watchdog FileSystemEventHandler that forwards events to FileWatcher.

    Separated from FileWatcher so the watchdog import stays inside this
    class (importable only after _require_watchdog() has been called).
    """

    def __init__(self, watcher: FileWatcher) -> None:
        self._watcher = watcher

    # NOTE: watchdog calls these methods directly on the handler object.
    # We inherit from FileSystemEventHandler dynamically to avoid importing
    # watchdog at module load time.

    def on_modified(self, event: object) -> None:  # type: ignore[override]
        if _is_file_event(event):
            self._watcher._schedule_reindex(_get_src_path(event))

    def on_created(self, event: object) -> None:  # type: ignore[override]
        if _is_file_event(event):
            self._watcher._schedule_reindex(_get_src_path(event))

    def on_deleted(self, event: object) -> None:  # type: ignore[override]
        if _is_file_event(event):
            self._watcher.handle_deleted(_get_src_path(event))

    def dispatch(self, event: object) -> None:
        """Watchdog calls dispatch() — route to on_* methods."""
        event_type = getattr(event, "event_type", None)
        if event_type == "modified":
            self.on_modified(event)
        elif event_type == "created":
            self.on_created(event)
        elif event_type == "deleted":
            self.on_deleted(event)


# ---------------------------------------------------------------------------
# Small helpers (avoid attribute access on untyped watchdog objects)
# ---------------------------------------------------------------------------


def _is_file_event(event: object) -> bool:
    return not getattr(event, "is_directory", True)


def _get_src_path(event: object) -> str:
    return str(getattr(event, "src_path", ""))
