"""
MultiRepoWatcher — watch all repos in RepoRegistry for file changes.

Uses watchfiles.awatch() with a single call over all paths simultaneously.
Debounce is handled by watchfiles' Rust layer (default 1600ms).
Hash guard prevents re-index loops when indexer writes to source tree.

Usage:
    registry = RepoRegistry.load()
    watcher = MultiRepoWatcher(registry)
    stop = asyncio.Event()
    # In a real program: signal.signal(SIGINT, lambda *a: stop.set())
    await watcher.run(stop)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.federation.registry import RepoRegistry
from trelix.indexing.indexer import Indexer

logger = logging.getLogger("trelix.indexing.multi_watcher")


def _require_watchfiles() -> None:
    try:
        import watchfiles  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'watch-all' command requires the 'watchfiles' package.\n"
            "Install it with:  pip install 'trelix[watch]'\n"
            "  or:             pip install watchfiles>=0.21"
        ) from exc


try:
    from watchfiles import awatch, Change
except ImportError:
    # Deferred import — will raise at runtime via _require_watchfiles()
    awatch = None  # type: ignore[assignment]
    Change = None  # type: ignore[assignment]


class MultiRepoWatcher:
    """
    Watch all repos registered in RepoRegistry for file changes.

    Uses a single watchfiles.awatch() call with all repo paths.
    Hash guard skips re-indexing if file content hasn't actually changed
    (prevents cascade loops when the indexer writes .trelix/ files).
    """

    def __init__(self, registry: RepoRegistry, debounce_ms: int = 1600) -> None:
        self._registry = registry
        self._debounce_ms = debounce_ms
        self._file_hashes: dict[str, str] = {}
        self._files_reindexed = 0
        self._files_skipped = 0

    def _require_watchfiles(self) -> None:
        _require_watchfiles()

    def _file_hash(self, path: str) -> str:
        """MD5 of file bytes — fast enough for hash guard, not cryptographic."""
        try:
            return hashlib.md5(Path(path).read_bytes()).hexdigest()
        except OSError:
            return ""

    def _is_unchanged(self, path: str) -> bool:
        """Return True if file content hash matches cached value."""
        current = self._file_hash(path)
        if not current:
            return False
        cached = self._file_hashes.get(path)
        if cached == current:
            return True
        self._file_hashes[path] = current
        return False

    def _get_repo_for_path(self, file_path: str) -> str | None:
        """Return the repo_path that contains this file_path, or None."""
        for entry in self._registry.list():
            if file_path.startswith(entry.path):
                return entry.path
        return None

    async def run(self, stop_event: asyncio.Event) -> None:
        """
        Watch all registered repos. Blocks until stop_event is set.
        No-op if registry is empty.
        """
        entries = self._registry.list()
        if not entries:
            logger.info("MultiRepoWatcher: no repos registered, exiting immediately")
            return

        _require_watchfiles()

        repo_paths = [entry.path for entry in entries]
        logger.info(
            "MultiRepoWatcher: watching %d repos: %s",
            len(repo_paths),
            ", ".join(entry.alias for entry in entries),
        )

        repo_indexers: dict[str, Indexer] = {}
        for entry in entries:
            try:
                config = IndexConfig.model_construct(repo_path=entry.path)
                repo_indexers[entry.path] = Indexer(config)
            except Exception as exc:
                logger.warning(
                    "MultiRepoWatcher: failed to create indexer for %s: %s",
                    entry.alias,
                    exc,
                )

        async for changes in awatch(
            *repo_paths,
            stop_event=stop_event,
            debounce=self._debounce_ms,
        ):
            for change_type, file_path in changes:
                repo_path = self._get_repo_for_path(file_path)
                if repo_path is None:
                    continue

                if change_type == Change.deleted:
                    # Remove deleted file from index
                    indexer = repo_indexers.get(repo_path)
                    if indexer:
                        try:
                            indexer.remove_file(file_path)
                        except Exception as exc:
                            logger.debug("Failed to remove %s from index: %s", file_path, exc)
                    self._file_hashes.pop(file_path, None)
                    continue

                # For added/modified: check hash to avoid cascade loops
                if self._is_unchanged(file_path):
                    self._files_skipped += 1
                    logger.debug("MultiRepoWatcher: skipped unchanged %s", file_path)
                    continue

                indexer = repo_indexers.get(repo_path)
                if indexer is None:
                    continue

                try:
                    # Use index_file if available (incremental), else full index
                    if hasattr(indexer, "index_file"):
                        indexer.index_file(file_path)
                    else:
                        indexer.index()
                    self._files_reindexed += 1
                    logger.info("MultiRepoWatcher: re-indexed %s", file_path)
                except Exception as exc:
                    logger.warning(
                        "MultiRepoWatcher: re-index failed for %s: %s", file_path, exc
                    )

    def stats(self) -> dict[str, int]:
        """Return watching statistics."""
        return {
            "repos_watched": len(self._registry.list()),
            "files_reindexed": self._files_reindexed,
            "files_skipped_unchanged": self._files_skipped,
        }
