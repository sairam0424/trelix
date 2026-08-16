"""
MultiRepoWatcher — watch all repos in RepoRegistry for file changes.

Uses watchfiles.awatch() with a single call over all paths simultaneously.
Debounce is handled by watchfiles' Rust layer (default 1600ms).
Hash guard prevents re-index loops when indexer writes to source tree.

Every event is routed through the owning repo's FileWalker before it reaches the
indexer (see `_should_index`). Until v3.1.3 it was not: `index_file()` only checks
language and content hash, so one `npm install` under a registered repo pushed the
whole of `node_modules/` — plus `.venv/`, `dist/` and `.git/` — into the embedder,
while `trelix index` and the single-repo `watch` path both refused those same paths.

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
from trelix.indexing.walker import FileWalker, detect_language

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
    from watchfiles import Change as Change
    from watchfiles import awatch as awatch
except ImportError:
    awatch = None  # type: ignore[assignment,misc]
    Change = None  # type: ignore[assignment,misc]


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
        self._files_ignored = 0

    def _require_watchfiles(self) -> None:
        _require_watchfiles()

    def _file_hash(self, path: str) -> str:
        """MD5 of file bytes — fast enough for hash guard, not cryptographic."""
        try:
            return hashlib.md5(Path(path).read_bytes()).hexdigest()
        except OSError:  # pragma: no cover
            return ""

    def _is_unchanged(self, path: str) -> bool:
        """Return True if file content hash matches cached value."""
        current = self._file_hash(path)
        if not current:  # pragma: no cover
            return False
        cached = self._file_hashes.get(path)
        if cached == current:  # pragma: no cover
            return True
        self._file_hashes[path] = current
        return False

    def _get_repo_for_path(self, file_path: str) -> str | None:
        """Return the repo_path that contains this file_path, or None."""
        for entry in self._registry.list():
            # Ensure exact directory boundary — prevent /repo matching /repo2/file
            repo_dir = entry.path.rstrip("/") + "/"
            if file_path.startswith(repo_dir) or file_path == entry.path.rstrip("/"):
                return entry.path
        return None  # pragma: no cover

    def _should_index(self, walker: FileWalker, file_path: str) -> bool:
        """Ask the repo's own FileWalker whether it would have yielded this path.

        Every verdict below is the walker's, read from its config rather than restated
        here: a new entry in `extra_ignore_dirs` or a new language changes `watch-all`
        and `trelix index` together, which is the whole point of going through the
        walker instead of keeping a fourth copy of the filter.

        The directory loop is the part a walk gets for free and a watch event does not.
        `extra_ignore_dirs` is enforced during traversal — `walk()` simply never
        descends into `node_modules/` — so NO per-file rule mentions those names, and
        `_is_ignored_file("repo/node_modules/left-pad/index.js")` returns False in a
        repo with no .gitignore. A watch event arrives as a bare path with no traversal
        behind it, so each directory between the repo root and the file has to be
        re-asked explicitly or the exclusion never fires.
        """
        path = Path(file_path)
        if not path.is_file():
            # Vanished between the event and now, or never a file (dir mkdir events).
            return False

        try:
            rel = path.relative_to(walker.repo_root)
        except ValueError:
            # _get_repo_for_path already proved textual containment, so this means the
            # registry path and the event path disagree (trailing slash, symlink, case).
            # Refusing is the safe answer, but it silently drops real edits — say so.
            logger.warning(
                "MultiRepoWatcher: %s is not under repo root %s — not indexing it",
                file_path,
                walker.repo_root,
            )
            return False

        current = walker.repo_root
        for part in rel.parts[:-1]:
            current = current / part
            if walker._is_ignored_dir(current):
                return False

        if walker._is_ignored_file(path):
            return False

        walker_config = walker.config.walker
        if path.name in set(walker_config.extra_ignore_filenames):
            return False
        if any(path.name.endswith(ext) for ext in walker_config.extra_ignore_extensions):
            return False

        # detect_language(), never a bare suffix lookup: an extensionless Dockerfile or
        # Makefile has no suffix, and the allow-list is what makes adding an extension
        # to EXTENSION_MAP insufficient on its own.
        if detect_language(path) not in set(walker_config.languages):
            return False

        try:
            if path.stat().st_size > walker_config.max_file_size_bytes:
                return False
        except OSError:
            return False

        return True

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
        # One walker per repo, sharing that repo's config, so ignore rules and the
        # parsed .gitignore chain are the same objects the batch index would use. The
        # walker caches parsed .gitignore files, so keeping it alive for the lifetime of
        # the watch means each one is read once rather than once per event.
        repo_walkers: dict[str, FileWalker] = {}
        for entry in entries:
            try:
                config = IndexConfig.model_construct(repo_path=entry.path)
                repo_indexers[entry.path] = Indexer(config)
                repo_walkers[entry.path] = FileWalker(config)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "MultiRepoWatcher: failed to create indexer for %s: %s",
                    entry.alias,
                    exc,
                )

        # awatch is module-level (patchable in tests); guarded by _require_watchfiles above
        async for changes in awatch(  # pragma: no cover
            *repo_paths,
            stop_event=stop_event,
            debounce=self._debounce_ms,
        ):
            for change_type, file_path in changes:
                repo_path = self._get_repo_for_path(file_path)
                if repo_path is None:
                    continue

                if Change is not None and change_type == Change.deleted:
                    # Remove deleted file from SQLite index + vectors
                    self._file_hashes.pop(file_path, None)
                    indexer = repo_indexers.get(repo_path)
                    if indexer is not None:
                        try:
                            rel = str(Path(file_path).relative_to(repo_path))
                            indexer.db.delete_file_by_path(
                                abs_path=file_path,
                                rel_path=rel,
                                vector_store=indexer.vector_store,
                            )
                            logger.info("MultiRepoWatcher: deleted %s from index", rel)
                        except Exception as exc:
                            logger.debug(
                                "MultiRepoWatcher: delete failed for %s: %s", file_path, exc
                            )
                    continue

                # Ignore filtering comes FIRST: it is what keeps node_modules/, .venv/
                # and dist/ out of the index, and it must run before the hash guard so a
                # 40 MB vendored bundle is never read just to decide we do not want it.
                # Deletions above are deliberately left unfiltered — a deleted path
                # cannot be inspected, and rows written before a rule existed still
                # have to be removable.
                walker = repo_walkers.get(repo_path)
                if walker is None or not self._should_index(walker, file_path):
                    self._files_ignored += 1
                    logger.debug("MultiRepoWatcher: ignored %s (walker filters)", file_path)
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
                    logger.warning("MultiRepoWatcher: re-index failed for %s: %s", file_path, exc)

    def stats(self) -> dict[str, int]:
        """Return watching statistics."""
        return {
            "repos_watched": len(self._registry.list()),
            "files_reindexed": self._files_reindexed,
            "files_skipped_unchanged": self._files_skipped,
            # Reported separately from "unchanged": a large number here is normal (one
            # npm install is thousands), but a non-zero count on a repo the user thinks
            # is fully watched is the signal that an ignore rule is too broad.
            "files_skipped_ignored": self._files_ignored,
        }
