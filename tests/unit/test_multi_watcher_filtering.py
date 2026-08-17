"""MultiRepoWatcher must apply the walker's ignore rules to watch events.

`watch-all` used to hand every path watchfiles reported straight to
`Indexer.index_file()`, which only checks language + content hash. A single `npm
install` under a registered repo therefore fed thousands of `node_modules/` files
into the embedder — work the batch walk and the single-repo `watch` path both
refuse to do. These tests pin the walker's verdict, not a hardcoded list of
directory names, so the three surfaces cannot drift apart again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.federation.registry import RepoEntry, RepoRegistry

watchfiles = pytest.importorskip("watchfiles", reason="watchfiles optional dep not installed")


def _registry(*paths: str) -> RepoRegistry:
    reg = RepoRegistry.__new__(RepoRegistry)
    reg._config_path = "/tmp/fake.json"
    reg._entries = [RepoEntry(alias=f"r{i}", path=p, weight=1.0) for i, p in enumerate(paths)]
    return reg


async def _run_one_change(repo: Path, changed: Path) -> MagicMock:
    """Drive one modify event for `changed` through a watcher over `repo`."""
    from trelix.indexing.multi_watcher import MultiRepoWatcher

    watcher = MultiRepoWatcher(_registry(str(repo)))
    stop_event = asyncio.Event()

    async def fake_awatch(*paths, stop_event=None, **kwargs):  # type: ignore[no-untyped-def]
        yield {(watchfiles.Change.modified, str(changed))}
        stop_event.set()

    mock_indexer = MagicMock()
    with patch("trelix.indexing.multi_watcher.awatch", new=fake_awatch):
        with patch("trelix.indexing.multi_watcher.Indexer", return_value=mock_indexer):
            await asyncio.wait_for(watcher.run(stop_event), timeout=5.0)

    mock_indexer.watcher_stats = watcher.stats()
    return mock_indexer


def _write(path: Path, text: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestIgnoredDirectoriesAreNotIndexed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rel",
        [
            "node_modules/left-pad/index.js",
            ".venv/lib/python3.12/site-packages/pkg/mod.py",
            "dist/bundle.js",
            ".git/COMMIT_EDITMSG.md",
            "__pycache__/mod.py",
            ".trelix/cache/notes.md",
        ],
    )
    async def test_ignored_dir_change_is_not_reindexed(self, tmp_path: Path, rel: str) -> None:
        """extra_ignore_dirs is a DIRECTORY rule, so a bare event path must be
        checked against every parent between the repo root and the file."""
        changed = _write(tmp_path / rel)

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_not_called()
        assert indexer.watcher_stats["files_reindexed"] == 0
        assert indexer.watcher_stats["files_skipped_ignored"] == 1

    @pytest.mark.asyncio
    async def test_nested_ignored_dir_is_not_reindexed(self, tmp_path: Path) -> None:
        """The ignored directory may be several levels below the repo root."""
        changed = _write(tmp_path / "frontend/app/node_modules/react/index.js")

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_not_called()


class TestGitignoreAndLanguageFilters:
    @pytest.mark.asyncio
    async def test_gitignored_file_is_not_reindexed(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("generated_*.py\n")
        changed = _write(tmp_path / "src/generated_pb2.py")

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_language_is_not_reindexed(self, tmp_path: Path) -> None:
        changed = _write(tmp_path / "build.log", "not source\n")

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignored_filename_is_not_reindexed(self, tmp_path: Path) -> None:
        """package-lock.json is a real .json the language allow-list accepts —
        only extra_ignore_filenames keeps it out."""
        changed = _write(tmp_path / "package-lock.json", '{"lockfileVersion": 3}\n')

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_oversized_file_is_not_reindexed(self, tmp_path: Path) -> None:
        changed = _write(tmp_path / "huge.py", "# pad\n" * 200_000)
        assert changed.stat().st_size > 500_000

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_not_called()


class TestRealSourceStillIndexed:
    @pytest.mark.asyncio
    async def test_ordinary_source_file_is_reindexed(self, tmp_path: Path) -> None:
        """Guard against over-filtering: the fix must not silence real edits."""
        changed = _write(tmp_path / "src/auth.py", "def login():\n    return True\n")

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_called_once_with(str(changed))
        assert indexer.watcher_stats["files_reindexed"] == 1
        assert indexer.watcher_stats["files_skipped_ignored"] == 0

    @pytest.mark.asyncio
    async def test_extensionless_dockerfile_is_reindexed(self, tmp_path: Path) -> None:
        """detect_language(), not a suffix lookup — a Dockerfile has no suffix."""
        changed = _write(tmp_path / "Dockerfile", "FROM python:3.12\n")

        indexer = await _run_one_change(tmp_path, changed)

        indexer.index_file.assert_called_once_with(str(changed))

    def test_predicate_agrees_with_walk_exactly(self, tmp_path: Path) -> None:
        """The anti-drift assertion: over a whole tree, the watch predicate accepts
        exactly the set FileWalker.walk() yields — no more, no less.

        Measured on this repository at the time of the fix: walk() yields 444 files,
        the predicate accepts the same 444, while the pre-fix gate (`detect_language`
        alone) accepted 90,385.
        """
        from trelix.core.config import IndexConfig
        from trelix.indexing.multi_watcher import MultiRepoWatcher
        from trelix.indexing.walker import FileWalker

        (tmp_path / ".gitignore").write_text("generated_*.py\n!generated_keep.py\n")
        for rel in [
            "src/auth.py",
            "src/ui/app.tsx",
            "README.md",
            "Dockerfile",
            "Makefile",
            "src/generated_pb2.py",  # gitignored
            "src/generated_keep.py",  # re-included by the negation
            "node_modules/left-pad/index.js",  # extra_ignore_dirs
            "frontend/node_modules/react/index.js",  # nested
            ".venv/lib/pkg/mod.py",
            "dist/bundle.js",
            "build.log",  # unknown language
            "package-lock.json",  # extra_ignore_filenames
        ]:
            _write(tmp_path / rel)

        config = IndexConfig.model_construct(repo_path=str(tmp_path))
        walked = {f.path for f in FileWalker(config).walk()}

        watcher = MultiRepoWatcher(_registry(str(tmp_path)))
        walker = FileWalker(config)
        accepted = {
            str(p)
            for p in tmp_path.rglob("*")
            if p.is_file() and watcher._should_index(walker, str(p))
        }

        assert accepted == walked
        assert tmp_path / "src/auth.py" in {Path(p) for p in accepted}

    @pytest.mark.asyncio
    async def test_deleted_ignored_path_still_purges_index(self, tmp_path: Path) -> None:
        """Deletions stay unfiltered: rows written before an ignore rule existed
        must still be removable, and a deleted path cannot be inspected anyway."""
        from trelix.indexing.multi_watcher import MultiRepoWatcher

        gone = tmp_path / "node_modules/left-pad/index.js"
        watcher = MultiRepoWatcher(_registry(str(tmp_path)))
        stop_event = asyncio.Event()

        async def fake_awatch(*paths, stop_event=None, **kwargs):  # type: ignore[no-untyped-def]
            yield {(watchfiles.Change.deleted, str(gone))}
            stop_event.set()

        mock_indexer = MagicMock()
        with patch("trelix.indexing.multi_watcher.awatch", new=fake_awatch):
            with patch("trelix.indexing.multi_watcher.Indexer", return_value=mock_indexer):
                await asyncio.wait_for(watcher.run(stop_event), timeout=5.0)

        mock_indexer.db.delete_file_by_path.assert_called_once()
