"""
How `_is_within_root` behaves when `entry.resolve()` cannot answer the question.

The containment check (`TRELIX_WALKER_FOLLOW_SYMLINKS=false`) is the only walker path
that calls `resolve()` on every entry, and it used to guard that call with
`except OSError` alone. On this project's floor interpreter that handler is dead code:
`pathlib.Path.resolve(strict=False)` calls `os.path.realpath(strict=False)` — which
never raises — then `stat()`s the result and SWALLOWS every OSError except ELOOP,
which it re-raises as `RuntimeError("Symlink loop from ...")`. Measured on 3.11.14:
a broken link, a nonexistent path and a path under a chmod-000 directory all resolve
silently; only a symlink loop raises, and it raises the one class the walker did not
catch. A mutual loop (`la -> lb`, `lb -> la`) therefore aborted the ENTIRE walk with an
uncaught RuntimeError, taking every legitimate sibling file with it.

The classes below also pin the deliberate asymmetry in what counts as a gap:
a policy exclusion is silent, a read error is recorded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.core.models import Language
from trelix.indexing.walker import FileWalker


def _contained_walker(repo: Path, *, follow: bool = False) -> FileWalker:
    """A walker over `repo` with containment enabled (i.e. follow_symlinks off)."""
    return FileWalker(
        IndexConfig(
            repo_path=str(repo),
            walker=WalkerConfig(follow_symlinks=follow, languages=[Language.PYTHON]),
        )
    )


def _repo_with_mutual_loop(base: Path) -> Path:
    """A repo holding `la -> lb -> la` plus real files sorted either side of them.

    `_iter_files` walks `sorted(root.iterdir())`, so a file whose name sorts AFTER the
    loop links is only reachable if the loop does not abort the traversal. `aa_keep.py`
    alone would pass even with the bug, because it is yielded before the raise.
    """
    repo = base / "repo"
    repo.mkdir()
    (repo / "aa_keep.py").write_text("first = 1\n", encoding="utf-8")
    (repo / "zz_keep.py").write_text("last = 2\n", encoding="utf-8")
    (repo / "la").symlink_to("lb")
    (repo / "lb").symlink_to("la")
    return repo


class TestSymlinkLoopUnderContainment:
    """A mutual symlink loop must not be fatal, and must not be called a gap."""

    def test_mutual_loop_does_not_abort_the_walk(self, tmp_path: Path) -> None:
        repo = _repo_with_mutual_loop(tmp_path)

        rels = sorted(f.rel_path for f in _contained_walker(repo).walk())

        assert rels == ["aa_keep.py", "zz_keep.py"], (
            f"a symlink loop cost the walk its legitimate siblings: {rels}"
        )

    def test_the_default_config_never_reaches_resolve(self, tmp_path: Path) -> None:
        """Exposure is limited to the opt-in flag: the default short-circuits.

        `_is_within_root` returns True before touching the filesystem when
        `follow_symlinks` is True, which is why this defect never reached a
        default-configured index.
        """
        repo = _repo_with_mutual_loop(tmp_path)

        walker = _contained_walker(repo, follow=True)
        rels = sorted(f.rel_path for f in walker.walk())

        assert rels == ["aa_keep.py", "zz_keep.py"]
        assert walker.walk_was_complete is True

    def test_a_loop_is_not_recorded_as_incomplete(self, tmp_path: Path) -> None:
        """ELOOP means resolution never terminates, so there is no file behind it.

        Same reasoning as the symlink-CYCLE skip in `_iter_files`: nothing real is
        missed, so marking the walk incomplete would disable `missing_is_trustworthy`
        (and any future --prune) for the lifetime of a user-created link that is not
        an error.
        """
        repo = _repo_with_mutual_loop(tmp_path)
        walker = _contained_walker(repo)

        list(walker.walk())

        assert walker.incomplete_paths == [], (
            f"a dangling symlink loop was reported as a gap: {walker.incomplete_paths}"
        )
        assert walker.walk_was_complete is True

    def test_a_broken_link_is_not_recorded_as_incomplete(self, tmp_path: Path) -> None:
        """Nor is a plain dangling link — for the same reason, no bytes exist."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        (repo / "dangling.py").symlink_to(tmp_path / "does_not_exist.py")
        walker = _contained_walker(repo)

        rels = sorted(f.rel_path for f in walker.walk())

        assert rels == ["main.py"]
        assert walker.walk_was_complete is True


class TestOutOfTreeRejectionIsPolicyNotGap:
    """Excluding a resolvable out-of-tree file is configuration, not a failure."""

    def test_containment_rejection_leaves_the_walk_complete(self, tmp_path: Path) -> None:
        """Recording it would make containment permanently distrust its own index.

        The exclusion is deliberate and reproducible, and `follow_symlinks` is one of
        the provenance fields (store/provenance.py), so a later walk with the flag
        flipped is already detected as a config change. Treating every out-of-tree
        link as an unreadable path would instead block deletion-side operations
        forever on any repo that legitimately symlinks outside itself.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        (outside / "secret.py").write_text("def out():\n    return 1\n", encoding="utf-8")
        (repo / "linked.py").symlink_to(outside / "secret.py")
        (repo / "linked_dir").symlink_to(outside, target_is_directory=True)
        walker = _contained_walker(repo)

        rels = sorted(f.rel_path for f in walker.walk())

        assert rels == ["main.py"], f"out-of-tree content leaked: {rels}"
        assert walker.walk_was_complete is True
        assert walker.incomplete_paths == []


class TestUnresolvableEntryIsRecorded:
    """A read error during resolution DOES drop real content, so it is a gap.

    `resolve(strict=False)` raises no OSError on CPython/POSIX today, so these use
    monkeypatch. The handler is not decorative: `resolve()` is documented as able to
    raise OSError, Windows resolution can fail with its own error codes, and 3.13+
    reimplemented the method on top of `os.path.realpath` — any of which can put a
    real OSError back on this path. Without recording, a whole subtree vanishes while
    `walk_was_complete` still says the index mirrors the repository.
    """

    @staticmethod
    def _fail_resolving(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        """Make `Path.resolve()` raise PermissionError for entries called `name`."""
        real_resolve = Path.resolve

        def fake_resolve(self: Path, strict: bool = False) -> Path:
            if self.name == name:
                raise PermissionError(13, "Permission denied", str(self))
            return real_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

    def test_unresolvable_file_marks_the_walk_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        (repo / "guarded.py").write_text("def guarded():\n    return 1\n", encoding="utf-8")
        walker = _contained_walker(repo)
        self._fail_resolving(monkeypatch, "guarded.py")

        rels = sorted(f.rel_path for f in walker.walk())

        assert rels == ["main.py"], "containment must exclude what it cannot adjudicate"
        assert walker.walk_was_complete is False, (
            "a file was dropped because it could not be resolved, and the walk still "
            "claimed to be complete"
        )
        assert walker.incomplete_paths == ["guarded.py"]

    def test_unresolvable_directory_marks_the_walk_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dir branch drops a whole subtree, which is the expensive case.

        `entry.is_dir()` has already succeeded here, so the target existed a moment
        ago — a resolve failure at this point is a race or a filesystem error, not a
        dangling link, and everything under it is silently missing from the index.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        sub = repo / "guarded_dir"
        sub.mkdir()
        (sub / "inner.py").write_text("def inner():\n    return 2\n", encoding="utf-8")
        walker = _contained_walker(repo, follow=True)  # default path: dir-branch only
        self._fail_resolving(monkeypatch, "guarded_dir")

        rels = sorted(f.rel_path for f in walker.walk())

        assert rels == ["main.py"]
        assert walker.walk_was_complete is False, (
            "an entire subtree was dropped and the walk still claimed to be complete"
        )
        assert walker.incomplete_paths == ["guarded_dir"]
