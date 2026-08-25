"""Fallback branches of four private `FileWalker` helpers, unreachable through `walk()`.

`_record_incomplete`, `_under_store_path` and `_report_conditional` each try
`path.relative_to(self.repo_root)` and catch `ValueError` for a path OUTSIDE the repo
root. Nothing in `FileWalker.walk()` can ever construct that condition: `_iter_files`
recurses with `entry` -- the lexical, unresolved path -- never with `entry.resolve()`, so
every path these three functions see during a real walk is `repo_root / ...` by
construction, even when the entry is a symlink whose target lives elsewhere (see
`test_walker_containment.py`'s `TestOutOfTreeRejectionIsPolicyNotGap`, where
`_is_within_root` rejects an out-of-tree symlink BEFORE any of these three ever run).
(A fourth function sharing the same try/except shape, `_gitignore_chain`, has no
surviving mutant on this branch today -- it is not tested below for that reason, per the
project's own rule that a mutation-testing report is closed by verifying it, not by
adding a redundant test.)

The condition these except-ValueError branches guard against is real, though: a symlink
inside the repo whose RESOLVED target lies outside `self.repo_root` is exactly the shape
their own comments name ("a walk can only get here through a symlink" --
`_under_store_path`). `_outside_target` below builds that symlink for real and returns its
resolved, out-of-root target, and each test calls its helper with that path directly --
the same way a caller auditing a RESOLVED path, rather than the lexical one `_iter_files`
always uses, would.

`_conditional_evidence`'s `except OSError` fallback (a failed `parent.iterdir()`) is a
separate, fourth case, guarding a different condition: it is only reached when
`self._conditional_cache` has no entry yet for `parent`. Every real `walk()` primes that
cache one call earlier via `_prime_conditional_cache`, so the cache is never cold when
`_is_ignored_dir` calls `_conditional_evidence` during a walk. `_conditional_evidence`'s
own docstring says the cold path exists "on the watcher paths," which call `_is_ignored_dir`
without a preceding `_iter_files` -- so a fresh `FileWalker` (empty cache) with
`parent.iterdir()` monkeypatched to fail models that directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.core.models import Language
from trelix.indexing.walker import FileWalker


def _walker(repo: Path) -> FileWalker:
    return FileWalker(
        IndexConfig(repo_path=str(repo), walker=WalkerConfig(languages=[Language.PYTHON]))
    )


def _outside_target(tmp_path: Path) -> Path:
    """`repo/link -> outside/node_modules`, returned as the RESOLVED, out-of-root path.

    `repo` and `outside` are siblings under `tmp_path`, so the resolved target shares no
    path prefix with `repo` at all -- the strict condition `relative_to` fails on.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    (outside / "node_modules").mkdir(parents=True)
    (outside / "node_modules" / "x.py").write_text("x = 1\n", encoding="utf-8")
    link = repo / "link"
    link.symlink_to(outside, target_is_directory=True)
    return (link / "node_modules").resolve()


class TestUnderStorePathOutsideRepoRoot:
    def test_a_resolved_symlink_target_outside_root_is_judged_on_its_own_segments(
        self, tmp_path: Path
    ) -> None:
        """MUTATION: the except-ValueError fallback `rel = path` mutated to `rel = None`
        (mutmut's own survivor here) makes the very next line, `rel.parts`, raise
        `AttributeError` -- so this also kills a mutant that simply crashes whenever the
        branch runs, which no existing test ever makes happen.

        No existing test calls `_under_store_path` with a path outside `self.repo_root`:
        every parametrisation in test_walker_store_path_segments.py builds its fixture
        INSIDE the repo, where `relative_to` always succeeds and the `try` branch runs.
        """
        outside_path = _outside_target(tmp_path)
        walker = _walker(tmp_path / "repo")

        assert walker._under_store_path(outside_path) is True

    def test_an_outside_path_with_no_store_segment_is_not_flagged(self, tmp_path: Path) -> None:
        """Discrimination control: the fallback must not just always return True."""
        repo = tmp_path / "repo"
        repo.mkdir()
        outside = tmp_path / "elsewhere" / "plain_dir"
        outside.mkdir(parents=True)
        walker = _walker(repo)

        assert walker._under_store_path(outside) is False


class TestRecordIncompleteOutsideRepoRoot:
    def test_an_out_of_root_path_is_recorded_by_its_own_absolute_string(
        self, tmp_path: Path
    ) -> None:
        """MUTATION: the except-ValueError fallback `rel = str(path)` mutated to
        `rel = None` or to `rel = str(None)` (mutmut's own two survivors here), either of
        which records a path-less (or identical-looking, wrong) entry instead of the
        actual path that was skipped.

        Every existing `TestWalkCompleteness` fixture (test_walker.py) puts the unreadable
        entry INSIDE the repo, so `relative_to` always succeeds there and only the `try`
        branch (a repo-relative string) is exercised.
        """
        outside_path = _outside_target(tmp_path)
        walker = _walker(tmp_path / "repo")

        walker._record_incomplete(outside_path, OSError("simulated read failure"))

        assert walker.incomplete_paths == [str(outside_path)]


class TestReportConditionalOutsideRepoRoot:
    def test_an_out_of_root_dir_is_deduped_by_its_own_absolute_string(self, tmp_path: Path) -> None:
        """MUTATION: the except-ValueError fallback `rel = str(dir_path)` mutated to
        `rel = None` or to `rel = str(None)` (mutmut's own two survivors here), either of
        which would dedupe every out-of-root directory under the same key instead of by
        its own location -- reporting the second one as already having been reported.

        No existing test calls `_report_conditional` with a path outside
        `self.repo_root`; every call inside a real walk reaches it through
        `_is_ignored_dir(entry)`, and `entry` is always lexically under `repo_root` (see
        the module docstring above).
        """
        outside_path = _outside_target(tmp_path)
        walker = _walker(tmp_path / "repo")

        walker._report_conditional(outside_path, "package.json")

        assert walker._reported_conditional == {str(outside_path)}


class TestConditionalEvidenceColdCache:
    def test_an_unreadable_parent_yields_no_evidence_without_crashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `_conditional_evidence`'s except-OSError fallback `entries = []`
        mutated to `entries = None` (mutmut's own survivor here) makes the very next
        line, `_classify_conditional_dirs(parent, entries)`'s dict-comprehension `for
        entry in entries`, raise `TypeError: 'NoneType' object is not iterable` --
        exactly the crash the task brief names ("would raise TypeError on a real
        directory-read failure").

        Only reached with a COLD `_conditional_cache`: every `walk()` call primes it one
        level up via `_prime_conditional_cache` before `_is_ignored_dir` ever asks, so no
        walk-level test can hit this branch (see the module docstring above). The watcher
        paths call `_is_ignored_dir` without that priming step, which is what a fresh,
        never-walked `FileWalker` plus a failing `iterdir()` models here.
        """
        repo = tmp_path / "repo"
        (repo / "packages").mkdir(parents=True)
        walker = _walker(repo)
        assert repo not in walker._conditional_cache, "precondition: the cache must be cold"

        real_iterdir = Path.iterdir

        def failing_iterdir(self: Path) -> Iterator[Path]:
            if self == repo:
                raise PermissionError(13, "Permission denied", str(self))
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", failing_iterdir)

        result = walker._conditional_evidence(repo / "packages")

        assert result is None
        assert walker._conditional_cache[repo] == {}
