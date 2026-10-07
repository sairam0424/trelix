"""The pinned clone of a suite: made at the pin, re-verified when it exists, never repaired.

The git environment and the hostile-configuration cases are in
`test_eval_suite_git_isolation.py`; the gold paths are in `test_eval_suite_gold.py`. The
"remote" is a real two-commit repository built in a temp directory (`eval_suite_harness.py`):
`first` has `src/app.py` and no `src/extra.py`, `second` has both, so a checkout of the branch
head and a checkout of the pin give opposite answers.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `checkout --detach <sha>` replaced by nothing (the branch head stays)   (TestAtThePin)
2. `check_clone`: the HEAD, origin URL, shallow, partial, worktree or `.git` test removed, or
   any ONE of the three partial-clone settings left out  (TestAnExistingCloneIsVerifiedNotRepaired)
3. `--ignored` or `--untracked-files=all` dropped from the status call   (the ignored case)
4. the stale-`.partial` refusal removed, or `.partial` deleted when found, or a parent that
   cannot be made blamed on a `.partial`, or a `.partial` beside a verified clone passed over
                                                                           (TestPartialDirectory)
5. the `_discard` on failure removed, or the `OSError` branch of the rename removed, or the
   check that HEAD is the pin after the checkout removed      (TestAFailedCloneLeavesNothing)
6. a timeout, a missing git or an OSError not turned into a SuiteError, or Ctrl-C leaving the
   `.partial` behind                                                       (TestGitFailures)
7. the clone made with `--depth` or `--filter`                   (test_the_clone_is_complete)
8. `clone_path` layout changed                                   (test_the_clone_lives_at_...)
9. `tracked_files` listing HEAD instead of the sha it is given   (test_the_files_at_the_pin_...)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.unit.eval_suite_harness import (
    Remote,
    clone_local,
    git,
    load_local,
    make_remote,
    record_git,
    write_suite,
)
from tests.unit.eval_suite_harness import remote as remote
from trelix.eval.suite import SuiteError, SuiteSpec
from trelix.eval.suite_git import check_clone, clone_path, tracked_files

_FIRST_APP = "def login(user):\n    return user\n"


def _spec(tmp_path: Path, url: Path | str, sha: str) -> SuiteSpec:
    return load_local(write_suite(tmp_path / "suite", str(url), sha))


@pytest.fixture(scope="module")
def cloned_cache(tmp_path_factory: pytest.TempPathFactory, remote: Remote) -> Path:
    """A cache holding one clone made by the code under test, at the first commit of `remote`.

    Tests that change a clone work on a copy, so each pays a directory copy and not a clone.
    """
    cache = tmp_path_factory.mktemp("eval-suite-cache") / "cache"
    suite = write_suite(cache.parent / "suite", str(remote.path), remote.first)
    clone_local(load_local(suite), cache)
    return cache


@pytest.fixture
def cloned(tmp_path: Path, remote: Remote, cloned_cache: Path) -> tuple[SuiteSpec, Path]:
    """A verified clone of `remote` at its first commit: the spec and the clone's path."""
    spec = _spec(tmp_path, remote.path, remote.first)
    shutil.copytree(cloned_cache, tmp_path / "cache", symlinks=True)
    return spec, clone_path(tmp_path / "cache", spec)


def _problems(spec: SuiteSpec, cache: Path) -> str:
    with pytest.raises(SuiteError) as caught:
        clone_local(spec, cache)
    return " | ".join(caught.value.problems)


class TestAtThePin:
    def test_the_clone_is_at_the_pin_not_at_the_branch_head(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        """The branch head is `second`, which has `src/extra.py`. A checkout of it fails this."""
        spec = _spec(tmp_path, remote.path, remote.first)
        clone = clone_local(spec, tmp_path / "cache")
        assert (clone / "src" / "app.py").read_text() == _FIRST_APP
        assert not (clone / "src" / "extra.py").exists()
        assert git(clone, "rev-parse", "HEAD") == remote.first
        assert git(clone, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"  # detached

    def test_a_pin_on_the_newer_commit_gets_the_newer_files(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        clone = clone_local(_spec(tmp_path, remote.path, remote.second), tmp_path / "cache")
        assert (clone / "src" / "extra.py").read_text() == "X = 1\n"

    def test_the_clone_lives_at_clones_sha_name(self, tmp_path: Path, remote: Remote) -> None:
        spec = _spec(tmp_path, remote.path, remote.first)
        clone = clone_local(spec, tmp_path / "cache")
        assert clone == tmp_path / "cache" / "clones" / remote.first / "demo"
        assert clone_path(tmp_path / "cache", spec) == clone
        assert [p.name for p in (tmp_path / "cache").iterdir()] == ["clones"]
        assert [p.name for p in (tmp_path / "cache" / "clones").iterdir()] == [remote.first]
        assert [p.name for p in clone.parent.iterdir()] == ["demo"]  # the .partial is gone

    def test_the_clone_is_complete(self, tmp_path: Path, remote: Remote) -> None:
        clone = clone_local(_spec(tmp_path, remote.path, remote.first), tmp_path / "cache")
        assert git(clone, "rev-parse", "--is-shallow-repository") == "false"
        assert git(clone, "config", "--local", "--list").count("promisor") == 0
        assert git(clone, "rev-list", "--count", "--all") == "2"

    def test_a_file_url_clones_the_same_way(self, tmp_path: Path, remote: Remote) -> None:
        spec = _spec(tmp_path, f"file://{remote.path}", remote.first)
        clone = clone_local(spec, tmp_path / "cache")
        assert git(clone, "config", "--get", "remote.origin.url") == f"file://{remote.path}"

    def test_the_files_at_the_pin_are_listed_not_the_files_at_the_head(
        self, cloned: tuple[SuiteSpec, Path], remote: Remote
    ) -> None:
        _, clone = cloned
        assert tracked_files(clone, remote.first) == {
            "README.md",
            "conftest.py",
            "setup.py",
            "sitecustomize.py",
            "src/app.py",
        }
        assert "src/extra.py" in tracked_files(clone, remote.second)  # HEAD is `first`

    def test_a_second_call_verifies_and_reuses_without_cloning(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec, clone = cloned
        calls = record_git(monkeypatch)
        assert clone_local(spec, tmp_path / "cache") == clone
        assert [call.argv[5] for call in calls] == ["rev-parse", "config", "status"]


class TestAnExistingCloneIsVerifiedNotRepaired:
    def test_a_moved_head_is_refused_naming_both_commits_and_left_alone(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path, remote: Remote
    ) -> None:
        spec, clone = cloned
        git(clone, "checkout", "-q", "--detach", remote.second)
        text = _problems(spec, tmp_path / "cache")
        assert f"HEAD is {remote.second}, not the pinned {remote.first}" in text
        assert f"remove {clone} and run again" in text
        assert git(clone, "rev-parse", "HEAD") == remote.second  # not repaired

    @pytest.mark.parametrize(
        ("label", "change", "entry"),
        [
            ("modified", lambda c: (c / "src" / "app.py").write_text("x = 2\n"), "M src/app.py"),
            ("untracked", lambda c: (c / "stray.txt").write_text("x"), "?? stray.txt"),
            ("deleted", lambda c: (c / "README.md").unlink(), "D README.md"),
            (
                "ignored",
                lambda c: (
                    (c / ".git" / "info" / "exclude").write_text("ignored.txt\n"),
                    (c / "ignored.txt").write_text("x"),
                ),
                "!! ignored.txt",
            ),
        ],
    )
    def test_a_worktree_that_is_not_pristine_is_refused_and_not_cleaned(
        self,
        cloned: tuple[SuiteSpec, Path],
        tmp_path: Path,
        label: str,
        change: Any,
        entry: str,
    ) -> None:
        """An ignored file is refused too: the walker indexes what git ignores."""
        spec, clone = cloned
        change(clone)
        before = sorted(p.name for p in clone.iterdir())
        text = _problems(spec, tmp_path / "cache")
        assert f"the worktree is not pristine (1 entries: {entry})" in text
        assert sorted(p.name for p in clone.iterdir()) == before

    def test_more_than_five_entries_are_counted(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        spec, clone = cloned
        for n in range(7):
            (clone / f"stray{n}.txt").write_text("x")
        text = _problems(spec, tmp_path / "cache")
        assert "7 entries: ?? stray0.txt; ?? stray1.txt; ?? stray2.txt; ?? stray3.txt; " in text
        assert "?? stray4.txt; and 2 more)" in text

    def test_an_untracked_directory_is_listed_file_by_file(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        spec, clone = cloned
        (clone / "newdir").mkdir()
        (clone / "newdir" / "a.txt").write_text("x")
        (clone / "newdir" / "b.txt").write_text("x")
        assert "the worktree is not pristine (2 entries: ?? newdir/a.txt; ?? newdir/b.txt)" in (
            _problems(spec, tmp_path / "cache")
        )

    def test_a_changed_origin_url_is_refused(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        spec, clone = cloned
        git(clone, "remote", "set-url", "origin", "/somewhere/else.git")
        assert "the origin URL is ['/somewhere/else.git'], not" in _problems(
            spec, tmp_path / "cache"
        )

    def test_a_second_origin_url_is_refused(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        spec, clone = cloned
        git(clone, "config", "--add", "remote.origin.url", "/second.git")
        assert "the origin URL is" in _problems(spec, tmp_path / "cache")

    def test_a_shallow_clone_is_refused(self, tmp_path: Path, remote: Remote) -> None:
        """At the pin, with the right URL and a pristine worktree: only `shallow` is wrong."""
        url = f"file://{remote.path}"
        spec = _spec(tmp_path, url, remote.second)
        final = clone_path(tmp_path / "cache", spec)
        final.parent.mkdir(parents=True)
        git(tmp_path, "clone", "-q", "--depth", "1", url, str(final))
        assert check_clone(final, spec) == [
            f"clone {final}: it is a shallow clone, which git may complete from the network"
        ]
        assert "it is a shallow clone" in _problems(spec, tmp_path / "cache")

    def test_a_partial_clone_is_refused(self, tmp_path: Path) -> None:
        source = make_remote(tmp_path, name="filterable")
        git(source.path, "config", "uploadpack.allowFilter", "true")
        url = f"file://{source.path}"
        spec = _spec(tmp_path, url, source.second)
        final = clone_path(tmp_path / "cache", spec)
        final.parent.mkdir(parents=True)
        git(tmp_path, "clone", "-q", "--filter=blob:none", url, str(final))
        assert check_clone(final, spec) == [
            f"clone {final}: it is a partial clone, which git may complete from the network"
        ]

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("extensions.partialclone", "origin"),
            ("remote.origin.promisor", "true"),
            ("remote.origin.partialclonefilter", "blob:none"),
            ("remote.upstream.promisor", "true"),
        ],
    )
    def test_any_one_partial_clone_setting_is_enough_to_refuse(
        self, cloned: tuple[SuiteSpec, Path], key: str, value: str
    ) -> None:
        """A complete clone plus ONE setting: a check on any other setting cannot refuse it."""
        spec, clone = cloned
        git(clone, "config", key, value)
        assert check_clone(clone, spec) == [
            f"clone {clone}: it is a partial clone, which git may complete from the network"
        ]

    def test_a_directory_without_a_git_directory_of_its_own_is_refused(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        """Inside a repository, git would answer for the enclosing one."""
        spec = _spec(tmp_path, remote.path, remote.first)
        enclosing = tmp_path / "enclosing"
        enclosing.mkdir()
        git(enclosing, "init", "-q")
        inner = enclosing / "inner"
        inner.mkdir()
        assert check_clone(inner, spec) == [
            f"clone {inner} is not a git repository of its own (it has no real .git directory)"
        ]

    def test_a_symlink_in_place_of_the_clone_is_refused(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        spec, clone = cloned
        elsewhere = tmp_path / "elsewhere"
        clone.rename(elsewhere)
        clone.symlink_to(elsewhere)
        assert "is not a git repository of its own" in _problems(spec, tmp_path / "cache")

    def test_a_git_directory_that_is_a_symlink_is_refused(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        spec, clone = cloned
        moved = tmp_path / "moved-git-directory"
        (clone / ".git").rename(moved)
        (clone / ".git").symlink_to(moved)
        assert check_clone(clone, spec) == [
            f"clone {clone} is not a git repository of its own (it has no real .git directory)"
        ]

    def test_a_dangling_symlink_in_place_of_the_clone_is_refused(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        spec = _spec(tmp_path, remote.path, remote.first)
        final = clone_path(tmp_path / "cache", spec)
        final.parent.mkdir(parents=True)
        final.symlink_to(tmp_path / "nowhere")
        assert "is not a git repository of its own" in _problems(spec, tmp_path / "cache")

    def test_a_broken_git_directory_is_reported_not_raised(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        spec, clone = cloned
        (clone / ".git" / "HEAD").unlink()  # the clone is broken either way: say so, don't crash
        assert _problems(spec, tmp_path / "cache").startswith(f"clone {clone}:")


class TestPartialDirectory:
    def test_a_stale_partial_is_refused_with_its_path_and_never_deleted(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        spec = _spec(tmp_path, remote.path, remote.first)
        partial = tmp_path / "cache" / "clones" / remote.first / "demo.partial"
        partial.mkdir(parents=True)
        (partial / "keep.txt").write_text("from a killed run")
        text = _problems(spec, tmp_path / "cache")
        assert f"{partial} is in the way" in text
        assert "eval-suite never deletes it" in text
        assert (partial / "keep.txt").read_text() == "from a killed run"
        assert not (partial.parent / "demo").exists()

    def test_a_stale_partial_beside_a_verified_clone_is_refused_with_the_same_text(
        self, cloned: tuple[SuiteSpec, Path], tmp_path: Path
    ) -> None:
        """A clone that verifies does not hide a `.partial` beside it: the operator hears about it
        on every run until it is removed, in the same words as when no clone exists yet."""
        spec, clone = cloned
        partial = clone.with_name("demo.partial")
        partial.mkdir()
        (partial / "keep.txt").write_text("from a killed run")
        assert _problems(spec, tmp_path / "cache") == (
            f"{partial} is in the way: a run that was killed left it, or another eval-suite "
            "is cloning now. eval-suite never deletes it; remove it and run again"
        )
        assert (partial / "keep.txt").read_text() == "from a killed run"
        assert check_clone(clone, spec) == []

    def test_a_partial_that_is_a_file_is_refused_too(self, tmp_path: Path, remote: Remote) -> None:
        spec = _spec(tmp_path, remote.path, remote.first)
        partial = tmp_path / "cache" / "clones" / remote.first / "demo.partial"
        partial.parent.mkdir(parents=True)
        partial.write_text("x")
        assert f"{partial} is in the way" in _problems(spec, tmp_path / "cache")

    def test_a_file_in_place_of_the_sha_directory_is_not_blamed_on_a_partial(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        """What exists is `<cache>/clones/<sha>`, as a file; no `.partial` does."""
        spec = _spec(tmp_path, remote.path, remote.first)
        sha_dir = tmp_path / "cache" / "clones" / remote.first
        sha_dir.parent.mkdir(parents=True)
        sha_dir.write_text("x")
        text = _problems(spec, tmp_path / "cache")
        assert text.startswith(f"cannot create {sha_dir / 'demo.partial'}: ")
        assert "is in the way" not in text
        assert sha_dir.read_text() == "x"


class TestAFailedCloneLeavesNothing:
    def _assert_nothing_left(self, cache: Path, spec: SuiteSpec) -> None:
        final = clone_path(cache, spec)
        assert not final.exists()
        assert not final.with_name("demo.partial").exists()

    def test_an_unknown_pin_is_refused_and_leaves_no_partial(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        spec = _spec(tmp_path, remote.path, "0" * 40)
        text = _problems(spec, tmp_path / "cache")
        assert "git checkout failed (exit 128)" in text
        self._assert_nothing_left(tmp_path / "cache", spec)

    def test_a_missing_repository_is_refused_and_leaves_no_partial(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path, tmp_path / "no-such-repository", "1" * 40)
        assert "git clone failed (exit 128)" in _problems(spec, tmp_path / "cache")
        self._assert_nothing_left(tmp_path / "cache", spec)

    def test_a_pin_that_is_an_annotated_tag_is_refused_by_the_check_after_the_checkout(
        self, tmp_path: Path
    ) -> None:
        """`checkout --detach <tag object id>` succeeds and lands on the tag's commit, so only
        the HEAD check that follows it can refuse a pin that is not a commit id."""
        tagged = make_remote(tmp_path, name="tagged")
        git(tagged.path, "tag", "-a", "-m", "v1", "v1", tagged.first)
        tag_object = git(tagged.path, "rev-parse", "v1")
        spec = _spec(tmp_path, tagged.path, tag_object)
        text = _problems(spec, tmp_path / "cache")
        assert f"HEAD is {tagged.first}, not the pinned {tag_object}" in text
        self._assert_nothing_left(tmp_path / "cache", spec)

    def test_a_commit_beyond_a_shallow_remote_is_refused_and_leaves_no_partial(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        shallow = tmp_path / "shallow-source"
        git(tmp_path, "clone", "-q", "--depth", "1", f"file://{remote.path}", str(shallow))
        spec = _spec(tmp_path, shallow, remote.first)  # `first` is below the boundary
        assert "git checkout failed" in _problems(spec, tmp_path / "cache")
        self._assert_nothing_left(tmp_path / "cache", spec)

    def test_a_pin_the_branch_no_longer_reaches_is_refused_and_leaves_no_partial(
        self, tmp_path: Path
    ) -> None:
        """A force-pushed branch: the pinned commit still exists in the remote, but it is not
        advertised, so the clone does not receive it. (A local-path clone copies every object,
        which is why this one goes through a `file://` URL, as an https remote would.)"""
        rewritten = make_remote(tmp_path, name="rewritten")
        git(rewritten.path, "checkout", "-q", "--orphan", "fresh")
        (rewritten.path / "new.txt").write_text("history was rewritten\n")
        git(rewritten.path, "add", "-A", "-f")
        git(rewritten.path, "commit", "-q", "-m", "fresh root")
        git(rewritten.path, "branch", "-q", "-D", "main")
        git(rewritten.path, "branch", "-q", "-m", "main")
        spec = _spec(tmp_path, f"file://{rewritten.path}", rewritten.first)
        assert "git checkout failed" in _problems(spec, tmp_path / "cache")
        self._assert_nothing_left(tmp_path / "cache", spec)

    def test_a_failed_rename_is_a_suite_error_and_leaves_no_partial(
        self, tmp_path: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(src: Any, dst: Any) -> None:
            raise PermissionError("read-only cache")

        monkeypatch.setattr("trelix.eval.suite_git.os.replace", refuse)
        spec = _spec(tmp_path, remote.path, remote.first)
        assert "cannot move the verified clone into place at" in _problems(spec, tmp_path / "cache")
        self._assert_nothing_left(tmp_path / "cache", spec)

    def test_a_partial_that_cannot_be_removed_is_logged_and_the_cause_still_raised(
        self,
        tmp_path: Path,
        remote: Remote,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def stuck(path: Any) -> None:
            raise PermissionError("busy")

        monkeypatch.setattr("trelix.eval.suite_git.shutil.rmtree", stuck)
        spec = _spec(tmp_path, remote.path, "0" * 40)
        with caplog.at_level(logging.WARNING, logger="trelix.eval.suite"):
            assert "git checkout failed" in _problems(spec, tmp_path / "cache")
        assert "could not remove the failed partial clone" in caplog.text


class _Raises:
    """A `subprocess` whose `run` raises `exc`; everything else is the real module's."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def run(self, *args: Any, **kwargs: Any) -> None:
        raise self._exc

    def __getattr__(self, name: str) -> Any:
        return getattr(subprocess, name)


class TestGitFailures:
    def _fail_with(self, monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
        monkeypatch.setattr("trelix.eval.suite_git.subprocess", _Raises(exc))

    def test_a_clone_that_times_out_is_a_suite_error_and_leaves_no_partial(
        self, tmp_path: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._fail_with(monkeypatch, subprocess.TimeoutExpired("git", 1800))
        spec = _spec(tmp_path, remote.path, remote.first)
        assert _problems(spec, tmp_path / "cache") == "git clone timed out after 1800 seconds"
        assert not clone_path(tmp_path / "cache", spec).with_name("demo.partial").exists()

    def test_an_interrupted_clone_removes_its_partial_and_is_not_swallowed(
        self, tmp_path: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._fail_with(monkeypatch, KeyboardInterrupt())
        spec = _spec(tmp_path, remote.path, remote.first)
        with pytest.raises(KeyboardInterrupt):
            clone_local(spec, tmp_path / "cache")
        assert not clone_path(tmp_path / "cache", spec).with_name("demo.partial").exists()

    def test_any_other_command_times_out_after_120_seconds(
        self, cloned: tuple[SuiteSpec, Path], remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, clone = cloned
        self._fail_with(monkeypatch, subprocess.TimeoutExpired("git", 120))
        with pytest.raises(SuiteError) as caught:
            tracked_files(clone, remote.first)
        assert caught.value.problems == ("git ls-tree timed out after 120 seconds",)

    def test_a_missing_git_is_a_clear_error(
        self, tmp_path: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        spec = _spec(tmp_path, remote.path, remote.first)
        assert _problems(spec, tmp_path / "cache") == (
            "git was not found on PATH: eval-suite needs it to clone the suite's repository"
        )

    def test_any_os_error_is_a_suite_error(
        self, tmp_path: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._fail_with(monkeypatch, PermissionError("denied"))
        spec = _spec(tmp_path, remote.path, remote.first)
        assert _problems(spec, tmp_path / "cache").startswith("cannot run git: ")
