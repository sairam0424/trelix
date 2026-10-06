"""Gold paths: in the repository at the pin, and in the set the run will index.

A gold file that exists in git and is not indexed scores 0 in every arm and looks exactly like
a retrieval miss. The default walker ignores a directory called `packages`, among others, and
follows symlinks without resolving them, so these tests use a repository with one of each.

The symlinks point at a file OUTSIDE the clone that holds a canary. The control shows the
default walker reads it; the test shows the configuration of the run does not. Nothing links
to a device file: a mutant that reads it would hang the suite.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `follow_symlinks=False` removed from the walker override     (TestSymlinksLeaveNoTrace)
2. the walker check removed, or run against the default config  (TestGoldPathsMustBeIndexed)
3. the list cut at six instead of five, or the count dropped   (test_at_most_five_paths_...)
4. the exists-at-the-pin check skipped, or made against HEAD    (test_a_path_missing_at_the_pin_...)
5. `walk_was_complete` ignored                                   (test_a_walk_that_could_not_...)
6. the `ValueError` of the configuration not turned into a SuiteError
                                                               (test_a_configuration_that_...)
7. `ValueError`, `RecursionError` or `re.error` left out of what a walk may raise, the walk
   not inside the `try`, the `from exc` dropped, the error type or the clip left out of the
   message                                        (TestAHostileRepositoryIsRefusedNotCrashed)
8. either name left out of the two the walker opens by name, the name compared case-sensitively,
   the `is_symlink()` test dropped, the check moved after the walk, the list cut at six or left
   unsorted, the count dropped, the clip left out        (TestLinksTheWalkerOpensByName)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest

from tests.unit.eval_suite_harness import (
    FIRST_FILES,
    Remote,
    clone_local,
    load_local,
    make_remote,
    write_suite,
)
from tests.unit.eval_suite_harness import remote as remote
from trelix.core.config import IndexConfig
from trelix.eval.suite import SuiteError, SuiteSpec
from trelix.eval.suite_gold import GoldCounts, check_gold, indexed_files
from trelix.indexing.walker import FileWalker

_CANARY = "CANARY = 'outside the clone'\n"
_MANY = [f"packages/pkg/m{n}.py" for n in range(1, 8)]
_FILES = {
    **FIRST_FILES,
    "packages/pkg/mod.py": "X = 1\n",
    **{path: "X = 1\n" for path in _MANY},
    "assets/app.min.js": "var x=1;\n",
    ".gitignore": "generated.py\n",
    "generated.py": "X = 1\n",
    "big.py": "x = 1\n" * 100_000,
}
_SECOND = {"src/extra.py": "X = 1\n"}

_NOT_INDEXED = (
    "is in the repository but the walker does not index it "
    "(an ignored directory, extension or language, or too large)"
)


@pytest.fixture(scope="module")
def gold_remote(tmp_path_factory: pytest.TempPathFactory) -> Remote:
    """A repository holding, besides the demo files, one gold file of each unindexed kind."""
    parent = tmp_path_factory.mktemp("eval-suite-gold")
    outside = parent / "outside"
    outside.mkdir()
    (outside / "canary.py").write_text(_CANARY, encoding="utf-8")
    links = {
        "src/leak.py": outside / "canary.py",
        "rootlink.py": outside / "canary.py",
        "linkdir": outside,
    }
    return make_remote(parent, _FILES, _SECOND, name="gold-remote", links=links)


def _suite(tmp_path: Path, remote: Remote, *paths: str) -> SuiteSpec:
    golden = [{"query": f"question {n}", "relevant_files": [p]} for n, p in enumerate(paths, 1)]
    suite = write_suite(tmp_path / "suite", str(remote.path), remote.first, golden=golden)
    return load_local(suite)


def _refusal(spec: SuiteSpec, tmp_path: Path) -> list[str]:
    clone = clone_local(spec, tmp_path / "cache")
    with pytest.raises(SuiteError) as caught:
        check_gold(spec, clone)
    return list(caught.value.problems)


class TestGoldPathsMustBeIndexed:
    def test_indexed_gold_files_are_counted(self, tmp_path: Path, remote: Remote) -> None:
        suite = write_suite(tmp_path / "suite", str(remote.path), remote.first)
        spec = load_local(suite)
        clone = clone_local(spec, tmp_path / "cache")
        assert check_gold(spec, clone) == GoldCounts(queries=2, files=2)

    def test_distinct_gold_files_are_counted_once(self, tmp_path: Path, remote: Remote) -> None:
        golden = [
            {"query": "a", "relevant_files": ["src/app.py", "README.md"]},
            {"query": "b", "relevant_files": ["src/app.py"]},
            {"query": "c", "relevant_files": ["README.md"]},
        ]
        suite = write_suite(tmp_path / "suite", str(remote.path), remote.first, golden=golden)
        spec = load_local(suite)
        clone = clone_local(spec, tmp_path / "cache")
        assert check_gold(spec, clone) == GoldCounts(queries=3, files=2)

    @pytest.mark.parametrize(
        "path",
        [
            "packages/pkg/mod.py",
            "assets/app.min.js",
            "generated.py",
            "big.py",
            "src/leak.py",
            "rootlink.py",
            "linkdir",
        ],
    )
    def test_a_gold_path_the_walker_does_not_index_is_refused_naming_the_path(
        self, tmp_path: Path, gold_remote: Remote, path: str
    ) -> None:
        """An ignored directory, an ignored extension, a .gitignore line, a file over 500,000
        bytes, and three tracked symlinks that leave the clone."""
        spec = _suite(tmp_path, gold_remote, path)
        assert _refusal(spec, tmp_path) == [f"golden: line 1: '{path}' {_NOT_INDEXED}"]

    def test_the_same_suite_passes_when_the_gold_file_is_indexed(
        self, tmp_path: Path, gold_remote: Remote
    ) -> None:
        spec = _suite(tmp_path, gold_remote, "src/app.py")
        clone = clone_local(spec, tmp_path / "cache")
        assert check_gold(spec, clone) == GoldCounts(queries=1, files=1)

    def test_at_most_five_paths_are_listed_and_the_rest_counted(
        self, tmp_path: Path, gold_remote: Remote
    ) -> None:
        spec = _suite(tmp_path, gold_remote, *_MANY)
        assert _refusal(spec, tmp_path) == [
            *(f"golden: line {n}: '{path}' {_NOT_INDEXED}" for n, path in enumerate(_MANY[:5], 1)),
            "golden: and 2 more gold paths are not indexed",
        ]

    def test_exactly_five_are_listed_without_a_count(
        self, tmp_path: Path, gold_remote: Remote
    ) -> None:
        spec = _suite(tmp_path, gold_remote, *_MANY[:5])
        assert len(_refusal(spec, tmp_path)) == 5

    def test_the_paths_of_one_entry_are_listed_in_sorted_order(
        self, tmp_path: Path, gold_remote: Remote
    ) -> None:
        """An entry holds its gold files as a frozenset, whose order changes with the hash seed of
        each process. With 8 paths an unsorted listing would put the five it shows in sorted order
        by chance about once in 6,700 runs (with 3 paths, once in 6)."""
        paths = ["packages/pkg/mod.py", *reversed(_MANY)]
        golden = [{"query": "q", "relevant_files": paths}]
        suite = write_suite(
            tmp_path / "suite", str(gold_remote.path), gold_remote.first, golden=golden
        )
        assert _refusal(load_local(suite), tmp_path) == [
            *(f"golden: line 1: '{path}' {_NOT_INDEXED}" for path in _MANY[:5]),
            "golden: and 3 more gold paths are not indexed",
        ]

    def test_a_path_missing_at_the_pin_is_refused_against_the_pin_not_the_head(
        self, tmp_path: Path, gold_remote: Remote
    ) -> None:
        """`src/extra.py` exists in the second commit only; the pin is the first."""
        spec = _suite(tmp_path, gold_remote, "src/extra.py")
        assert _refusal(spec, tmp_path) == [
            f"golden: line 1: \"relevant_files\" path 'src/extra.py' does not exist at "
            f"{gold_remote.first}"
        ]


class TestSymlinksLeaveNoTrace:
    def test_the_default_walker_reads_the_file_outside_the_clone(
        self, tmp_path: Path, gold_remote: Remote
    ) -> None:
        """The control: without the override, the canary is indexed under a name inside."""
        clone = clone_local(_suite(tmp_path, gold_remote, "src/app.py"), tmp_path / "cache")
        walked = {f.rel_path for f in FileWalker(IndexConfig(repo_path=str(clone))).walk()}
        assert {"src/leak.py", "rootlink.py", "linkdir/canary.py"} <= walked
        assert (clone / "src" / "leak.py").read_text() == _CANARY

    def test_the_configuration_of_the_run_walks_nothing_outside_the_clone(
        self, tmp_path: Path, gold_remote: Remote
    ) -> None:
        clone = clone_local(_suite(tmp_path, gold_remote, "src/app.py"), tmp_path / "cache")
        walked = indexed_files(clone)
        assert "src/app.py" in walked
        assert not {"src/leak.py", "rootlink.py", "linkdir", "linkdir/canary.py"} & walked
        assert all("canary" not in path for path in walked)


class TestTheWalkMustBeComplete:
    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a directory with no permissions")
    def test_a_walk_that_could_not_read_a_directory_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "ok.py").write_text("X = 1\n")
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "hidden.py").write_text("X = 1\n")
        locked.chmod(0)
        try:
            with pytest.raises(SuiteError) as caught:
                indexed_files(tmp_path)
        finally:
            locked.chmod(0o700)
        assert caught.value.problems == (
            "the walker could not read 1 path(s) of the clone, so the set of indexed files is "
            "incomplete: locked",
        )

    def test_a_configuration_that_cannot_be_built_is_a_suite_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_WALKER_MAX_FILE_SIZE_BYTES", "a lot")
        with pytest.raises(SuiteError) as caught:
            indexed_files(tmp_path)
        assert caught.value.problems[0].startswith("cannot build the index configuration: ")


class TestAHostileRepositoryIsRefusedNotCrashed:
    """The clone is untrusted: what its own `.gitignore` or `package.json` makes the walker raise
    is a refusal, and not a traceback that comes back on every run.

    Each case plants a file the walker does not survive, and shows the error it raises: pathspec
    rejects the lone `!` with a ValueError, compiles `[z-a]` into a regex `re` rejects (a
    `re.error`, which is not a ValueError), and the JSON decoder gives up on a `package.json`
    nested 200,000 levels deep (a RecursionError, which is not either). The `packages` directory
    beside it is what makes the walker read that `package.json`.
    """

    @pytest.mark.parametrize(
        ("files", "raised"),
        [
            ({".gitignore": "build/\n!\n"}, ValueError),
            ({".gitignore": "[z-a]\n"}, re.error),
            ({"package.json": "[" * 200_000, "packages/pkg/mod.py": "X = 1\n"}, RecursionError),
        ],
        ids=["gitignore-lone-bang", "gitignore-bad-range", "package-json-too-deep"],
    )
    def test_what_the_walk_raises_on_the_clone_is_a_refusal(
        self, tmp_path: Path, files: dict[str, str], raised: type[Exception]
    ) -> None:
        (tmp_path / "ok.py").write_text("X = 1\n", encoding="utf-8")
        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text, encoding="utf-8")
        with pytest.raises(SuiteError) as caught:
            indexed_files(tmp_path)
        assert isinstance(caught.value.__cause__, raised)
        [problem] = caught.value.problems
        assert problem.startswith("the walker failed on the clone: ")

    def test_the_refusal_names_the_error_and_is_one_clipped_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A message that came from the clone cannot start a line of its own, and is cut."""

        def fail(self: FileWalker) -> None:
            raise ValueError("first\nrefused: forged " + "x" * 300)

        monkeypatch.setattr(FileWalker, "walk", fail)
        with pytest.raises(SuiteError) as caught:
            indexed_files(tmp_path)
        start = "the walker failed on the clone: ValueError: first refused: forged "
        assert caught.value.problems == (start + "x" * 175 + "...",)


_LEAN = {**FIRST_FILES, "packages/pkg/mod.py": "X = 1\n"}
_LINKED = (
    "the clone tracks {count} symlink(s) named .gitignore or package.json, which the walker "
    "opens by name and reads wherever they point: {listed}"
)


def _linked_remote(tmp_path: Path, target_text: str, *links: str) -> Remote:
    """The demo repository plus `links`, each a tracked symlink to one file outside it."""
    target = tmp_path / "outside" / "target.txt"
    target.parent.mkdir()
    target.write_text(target_text, encoding="utf-8")
    return make_remote(tmp_path, _LEAN, _SECOND, name="linked", links=dict.fromkeys(links, target))


def _no_walk(config: IndexConfig) -> None:
    raise AssertionError("the walker was started")


class TestLinksTheWalkerOpensByName:
    """`follow_symlinks = False` covers the files the walker iterates. It opens two more by name:
    `.gitignore` in every directory it enters and `package.json` beside a `packages` or `bin`
    directory, and it reads them wherever a tracked symlink points. So a tracked symlink with
    either name is refused, and before the walk starts: a `package.json` linked to a device or a
    FIFO never ends. Every target here is a plain file holding text, so a mutant that walks reads
    it and ends. The name is matched in any case: the walker asks for `.gitignore` by its fixed
    name, and a case-insensitive filesystem (the one these tests were written on) answers with a
    tracked `.GITIGNORE`; the control for that runs there only.
    """

    def test_the_walk_reads_a_linked_gitignore_outside_the_clone(self, tmp_path: Path) -> None:
        """The control: the target hides the gold file of the suite, and the walk obeys it."""
        remote = _linked_remote(tmp_path, "src/app.py\n", ".gitignore")
        clone = clone_local(_suite(tmp_path, remote, "src/app.py"), tmp_path / "cache")
        assert (clone / ".gitignore").is_symlink()
        assert "src/app.py" not in indexed_files(clone)

    def test_the_walk_reads_a_linked_gitignore_in_upper_case(self, tmp_path: Path) -> None:
        """The control for the case rule: on a case-insensitive filesystem the fixed name
        `.gitignore` the walker asks for is the tracked `.GITIGNORE`, and its target hides the
        gold file. A case-sensitive filesystem opens no such link, so the test skips there."""
        (tmp_path / "caseprobe").write_text("x", encoding="utf-8")
        if not (tmp_path / "CASEPROBE").exists():
            pytest.skip("case-sensitive filesystem: .GITIGNORE is not opened as .gitignore")
        remote = _linked_remote(tmp_path, "src/app.py\n", ".GITIGNORE")
        clone = clone_local(_suite(tmp_path, remote, "src/app.py"), tmp_path / "cache")
        assert (clone / ".GITIGNORE").is_symlink()
        assert "src/app.py" not in indexed_files(clone)

    def test_the_walk_reads_a_linked_package_json_outside_the_clone(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The control: the target declares a workspace, and the walker says so beside `packages`
        (it does not index that directory either way; it reports the manifest it read)."""
        remote = _linked_remote(tmp_path, '{"workspaces": ["packages/*"]}', "package.json")
        clone = clone_local(_suite(tmp_path, remote, "src/app.py"), tmp_path / "cache")
        assert (clone / "package.json").is_symlink()
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.walker"):
            indexed_files(clone)
        assert "package.json beside it declares this directory as first-party source" in caplog.text

    @pytest.mark.parametrize(
        "link",
        [
            ".gitignore",
            "package.json",
            "sub/.gitignore",
            "sub/package.json",
            ".GITIGNORE",
            "sub/PACKAGE.JSON",
        ],
    )
    def test_a_tracked_link_with_either_name_is_refused_before_the_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link: str
    ) -> None:
        monkeypatch.setattr("trelix.eval.suite_gold.FileWalker", _no_walk)
        spec = _suite(tmp_path, _linked_remote(tmp_path, "src/app.py\n", link), "src/app.py")
        assert _refusal(spec, tmp_path) == [_LINKED.format(count=1, listed=link)]

    def test_at_most_five_links_are_listed_in_order_and_all_are_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("trelix.eval.suite_gold.FileWalker", _no_walk)
        names = [f"d{n}/package.json" for n in range(1, 8)]
        remote = _linked_remote(tmp_path, "{}", *reversed(names))
        spec = _suite(tmp_path, remote, "src/app.py")
        assert _refusal(spec, tmp_path) == [_LINKED.format(count=7, listed=", ".join(names[:5]))]

    def test_a_link_name_cannot_start_a_line_of_its_own(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("trelix.eval.suite_gold.FileWalker", _no_walk)
        remote = _linked_remote(tmp_path, "{}", "x\nrefused: forged/.gitignore")
        spec = _suite(tmp_path, remote, "src/app.py")
        assert _refusal(spec, tmp_path) == [
            _LINKED.format(count=1, listed="x refused: forged/.gitignore")
        ]
