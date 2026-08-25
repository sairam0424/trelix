"""
The walker filters nothing tested it: minified bundles, the language allow-list,
dangling symlinks and the size cap measured through a symlink.

Mutation-verified. Every test below names, in its docstring, the exact single-line
change to `src/trelix/indexing/walker.py` that must make it fail. Before these tests
existed, all four of those changes left the walker's own suite green:

  * `endswith` -> `startswith` in the extension filter, and deleting that filter
    outright, and matching it on `path.suffix` instead of `path.name` — 269 tests
    passed on all three. The overlap between `EXTENSION_MAP` and the default
    `extra_ignore_extensions` is exactly `.min.js` and `.min.css`, so those two are
    the ONLY files that filter removes, and no fixture anywhere had one.
  * `language not in allowed_languages` -> `language is Language.UNKNOWN`, i.e. the
    `languages` allow-list stops being consulted at all. Every existing fixture
    either lists every language it creates files in, or creates files in only one.
  * `elif entry.is_file():` -> `else:` in `_iter_files`. Both existing broken-symlink
    tests set `follow_symlinks=False`, where the containment check rejects the
    dangling entry *before* the is_file() branch is reached — so the mutant is
    invisible under the setting that is not the default.
  * `path.stat()` -> `path.lstat()` in the size check, which measures the length of
    the symlink instead of the file it points at.

These are behaviour tests: they build a repo, walk it, and compare the yielded set
against a literal expected set. No value is imported from `walker.py`, and nothing
here iterates a collection defined in `walker.py` to build its own expectation.
"""

from __future__ import annotations

from pathlib import Path

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.core.models import IndexedFile, Language
from trelix.indexing.walker import FileWalker


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _walk(config: IndexConfig) -> list[IndexedFile]:
    return list(FileWalker(config).walk())


class TestMinifiedBundlesAreExcluded:
    """`extra_ignore_extensions` is matched on the full NAME, not the suffix."""

    def test_min_js_and_min_css_are_excluded_while_their_siblings_are_indexed(
        self, tmp_path: Path
    ) -> None:
        """Kills: `path.name.endswith(ext)` -> `path.name.startswith(ext)`, the
        deletion of the whole extension filter, and `path.suffix in ignore_extensions`.

        A minified bundle is one enormous line. It cannot be usefully chunked, it
        embeds as noise, and `jquery.min.js` is 90 KB of third-party code priced per
        token — which is why `.min.js`/`.min.css` are the two entries in the shipped
        `extra_ignore_extensions` that a compound extension makes load-bearing:
        `Path("app.min.js").suffix` is `".js"`, so a suffix-based filter, or no filter
        at all, indexes every vendored bundle in the repository.

        Uses the SHIPPED default WalkerConfig deliberately — the defect is that the
        default does nothing, so a fixture that hand-writes the ignore list would
        prove less than nothing.
        """
        repo = tmp_path / "repo"
        # NOT `vendor/`: that name is in the default extra_ignore_dirs, which would
        # exclude the bundles for an unrelated reason and make this test vacuous.
        _write(repo / "main.py", "def main():\n    return 0\n")
        _write(repo / "static/app.js", "export const app = () => 1;\n")
        _write(repo / "static/app.min.js", "export const app=()=>1;\n")
        _write(repo / "static/site.css", ".a { color: red; }\n")
        _write(repo / "static/site.min.css", ".a{color:red}\n")

        # Preconditions: the two files under test must be real, non-empty and small
        # enough that no other filter can be the reason they are absent.
        for name in ("static/app.min.js", "static/site.min.css"):
            size = (repo / name).stat().st_size
            assert 0 < size < 500_000, (
                f"fixture {name} is {size} bytes — it must be non-empty and under the "
                "size cap, or its absence proves nothing about the extension filter"
            )

        rels = {f.rel_path for f in _walk(IndexConfig(repo_path=str(repo)))}

        assert "static/app.js" in rels and "static/site.css" in rels, (
            "precondition failed: the fixture's NON-minified siblings static/app.js "
            "and static/site.css were not indexed either, so this test can no longer "
            "tell the extension filter from a broken fixture"
        )
        assert rels == {"main.py", "static/app.js", "static/site.css"}, (
            f"minified bundles leaked into the index: {sorted(rels)}"
        )


class TestLanguageAllowListIsHonoured:
    """`walker.languages` is an allow-list, and it has to actually be applied."""

    def test_only_the_configured_languages_are_yielded(self, tmp_path: Path) -> None:
        """Kills: `if language not in allowed_languages:` -> `if language is
        Language.UNKNOWN:`.

        `languages` is the one knob that bounds what an index costs — narrowing it to
        Python is how a user keeps a polyglot monorepo's markdown, JSON and TypeScript
        out of a paid embedding run. A walker that only rejects UNKNOWN honours the
        setting in the config file and ignores it on disk, so the run silently costs
        several times what was asked for and the index answers questions about files
        the user excluded.

        The second walk is the precondition: it proves the same three fixture files
        ARE all indexable, so the first walk's result is the allow-list at work and
        not a fixture that stopped discriminating.
        """
        repo = tmp_path / "repo"
        _write(repo / "main.py", "def main():\n    return 0\n")
        _write(repo / "app.ts", "export const greet = () => 'hi';\n")
        _write(repo / "README.md", "# readme\n")

        python_only = IndexConfig(
            repo_path=str(repo), walker=WalkerConfig(languages=[Language.PYTHON])
        )
        all_three = IndexConfig(
            repo_path=str(repo),
            walker=WalkerConfig(
                languages=[Language.PYTHON, Language.TYPESCRIPT, Language.MARKDOWN]
            ),
        )

        widened = {f.rel_path for f in _walk(all_three)}
        assert widened == {"README.md", "app.ts", "main.py"}, (
            "precondition failed: the fixture's app.ts / README.md are not indexable "
            "even with their languages allowed, so narrowing the list proves nothing: "
            f"{sorted(widened)}"
        )

        narrowed = {f.rel_path for f in _walk(python_only)}

        assert narrowed == {"main.py"}, (
            f"languages=[PYTHON] still indexed other languages: {sorted(narrowed)}"
        )


class TestDanglingSymlinkUnderTheDefaultConfig:
    """A broken symlink must not be yielded, and must not be called a gap.

    Both existing broken-link tests run with `follow_symlinks=False`, where
    `_is_within_root` rejects the entry before `_iter_files` ever asks whether it is
    a file. Under the DEFAULT (`follow_symlinks=True`) that check returns True
    without touching the filesystem, so the `elif entry.is_file()` branch is the only
    thing standing between a dangling link and the rest of the walk.
    """

    def test_a_dangling_symlink_leaves_the_walk_complete(self, tmp_path: Path) -> None:
        """Kills: `elif entry.is_file():` -> `else:` in `_iter_files`.

        With `else`, a dangling `*.py` link is yielded, passes every filter (its name
        looks like Python), and then fails `stat()` — which `walk()` records as an
        incomplete path. One deliberately-broken symlink therefore flips
        `walk_was_complete` to False for the whole repository, and that flag is what
        `--prune` and `compute_drift` consult before deleting index rows. The user's
        punishment for a stale link is that deletion-side reconciliation refuses to
        run for ever, plus a WARNING claiming the index is incomplete when nothing is
        actually missing.
        """
        repo = tmp_path / "repo"
        _write(repo / "main.py", "def entry():\n    return 0\n")
        (repo / "dangling.py").symlink_to(tmp_path / "never_existed.py")

        config = IndexConfig(repo_path=str(repo))
        assert config.walker.follow_symlinks is True, (
            "precondition failed: this fixture needs the DEFAULT follow_symlinks=True. "
            "With it False the containment check rejects dangling.py before the "
            "is_file() branch runs, and this test passes for the wrong reason"
        )
        assert (repo / "dangling.py").is_symlink() and not (repo / "dangling.py").exists(), (
            "precondition failed: fixture dangling.py is not a broken symlink"
        )

        walker = FileWalker(config)
        rels = {f.rel_path for f in walker.walk()}

        assert rels == {"main.py"}, f"a dangling symlink was yielded as a file: {sorted(rels)}"
        assert walker.incomplete_paths == [], (
            f"a dangling symlink was recorded as a gap in the index: {walker.incomplete_paths}"
        )
        assert walker.walk_was_complete is True, (
            "a dangling symlink made the walk declare itself incomplete, which blocks "
            "every deletion-side operation on the repository"
        )


class TestSizeCapIsMeasuredOnTheTarget:
    """The size filter must weigh the file, not the symlink pointing at it."""

    def test_an_oversized_file_reached_through_a_symlink_is_still_excluded(
        self, tmp_path: Path
    ) -> None:
        """Kills: `size = path.stat().st_size` -> `size = path.lstat().st_size`.

        `lstat()` reports the length of the link's target STRING — 6 bytes for
        `big.py` — so every oversized file reachable through a symlink walks straight
        past `max_file_size_bytes`. Then `_compute_hash` does follow the link, so the
        walker reads the whole file into memory to hash it and hands the indexer a
        file it had already decided was too big; and the `size_bytes` written to the
        index is the length of a path string, so every consumer that trusts it (the
        dry-run token estimate, the drift report) is reading a number about the wrong
        object.
        """
        repo = tmp_path / "repo"
        _write(repo / "small.py", "x = 1\n")
        _write(repo / "big.py", "# pad\n" * 400)
        (repo / "link_big.py").symlink_to("big.py")
        (repo / "alias_small.py").symlink_to("small.py")

        cap = 1_000
        real_size = (repo / "small.py").stat().st_size
        assert (repo / "big.py").stat().st_size > cap, "fixture big.py is not over the cap"
        assert (repo / "link_big.py").lstat().st_size <= cap, (
            "precondition failed: fixture link_big.py's own lstat size is already over "
            "the cap, so stat() and lstat() no longer disagree and this test cannot "
            "detect the difference"
        )
        assert (repo / "alias_small.py").lstat().st_size != real_size, (
            "precondition failed: fixture alias_small.py's link length coincidentally "
            "equals its target's size, so size_bytes cannot discriminate — rename it"
        )

        config = IndexConfig(
            repo_path=str(repo),
            walker=WalkerConfig(max_file_size_bytes=cap, languages=[Language.PYTHON]),
        )
        walked = _walk(config)
        by_rel = {f.rel_path: f for f in walked}

        assert set(by_rel) == {"small.py", "alias_small.py"}, (
            f"the size cap was applied to the symlink instead of its target: {sorted(by_rel)}"
        )
        assert by_rel["alias_small.py"].size_bytes == real_size, (
            "size_bytes for a symlinked file recorded the length of the link, not the "
            f"size of the file: {by_rel['alias_small.py'].size_bytes} != {real_size}"
        )
