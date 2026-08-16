"""
Unit tests for trelix.indexing.walker.FileWalker.

Uses a synthetic repo in tmp_path to verify:
- .gitignore-ignored files are skipped when respect_gitignore=True
- node_modules/ files are never returned
- .trelix/ files (self-index dir) are never returned
- EXTENSION_MAP maps extensions to the correct Language enum values
- Files above max_file_size_bytes are skipped
- SHA-256 hash is deterministic (same content = same hash)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.core.models import Language
from trelix.indexing.walker import EXTENSION_MAP, FileWalker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    repo_path: Path,
    *,
    respect_gitignore: bool = True,
    max_file_size_bytes: int = 500_000,
    languages: list[Language] | None = None,
) -> IndexConfig:
    """Build a minimal IndexConfig pointing at `repo_path`."""
    walker_cfg = WalkerConfig(
        respect_gitignore=respect_gitignore,
        max_file_size_bytes=max_file_size_bytes,
        languages=languages
        or [
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.TSX,
            Language.GO,
            Language.RUST,
            Language.JAVA,
            Language.KOTLIN,
            Language.RUBY,
            Language.CPP,
            Language.C,
            Language.CSHARP,
            Language.RAZOR,
            Language.CSHTML,
            Language.CSPROJ,
            Language.MARKDOWN,
            Language.JSON,
            Language.YAML,
            Language.TOML,
            Language.HTML,
            Language.CSS,
        ],
    )
    return IndexConfig(repo_path=str(repo_path), walker=walker_cfg)


def _build_synthetic_repo(base: Path) -> dict[str, Path]:
    """
    Create a small synthetic repository structure under `base`:

        base/
          main.py            <- Python file (should be indexed)
          app.ts             <- TypeScript file (should be indexed)
          .gitignore         <- ignores secret.py
          secret.py          <- should be skipped when respect_gitignore=True
          node_modules/
            lib.js           <- should always be skipped (in extra_ignore_dirs)
          .trelix/
            index.db         <- should always be skipped (in extra_ignore_dirs)
    """
    # main.py
    main_py = base / "main.py"
    main_py.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    # app.ts
    app_ts = base / "app.ts"
    app_ts.write_text("export const greet = () => 'hello';\n", encoding="utf-8")

    # .gitignore -- ignores secret.py
    gitignore = base / ".gitignore"
    gitignore.write_text("secret.py\n", encoding="utf-8")

    # secret.py (should be gitignore-excluded)
    secret_py = base / "secret.py"
    secret_py.write_text("SENSITIVE_VALUE = 'ignored-by-gitignore'\n", encoding="utf-8")

    # node_modules/lib.js
    node_modules = base / "node_modules"
    node_modules.mkdir()
    node_lib = node_modules / "lib.js"
    node_lib.write_text("module.exports = {};\n", encoding="utf-8")

    # .trelix/index.db (self-index dir)
    trelix_dir = base / ".trelix"
    trelix_dir.mkdir()
    trelix_db = trelix_dir / "index.db"
    trelix_db.write_bytes(b"\x00\x01trelix-db-stub")

    return {
        "main_py": main_py,
        "app_ts": app_ts,
        "gitignore": gitignore,
        "secret_py": secret_py,
        "node_lib": node_lib,
        "trelix_db": trelix_db,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGitignoreFiltering:
    def test_gitignore_ignored_files_are_skipped(self, tmp_path: Path) -> None:
        """Files matched by .gitignore must NOT appear when respect_gitignore=True."""
        _build_synthetic_repo(tmp_path)
        config = make_config(tmp_path, respect_gitignore=True)
        walker = FileWalker(config)

        found_rel_paths = {f.rel_path for f in walker.walk()}

        assert "secret.py" not in found_rel_paths, (
            "secret.py is listed in .gitignore -- it must be skipped"
        )

    def test_gitignore_respected_includes_non_ignored(self, tmp_path: Path) -> None:
        """Non-ignored Python files must still be returned when respect_gitignore=True."""
        _build_synthetic_repo(tmp_path)
        config = make_config(tmp_path, respect_gitignore=True)
        walker = FileWalker(config)

        found_rel_paths = {f.rel_path for f in walker.walk()}

        assert "main.py" in found_rel_paths, "main.py is not gitignored -- it must be indexed"

    def test_gitignore_disabled_includes_secret(self, tmp_path: Path) -> None:
        """When respect_gitignore=False, .gitignore patterns are ignored."""
        _build_synthetic_repo(tmp_path)
        config = make_config(tmp_path, respect_gitignore=False)
        walker = FileWalker(config)

        found_rel_paths = {f.rel_path for f in walker.walk()}

        assert "secret.py" in found_rel_paths, (
            "With respect_gitignore=False, secret.py must be included"
        )


class TestIgnoredDirectories:
    def test_node_modules_files_never_returned(self, tmp_path: Path) -> None:
        """Files inside node_modules/ must never appear regardless of gitignore setting."""
        _build_synthetic_repo(tmp_path)

        for respect_gitignore in (True, False):
            config = make_config(tmp_path, respect_gitignore=respect_gitignore)
            walker = FileWalker(config)
            # Use rel_path (relative to repo root) so the check is unambiguous
            found_rel_paths = [f.rel_path for f in walker.walk()]
            assert not any("node_modules" in rp for rp in found_rel_paths), (
                f"node_modules file appeared with respect_gitignore={respect_gitignore}: "
                f"{[rp for rp in found_rel_paths if 'node_modules' in rp]}"
            )

    def test_trelix_dir_files_never_returned(self, tmp_path: Path) -> None:
        """.trelix/ is in extra_ignore_dirs -- its files must never be indexed."""
        _build_synthetic_repo(tmp_path)

        for respect_gitignore in (True, False):
            config = make_config(tmp_path, respect_gitignore=respect_gitignore)
            walker = FileWalker(config)
            # Use rel_path (relative to repo root) so the check is unambiguous
            found_rel_paths = [f.rel_path for f in walker.walk()]
            assert not any(".trelix" in rp for rp in found_rel_paths), (
                f".trelix file appeared with respect_gitignore={respect_gitignore}: "
                f"{[rp for rp in found_rel_paths if '.trelix' in rp]}"
            )

    def test_trelix_in_extra_ignore_dirs_default(self) -> None:
        """Confirm .trelix is present in the default extra_ignore_dirs list."""
        walker_cfg = WalkerConfig()
        assert ".trelix" in walker_cfg.extra_ignore_dirs


class TestExtensionMap:
    def test_py_maps_to_python(self) -> None:
        assert EXTENSION_MAP[".py"] == Language.PYTHON

    def test_ts_maps_to_typescript(self) -> None:
        assert EXTENSION_MAP[".ts"] == Language.TYPESCRIPT

    def test_tsx_maps_to_tsx(self) -> None:
        assert EXTENSION_MAP[".tsx"] == Language.TSX

    def test_js_maps_to_javascript(self) -> None:
        assert EXTENSION_MAP[".js"] == Language.JAVASCRIPT

    def test_go_maps_to_go(self) -> None:
        assert EXTENSION_MAP[".go"] == Language.GO

    def test_rs_maps_to_rust(self) -> None:
        assert EXTENSION_MAP[".rs"] == Language.RUST

    def test_java_maps_to_java(self) -> None:
        assert EXTENSION_MAP[".java"] == Language.JAVA

    def test_md_maps_to_markdown(self) -> None:
        assert EXTENSION_MAP[".md"] == Language.MARKDOWN

    def test_json_maps_to_json(self) -> None:
        assert EXTENSION_MAP[".json"] == Language.JSON

    def test_yaml_maps_to_yaml(self) -> None:
        assert EXTENSION_MAP[".yaml"] == Language.YAML

    def test_toml_maps_to_toml(self) -> None:
        assert EXTENSION_MAP[".toml"] == Language.TOML

    def test_css_maps_to_css(self) -> None:
        assert EXTENSION_MAP[".css"] == Language.CSS

    def test_html_maps_to_html(self) -> None:
        assert EXTENSION_MAP[".html"] == Language.HTML


class TestFileSizeFilter:
    def test_file_above_max_size_is_skipped(self, tmp_path: Path) -> None:
        """Files larger than max_file_size_bytes must not be yielded."""
        big_file = tmp_path / "big.py"
        big_file.write_bytes(b"x" * 1001)

        config = make_config(tmp_path, max_file_size_bytes=1000)
        walker = FileWalker(config)
        found = list(walker.walk())

        assert not any(f.rel_path == "big.py" for f in found), (
            "big.py exceeds max_file_size_bytes=1000 -- it must be skipped"
        )

    def test_file_at_max_size_is_included(self, tmp_path: Path) -> None:
        """Files exactly at the max_file_size_bytes limit must be included."""
        exact_file = tmp_path / "exact.py"
        exact_file.write_bytes(b"x" * 1000)

        config = make_config(tmp_path, max_file_size_bytes=1000)
        walker = FileWalker(config)
        found = list(walker.walk())

        assert any(f.rel_path == "exact.py" for f in found), (
            "exact.py is exactly max_file_size_bytes=1000 -- it must be included"
        )

    def test_file_below_max_size_is_included(self, tmp_path: Path) -> None:
        """Files smaller than max_file_size_bytes must be yielded."""
        small_file = tmp_path / "small.py"
        small_file.write_text("pass\n", encoding="utf-8")

        config = make_config(tmp_path, max_file_size_bytes=1000)
        walker = FileWalker(config)
        found = list(walker.walk())

        assert any(f.rel_path == "small.py" for f in found), (
            "small.py is well under max_file_size_bytes=1000 -- it must be included"
        )


class TestHashDeterminism:
    def test_same_content_produces_same_hash(self, tmp_path: Path) -> None:
        """Two files with identical content must have the same SHA-256 hash."""
        content = b"def foo():\n    return 42\n"

        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.write_bytes(content)
        file_b.write_bytes(content)

        config = make_config(tmp_path)
        walker = FileWalker(config)
        results = {f.rel_path: f.hash for f in walker.walk()}

        assert "a.py" in results
        assert "b.py" in results
        assert results["a.py"] == results["b.py"], (
            "Files with identical content must produce the same SHA-256 hash"
        )

    def test_different_content_produces_different_hash(self, tmp_path: Path) -> None:
        """Two files with different content must have different SHA-256 hashes."""
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.write_bytes(b"def foo(): return 1\n")
        file_b.write_bytes(b"def foo(): return 2\n")

        config = make_config(tmp_path)
        walker = FileWalker(config)
        results = {f.rel_path: f.hash for f in walker.walk()}

        assert results["a.py"] != results["b.py"], (
            "Files with different content must produce different SHA-256 hashes"
        )

    def test_hash_matches_manual_sha256(self, tmp_path: Path) -> None:
        """Walker hash must match a manually computed SHA-256."""
        content = b"print('hello trelix')\n"
        py_file = tmp_path / "hello.py"
        py_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()

        config = make_config(tmp_path)
        walker = FileWalker(config)
        results = {f.rel_path: f.hash for f in walker.walk()}

        assert results["hello.py"] == expected, (
            "Walker hash must match hashlib.sha256(content).hexdigest()"
        )

    def test_hash_is_stable_across_walks(self, tmp_path: Path) -> None:
        """Walking the same repo twice must produce identical hashes."""
        py_file = tmp_path / "stable.py"
        py_file.write_text("x = 1\n", encoding="utf-8")

        config = make_config(tmp_path)

        walker1 = FileWalker(config)
        walk1 = {f.rel_path: f.hash for f in walker1.walk()}

        walker2 = FileWalker(config)
        walk2 = {f.rel_path: f.hash for f in walker2.walk()}

        assert walk1 == walk2, "Hashes must be stable across repeated walks"


class TestIndexedFileFields:
    def test_indexed_file_has_correct_language(self, tmp_path: Path) -> None:
        """IndexedFile.language must match the EXTENSION_MAP entry."""
        (tmp_path / "module.py").write_text("pass\n", encoding="utf-8")
        (tmp_path / "component.ts").write_text("export default {};\n", encoding="utf-8")

        config = make_config(tmp_path)
        walker = FileWalker(config)
        results = {f.rel_path: f for f in walker.walk()}

        assert results["module.py"].language == Language.PYTHON
        assert results["component.ts"].language == Language.TYPESCRIPT

    def test_indexed_file_rel_path_is_relative(self, tmp_path: Path) -> None:
        """IndexedFile.rel_path must be relative (not absolute)."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "util.py").write_text("pass\n", encoding="utf-8")

        config = make_config(tmp_path)
        walker = FileWalker(config)
        results = list(walker.walk())

        assert len(results) == 1
        assert not Path(results[0].rel_path).is_absolute(), (
            "rel_path must be relative to the repo root"
        )
        assert results[0].rel_path == str(Path("src") / "util.py")

    def test_indexed_file_path_is_absolute(self, tmp_path: Path) -> None:
        """IndexedFile.path must be an absolute path."""
        (tmp_path / "abs.py").write_text("pass\n", encoding="utf-8")

        config = make_config(tmp_path)
        walker = FileWalker(config)
        results = list(walker.walk())

        assert len(results) == 1
        assert Path(results[0].path).is_absolute(), "IndexedFile.path must be absolute"

    def test_size_bytes_matches_actual_file_size(self, tmp_path: Path) -> None:
        """IndexedFile.size_bytes must match the actual on-disk file size."""
        content = b"print('size test')\n"
        py_file = tmp_path / "size.py"
        py_file.write_bytes(content)

        config = make_config(tmp_path)
        walker = FileWalker(config)
        results = list(walker.walk())

        assert len(results) == 1
        assert results[0].size_bytes == len(content)


class TestSymlinkContainment:
    """`follow_symlinks` decides whether the walk is confined to `repo_path`.

    Historically the walker had NO symlink handling at all: `_iter_files` used
    `entry.is_dir()` / `entry.is_file()` (both of which follow symlinks) and
    `walk()` computed `rel_path` with `relative_to(repo_root)` on the UNRESOLVED
    path. A file outside the repository was therefore indexed and then presented
    under a path that looks like it is inside — `linked_dir/secret.py` for a file
    that lives nowhere near the repo.

    SECURITY.md asserted the opposite ("does not follow symlinks outside the repo
    boundary") from v2.x through v3.1.0. These tests pin the real behaviour of
    both settings so the document and the code cannot drift apart again.

    The default is True — unchanged from every previous release — because
    confining by default would silently drop files from any repository that
    symlinks to shared or vendored directories, with no error to explain it.
    """

    @staticmethod
    def _repo_with_escaping_symlinks(base: Path) -> Path:
        """A repo with one dir symlink and one file symlink pointing outside it."""
        repo = base / "repo"
        repo.mkdir()
        outside = base / "outside"
        outside.mkdir()

        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        (outside / "secret.py").write_text("def out_of_tree():\n    return 1\n", encoding="utf-8")
        (repo / "linked_dir").symlink_to(outside, target_is_directory=True)
        (repo / "linked.py").symlink_to(outside / "secret.py")
        return repo

    def _walk(self, repo: Path, *, follow: bool) -> list[str]:
        cfg = make_config(repo)
        cfg.walker.follow_symlinks = follow
        return sorted(f.rel_path for f in FileWalker(cfg).walk())

    def test_default_is_to_follow_symlinks(self) -> None:
        """The default must not change: existing indexes were built with it."""
        assert WalkerConfig().follow_symlinks is True

    def test_following_reaches_outside_the_repo(self, tmp_path: Path) -> None:
        """Documents the real default behaviour, including the misleading rel_path."""
        repo = self._repo_with_escaping_symlinks(tmp_path)

        rels = self._walk(repo, follow=True)

        assert "main.py" in rels
        # Both symlink kinds are traversed...
        assert "linked.py" in rels
        assert "linked_dir/secret.py" in rels
        # ...and the out-of-tree file is presented as though it sat in the repo.
        assert not (repo / "linked_dir" / "secret.py").resolve().is_relative_to(repo.resolve())

    def test_containment_excludes_both_symlink_kinds(self, tmp_path: Path) -> None:
        repo = self._repo_with_escaping_symlinks(tmp_path)

        rels = self._walk(repo, follow=False)

        assert rels == ["main.py"], f"out-of-tree content leaked: {rels}"

    def test_containment_keeps_in_tree_symlinks(self, tmp_path: Path) -> None:
        """Confinement is about the boundary, not about symlinks per se.

        A symlink whose target is inside the repo resolves inside the repo, so it
        must still be indexed — otherwise the flag would be a blunt "ignore all
        symlinks" switch, which is not what it claims to be.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "real.py").write_text("def real():\n    return 1\n", encoding="utf-8")
        (repo / "alias.py").symlink_to(repo / "real.py")

        rels = self._walk(repo, follow=False)

        assert "real.py" in rels
        assert "alias.py" in rels, "an in-tree symlink was wrongly excluded"

    def test_containment_survives_an_unresolved_symlinked_repo_root(self, tmp_path: Path) -> None:
        """A symlinked repo_root must not cause the walker to reject its own repo.

        Both sides of the containment comparison have to be resolved. Comparing a
        resolved entry against an UNRESOLVED root resolves every file to its real
        location, matches it against the symlink path, and excludes everything.

        Reaching that state requires bypassing IndexConfig's repo_path validator,
        which resolves the path — so this builds the config with
        `model_construct`, exactly as `federation/retriever.py` and
        `indexing/multi_watcher.py` do. Without model_construct the test is
        vacuous: it passes even with the root left unresolved.
        """
        real = tmp_path / "real_repo"
        real.mkdir()
        (real / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        link_to_repo = tmp_path / "repo_via_link"
        link_to_repo.symlink_to(real, target_is_directory=True)

        # model_construct skips the validator, so repo_path stays symlinked.
        cfg = IndexConfig.model_construct(
            repo_path=str(link_to_repo), walker=WalkerConfig(follow_symlinks=False)
        )
        walker = FileWalker(cfg)
        assert str(walker.repo_root) != str(walker._resolved_root), (
            "precondition failed: repo_root was already resolved, so this test "
            "cannot detect an unresolved-root comparison"
        )

        rels = sorted(f.rel_path for f in walker.walk())

        assert rels == ["main.py"], "a symlinked repo_root wrongly excluded its own files"

    def test_broken_symlink_is_excluded_not_fatal(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        (repo / "dangling.py").symlink_to(tmp_path / "does_not_exist.py")

        rels = self._walk(repo, follow=False)

        assert rels == ["main.py"]


class TestNestedGitignore:
    """A `.gitignore` in a subdirectory must apply to that subdirectory.

    `_load_gitignore_spec` read only `repo_root/.gitignore`, while the module
    docstring advertised "respects nested .gitignore files". Every nested
    `.gitignore` in every indexed repository was therefore ignored.

    The cost of that gap was measured on trelix's own repository:
    `workspace-vscode/.gitignore` excludes `.vscode-test/`, which
    `@vscode/test-electron` fills with a 2.6 GB VS Code application bundle. The
    walker descended into it and indexed 543 of 915 files (59%) and 23,865 of
    32,337 chunks (74%) from that bundle — minified single-letter symbols that
    then competed with real code in search results.

    These tests pin nested-`.gitignore` semantics as git defines them, so the
    docstring and the code cannot drift apart again.
    """

    @staticmethod
    def _repo(base: Path) -> Path:
        """A repo whose nested .gitignore excludes a build directory.

        Mirrors the shape that caused the bug:

            repo/
              .gitignore        <- "root_only.py"
              main.py
              root_only.py      <- excluded by the ROOT .gitignore
              sub/
                .gitignore      <- "harness/" and "local.py"
                keep.py         <- must survive
                local.py        <- excluded by the NESTED .gitignore
                harness/
                  bundle.js     <- excluded because its DIR is nested-ignored
        """
        repo = base / "repo"
        repo.mkdir()
        (repo / ".gitignore").write_text("root_only.py\n", encoding="utf-8")
        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")
        (repo / "root_only.py").write_text("ROOT = 1\n", encoding="utf-8")

        sub = repo / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("harness/\nlocal.py\n", encoding="utf-8")
        (sub / "keep.py").write_text("KEEP = 1\n", encoding="utf-8")
        (sub / "local.py").write_text("LOCAL = 1\n", encoding="utf-8")

        harness = sub / "harness"
        harness.mkdir()
        (harness / "bundle.js").write_text("var a=1;\n", encoding="utf-8")
        return repo

    def _walk(self, repo: Path, *, respect: bool = True) -> list[str]:
        return sorted(
            f.rel_path for f in FileWalker(make_config(repo, respect_gitignore=respect)).walk()
        )

    def test_nested_gitignore_excludes_directory(self, tmp_path: Path) -> None:
        """`sub/.gitignore` saying `build/` must exclude `sub/harness/bundle.js`.

        This is the .vscode-test case reduced to its essentials.
        """
        rels = self._walk(self._repo(tmp_path))
        assert "sub/harness/bundle.js" not in rels, (
            "sub/.gitignore excludes harness/ — the walker descended into it anyway"
        )

    def test_nested_gitignore_excludes_file(self, tmp_path: Path) -> None:
        """A plain filename in a nested .gitignore must exclude that file."""
        rels = self._walk(self._repo(tmp_path))
        assert "sub/local.py" not in rels, "sub/.gitignore lists local.py — it must be skipped"

    def test_non_ignored_siblings_survive(self, tmp_path: Path) -> None:
        """Honouring nested files must not over-exclude their siblings."""
        rels = self._walk(self._repo(tmp_path))
        assert "sub/keep.py" in rels, "keep.py is not ignored anywhere — it must be indexed"
        assert "main.py" in rels, "main.py is not ignored anywhere — it must be indexed"

    def test_root_gitignore_still_applies(self, tmp_path: Path) -> None:
        """Regression: adding nested support must not break root-level patterns."""
        rels = self._walk(self._repo(tmp_path))
        assert "root_only.py" not in rels, "the root .gitignore stopped being honoured"

    def test_nested_patterns_are_relative_to_their_own_directory(self, tmp_path: Path) -> None:
        """A nested pattern must not match the same name at the repo root.

        git anchors each .gitignore's patterns to the directory containing it. A
        naive implementation that concatenates all pattern files into one spec
        would wrongly exclude `local.py` at the root too.
        """
        repo = self._repo(tmp_path)
        (repo / "local.py").write_text("ROOT_LOCAL = 1\n", encoding="utf-8")

        rels = self._walk(repo)

        assert "local.py" in rels, (
            "sub/.gitignore's `local.py` leaked upward and excluded the root local.py"
        )
        assert "sub/local.py" not in rels, "sub/local.py must still be excluded"

    def test_nested_patterns_do_not_leak_into_sibling_directories(self, tmp_path: Path) -> None:
        """`sub/.gitignore` must have no effect on `other/`."""
        repo = self._repo(tmp_path)
        other = repo / "other"
        other.mkdir()
        (other / "local.py").write_text("OTHER = 1\n", encoding="utf-8")
        (other / "harness").mkdir()
        (other / "harness" / "keep.js").write_text("var b=2;\n", encoding="utf-8")

        rels = self._walk(repo)

        assert "other/local.py" in rels, "sub/.gitignore wrongly excluded other/local.py"
        assert "other/harness/keep.js" in rels, "sub/.gitignore wrongly excluded other/harness/"

    def test_deeper_gitignore_negation_re_includes_a_file(self, tmp_path: Path) -> None:
        """A deeper .gitignore's `!` rule must override a shallower exclusion.

        git resolves conflicts by proximity: the .gitignore closest to the file
        wins. Here `sub/.gitignore` excludes `local.py`, and
        `sub/deep/.gitignore` re-includes it.
        """
        repo = self._repo(tmp_path)
        deep = repo / "sub" / "deep"
        deep.mkdir()
        (deep / ".gitignore").write_text("!local.py\n", encoding="utf-8")
        (deep / "local.py").write_text("DEEP = 1\n", encoding="utf-8")

        rels = self._walk(repo)

        assert "sub/deep/local.py" in rels, (
            "sub/deep/.gitignore re-includes local.py — the deeper file must win"
        )

    def test_respect_gitignore_false_disables_nested_files_too(self, tmp_path: Path) -> None:
        """Opting out of .gitignore must opt out of nested ones as well."""
        rels = self._walk(self._repo(tmp_path), respect=False)

        for expected in ("root_only.py", "sub/local.py", "sub/harness/bundle.js"):
            assert expected in rels, f"{expected} was excluded even though respect_gitignore=False"

    def test_gitignore_in_an_unindexed_directory_still_counts(self, tmp_path: Path) -> None:
        """A .gitignore need not sit beside indexable files to apply.

        workspace-vscode/ contains almost nothing trelix indexes directly, yet
        its .gitignore is what excludes the 2.6 GB bundle underneath.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def entry():\n    return 0\n", encoding="utf-8")

        holder = repo / "holder"
        holder.mkdir()
        (holder / ".gitignore").write_text(".vscode-test/\n", encoding="utf-8")
        bundle = holder / ".vscode-test" / "app" / "resources"
        bundle.mkdir(parents=True)
        (bundle / "minified.js").write_text("var J=1,j=2;\n", encoding="utf-8")

        rels = self._walk(repo)

        assert rels == ["main.py"], (
            f"the nested .gitignore's .vscode-test/ exclusion was not honoured: {rels}"
        )


class TestWalkCompleteness:
    """An unreadable directory must not vanish from the index in silence.

    `_iter_files` catches `PermissionError` from `root.iterdir()` and bare-`return`s,
    which drops the ENTIRE subtree below it, and catches `OSError` per file with a
    `continue`. Neither left any trace: the walk simply yielded fewer files and the
    index reported success over a corpus with a hole in it.

    That matters beyond the missing files. Any future reconciliation pass — deleting
    index rows for files the walk no longer yields — would read a truncated walk as
    "these files were deleted" and remove embeddings the user paid to compute. A
    percentage threshold cannot defend against it either: one unreadable directory is
    almost always far below any sane threshold, so the guard passes and the data goes.

    `FileWalker.incomplete_paths` records what was skipped, so a caller can refuse to
    act on a walk that is not trustworthy.
    """

    @staticmethod
    def _repo(base: Path) -> Path:
        repo = base / "repo"
        repo.mkdir()
        (repo / "visible.py").write_text("def visible(): pass\n", encoding="utf-8")
        locked = repo / "locked"
        locked.mkdir()
        (locked / "hidden.py").write_text("def hidden(): pass\n", encoding="utf-8")
        return repo

    def test_a_complete_walk_reports_no_gaps(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        walker = FileWalker(make_config(repo))

        rels = sorted(f.rel_path for f in walker.walk())

        assert rels == ["locked/hidden.py", "visible.py"]
        assert walker.incomplete_paths == []
        assert walker.walk_was_complete is True

    def test_an_unreadable_directory_is_recorded(self, tmp_path: Path) -> None:
        """The headline case: a whole subtree disappears, and now says so."""
        import os
        import stat

        repo = self._repo(tmp_path)
        locked = repo / "locked"
        os.chmod(locked, 0o000)
        try:
            walker = FileWalker(make_config(repo))
            rels = sorted(f.rel_path for f in walker.walk())
        finally:
            os.chmod(locked, stat.S_IRWXU)

        if rels == ["locked/hidden.py", "visible.py"]:
            pytest.skip("this filesystem/user ignores chmod 000 (root, or a permissive FS)")

        assert rels == ["visible.py"], f"expected the locked subtree to be skipped: {rels}"
        assert walker.walk_was_complete is False, (
            "the walk silently dropped a subtree and still claimed to be complete"
        )
        assert any("locked" in p for p in walker.incomplete_paths), (
            f"the skipped path was not recorded: {walker.incomplete_paths}"
        )

    def test_the_record_resets_between_walks(self, tmp_path: Path) -> None:
        """A second walk must not inherit the first one's gaps."""
        repo = self._repo(tmp_path)
        walker = FileWalker(make_config(repo))
        walker._incomplete_paths.append("stale/entry")

        list(walker.walk())

        assert walker.incomplete_paths == [], "gaps from a previous walk leaked forward"
