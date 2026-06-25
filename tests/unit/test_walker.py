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
        languages=languages or [
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

        assert "main.py" in found_rel_paths, (
            "main.py is not gitignored -- it must be indexed"
        )

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
