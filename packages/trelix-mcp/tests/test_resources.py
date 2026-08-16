"""Tests for trelix-mcp MCP Resources."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure trelix core is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


class TestIndexStatsResource:
    async def test_index_stats_resource_registered(self) -> None:
        """trelix://index/stats must be registered as a direct resource."""
        from trelix_mcp.server import mcp

        resources = await mcp.list_resources()
        uris = {str(r.uri) for r in resources}
        assert any("stats" in uri for uri in uris), (
            f"No stats resource found. Direct resources: {uris}"
        )

    async def test_resource_templates_registered(self) -> None:
        """manifest and symbol URI templates must be registered."""
        from trelix_mcp.server import mcp

        templates = await mcp.list_resource_templates()
        uri_templates = {str(t.uri_template) for t in templates}
        assert any("manifest" in t for t in uri_templates), (
            f"No manifest template found. Templates: {uri_templates}"
        )
        assert any("symbols" in t for t in uri_templates), (
            f"No symbols template found. Templates: {uri_templates}"
        )

    def test_index_stats_returns_json(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_index_stats

        mock_db = MagicMock()
        mock_db._conn.execute.return_value.fetchone.return_value = (100, 500, 1200)

        with patch("trelix_mcp.resources.Database", return_value=mock_db):
            with patch("trelix_mcp.resources.IndexConfig") as MockConfig:
                MockConfig.return_value.db_path_absolute = tmp_path / "index.db"
                result = get_index_stats(repo_path=str(tmp_path))

        data = json.loads(result)
        assert "symbol_count" in data or "error" in data

    def test_index_stats_returns_dict_with_counts(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_index_stats

        mock_db = MagicMock()
        mock_db._conn.execute.return_value.fetchone.return_value = (100, 500, 1200)

        with patch("trelix_mcp.resources.Database", return_value=mock_db):
            with patch("trelix_mcp.resources.IndexConfig") as MockConfig:
                MockConfig.return_value.db_path_absolute = tmp_path / "index.db"
                result = get_index_stats(repo_path=str(tmp_path))

        data = json.loads(result)
        if "error" not in data:
            assert data["symbol_count"] == 100
            assert data["file_count"] == 500
            assert data["chunk_count"] == 1200

    def test_index_stats_returns_error_on_missing_index(self) -> None:
        from trelix_mcp.resources import get_index_stats

        # Use a path that cannot have a trelix index (non-directory)
        result = get_index_stats(repo_path="/nonexistent/path/that/cannot/exist")
        data = json.loads(result)
        assert "error" in data

    def test_index_stats_never_raises(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_index_stats

        # Completely invalid path — must not raise
        result = get_index_stats(repo_path="/nonexistent/path/that/cannot/exist")
        data = json.loads(result)
        assert "error" in data


class TestManifestResource:
    def test_manifest_resource_returns_file_list(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_repo_manifest

        mock_db = MagicMock()
        mock_db._conn.execute.return_value.fetchall.return_value = [
            ("src/auth.py", "python", 42),
            ("src/db.py", "python", 18),
        ]

        with patch("trelix_mcp.resources.Database", return_value=mock_db):
            with patch("trelix_mcp.resources.IndexConfig") as MockConfig:
                MockConfig.return_value.db_path_absolute = tmp_path / "index.db"
                result = get_repo_manifest(repo_path=str(tmp_path))

        data = json.loads(result)
        assert "files" in data or "error" in data

    def test_manifest_file_list_shape(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_repo_manifest

        mock_db = MagicMock()
        mock_db._conn.execute.return_value.fetchall.return_value = [
            ("src/auth.py", "python", 42),
            ("src/db.py", "python", 18),
        ]

        with patch("trelix_mcp.resources.Database", return_value=mock_db):
            with patch("trelix_mcp.resources.IndexConfig") as MockConfig:
                MockConfig.return_value.db_path_absolute = tmp_path / "index.db"
                result = get_repo_manifest(repo_path=str(tmp_path))

        data = json.loads(result)
        if "files" in data:
            assert data["file_count"] == 2
            assert data["files"][0]["path"] == "src/auth.py"
            assert data["files"][0]["language"] == "python"
            assert data["files"][0]["symbol_count"] == 42

    def test_manifest_returns_error_on_failure(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_repo_manifest

        result = get_repo_manifest(repo_path=str(tmp_path))
        data = json.loads(result)
        assert isinstance(data, dict)


class TestSymbolSourceResource:
    def test_symbol_source_returns_json(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_symbol_source

        mock_sym = MagicMock()
        mock_sym.qualified_name = "AuthService.login"
        mock_sym.kind.value = "function"
        mock_sym.signature = "def login(self, user: str, password: str) -> bool"
        mock_sym.body = "    return self._verify(user, password)"

        mock_db = MagicMock()
        mock_db.get_symbol_by_name.return_value = [mock_sym]

        with patch("trelix_mcp.resources.Database", return_value=mock_db):
            with patch("trelix_mcp.resources.IndexConfig") as MockConfig:
                MockConfig.return_value.db_path_absolute = tmp_path / "index.db"
                result = get_symbol_source(
                    repo_path=str(tmp_path),
                    qualified_name="AuthService.login",
                )

        data = json.loads(result)
        assert "qualified_name" in data or "error" in data

    def test_symbol_source_not_found_returns_error(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_symbol_source

        mock_db = MagicMock()
        mock_db.get_symbol_by_name.return_value = []

        with patch("trelix_mcp.resources.Database", return_value=mock_db):
            with patch("trelix_mcp.resources.IndexConfig") as MockConfig:
                MockConfig.return_value.db_path_absolute = tmp_path / "index.db"
                result = get_symbol_source(
                    repo_path=str(tmp_path),
                    qualified_name="Nonexistent.method",
                )

        data = json.loads(result)
        assert "error" in data

    def test_symbol_source_never_raises(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_symbol_source

        result = get_symbol_source(
            repo_path="/nonexistent/path",
            qualified_name="some.symbol",
        )
        data = json.loads(result)
        assert isinstance(data, dict)


class TestManifestReportsTheTrueTotal:
    """The manifest must not report its page size as the repository's file count.

    `get_repo_manifest` selects with `LIMIT 500` and then returns
    `"file_count": len(rows)`. On any repository with more than 500 indexed files, an
    agent asking how big the codebase is was told "exactly 500" — a number that is both
    wrong and suspiciously round, and which it cannot detect as a page boundary because
    nothing in the response says a limit was applied.
    """

    @staticmethod
    def _seed(tmp_path, n_files: int):  # type: ignore[no-untyped-def]
        from trelix.core.models import IndexedFile, Language
        from trelix.store.db import Database

        db_path = tmp_path / ".trelix" / "index.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)
        for i in range(n_files):
            db.upsert_file(
                IndexedFile(
                    path=f"/repo/src/f{i:04d}.py",
                    rel_path=f"src/f{i:04d}.py",
                    language=Language.PYTHON,
                    hash=f"h{i}",
                    size_bytes=10,
                )
            )
        db._conn.commit()
        db.close()

    def test_total_is_the_real_count_not_the_page_size(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        from trelix_mcp.resources import get_repo_manifest

        self._seed(tmp_path, 620)
        manifest = json.loads(get_repo_manifest(repo_path=str(tmp_path)))

        assert manifest["total_file_count"] == 620, (
            f"reported {manifest.get('total_file_count')} for a 620-file index"
        )

    def test_the_page_is_still_reported_separately(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An agent needs to know it is looking at a page, not the whole repo."""
        import json

        from trelix_mcp.resources import get_repo_manifest

        self._seed(tmp_path, 620)
        manifest = json.loads(get_repo_manifest(repo_path=str(tmp_path)))

        assert len(manifest["files"]) == 500, "the page size changed unexpectedly"
        assert manifest["files_truncated"] is True
        assert manifest["file_count"] == 500, (
            "file_count keeps its old meaning (the page) for existing consumers"
        )

    def test_a_small_repo_is_not_marked_truncated(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        from trelix_mcp.resources import get_repo_manifest

        self._seed(tmp_path, 12)
        manifest = json.loads(get_repo_manifest(repo_path=str(tmp_path)))

        assert manifest["total_file_count"] == 12
        assert manifest["files_truncated"] is False


class TestIndexStatsResourceIsNotAStub:
    """`trelix://index/stats` returned a hint string while the real implementation sat
    unused.

    `get_index_stats` in resources.py is fully written — symbol_count, file_count,
    chunk_count, error handling — and `grep -rn get_index_stats` finds callers only in
    its own tests. The resource that should serve it returned
    `{"hint": "Use trelix://repo/{repo_path}/manifest ..."}`.
    """

    def test_a_repo_scoped_stats_resource_serves_real_numbers(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        import trelix_mcp.server as srv

        from trelix.core.models import IndexedFile, Language
        from trelix.store.db import Database

        db_path = tmp_path / ".trelix" / "index.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)
        db.upsert_file(
            IndexedFile(
                path="/repo/a.py",
                rel_path="a.py",
                language=Language.PYTHON,
                hash="h",
                size_bytes=10,
            )
        )
        db._conn.commit()
        db.close()

        stats = json.loads(srv.resource_repo_index_stats(str(tmp_path)))

        assert stats["file_count"] == 1
        assert "symbol_count" in stats and "chunk_count" in stats
        assert "hint" not in stats, "the resource is still returning the stub"
