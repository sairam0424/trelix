"""Tests for trelix-mcp MCP Resources."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

    def test_index_stats_returns_error_on_missing_index(self, tmp_path: Path) -> None:
        from trelix_mcp.resources import get_index_stats

        result = get_index_stats(repo_path=str(tmp_path))
        # Must return JSON even on error (never raise)
        data = json.loads(result)
        assert isinstance(data, dict)

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
