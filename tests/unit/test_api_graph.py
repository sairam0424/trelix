"""Tests for graph REST API endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from trelix.api.app import create_app
from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import Database


def _make_indexed_repo(tmp_path: Path) -> Path:
    """Create a minimal indexed repo under tmp_path.

    IndexConfig(repo_path=tmp_path) resolves db_path_absolute to
    tmp_path/.trelix/index.db, so we create the DB there.
    """
    db_path = tmp_path / ".trelix" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    fid = db.upsert_file(
        IndexedFile(
            path=str(tmp_path / "a.py"),
            rel_path="a.py",
            language=Language.PYTHON,
            hash="x",
            size_bytes=10,
        )
    )
    db.insert_symbol(
        Symbol(
            file_id=fid,
            name="fn",
            qualified_name="fn",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=5,
            signature="def fn()",
            body="def fn(): pass",
        )
    )
    db.close()
    return tmp_path


class TestGraphApiEndpoints:
    def test_graph_stats(self, tmp_path: Path) -> None:
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        response = client.get(f"/graph?repo={repo}")
        assert response.status_code == 200
        data = response.json()
        assert "node_count" in data
        assert "edge_count" in data
        assert "community_count" in data

    def test_graph_communities(self, tmp_path: Path) -> None:
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        response = client.get(f"/graph/communities?repo={repo}")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_graph_search_endpoint(self, tmp_path: Path) -> None:
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        # symbol_id=1 is the first inserted symbol; single-node graph has no
        # neighbors, so an empty list is the correct result.
        response = client.get(f"/graph/search?repo={repo}&symbol_id=1&depth=1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
