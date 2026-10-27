"""The REST read routes refuse a repository that has no index, and create nothing.

Every read route builds a Retriever, GraphBuilder or Database on `repo`, and opening a
missing index creates it: `.trelix/index.db` with its schema and an empty vec0 table. The
route then answered 200 with zero results and left a repository that was never indexed
looking indexed, so a later `trelix search` said "no results" instead of "not indexed".
The routes now answer 400 with the CLI's words before anything is opened. `POST /index`
(which creates the index) and `POST /parse` (which never opens one) are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.indexed import mark_indexed
from trelix.api.app import create_app
from trelix.store.db import Database

_READ_ROUTES = [
    pytest.param("/search", {"query": "add"}, id="search"),
    pytest.param("/ask", {"query": "add"}, id="ask"),
    pytest.param("/stats", {}, id="stats"),
    pytest.param("/graph", {}, id="graph"),
    pytest.param("/graph/communities", {}, id="graph-communities"),
    pytest.param("/graph/visualize", {}, id="graph-visualize"),
    pytest.param("/graph/search", {"symbol_id": "1"}, id="graph-search"),
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository with a source file and NO .trelix/, declared as a served root."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(root))
    return root


@pytest.fixture
def client(repo: Path) -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _expected_detail(repo: Path) -> str:
    resolved = repo.resolve()
    return f"No index found at {resolved}/.trelix/index.db. Run trelix index {resolved} first."


@pytest.mark.parametrize(("path", "extra"), _READ_ROUTES)
def test_read_route_on_an_unindexed_repo_is_400_and_creates_nothing(
    client: TestClient, repo: Path, path: str, extra: dict[str, str]
) -> None:
    resp = client.get(path, params={"repo": str(repo), **extra})

    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": _expected_detail(repo)}
    assert not (repo / ".trelix").exists(), (
        "a read route must not create .trelix/ (or index.db) in an unindexed repo"
    )


@pytest.mark.parametrize(("path", "extra"), _READ_ROUTES)
def test_read_route_refuses_before_opening_any_store(
    client: TestClient, repo: Path, path: str, extra: dict[str, str]
) -> None:
    with (
        patch("trelix.api.app.Retriever") as retriever,
        patch("trelix.graph.builder.GraphBuilder") as builder,
        patch("trelix.store.db.Database") as database,
    ):
        resp = client.get(path, params={"repo": str(repo), **extra})

    assert resp.status_code == 400, resp.text
    retriever.assert_not_called()
    builder.assert_not_called()
    database.assert_not_called()


def test_stats_on_an_indexed_repo_still_answers(client: TestClient, repo: Path) -> None:
    mark_indexed(repo)

    resp = client.get("/stats", params={"repo": str(repo)})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"files": 0, "symbols": 0, "chunks": 0}


def test_search_on_an_indexed_repo_still_reaches_the_retriever(
    client: TestClient, repo: Path
) -> None:
    mark_indexed(repo)
    context = MagicMock()
    context.results = []

    with patch("trelix.api.app.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = context
        resp = client.get("/search", params={"repo": str(repo), "query": "add"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"results": [], "next_cursor": None, "total_available": 0}
    retriever.assert_called_once()


def test_graph_on_an_indexed_repo_still_builds(client: TestClient, repo: Path) -> None:
    mark_indexed(repo)

    resp = client.get("/graph", params={"repo": str(repo)})

    assert resp.status_code == 200, resp.text
    assert resp.json()["node_count"] == 0


class _CreatingIndexer:
    """Stands in for Indexer: like the real one, it creates the index when it runs."""

    def __init__(self, config: Any) -> None:  # noqa: ANN401 - mirrors Indexer(config)
        self._config = config

    def index(self) -> dict[str, Any]:
        Database(self._config.db_path_absolute).close()
        return {
            "files_found": 1,
            "files_indexed": 1,
            "files_skipped": 0,
            "symbols_extracted": 1,
            "chunks_total": 1,
            "chunks_embedded": 1,
            "errors": 0,
            "elapsed_seconds": 0.5,
        }


def test_post_index_still_creates_the_index_that_the_read_routes_then_accept(
    client: TestClient, repo: Path
) -> None:
    """Indexing is the one route that must work on a repo with no index yet."""
    with patch("trelix.indexing.indexer.Indexer", _CreatingIndexer):
        indexed = client.post("/index", json={"repo_path": str(repo)})

    assert indexed.status_code == 200, indexed.text
    assert (repo / ".trelix" / "index.db").exists()
    stats = client.get("/stats", params={"repo": str(repo)})
    assert stats.status_code == 200, stats.text


def test_post_parse_on_an_unindexed_repo_still_works_and_creates_nothing(
    client: TestClient, repo: Path
) -> None:
    resp = client.post(
        "/parse",
        json={"repo_path": str(repo), "content": "def f():\n    return 1\n", "file_name": "f.py"},
    )

    assert resp.status_code == 200, resp.text
    assert [s["name"] for s in resp.json()["symbols"]] == ["f"]
    assert not (repo / ".trelix").exists()


def test_a_repo_that_does_not_exist_is_still_the_routes_own_error(
    client: TestClient, repo: Path
) -> None:
    """Not this check's business: /ask streams the error frame, the other routes answer 500,
    exactly as before the check existed."""
    ghost = repo / "ghost"

    ask = client.get("/ask", params={"repo": str(ghost), "query": "add"})
    stats = client.get("/stats", params={"repo": str(ghost)})

    assert ask.status_code == 200, ask.text
    assert "[ERROR: 1 validation error for IndexConfig" in ask.text
    assert stats.status_code == 500, stats.text
    assert not ghost.exists()


def test_health_is_unaffected(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_a_caller_without_the_credential_learns_nothing_about_the_index(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authentication comes first: 401, not the 400 that would confirm the repo is unindexed."""
    monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", "test-k")
    client = TestClient(create_app(), raise_server_exceptions=False)

    resp = client.get("/stats", params={"repo": str(repo)})

    assert resp.status_code == 401, resp.text
    assert not (repo / ".trelix").exists()


def test_a_repo_outside_every_served_root_is_403_not_400(
    repo: Path, tmp_path: Path, client: TestClient
) -> None:
    """Containment comes before the index check, so an out-of-root path is not probed."""
    outside = tmp_path / "outside"
    outside.mkdir()

    resp = client.get("/stats", params={"repo": str(outside)})

    assert resp.status_code == 403, resp.text
    assert not (outside / ".trelix").exists()
