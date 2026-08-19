"""Tests for graph REST API endpoints."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trelix.api.app import create_app
from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import Database

from .test_graph_visualizer import _make_pyvis_mock


@pytest.fixture
def allow_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Register this test's ``tmp_path`` as an allowed repository root.

    Opt-in per class, never autouse — see the twin fixture in ``test_api.py``
    for why an autouse version would make the suite structurally unable to
    notice a containment regression. ``TestGraphVisualizeContainment``'s last
    case deliberately does not go through it.
    """
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(tmp_path))
    return tmp_path


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
    db._conn.commit()
    db.close()
    return tmp_path


@pytest.mark.usefixtures("allow_repo_root")
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
        assert data["node_count"] == 1  # fixture inserts exactly 1 symbol

    def test_graph_communities(self, tmp_path: Path) -> None:
        """The fixture's single symbol forms a SINGLETON community.

        `min_community_size=1` is explicit here because the endpoint now defaults to 2:
        on a real repo 99.1% of detected communities are singletons and they were the
        bulk of a ~1.16 MB response. A one-symbol fixture produces nothing but that
        case, so the endpoint's plumbing has to be tested with the filter opened up.
        """
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        response = client.get(f"/graph/communities?repo={repo}&min_community_size=1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # at least one community from 1 symbol

    def test_graph_communities_excludes_singletons_by_default(self, tmp_path: Path) -> None:
        """The default filters the noise that dominated the payload."""
        repo = _make_indexed_repo(tmp_path)
        client = TestClient(create_app())

        default = client.get(f"/graph/communities?repo={repo}")
        opened = client.get(f"/graph/communities?repo={repo}&min_community_size=1")

        assert default.status_code == 200
        assert default.json() == [], (
            "a singleton-only graph should return nothing under the default filter"
        )
        assert len(opened.json()) >= 1, "the opt-out must still return the singleton"

    def test_graph_communities_response_is_still_a_bare_list(self, tmp_path: Path) -> None:
        """Only the LENGTH changes for an existing consumer, never the shape."""
        repo = _make_indexed_repo(tmp_path)
        client = TestClient(create_app())
        body = client.get(f"/graph/communities?repo={repo}&min_community_size=1").json()

        assert isinstance(body, list)
        assert all(isinstance(entry, dict) for entry in body)

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

    def test_graph_search_with_connected_nodes(self, tmp_path: Path) -> None:
        from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
        from trelix.store.db import Database

        db_dir = tmp_path / ".trelix"
        db_dir.mkdir(parents=True)
        db = Database(db_dir / "index.db")
        fid = db.upsert_file(
            IndexedFile(
                path=str(tmp_path / "a.py"),
                rel_path="a.py",
                language=Language.PYTHON,
                hash="x",
                size_bytes=10,
            )
        )
        sid1 = db.insert_symbol(
            Symbol(
                file_id=fid,
                name="fn_a",
                qualified_name="fn_a",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=3,
                signature="def fn_a()",
                body="def fn_a(): pass",
            )
        )
        sid2 = db.insert_symbol(
            Symbol(
                file_id=fid,
                name="fn_b",
                qualified_name="fn_b",
                kind=SymbolKind.FUNCTION,
                line_start=5,
                line_end=7,
                signature="def fn_b()",
                body="def fn_b(): pass",
            )
        )
        db.insert_call_edges([CallEdge(caller_id=sid1, callee_name="fn_b", callee_id=sid2, line=2)])
        db.insert_chunk_for_symbol(sid1, "def fn_a(): pass", 5)
        db.insert_chunk_for_symbol(sid2, "def fn_b(): pass", 5)
        db._conn.commit()
        db.close()
        app = create_app()
        client = TestClient(app)
        response = client.get(f"/graph/search?repo={tmp_path}&symbol_id={sid1}&depth=1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # fn_b is a neighbor of fn_a


@pytest.mark.usefixtures("allow_repo_root")
class TestGraphVisualizeContainment:
    """Regression tests for the output-path containment check.

    Before the fix, the check was a raw string prefix match
    (str(requested).startswith(str(allowed))), which wrongly accepts a
    sibling directory that merely starts with the same characters as
    "<repo>/.trelix" (e.g. "<repo>/.trelix-evil"). Path.is_relative_to()
    correctly rejects it.

    Those four cases are kept verbatim: they are a genuine regression suite for
    the prefix-match bug. They say nothing about the ROOT the comparison is
    anchored to, which is the point — ``allowed = Path(repo).resolve()/".trelix"``
    was derived from the caller's own ``repo``, so an unauthenticated GET could
    create ``<any-existing-directory>/.trelix/<name>.html`` anywhere on the
    host and the four tests below would still have passed. The fifth case, which
    does not go through this file's allow-list fixture, covers that.
    """

    def test_default_output_path_is_accepted(self, tmp_path: Path) -> None:
        _make_pyvis_mock()
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        response = client.get(f"/graph/visualize?repo={repo}")
        assert response.status_code == 200
        data = response.json()
        assert data["path"] == str(repo / ".trelix" / "graph.html")
        assert Path(data["path"]).exists()

    def test_output_inside_trelix_dir_is_accepted(self, tmp_path: Path) -> None:
        _make_pyvis_mock()
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        output = str(repo / ".trelix" / "custom.html")
        response = client.get(f"/graph/visualize?repo={repo}&output={output}")
        assert response.status_code == 200
        assert response.json()["path"] == output

    def test_sibling_directory_sharing_trelix_prefix_is_rejected(self, tmp_path: Path) -> None:
        """<repo>/.trelix-evil/x.html starts with the same characters as
        <repo>/.trelix but is not inside it — must be rejected."""
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        evil_dir = repo / ".trelix-evil"
        evil_dir.mkdir()
        output = str(evil_dir / "x.html")
        response = client.get(f"/graph/visualize?repo={repo}&output={output}")
        assert response.status_code == 400
        assert "must be inside" in response.json()["detail"]
        assert not Path(output).exists()

    def test_output_outside_repo_entirely_is_rejected(self, tmp_path: Path) -> None:
        """Still refused; the status moved from 400 to 403 and that is correct.

        ``tmp_path.parent`` is above this test's allowed root, so the shared
        containment dependency answers before the route's repo-relative check
        gets a turn. The repo-relative layer is still exercised on its own by
        the sibling-prefix case above, whose target IS inside the allowed root.
        """
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        outside = tmp_path.parent / "outside.html"
        response = client.get(f"/graph/visualize?repo={repo}&output={outside}")
        assert response.status_code == 403
        assert not outside.exists()

    def test_repo_the_server_was_never_pointed_at_cannot_be_written_to(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The root itself — the case the four above structurally cannot reach.

        ``repo`` names a directory this server was never pointed at, and
        ``output`` is dutifully inside that directory's ``.trelix``, so the
        repo-relative check is satisfied and used to answer 200: an
        unauthenticated GET that created a directory and a ~12 KB file on the
        host. The allow-list refuses it before the graph is built.

        The allow-list is pointed at a DIFFERENT existing directory rather than
        left unset, so this proves "not this root" and not merely the weaker
        "no root configured".
        """
        _make_pyvis_mock()
        served = tmp_path / "served"
        served.mkdir()
        monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(served))

        victim = tmp_path / "victim"
        victim.mkdir()
        output = victim / ".trelix" / "planted.html"

        client = TestClient(create_app())
        response = client.get(f"/graph/visualize?repo={victim}&output={output}")

        assert response.status_code == 403
        assert not output.exists()
        assert os.listdir(victim) == [], "a refused route created a directory on disk"
