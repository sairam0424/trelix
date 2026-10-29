"""A symlinked index store never leaves the allowed roots (REST).

A repository is untrusted content, and git tracks symlinks. A ``.trelix`` directory, an
``index.db`` file or any other entry directly inside ``.trelix`` that is a symlink moves the
store (and everything trelix writes beside it) outside the allowed roots without the request
naming anything outside them: ``confine_repo`` compares the path the caller named, and the
repository itself is inside the root.

Each case builds the hostile repository, asserts a refusal, and compares the bytes that must not
move before and after. The first proposed rule, "check the parent of the resolved database
path", passes a real ``.trelix/`` whose ``index.db`` is the link, so that case has its own test.
Every expected value is a literal.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.api_confinement import CreatingIndexer, pyvis_without_asserts, tree
from tests.fixtures.indexed import mark_indexed
from trelix.api.app import create_app
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# A symlinked index store never leaves the allowed roots
# ---------------------------------------------------------------------------

# Root, and Windows, ignore directory permission bits: the denial the tests below need cannot be
# made there.
_CANNOT_DENY_ACCESS = sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0

_READ_ROUTES = [
    pytest.param("/search", {"query": "add"}, id="search"),
    pytest.param("/ask", {"query": "add"}, id="ask"),
    pytest.param("/stats", {}, id="stats"),
    pytest.param("/graph", {}, id="graph"),
    pytest.param("/graph/communities", {}, id="graph-communities"),
    pytest.param("/graph/visualize", {}, id="graph-visualize"),
    pytest.param("/graph/search", {"symbol_id": "1"}, id="graph-search"),
]


class _StoreWorld(NamedTuple):
    root: Path  # the only allowed root
    repo: Path  # a repository inside it
    foreign: Path  # a repository OUTSIDE it, holding a real index
    client: TestClient


@pytest.fixture
def store_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _StoreWorld:
    root = tmp_path / "root"
    repo = root / "hostile"
    repo.mkdir(parents=True)
    foreign = mark_indexed(tmp_path / "foreign")
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(root))
    client = TestClient(create_app(), raise_server_exceptions=False)
    return _StoreWorld(root, repo, foreign, client)


class TestSymlinkedStoreIsRefused:
    @pytest.mark.parametrize(("path", "extra"), _READ_ROUTES)
    def test_a_trelix_directory_symlink_is_refused_by_every_read_route(
        self, store_world: _StoreWorld, path: str, extra: dict[str, str]
    ) -> None:
        (store_world.repo / ".trelix").symlink_to(store_world.foreign / ".trelix")
        before = tree(store_world.foreign)

        with patch("trelix.api.app.Retriever") as retriever:
            retriever.return_value.retrieve.return_value = MagicMock(results=[])
            resp = store_world.client.get(path, params={"repo": str(store_world.repo), **extra})

        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": "repo has a symlinked index store"}
        retriever.assert_not_called()
        assert tree(store_world.foreign) == before

    @pytest.mark.parametrize(("path", "extra"), _READ_ROUTES)
    def test_an_index_db_file_symlink_is_refused_by_every_read_route(
        self, store_world: _StoreWorld, path: str, extra: dict[str, str]
    ) -> None:
        """A real ``.trelix/`` whose ``index.db`` is the link: a parent-directory check passes."""
        (store_world.repo / ".trelix").mkdir()
        (store_world.repo / ".trelix" / "index.db").symlink_to(
            store_world.foreign / ".trelix" / "index.db"
        )
        before = tree(store_world.foreign)

        with patch("trelix.api.app.Retriever") as retriever:
            retriever.return_value.retrieve.return_value = MagicMock(results=[])
            resp = store_world.client.get(path, params={"repo": str(store_world.repo), **extra})

        assert resp.status_code == 403, resp.text
        retriever.assert_not_called()
        assert tree(store_world.foreign) == before

    def test_a_dangling_trelix_symlink_is_refused_and_creates_nothing(
        self, store_world: _StoreWorld, tmp_path: Path
    ) -> None:
        nowhere = tmp_path / "nowhere"
        (store_world.repo / ".trelix").symlink_to(nowhere)

        stats = store_world.client.get("/stats", params={"repo": str(store_world.repo)})
        with patch("trelix.indexing.indexer.Indexer", CreatingIndexer):
            index = store_world.client.post("/index", json={"repo_path": str(store_world.repo)})

        assert stats.status_code == 403, stats.text
        assert index.status_code == 403, index.text
        assert not nowhere.exists()

    def test_the_log_names_the_link_and_the_response_does_not(
        self, store_world: _StoreWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The 403 omits the path on purpose (it is unauthenticated on the shipped default), so
        the server log is the only place the operator learns which entry was refused."""
        (store_world.repo / ".trelix").symlink_to(store_world.foreign / ".trelix")
        caplog.set_level(logging.WARNING, logger="trelix.api")

        resp = store_world.client.get("/stats", params={"repo": str(store_world.repo)})

        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": "repo has a symlinked index store"}
        assert ".trelix" not in resp.text
        refusals = [r.getMessage() for r in caplog.records if r.name == "trelix.api"]
        assert len(refusals) == 1
        link = str(store_world.repo.resolve() / ".trelix")
        assert refusals[0].endswith(f"has a symlinked index store entry at {link!r}")

    @pytest.mark.skipif(sys.platform == "win32", reason="a Windows file name cannot hold a newline")
    def test_an_entry_name_with_a_newline_cannot_forge_a_log_record(
        self, store_world: _StoreWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Entry names under ``.trelix`` come from the repository, so the name is logged quoted."""
        store = store_world.repo / ".trelix"
        store.mkdir()
        (store / "x\nWARNING trelix.api:forged line").symlink_to(store_world.foreign / ".trelix")
        caplog.set_level(logging.WARNING, logger="trelix.api")

        resp = store_world.client.get("/stats", params={"repo": str(store_world.repo)})

        assert resp.status_code == 403, resp.text
        refusals = [r.getMessage() for r in caplog.records if r.name == "trelix.api"]
        assert len(refusals) == 1
        assert len(refusals[0].splitlines()) == 1
        assert r"x\nWARNING trelix.api:forged line" in refusals[0]

    def test_a_repository_outside_the_roots_never_learns_about_its_store(
        self, store_world: _StoreWorld, tmp_path: Path
    ) -> None:
        """The roots check answers first, so the 403 for a path outside them cannot be told
        apart by what that directory holds (it would otherwise be an oracle for ``<dir>/.trelix``
        being a symlink, on an unauthenticated default)."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / ".trelix").symlink_to(store_world.foreign / ".trelix")

        stats = store_world.client.get("/stats", params={"repo": str(outside)})
        index = store_world.client.post("/index", json={"repo_path": str(outside)})

        assert stats.status_code == 403, stats.text
        assert stats.json() == {"detail": "repo is not inside an allowed repository root"}
        assert index.status_code == 403, index.text
        assert index.json() == {"detail": "repo_path is not inside an allowed repository root"}

    def test_a_dangling_index_db_symlink_is_refused(
        self, store_world: _StoreWorld, tmp_path: Path
    ) -> None:
        (store_world.repo / ".trelix").mkdir()
        (store_world.repo / ".trelix" / "index.db").symlink_to(tmp_path / "nowhere.db")

        resp = store_world.client.get("/stats", params={"repo": str(store_world.repo)})

        assert resp.status_code == 403, resp.text
        assert not (tmp_path / "nowhere.db").exists()

    def test_post_index_on_a_symlinked_trelix_writes_nothing_outside(
        self, store_world: _StoreWorld, tmp_path: Path
    ) -> None:
        """``POST /index`` is gated but needs no index, so it was the widest way in."""
        foreign_dir = tmp_path / "foreign_dir"
        foreign_dir.mkdir()
        (store_world.repo / ".trelix").symlink_to(foreign_dir)

        with patch("trelix.indexing.indexer.Indexer", CreatingIndexer):
            resp = store_world.client.post("/index", json={"repo_path": str(store_world.repo)})

        assert resp.status_code == 403, resp.text
        assert list(foreign_dir.iterdir()) == []

    def test_a_dangling_gitignore_symlink_cannot_create_a_file_elsewhere(
        self, store_world: _StoreWorld, tmp_path: Path
    ) -> None:
        """The store prepares ``.trelix/.gitignore``; ``exists()`` is False for a dangling
        link, so it wrote through the link and created the target."""
        (store_world.repo / ".trelix").mkdir()
        target = tmp_path / "created_elsewhere"
        (store_world.repo / ".trelix" / ".gitignore").symlink_to(target)

        with patch("trelix.indexing.indexer.Indexer", CreatingIndexer):
            resp = store_world.client.post("/index", json={"repo_path": str(store_world.repo)})

        assert resp.status_code == 403, resp.text
        assert not target.exists()

    def test_a_graph_html_symlink_cannot_overwrite_a_file_elsewhere(
        self, store_world: _StoreWorld, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default output (no ``output=``) was never resolved, so it wrote through a link."""
        pyvis_without_asserts(monkeypatch)
        mark_indexed(store_world.repo)
        victim = tmp_path / "victim.txt"
        victim.write_text("precious\n", encoding="utf-8")
        (store_world.repo / ".trelix" / "graph.html").symlink_to(victim)

        resp = store_world.client.get("/graph/visualize", params={"repo": str(store_world.repo)})

        assert resp.status_code == 403, resp.text
        assert victim.read_text(encoding="utf-8") == "precious\n"

    def test_a_debug_directory_symlink_is_refused(
        self, store_world: _StoreWorld, tmp_path: Path
    ) -> None:
        """Retrieval writes a trace file under ``.trelix/debug`` on every query."""
        mark_indexed(store_world.repo)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (store_world.repo / ".trelix" / "debug").symlink_to(elsewhere)

        resp = store_world.client.get("/stats", params={"repo": str(store_world.repo)})

        assert resp.status_code == 403, resp.text
        assert list(elsewhere.iterdir()) == []

    def test_post_parse_is_confined_too(self, store_world: _StoreWorld) -> None:
        (store_world.repo / ".trelix").symlink_to(store_world.foreign / ".trelix")

        resp = store_world.client.post(
            "/parse",
            json={"repo_path": str(store_world.repo), "content": "x = 1\n", "file_name": "x.py"},
        )

        assert resp.status_code == 403, resp.text

    def test_a_path_too_long_to_examine_is_a_403_not_a_500(self, store_world: _StoreWorld) -> None:
        """``lstat`` on a 300-character name raises ENAMETOOLONG, an ``OSError`` that is not a
        ``PermissionError``: the guard must refuse on any failure to look, not only on EACCES."""
        repo = store_world.root / ("a" * 300)

        stats = store_world.client.get("/stats", params={"repo": str(repo)})
        index = store_world.client.post("/index", json={"repo_path": str(repo)})

        assert stats.status_code == 403, stats.text
        assert stats.json() == {"detail": "repo has an index store that cannot be examined"}
        assert index.status_code == 403, index.text
        assert index.json() == {"detail": "repo_path has an index store that cannot be examined"}


@pytest.fixture(params=["absolute", "relative"])
def moved_store(
    request: pytest.FixtureRequest,
    store_world: _StoreWorld,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _StoreWorld:
    """``store_world`` with the operator's store moved off ``<repo>/.trelix/index.db``.

    ``confine_repo`` examines the configured store path, so here it never looks at ``.trelix``.
    The pyvis stand-in writes wherever it is told, as real pyvis does under ``python -O``.
    """
    db_path = str(tmp_path / "operator" / "index.db") if request.param == "absolute" else "data/db"
    monkeypatch.setenv("TRELIX_STORE_DB_PATH", db_path)
    mark_indexed(store_world.repo)
    pyvis_without_asserts(monkeypatch)
    return store_world


class TestGraphVisualizeWritesInsideTheRepository:
    """``GET /graph/visualize`` writes ``<repo>/.trelix/graph.html`` whatever the store path is."""

    def test_a_graph_html_symlink_is_refused(
        self, moved_store: _StoreWorld, tmp_path: Path
    ) -> None:
        victim = tmp_path / "victim.txt"
        victim.write_text("precious\n", encoding="utf-8")
        (moved_store.repo / ".trelix").mkdir()
        (moved_store.repo / ".trelix" / "graph.html").symlink_to(victim)

        resp = moved_store.client.get("/graph/visualize", params={"repo": str(moved_store.repo)})

        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": "repo has a symlinked index store"}
        assert victim.read_text(encoding="utf-8") == "precious\n"

    def test_a_trelix_directory_symlink_is_refused(
        self, moved_store: _StoreWorld, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (moved_store.repo / ".trelix").symlink_to(elsewhere, target_is_directory=True)

        resp = moved_store.client.get("/graph/visualize", params={"repo": str(moved_store.repo)})

        assert resp.status_code == 403, resp.text
        assert resp.json() == {"detail": "repo has a symlinked index store"}
        assert list(elsewhere.iterdir()) == []

    def test_a_repository_without_links_still_gets_its_graph(
        self, moved_store: _StoreWorld
    ) -> None:
        resp = moved_store.client.get("/graph/visualize", params={"repo": str(moved_store.repo)})

        assert resp.status_code == 200, resp.text
        assert (moved_store.repo / ".trelix" / "graph.html").is_file()


@pytest.mark.skipif(_CANNOT_DENY_ACCESS, reason="root and Windows ignore directory permissions")
class TestAnUnexaminableStoreIsRefused:
    """The guard is a security check: when it cannot look, it refuses, it does not crash or pass."""

    def test_an_unreadable_trelix_is_a_403_not_a_500(self, store_world: _StoreWorld) -> None:
        """``POST /parse`` never opens the store, but the guard still has to decide."""
        mark_indexed(store_world.repo)
        store = store_world.repo / ".trelix"
        store.chmod(0o000)
        try:
            stats = store_world.client.get("/stats", params={"repo": str(store_world.repo)})
            parse = store_world.client.post(
                "/parse",
                json={
                    "repo_path": str(store_world.repo),
                    "content": "x = 1\n",
                    "file_name": "x.py",
                },
            )
        finally:
            store.chmod(0o700)  # tmp_path cleanup must be able to remove it

        assert stats.status_code == 403, stats.text
        assert stats.json() == {"detail": "repo has an index store that cannot be examined"}
        assert parse.status_code == 403, parse.text
        assert parse.json() == {"detail": "repo_path has an index store that cannot be examined"}

    def test_a_trelix_that_can_be_entered_but_not_listed_cannot_hide_a_link(
        self, store_world: _StoreWorld, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Entering a directory needs no read permission, so ``graph.html`` can still be
        opened by name even though the directory cannot be listed."""
        pyvis_without_asserts(monkeypatch)
        mark_indexed(store_world.repo)
        victim = tmp_path / "victim.txt"
        victim.write_text("precious\n", encoding="utf-8")
        store = store_world.repo / ".trelix"
        (store / "graph.html").symlink_to(victim)
        store.chmod(0o100)
        try:
            resp = store_world.client.get(
                "/graph/visualize", params={"repo": str(store_world.repo)}
            )
        finally:
            store.chmod(0o700)

        assert resp.status_code == 403, resp.text
        assert victim.read_text(encoding="utf-8") == "precious\n"

    def test_the_log_names_the_entry_that_could_not_be_examined(
        self, store_world: _StoreWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        mark_indexed(store_world.repo)
        store = store_world.repo / ".trelix"
        store.chmod(0o000)
        caplog.set_level(logging.WARNING, logger="trelix.api")
        try:
            resp = store_world.client.get("/stats", params={"repo": str(store_world.repo)})
        finally:
            store.chmod(0o700)

        assert ".trelix" not in resp.text
        refusals = [r.getMessage() for r in caplog.records if r.name == "trelix.api"]
        assert len(refusals) == 1
        assert "cannot be examined" in refusals[0]
        # Python 3.12 and 3.13 raise on ``.trelix/index.db`` (is_symlink reports EACCES); 3.14
        # swallows it and the listing of ``.trelix`` raises instead. The directory is the part
        # of the entry's path that every version names.
        assert str(store_world.repo.resolve() / ".trelix") in refusals[0]


class TestOrdinaryRepositoriesStillWork:
    def test_a_real_trelix_directory_is_served(
        self, store_world: _StoreWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyvis_without_asserts(monkeypatch)
        mark_indexed(store_world.repo)
        (store_world.repo / "vendor").symlink_to(store_world.foreign, target_is_directory=True)

        stats = store_world.client.get("/stats", params={"repo": str(store_world.repo)})
        graph = store_world.client.get("/graph/visualize", params={"repo": str(store_world.repo)})

        assert stats.status_code == 200, stats.text
        assert graph.status_code == 200, graph.text
        assert (store_world.repo / ".trelix" / "graph.html").is_file()

    def test_a_symlinked_repository_root_is_served(self, store_world: _StoreWorld) -> None:
        """The check is on what is inside the repository, not on how the repository is named."""
        real = mark_indexed(store_world.root / "real")
        link = store_world.root / "link"
        link.symlink_to(real, target_is_directory=True)

        resp = store_world.client.get("/stats", params={"repo": str(link)})

        assert resp.status_code == 200, resp.text

    def test_an_operator_chosen_absolute_store_is_not_second_guessed(
        self, store_world: _StoreWorld, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``TRELIX_STORE_DB_PATH`` is operator configuration, set out of band; an absolute
        value is where the index lives whatever the repository contains, and whatever
        links lead to it (the operator's disk layout is not the repository's business)."""
        real_dir = tmp_path / "operator_real"
        real_dir.mkdir()
        (tmp_path / "operator").symlink_to(real_dir, target_is_directory=True)
        operator_db = tmp_path / "operator" / "index.db"
        Database(operator_db).close()
        monkeypatch.setenv("TRELIX_STORE_DB_PATH", str(operator_db))
        (store_world.repo / ".trelix").symlink_to(store_world.foreign / ".trelix")

        resp = store_world.client.get("/stats", params={"repo": str(store_world.repo)})

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"files": 0, "symbols": 0, "chunks": 0}
