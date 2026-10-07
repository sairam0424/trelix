from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import trelix_mcp.server as srv
from budget_support import seed


@pytest.fixture
def mark_indexed():
    """Return a function that gives a repo directory an empty but real index.

    Every read tool refuses a repository with no `.trelix/index.db`. A test that patches
    the Retriever or GraphBuilder behind a tool and points it at a bare `tmp_path` calls
    this first to say "this repository is indexed".
    """

    def _mark(repo: Path) -> Path:
        from trelix.core.config import IndexConfig
        from trelix.store.db import Database

        repo.mkdir(parents=True, exist_ok=True)
        Database(IndexConfig(repo_path=str(repo)).db_path_absolute).close()
        return repo

    return _mark


@pytest.fixture(autouse=True)
def _reset_retriever_cache():
    """Clear server.py's cross-call Retriever cache before every test.

    Many tests reuse the literal path "/fake/repo" with a fresh
    `patch("trelix_mcp.server.Retriever")` each time. Without this, the
    cache added for real MCP server processes (see server.py's
    `_get_retriever`) would let one test's mocked Retriever leak into a
    later test that expects its own mock to be the one actually called —
    exactly the collision a real long-lived server process is designed to
    avoid, but tests deliberately construct a "logically fresh" retriever
    under the same fake path every time.
    """
    import trelix_mcp.server as server

    server._retriever_cache.clear()
    yield
    server._retriever_cache.clear()


@pytest.fixture(autouse=True)
def _restore_confinement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo what main() or a test does to the singleton server's confinement.

    `_install_confinement` sets `_allowed_roots` and appends a RepoConfinementMiddleware to
    `mcp.middleware`. The test gets a copy of the middleware list, so what it appends never reaches
    the next test, and `_allowed_roots` goes back to its value. An operator's exported
    TRELIX_ALLOWED_REPO_ROOTS must not confine a test either.
    """
    monkeypatch.setattr(srv.mcp, "middleware", list(srv.mcp.middleware))
    monkeypatch.setattr(srv, "_allowed_roots", srv._allowed_roots)
    monkeypatch.delenv("TRELIX_ALLOWED_REPO_ROOTS", raising=False)


@pytest.fixture(autouse=True)
def _clean_output_limit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator's TRELIX_MCP_MAX_K, TRELIX_MCP_MAX_RESULT_CHARS or
    TRELIX_MCP_RETRIEVER_CACHE_SIZE must not reach a test."""
    monkeypatch.delenv("TRELIX_MCP_MAX_K", raising=False)
    monkeypatch.delenv("TRELIX_MCP_MAX_RESULT_CHARS", raising=False)
    monkeypatch.delenv("TRELIX_MCP_RETRIEVER_CACHE_SIZE", raising=False)


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Serve `backends.hits` from search_code, graph_search_mcp and federation_search_all."""
    state = SimpleNamespace(hits=[], graph_max_results=None, retriever=MagicMock())
    state.retriever.retrieve.side_effect = lambda *args, **kwargs: SimpleNamespace(
        results=state.hits
    )
    monkeypatch.setattr(srv, "_get_retriever", lambda repo_path: state.retriever)

    def graph_search(db, graph, seeds, depth, max_results):  # type: ignore[no-untyped-def]
        state.graph_max_results = max_results
        return state.hits[:max_results]

    monkeypatch.setattr("trelix.graph.search.graph_search", graph_search)
    monkeypatch.setattr("trelix.graph.builder.GraphBuilder", MagicMock())
    registry = MagicMock()
    registry.load.return_value.list.return_value = [SimpleNamespace(alias="repo-a")]
    monkeypatch.setattr(srv, "RepoRegistry", registry)
    federated = MagicMock()
    federated.return_value.repos_queried_count.return_value = 1
    federated.return_value.unindexed_repos.return_value = []
    federated.return_value.retrieve.side_effect = lambda query, k: state.hits[:k]
    monkeypatch.setattr(srv, "FederatedRetriever", federated)
    return state


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Serve `sessions.items` from agent_list_sessions.

    Replaces the server's repo_path check, IndexConfig and Database (the tests' repo path is not
    a directory), so a test that also needs a real index (blast_radius, get_symbol) must not use it.
    """
    state = SimpleNamespace(items=[], database=MagicMock())
    state.database.list_agent_sessions.side_effect = lambda limit: state.items[:limit]
    config = MagicMock()
    config.return_value.retrieval.agent_session_max_age_seconds = 604_800.0
    monkeypatch.setattr(srv, "check_repo_dir", lambda repo_path, name="repo_path": None)
    monkeypatch.setattr(srv, "IndexConfig", config)
    monkeypatch.setattr(srv, "Database", MagicMock(return_value=state.database))
    return state


@pytest.fixture(scope="module")
def dependents_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real index: `target.run` with 520 dependents, one per file."""
    return seed(tmp_path_factory.mktemp("dependents"), 520)


@pytest.fixture(scope="module")
def long_path_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real index: `target.run` with 520 dependents whose files have long (72-character) paths."""
    return seed(
        tmp_path_factory.mktemp("long-paths"),
        520,
        caller_dir="src/some/deeply/nested/package/directory/structure",
    )


@pytest.fixture(scope="module")
def small_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real index: `target.run` with 3 dependents."""
    return seed(tmp_path_factory.mktemp("small"), 3)


@pytest.fixture(scope="module")
def lonely_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real index: `target.run` is known but has no dependents at all."""
    return seed(tmp_path_factory.mktemp("lonely"), 0)
