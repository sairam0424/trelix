from pathlib import Path

import pytest


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
