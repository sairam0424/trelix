import pytest


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
