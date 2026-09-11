"""Real subprocess, real stdio JSON-RPC round-trip against `trelix-mcp`.

No test anywhere in this repo previously spawned `trelix-mcp` as a real OS
subprocess — packages/trelix-mcp/tests/test_server_cli.py patches sys.argv and
mocks server_module.mcp.run() entirely in-process. The MCP Python SDK's own
"standard test client" (an in-memory Client + in-memory transport) is the same
kind of double: the SDK's own docs describe it as analogous to FastAPI's
TestClient. Neither can see a defect that only exists at the real process
boundary — which is exactly where trelix-mcp's 3.2.1 defect lived (the
console script silently ignoring --help/--version/any flag).

Uses mcp.client.stdio.stdio_client, which genuinely spawns `command`+`args` as
a child process and speaks real JSON-RPC over its real stdin/stdout — not the
SDK's in-memory alternative. Verified against the pinned versions this repo
actually installs (mcp==1.29.0, fastmcp==3.4.7) before writing this file; if a
future SDK major bump changes this API, re-verify rather than trust this
docstring.

MUTATION: rename search_code's @mcp.tool() registration in server.py to
something else and test_tools_list_includes_the_real_tools fails.

MUTATION (search_code itself): make search_code return
{"results": [], "next_cursor": None, "total_available": 0} unconditionally and
test_search_code_finds_real_results_in_a_real_index fails on the
total_available assertion — the earlier tests above only prove the tool is
listed, never that it does real retrieval against a real index.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer

_TRELIX_MCP_EXE = shutil.which("trelix-mcp")


def _sdk_field(obj: Any, new_name: str, old_name: str) -> Any:
    """Read a pydantic result field whose name mcp 2.0 renamed.

    mcp 2.0's `mcp.types` renamed several wire-model fields from camelCase to
    snake_case (`serverInfo`->`server_info`, `isError`->`is_error`,
    `structuredContent`->`structured_content`); mcp 1.x only ever exposed the
    camelCase name, and pydantic's default `BaseModel.__getattr__` raises
    `AttributeError` for the name that doesn't exist on the installed
    version rather than returning `None`. Trying the 2.x name first keeps
    this suite passing against both SDK major versions without importing
    `mcp.version`/sniffing it.
    """
    return getattr(obj, new_name) if hasattr(obj, new_name) else getattr(obj, old_name)


# A generous ceiling for process spawn + MCP handshake on a cold interpreter —
# not a tight bound, just a guard against a genuine hang reading forever.
_HANDSHAKE_TIMEOUT = 30.0

# search_code's own real round-trip does far more than a handshake: the
# trelix-mcp SUBPROCESS constructs its own Retriever, which loads a real
# local SentenceTransformer model independently of the fixture's own
# in-process indexing — measured at ~23s warm-cache locally (on top of the
# fixture's own ~37s indexing setup), because even a cache hit still makes a
# dozen-plus HTTP HEAD requests to the HF Hub to validate freshness (this repo
# does not set HF_HUB_OFFLINE for tests/e2e/, unlike ci.yml's `test` job — see
# that job's "Prefetch the local embedder model" step). A cold cache on a CI
# runner is plausibly several times slower, so this is a real ceiling, not a
# tight one — matching tests/regressions/test_regressions.py's precedent for
# overriding the suite's 60s pytest-timeout default on tests whose slowness is
# model-loading-related rather than a genuine hang.
_SEARCH_CODE_TIMEOUT = 90.0

_MINI_REPO = Path(__file__).parent.parent / "fixtures" / "mini_repo"


@pytest.fixture(scope="module")
def indexed_mini_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy ``tests/fixtures/mini_repo/`` into a fresh dir and index it with the
    local embedder — mirrors tests/integration/test_eval.py's
    ``mini_repo_dir``/``mini_repo_config`` pattern exactly (same copy-then-
    ``Indexer(...).index()`` shape), so ``search_code`` has a real index to
    query rather than hitting the "no index found" error path."""
    dest = tmp_path_factory.mktemp("mcp_search_code_e2e")
    for f in _MINI_REPO.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    config = IndexConfig(
        repo_path=str(dest),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
    )
    Indexer(config, quiet=True).index()
    return dest


@asynccontextmanager
async def _connected_session() -> AsyncIterator[ClientSession]:
    assert _TRELIX_MCP_EXE is not None, (
        "the `trelix-mcp` console script is not on PATH — this test needs a real "
        "installed trelix-mcp package, not just an importable src/ tree"
    )
    params = StdioServerParameters(command=_TRELIX_MCP_EXE, args=[])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            yield session


async def test_initialize_identifies_the_trelix_server() -> None:
    async def run() -> None:
        async with _connected_session() as session:
            result = await session.initialize()
            assert _sdk_field(result, "server_info", "serverInfo").name == "trelix"

    await asyncio.wait_for(run(), timeout=_HANDSHAKE_TIMEOUT)


async def test_tools_list_includes_the_real_tools() -> None:
    async def run() -> None:
        async with _connected_session() as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert "search_code" in names
            assert "agent_list_sessions" in names

    await asyncio.wait_for(run(), timeout=_HANDSHAKE_TIMEOUT)


@pytest.mark.timeout(180)
async def test_search_code_finds_real_results_in_a_real_index(indexed_mini_repo: Path) -> None:
    async def run() -> None:
        async with _connected_session() as session:
            await session.initialize()
            result = await session.call_tool(
                "search_code",
                {"query": "authenticate user", "repo_path": str(indexed_mini_repo)},
            )
            assert _sdk_field(result, "is_error", "isError") is False
            # Confirmed live against the installed mcp==1.29.1 / fastmcp==3.4.7:
            # a dict returned from an @mcp.tool()-decorated function is marshaled
            # into BOTH `result.content` (a single TextContent whose `.text` is
            # the JSON-encoded dict) AND `result.structuredContent` (the dict
            # itself, already parsed) — structuredContent is the direct,
            # no-reparsing access path. mcp 2.0 renamed the latter to
            # `structured_content`; see `_sdk_field`.
            payload = _sdk_field(result, "structured_content", "structuredContent")
            assert payload is not None
            assert payload["total_available"] > 0
            files = {r["file"] for r in payload["results"]}
            assert "auth.py" in files

    await asyncio.wait_for(run(), timeout=_SEARCH_CODE_TIMEOUT)
