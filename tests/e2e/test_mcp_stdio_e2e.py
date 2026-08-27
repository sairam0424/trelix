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
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_TRELIX_MCP_EXE = shutil.which("trelix-mcp")

# A generous ceiling for process spawn + MCP handshake on a cold interpreter —
# not a tight bound, just a guard against a genuine hang reading forever.
_HANDSHAKE_TIMEOUT = 30.0


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
            assert result.serverInfo.name == "trelix"

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
