"""Bidirectional parity between the registered MCP tools and `server.json`.

trelix-mcp's server.py registers 15 tools with `@mcp.tool()` (`test_server.py`'s
`test_four_tools_registered` already pins that count against the FastMCP
runtime object). Separately, `packages/trelix-mcp/server.json` -- the MCP
registry manifest published alongside the package -- carries its OWN,
hand-written list of tool names+descriptions under its `"tools"` key. Nothing
before this file checked that the two lists agree: `grep -rl "server.json"`
inside `packages/trelix-mcp/tests/` and this repo's `tests/` returns only
`tests/unit/test_release_version_gate.py`, which reads server.json's
`"version"` field, never its `"tools"` list.

That is a real drift surface, not a hypothetical one: a new `@mcp.tool()` in
server.py ships to users immediately (FastMCP registers it at import time) but
would be invisible to any registry/marketplace UI that renders server.json's
`tools` array, and a tool removed from server.py without removing its
server.json entry advertises a capability that 404s on the first call.

FALSIFIED BY (verified live below, not assumed -- see
`test_a_tool_missing_from_server_json_is_caught` and
`test_a_stale_server_json_entry_is_caught`):
  * adding a tool to server.py's registered set without adding it to
    server.json's `tools` array
  * adding an entry to server.json's `tools` array that names no real
    `@mcp.tool()`
Both are simulated in-process against the real JSON parsed from disk and the
real `FastMCP.list_tools()` result -- no mock stands in for either side of the
comparison (rule 3): the "mutation" is a small in-memory copy-and-edit of the
set actually read from `server.json`, compared against the real registered
name set, which is exactly what a real drift would do to this same assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_SERVER_JSON = Path(__file__).parent.parent / "server.json"


def _server_json_tool_names() -> frozenset[str]:
    data = json.loads(_SERVER_JSON.read_text())
    return frozenset(entry["name"] for entry in data["tools"])


@pytest.mark.asyncio
async def test_server_json_tools_match_registered_tools_bidirectionally() -> None:
    """Set equality both ways (rule 2): a tool on either side alone is a defect.

    This is the real assertion the module docstring's "may already be fine"
    caveat resolves: it is NOT already fine as a covered case -- no test
    before this one read server.json's `tools` key at all.
    """
    import trelix_mcp.server as srv

    registered = frozenset(t.name for t in await srv.mcp.list_tools())
    declared = _server_json_tool_names()

    assert registered - declared == frozenset(), (
        f"tool(s) registered in server.py but missing from server.json's "
        f"'tools' array: {sorted(registered - declared)}"
    )
    assert declared - registered == frozenset(), (
        f"server.json advertises tool(s) that server.py does not register: "
        f"{sorted(declared - registered)}"
    )


@pytest.mark.asyncio
async def test_a_tool_missing_from_server_json_is_caught() -> None:
    """CONTROL: simulate server.py gaining a tool server.json never learned about.

    Never edits the file on disk -- the control operates on the real name set
    read from the real registered tools and the real server.json, exactly as
    the assertion above does, so a drift there would fail the SAME comparison
    this test performs.
    """
    import trelix_mcp.server as srv

    registered = frozenset(t.name for t in await srv.mcp.list_tools())
    declared_missing_one_registered_tool = _server_json_tool_names() - {"search_code"}

    with pytest.raises(AssertionError):
        assert registered - declared_missing_one_registered_tool == frozenset()


@pytest.mark.asyncio
async def test_a_stale_server_json_entry_is_caught() -> None:
    """CONTROL: simulate server.json advertising a tool that was removed from server.py."""
    import trelix_mcp.server as srv

    registered = frozenset(t.name for t in await srv.mcp.list_tools())
    declared_with_phantom_tool = _server_json_tool_names() | {"this_tool_does_not_exist"}

    with pytest.raises(AssertionError):
        assert declared_with_phantom_tool - registered == frozenset()


def test_server_json_tools_array_is_non_empty_so_the_comparison_is_not_vacuous() -> None:
    """Precondition guard (rule 4): if server.json ever loses its `tools` key or it
    goes empty, the two assertions above would pass by both sides being `{}`,
    which is the fixture-makes-it-true-by-construction failure mode. Fail loudly
    here instead of silently validating nothing."""
    declared = _server_json_tool_names()
    assert len(declared) >= 1, "server.json's 'tools' array is empty — parity check is vacuous"
