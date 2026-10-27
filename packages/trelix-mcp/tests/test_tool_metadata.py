"""What `tools/list` and the connect handshake tell a client about trelix-mcp's tools.

Annotations, tool order, server instructions and the listing cache hint are all additive:
no tool is renamed or removed and no argument or result changes. Every expectation below is
a literal written out here, not read back from `trelix_mcp.tool_metadata`, and every check
reads what reaches the client through an in-process `fastmcp.Client` (the real registered
tools, the real transform), not the module's tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import trelix_mcp.server as srv
from fastmcp import Client

# The tools this server registered before tool metadata existed: the whole surface, none
# added or removed by it.
PRE_METADATA_TOOLS = frozenset(
    {
        "subscribe_resource",
        "unsubscribe_resource",
        "search_code",
        "index_codebase",
        "get_symbol",
        "blast_radius",
        "build_knowledge_graph",
        "graph_search_mcp",
        "federation_list_repos",
        "federation_add_repo",
        "federation_remove_repo",
        "federation_search_all",
        "ask_agent",
        "agent_list_sessions",
        "agent_clear_session",
    }
)


def _hints(read_only: bool, destructive: bool, idempotent: bool) -> dict[str, bool]:
    return {
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": idempotent,
        "openWorldHint": False,
    }


EXPECTED_ANNOTATIONS: dict[str, dict[str, bool]] = {
    "index_codebase": _hints(read_only=False, destructive=False, idempotent=True),
    "search_code": _hints(read_only=True, destructive=False, idempotent=False),
    "get_symbol": _hints(read_only=True, destructive=False, idempotent=False),
    "blast_radius": _hints(read_only=True, destructive=False, idempotent=False),
    "build_knowledge_graph": _hints(read_only=False, destructive=False, idempotent=False),
    "graph_search_mcp": _hints(read_only=False, destructive=False, idempotent=False),
    "ask_agent": _hints(read_only=False, destructive=False, idempotent=False),
    "agent_list_sessions": _hints(read_only=False, destructive=False, idempotent=False),
    "agent_clear_session": _hints(read_only=False, destructive=True, idempotent=False),
    "federation_list_repos": _hints(read_only=False, destructive=False, idempotent=False),
    "federation_add_repo": _hints(read_only=False, destructive=False, idempotent=False),
    "federation_remove_repo": _hints(read_only=False, destructive=True, idempotent=False),
    "federation_search_all": _hints(read_only=False, destructive=False, idempotent=False),
    "subscribe_resource": _hints(read_only=False, destructive=False, idempotent=False),
    "unsubscribe_resource": _hints(read_only=False, destructive=False, idempotent=False),
}

EXPECTED_ORDER = [
    "index_codebase",
    "search_code",
    "get_symbol",
    "blast_radius",
    "build_knowledge_graph",
    "graph_search_mcp",
    "ask_agent",
    "agent_list_sessions",
    "agent_clear_session",
    "federation_list_repos",
    "federation_add_repo",
    "federation_remove_repo",
    "federation_search_all",
    "subscribe_resource",
    "unsubscribe_resource",
]

# Words in the instructions that look like identifiers but are argument or field names.
INSTRUCTION_ARGUMENT_WORDS = frozenset(
    {"repo_path", "qualified_name", "symbol_name", "next_cursor", "session_id"}
)


async def _wire_tools() -> list[dict[str, object]]:
    """The tools exactly as a modern-protocol client receives them."""
    async with Client(srv.mcp) as client:
        result = await client.list_tools_mcp()
    return [tool.model_dump(by_alias=True, exclude_none=True) for tool in result.tools]


@pytest.mark.asyncio
async def test_the_tool_surface_is_the_one_from_before_tool_metadata() -> None:
    """Nothing was added, renamed or removed: the same 15 names."""
    names = [tool["name"] for tool in await _wire_tools()]

    assert len(names) == 15
    assert set(names) == PRE_METADATA_TOOLS


@pytest.mark.asyncio
async def test_annotations_table_and_registered_tools_match_both_ways() -> None:
    """A tool added without a table row, or a row for a tool that is gone, fails here."""
    registered = {tool["name"] for tool in await _wire_tools()}
    expected = set(EXPECTED_ANNOTATIONS)

    assert registered - expected == set(), (
        f"registered tools with no annotations row: {sorted(registered - expected)}"
    )
    assert expected - registered == set(), (
        f"annotation rows for tools that are not registered: {sorted(expected - registered)}"
    )


@pytest.mark.asyncio
async def test_module_table_names_exactly_the_registered_tools() -> None:
    """The table in tool_metadata.py itself: every tool has a row and no row is extra."""
    from trelix_mcp.tool_metadata import TOOL_ANNOTATIONS

    registered = {tool["name"] for tool in await _wire_tools()}

    assert set(TOOL_ANNOTATIONS) == registered


@pytest.mark.asyncio
async def test_every_tool_reaches_the_wire_with_its_literal_annotations() -> None:
    wire = {tool["name"]: tool.get("annotations") for tool in await _wire_tools()}

    assert wire == EXPECTED_ANNOTATIONS


@pytest.mark.asyncio
async def test_a_single_tool_lookup_carries_the_same_annotations_as_the_listing() -> None:
    """`mcp.get_tool(name)` (what a tool call resolves through) agrees with `tools/list`."""
    tool = await srv.mcp.get_tool("agent_clear_session")

    assert tool is not None
    assert tool.annotations is not None
    expected = EXPECTED_ANNOTATIONS["agent_clear_session"]
    assert tool.annotations.model_dump(by_alias=True, exclude_none=True) == expected
    assert await srv.mcp.get_tool("no_such_tool") is None


@pytest.mark.asyncio
async def test_only_the_verified_tools_are_read_only_and_only_two_are_destructive() -> None:
    """Flipping a writer to readOnlyHint=True, or a destructive tool to False, fails here."""
    wire = {tool["name"]: tool["annotations"] for tool in await _wire_tools()}

    read_only = {name for name, hints in wire.items() if hints["readOnlyHint"]}
    destructive = {name for name, hints in wire.items() if hints["destructiveHint"]}
    idempotent = {name for name, hints in wire.items() if hints["idempotentHint"]}
    open_world = {name for name, hints in wire.items() if hints["openWorldHint"]}

    assert read_only == {"search_code", "get_symbol", "blast_radius"}
    assert destructive == {"federation_remove_repo", "agent_clear_session"}
    assert idempotent == {"index_codebase"}
    assert open_world == set()


@pytest.mark.asyncio
async def test_tool_order_is_stable() -> None:
    """tools/list follows the literal list, not the order the decorators appear in server.py."""
    wire_order = [tool["name"] for tool in await _wire_tools()]
    in_process_order = [tool.name for tool in await srv.mcp.list_tools()]

    assert wire_order == EXPECTED_ORDER
    assert in_process_order == EXPECTED_ORDER


@pytest.mark.asyncio
async def test_tool_order_covers_every_registered_tool_once() -> None:
    from trelix_mcp.tool_metadata import TOOL_ORDER

    registered = {tool["name"] for tool in await _wire_tools()}

    assert len(TOOL_ORDER) == len(set(TOOL_ORDER))
    assert set(TOOL_ORDER) == registered


@pytest.mark.asyncio
async def test_a_tool_missing_from_the_order_list_is_listed_after_the_known_ones() -> None:
    """Fail soft: an unlisted tool still appears, last, instead of breaking or jumping the queue."""
    from fastmcp.tools.base import Tool
    from trelix_mcp.tool_metadata import ToolMetadata

    def _answer() -> int:
        return 1

    unlisted = Tool.from_function(_answer, name="zz_unlisted_tool")
    known = Tool.from_function(_answer, name="search_code")

    listed = await ToolMetadata().list_tools([unlisted, known])

    assert [tool.name for tool in listed] == ["search_code", "zz_unlisted_tool"]
    assert listed[0].annotations is not None
    assert listed[1].annotations is None


def test_instructions_are_at_most_two_thousand_characters() -> None:
    assert 0 < len(srv.mcp.instructions) <= 2000


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_instructions_reach_the_client_in_both_protocol_eras(mode: str) -> None:
    async with Client(srv.mcp, mode=mode) as client:
        received = client.instructions

    assert received == srv.mcp.instructions
    assert received is not None
    assert len(received) <= 2000


@pytest.mark.asyncio
async def test_instructions_name_only_tools_that_exist() -> None:
    """A tool name in the instructions must be registered (no repo_map/exact_search yet)."""
    registered = {tool["name"] for tool in await _wire_tools()}
    identifiers = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", srv.mcp.instructions))

    unknown = identifiers - registered - INSTRUCTION_ARGUMENT_WORDS
    assert unknown == set(), f"instructions mention names that are not tools: {sorted(unknown)}"
    assert "repo_map" not in srv.mcp.instructions
    assert "exact_search" not in srv.mcp.instructions


@pytest.mark.asyncio
async def test_instructions_mention_every_tool_the_workflow_depends_on() -> None:
    mentioned = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", srv.mcp.instructions))
    workflow = {
        "index_codebase",
        "search_code",
        "get_symbol",
        "blast_radius",
        "build_knowledge_graph",
        "graph_search_mcp",
        "ask_agent",
    }

    assert workflow <= mentioned
    assert srv.mcp.instructions.index("index_codebase") < srv.mcp.instructions.index("search_code")


@pytest.mark.asyncio
async def test_calls_shown_in_the_instructions_use_the_registered_argument_names() -> None:
    """Every `tool(arg, ...)` in the instructions names arguments that tool's schema has."""
    wire = await _wire_tools()
    schemas = {tool["name"]: set(tool["inputSchema"]["properties"]) for tool in wire}
    descriptions = {tool["name"]: tool["description"] for tool in wire}
    text = " ".join(srv.mcp.instructions.split())

    calls = re.findall(r"\b([a-z]+(?:_[a-z]+)+)\(([a-z_, ]*)\)", text)

    assert {name for name, _ in calls} == {
        "index_codebase",
        "search_code",
        "get_symbol",
        "blast_radius",
        "graph_search_mcp",
        "build_knowledge_graph",
        "ask_agent",
        "agent_list_sessions",
        "agent_clear_session",
    }
    for name, arguments in calls:
        shown = {argument.strip() for argument in arguments.split(",")}
        assert shown <= schemas[name], f"{name}({arguments}) shows arguments it does not take"
    assert "index_codebase(repo_path) first." in text
    assert "the update is incremental" in text
    assert "incremental" in descriptions["index_codebase"]
    assert "pass next_cursor back as cursor" in text
    assert "cursor" in schemas["search_code"]


@pytest.mark.asyncio
async def test_the_unindexed_answer_quoted_in_the_instructions_is_what_a_read_tool_returns(
    tmp_path: Path,
) -> None:
    arguments = {"qualified_name": "target", "repo_path": str(tmp_path)}
    async with Client(srv.mcp) as client:
        result = await client.call_tool("get_symbol", arguments, raise_on_error=False)

    assert result.is_error is True
    assert result.content[0].text.startswith("No index found")
    assert 'answer "No index found"' in " ".join(srv.mcp.instructions.split())


def test_instructions_do_not_chain_the_graph_tools_or_give_every_tool_a_repo_path() -> None:
    """graph_search_mcp builds the graph itself (two builds if a model runs both); the
    federation and subscription tools take no repo_path."""
    text = " ".join(srv.mcp.instructions.split())

    assert (
        "graph_search_mcp(query, repo_path) to follow call, import and type relationships "
        "from a search hit. It builds the graph itself."
    ) in text
    assert "build_knowledge_graph(repo_path), then" not in text
    assert "Each tool in the numbered list below takes repo_path" in text


def test_instructions_do_not_promise_subscription_delivery() -> None:
    """The server sends no resource update notifications; the text must not imply it does."""
    text = " ".join(srv.mcp.instructions.split())

    assert "The server sends no resource update notifications, so do not rely on them." in text
    assert "resources.subscribe" not in text
    assert "resources/subscribe" not in text
    assert "will be notified" not in text


@pytest.mark.asyncio
async def test_the_tool_listing_carries_a_private_five_minute_cache_hint() -> None:
    """cache_ttl=300, cache_scope="private": ttlMs/cacheScope on a modern-protocol listing."""
    async with Client(srv.mcp, mode="auto") as client:
        result = await client.list_tools_mcp()

    wire = result.model_dump(by_alias=True, exclude_none=True)
    assert wire["ttlMs"] == 300000
    assert wire["cacheScope"] == "private"
