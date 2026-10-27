"""What the trelix MCP server says about its own tools.

Four literal pieces, all additive (no tool is renamed, removed or changes its arguments or
result): the annotation hints every client can read in `tools/list`, the order `tools/list`
returns them in, the `instructions` string a model is shown at connect time, and the
`--tools core` subset. Kept out of server.py on purpose: this file is a table a reviewer
reads top to bottom, and a new tool must get a row here (a test fails until it does).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from fastmcp.server.transforms import Transform
from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from fastmcp.server.transforms import GetToolNext
    from fastmcp.tools.base import Tool
    from fastmcp.utilities.versions import VersionSpec

# `tools/list` answers carry a freshness hint. FastMCP has one server-wide setting for it, so
# the same hint is also on the prompt and resource lists and on resources/read (measured).
LISTING_CACHE_TTL_SECONDS = 300
LISTING_CACHE_SCOPE: Literal["public", "private"] = "private"

TOOL_PROFILES = ("core", "full")


def _hints(
    *, read_only: bool, destructive: bool = False, idempotent: bool = False
) -> ToolAnnotations:
    """One row of the table.

    openWorldHint is False for every tool: they act on a local index and registry, and the
    network calls they may make go to a closed list of destinations. ask_agent's LLM calls go
    to the provider the operator configured. index_codebase's embedding calls go to the
    provider its `provider` argument names (local, openai, azure, voyage or local-code), so
    the model picks among those; the hosted ones need credentials the operator has set.
    """
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=False,
    )


# readOnlyHint=True is claimed only where tests/test_tool_readonly.py hashes the index
# database before and after a real call and finds it byte-identical. That measurement is on an
# index already at the current schema with telemetry off (the default). It is a claim about the
# index database, not about every file. Three things write, and that file records each:
# TRELIX_TELEMETRY_ENABLED=true adds a query_telemetry row per search_code, the first open of an
# index written by an older trelix migrates it, and every search_code writes a JSON trace of the
# query to a debug/ directory beside the index (.trelix/debug/ for the default db_path; one new
# file per call; get_symbol and blast_radius write none). The
# federation reader tools (federation_list_repos, federation_search_all) are not covered by
# that test yet, so they are listed as not read-only, which is the safe way to be wrong.
TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    "index_codebase": _hints(read_only=False, idempotent=True),
    "search_code": _hints(read_only=True),
    "get_symbol": _hints(read_only=True),
    "blast_radius": _hints(read_only=True),
    # Both rebuild and save the graph metadata (GraphBuilder.build), so neither is read-only.
    "build_knowledge_graph": _hints(read_only=False),
    "graph_search_mcp": _hints(read_only=False),
    "ask_agent": _hints(read_only=False),
    # Evicts sessions older than the configured maximum age before it lists.
    "agent_list_sessions": _hints(read_only=False),
    "agent_clear_session": _hints(read_only=False, destructive=True),
    "federation_list_repos": _hints(read_only=False),
    "federation_add_repo": _hints(read_only=False),
    "federation_remove_repo": _hints(read_only=False, destructive=True),
    "federation_search_all": _hints(read_only=False),
    "subscribe_resource": _hints(read_only=False),
    "unsubscribe_resource": _hints(read_only=False),
}

# The order `tools/list` returns, workflow first and the subscription tools, which deliver
# nothing today, last. It is the same for every client and every run.
TOOL_ORDER: tuple[str, ...] = (
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
)

# `--tools core`. The design lists nine tools; `repo_map` and `exact_search` do not exist in
# this server yet, so the profile is the seven that do.
CORE_TOOLS: frozenset[str] = frozenset(
    {
        "index_codebase",
        "search_code",
        "get_symbol",
        "blast_radius",
        "build_knowledge_graph",
        "graph_search_mcp",
        "ask_agent",
    }
)

# What a model is told at connect time. `--tools core` sends only the first part, so the text
# never names a tool that profile hides.
CORE_INSTRUCTIONS = """\
trelix searches a local code repository through an index it keeps in <repo_path>/.trelix. \
Each tool in the numbered list below takes repo_path, an absolute path.

Use the tools in this order:
1. index_codebase(repo_path) first. Until a repository is indexed, the tools that read it \
answer "No index found". Run it again after large code changes; the update is incremental.
2. search_code(query, repo_path) for a question in plain language, such as "where are \
sessions validated". Results come in pages: pass next_cursor back as cursor for the next page.
3. get_symbol(qualified_name, repo_path) when you already know a symbol's name and want its \
full source.
4. blast_radius(symbol_name, repo_path) before you change a symbol: it lists the files that \
call or import it.
5. graph_search_mcp(query, repo_path) to follow call, import and type relationships from a \
search hit. It builds the graph itself. build_knowledge_graph(repo_path) is a separate call \
that returns the architecture clusters.
6. ask_agent(query, repo_path) for a question that needs several searches. It needs an LLM \
provider configured on the server. Pass the returned session_id to continue the conversation.
"""

_FULL_ONLY_INSTRUCTIONS = """
agent_list_sessions(repo_path) lists saved agent sessions and agent_clear_session(repo_path, \
session_id) deletes one.

To search several repositories at once, register each with federation_add_repo (index it \
separately), then call federation_search_all. federation_list_repos and \
federation_remove_repo manage the list.

subscribe_resource and unsubscribe_resource only record a URI in this server's memory. The \
server sends no resource update notifications, so do not rely on them.
"""

SERVER_INSTRUCTIONS = CORE_INSTRUCTIONS + _FULL_ONLY_INSTRUCTIONS


def _rank(name: str) -> int:
    """Position in TOOL_ORDER; a tool missing from it sorts last (a test forbids that)."""
    return TOOL_ORDER.index(name) if name in TOOL_ORDER else len(TOOL_ORDER)


def _annotated(tool: Tool) -> Tool:
    annotations = TOOL_ANNOTATIONS.get(tool.name)
    if annotations is None:
        return tool
    return tool.model_copy(update={"annotations": annotations})


class ToolMetadata(Transform):
    """Attach the literal annotations to every tool and list the tools in TOOL_ORDER."""

    async def list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        return [_annotated(tool) for tool in sorted(tools, key=lambda tool: _rank(tool.name))]

    async def get_tool(
        self, name: str, call_next: GetToolNext, *, version: VersionSpec | None = None
    ) -> Tool | None:
        tool = await call_next(name, version=version)
        return None if tool is None else _annotated(tool)


def apply_tool_profile(server: FastMCP, profile: str) -> None:
    """Hide every tool outside CORE_TOOLS when `profile` is "core"; "full" changes nothing.

    Hidden tools stay registered: they disappear from `tools/list` and a call to one is
    answered as an unknown tool, and the instructions stop naming them. Called once, from
    main(), before the server starts.
    """
    if profile == "core":
        server.disable(names=set(TOOL_ORDER) - CORE_TOOLS, components={"tool"})
        server.instructions = CORE_INSTRUCTIONS
