"""What a client reads: result shape, the keys the VS Code extension reads, schemas, wire size.

Everything here goes through an in-process `fastmcp.Client` and inspects the whole
`CallToolResult`, the way a connected client sees it. Each case sets up only the fixtures it needs:
`sessions` replaces the server's Database, so it must not be active for a real-index tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import trelix_mcp.server as srv
from budget_support import REPO, call, hit, session, structured, texts
from fastmcp import Client

# Data that needs the cut even at default arguments: bodies full of quotes (each one is escaped in
# the text block, so the response is far more than twice its text), long paths, long queries.
JSON_LIKE_BODY = '{"a": "b"}' * 80
QUOTED_BODY = '"' * 100_000
WIRE_CASES = [
    ("search_code", "json-like"),
    ("search_code", "quotes"),
    ("graph_search_mcp", "quotes"),
    ("federation_search_all", "json-like"),
    ("federation_search_all", "quotes"),
    ("blast_radius", "long-paths"),
    ("agent_list_sessions", "long-queries"),
]
# How many rows each tool is asked for at (default arguments, maximum arguments).
ROWS_ASKED_FOR = {
    "search_code": (10, 50),
    "graph_search_mcp": (10, 50),
    "federation_search_all": (10, 50),
    "blast_radius": (100, 500),
    "agent_list_sessions": (50, 50),
}


def _retrieval_arguments(tool: str) -> dict[str, Any]:
    return {
        "search_code": {"query": "q", "repo_path": REPO},
        "graph_search_mcp": {"query": "q", "repo_path": REPO},
        "federation_search_all": {"query": "q"},
    }[tool]


def _blast_arguments(repo: Path) -> dict[str, Any]:
    return {"symbol_name": "target.run", "repo_path": str(repo)}


@pytest.mark.parametrize(
    "case",
    [
        "search_code",
        "search_code_cut",
        "graph_search_mcp",
        "federation_search_all",
        "blast_radius",
        "blast_radius_cut",
        "agent_list_sessions",
        "get_symbol",
    ],
)
async def test_text_block_matches_structured_content(
    case: str, request: pytest.FixtureRequest
) -> None:
    tool, arguments = case.removesuffix("_cut"), {}
    if tool in {"search_code", "graph_search_mcp", "federation_search_all"}:
        backends = request.getfixturevalue("backends")
        backends.hits = [hit(i, body="b" * 800) for i in range(30 if case.endswith("_cut") else 3)]
        arguments = {**_retrieval_arguments(tool), **({"k": 30} if case.endswith("_cut") else {})}
    elif tool == "agent_list_sessions":
        request.getfixturevalue("sessions").items = [session(0)]
        arguments = {"repo_path": REPO}
    else:
        repo = request.getfixturevalue("dependents_repo" if case.endswith("_cut") else "small_repo")
        arguments = _blast_arguments(repo)
        if tool == "get_symbol":
            arguments = {"qualified_name": "target.run", "repo_path": str(repo)}

    result = await call(tool, **arguments)

    assert result.is_error is False
    assert len(texts(result)) >= 1
    assert json.loads(texts(result)[0]) == structured(result)


async def test_an_empty_result_has_a_text_block_that_matches(
    backends: Any, small_repo: Path, lonely_repo: Path
) -> None:
    """FastMCP sends `[]` and `null` with no text block at all; the server adds one.

    The VS Code extension reads the first text block, so until this an empty blast radius or an
    unknown symbol gave it nothing to parse (MCP_GUIDE used to document that as the one exception).
    Both empty paths of blast_radius are covered: a symbol the index does not know, and one it
    knows that nothing depends on.
    """
    backends.hits = []
    results = [
        await call("blast_radius", symbol_name="nothing.here", repo_path=str(small_repo)),
        await call("blast_radius", symbol_name="target.run", repo_path=str(lonely_repo)),
        await call("graph_search_mcp", query="q", repo_path=REPO),
    ]

    for result in results:
        assert (result.is_error, texts(result), structured(result)) == (False, ["[]"], [])

    symbol = await call("get_symbol", qualified_name="nothing.here", repo_path=str(small_repo))
    assert (symbol.is_error, texts(symbol), structured(symbol)) == (False, ["null"], None)


async def test_vscode_contract_keys(backends: Any, small_repo: Path, dependents_repo: Path) -> None:
    """The keys workspace-vscode/src/mcp-client.ts reads, from the first text block."""
    backends.hits = [hit(1)]

    search = await call("search_code", query="q", repo_path=REPO)
    page = json.loads(next(b.text for b in search.content if b.type == "text"))
    assert {"results", "next_cursor", "total_available"} <= set(page)
    assert set(page["results"][0]) == {
        "symbol",
        "file",
        "kind",
        "lines",
        "score",
        "source",
        "body",
        "language",
    }

    for repo in (small_repo, dependents_repo):
        blast = await call("blast_radius", **_blast_arguments(repo))
        entries = json.loads(next(b.text for b in blast.content if b.type == "text"))
        assert isinstance(entries, list)
        assert set(entries[0]) == {"file", "symbol", "kind", "line_start", "language"}

    symbol = await call("get_symbol", qualified_name="target.run", repo_path=str(small_repo))
    assert {
        "name",
        "qualified_name",
        "kind",
        "file",
        "line_start",
        "line_end",
        "signature",
        "docstring",
        "body",
        "language",
    } <= set(json.loads(texts(symbol)[0]))


async def test_new_arguments_appear_in_the_tool_schemas() -> None:
    async with Client(srv.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    for name in ("search_code", "graph_search_mcp", "federation_search_all"):
        detail = tools[name].input_schema["properties"]["detail"]
        assert (detail["enum"], detail["default"]) == (["concise", "detailed"], "detailed")
    assert tools["blast_radius"].input_schema["properties"]["limit"]["default"] == 100
    assert tools["get_symbol"].input_schema["properties"]["max_body_chars"]["default"] == 20000
    # A cut array travels as a ToolResult; the schema is still the one a plain list gets.
    for name in ("blast_radius", "graph_search_mcp"):
        assert tools[name].output_schema == {
            "properties": {
                "result": {
                    "items": {"additionalProperties": True, "type": "object"},
                    "type": "array",
                }
            },
            "required": ["result"],
            "type": "object",
            "x-fastmcp-wrap-result": True,
        }


def _rows_and_whether_cut(result: Any) -> tuple[int, bool]:
    """How many rows the result carries, and whether it says rows were left out."""
    body = json.loads(texts(result)[0])
    if isinstance(body, list):
        return len(body), "trelix" in (result.meta or {})
    return len(body.get("results", body.get("sessions"))), body["truncated"]


@pytest.mark.parametrize("at_maximum", [False, True], ids=["defaults", "maximum-arguments"])
@pytest.mark.parametrize(("tool", "data"), WIRE_CASES, ids=[f"{t}-{d}" for t, d in WIRE_CASES])
async def test_wire_size_stays_inside_the_budget(
    tool: str, data: str, at_maximum: bool, request: pytest.FixtureRequest
) -> None:
    """The whole CallToolResult is at most 30,000 characters, and 15,000 of text, every time.

    The data needs the cut at default arguments too, so the ceiling is a property of the
    server and not of a fixture that happens to fit.
    """
    size_argument = "limit" if tool in {"blast_radius", "agent_list_sessions"} else "k"
    if tool == "blast_radius":
        arguments = _blast_arguments(request.getfixturevalue("long_path_repo"))
    elif tool == "agent_list_sessions":
        request.getfixturevalue("sessions").items = [session(i, "q" * 200) for i in range(80)]
        arguments = {"repo_path": REPO}
    else:
        body = JSON_LIKE_BODY if data == "json-like" else QUOTED_BODY
        request.getfixturevalue("backends").hits = [hit(i, body=body) for i in range(300)]
        arguments = _retrieval_arguments(tool)
    if at_maximum:
        arguments[size_argument] = 100_000

    result = await call(tool, **arguments)

    assert result.is_error is False
    rows, was_cut = _rows_and_whether_cut(result)
    assert 1 <= rows < ROWS_ASKED_FOR[tool][at_maximum], "the budget must have cut rows"
    assert was_cut
    assert len(result.model_dump_json(by_alias=True)) <= 30_000
    assert sum(len(text) for text in texts(result)) <= 15_000


@pytest.mark.parametrize(
    "prompt",
    ["q" * 20_000, "q" * 100_000, '"' * 100_000],
    ids=["letters-20k", "letters-100k", "quotes-100k"],
)
async def test_a_huge_first_prompt_stays_inside_the_ceilings(sessions: Any, prompt: str) -> None:
    """A session's latest prompt is unbounded: the listing cuts it (one row stays under 30,000)."""
    sessions.items = [session(0, prompt), session(1)]

    result = await call("agent_list_sessions", repo_path=REPO)

    rows = json.loads(texts(result)[0])["sessions"]
    assert len(rows) == 2
    assert (len(rows[0]["query"]), rows[0]["query_truncated"]) == (300, True)
    assert len(result.model_dump_json(by_alias=True)) <= 30_000
    assert sum(len(text) for text in texts(result)) <= 15_000
