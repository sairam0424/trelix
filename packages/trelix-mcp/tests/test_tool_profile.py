"""`trelix-mcp --tools core|full`: which tools the server lists.

`full` (the default) is every tool, exactly the set registered before this flag existed.
`core` is the everyday search and indexing tools; the others are hidden, not deleted. The
design asks for nine core tools; `repo_map` and `exact_search` do not exist in this server
yet, so `core` is the seven that do. Same style as test_server_cli.py: patch `sys.argv`
and `mcp.run`, so whichever branch runs, the test sees what the server would have served.

`main()` adds a visibility rule to the module-level `mcp` for `core`. The fixture gives each
test its own copy of the server's transform list, so no test leaks a hidden tool into the next.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import trelix_mcp.server as server_module
from fastmcp import Client

ALL_TOOLS = frozenset(
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

CORE_TOOL_NAMES = frozenset(
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


@pytest.fixture(autouse=True)
def _isolated_server_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a private copy of what `--tools core` changes; monkeypatch restores it.

    That is the server's transform list (the visibility rule) and its instructions text.
    """
    monkeypatch.setattr(server_module.mcp, "_transforms", list(server_module.mcp.transforms))
    monkeypatch.setattr(server_module.mcp, "instructions", server_module.mcp.instructions)


@pytest.fixture
def served_tool_names(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[str]], list[str]]:
    """Run main() with the given argv and return the tool names the server would then list.

    `mcp.run` is replaced by a function that reads `tools/list` the way a client would, so
    the names are what the server lists at the moment it starts serving, after main() has
    applied the profile.
    """

    def _run(argv: list[str]) -> list[str]:
        monkeypatch.setattr("sys.argv", ["trelix-mcp", *argv])
        served: list[list[str]] = []

        def _serve(**_kwargs: object) -> None:
            async def _list() -> list[str]:
                async with Client(server_module.mcp) as client:
                    return [tool.name for tool in await client.list_tools()]

            served.append(asyncio.run(_list()))

        monkeypatch.setattr(server_module.mcp, "run", _serve)
        server_module.main()
        assert len(served) == 1, "main() must start the server exactly once"
        return served[0]

    return _run


def test_default_is_full_and_lists_every_tool(
    served_tool_names: Callable[[list[str]], list[str]],
) -> None:
    names = served_tool_names([])

    assert set(names) == ALL_TOOLS
    assert len(names) == 15


def test_explicit_full_lists_every_tool(
    served_tool_names: Callable[[list[str]], list[str]],
) -> None:
    assert set(served_tool_names(["--tools", "full"])) == ALL_TOOLS


def test_core_lists_exactly_the_seven_tools_that_exist(
    served_tool_names: Callable[[list[str]], list[str]],
) -> None:
    names = served_tool_names(["--tools", "core"])

    assert set(names) == CORE_TOOL_NAMES
    assert len(names) == 7
    assert "repo_map" not in names
    assert "exact_search" not in names


def test_core_keeps_the_literal_tool_order(
    served_tool_names: Callable[[list[str]], list[str]],
) -> None:
    assert served_tool_names(["--tools", "core"]) == [
        "index_codebase",
        "search_code",
        "get_symbol",
        "blast_radius",
        "build_knowledge_graph",
        "graph_search_mcp",
        "ask_agent",
    ]


def test_core_starts_the_server_over_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--tools", "core"])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    server_module.main()

    mock_run.assert_called_once_with(transport="stdio")


def _registered_listing() -> set[str]:
    """`tools/list` as the server answers it now."""

    async def _names() -> set[str]:
        return {tool.name for tool in await server_module.mcp.list_tools()}

    return asyncio.run(_names())


def test_core_hides_the_other_tools_without_deleting_them(
    served_tool_names: Callable[[list[str]], list[str]],
) -> None:
    """Hidden tools are still registered: enabling one by name brings it back."""
    served_tool_names(["--tools", "core"])

    assert _registered_listing() == CORE_TOOL_NAMES

    server_module.mcp.enable(names={"federation_list_repos"}, components={"tool"})

    assert _registered_listing() == CORE_TOOL_NAMES | {"federation_list_repos"}


def test_core_refuses_a_call_to_a_hidden_tool(
    served_tool_names: Callable[[list[str]], list[str]],
) -> None:
    served_tool_names(["--tools", "core"])

    async def _call() -> tuple[bool, str]:
        async with Client(server_module.mcp) as client:
            result = await client.call_tool("federation_list_repos", {}, raise_on_error=False)
        return result.is_error, str(result.content)

    is_error, content = asyncio.run(_call())

    assert is_error is True
    assert "Unknown tool: 'federation_list_repos'" in content


# Words in the instructions that look like identifiers but are argument or field names.
ARGUMENT_WORDS = frozenset(
    {"repo_path", "qualified_name", "symbol_name", "next_cursor", "session_id"}
)


def _instructions_after(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("sys.argv", ["trelix-mcp", *argv])
    monkeypatch.setattr(server_module.mcp, "run", MagicMock())
    server_module.main()
    return server_module.mcp.instructions


def test_core_instructions_name_exactly_the_core_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model on `core` is never told about a tool that profile hides."""
    text = _instructions_after(["--tools", "core"], monkeypatch)

    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text)) - ARGUMENT_WORDS
    assert named == CORE_TOOL_NAMES
    assert "federation" not in text
    assert "subscribe" not in text
    assert "agent_list_sessions" not in text
    assert 0 < len(text) <= 2000


def test_full_keeps_the_complete_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    before = server_module.mcp.instructions

    text = _instructions_after(["--tools", "full"], monkeypatch)

    assert text == before
    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", text)) - ARGUMENT_WORDS
    assert named == ALL_TOOLS


@pytest.mark.parametrize("value", ["minimal", "CORE", ""])
def test_an_invalid_profile_is_a_usage_error_and_does_not_start_the_server(
    value: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--tools", value])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    error_lines = [line for line in capsys.readouterr().err.splitlines() if "error:" in line]
    assert len(error_lines) == 1
    assert error_lines[0].startswith("trelix-mcp: error: argument --tools: invalid choice:")


def test_tools_flag_without_a_value_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--tools"])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    assert "expected one argument" in capsys.readouterr().err


def test_help_documents_the_tools_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--help"])
    monkeypatch.setattr(server_module.mcp, "run", MagicMock())

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--tools {core,full}" in out
    assert "'full' (default)" in out
