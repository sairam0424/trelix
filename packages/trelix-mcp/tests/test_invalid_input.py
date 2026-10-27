"""Every invalid input is a tool error that names the argument, what was given and what is valid.

Driven through an in-process `fastmcp.Client`, so each case checks what reaches a client: `isError`
true, one text block, nothing opened and nothing created on disk. Every expected text is a literal.

Recorded before these checks existed: a repo_path that did not exist was not a tool result at all
but a JSON-RPC "Invalid request parameters" error (IndexConfig's pydantic error, which FastMCP does
not mask into a result), a blank repo_path resolved to the server's working directory (the no-index
message named it and index_codebase indexed it), a file was told to run `trelix index <file>`, a
blank query was searched and ask_agent persisted a session for it, get_symbol and blast_radius
answered a blank name with an empty result, federation_add_repo registered a blank alias, a
relative, missing, blank or file path, and a weight of 0 or less, and the four federation tools
answered a blank config_path (Path("").resolve() is the server's working directory) with a 200
`error` dict that named the allowed roots and that directory, none of which the caller had passed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import trelix_mcp.server as srv
from budget_support import call, texts
from fastmcp.exceptions import ToolError

REPO_TOOLS = [
    ("search_code", {"query": "add numbers"}),
    ("get_symbol", {"qualified_name": "add"}),
    ("blast_radius", {"symbol_name": "add"}),
    ("build_knowledge_graph", {}),
    ("graph_search_mcp", {"query": "add"}),
    ("ask_agent", {"query": "what does add do"}),
    ("agent_list_sessions", {}),
    ("agent_clear_session", {"session_id": "s1"}),
    ("index_codebase", {}),
]
ROOT_HINT = "pass the absolute path of the repository root."


@pytest.fixture
def indexed(tmp_path: Path, mark_indexed: Any) -> str:
    return str(mark_indexed(tmp_path / "indexed"))


@pytest.fixture
def a_file(tmp_path: Path) -> str:
    path = tmp_path / "module.py"
    path.write_text("x = 1\n", encoding="utf-8")
    return str(path)


async def error_text(tool: str, **arguments: Any) -> str:
    """The one text block of a tool error, having checked that it is one."""
    result = await call(tool, **arguments)
    assert result.is_error is True
    (text,) = texts(result)
    return text


# ---------------------------------------------------------------------------
# repo_path: blank, missing, a file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tool", "arguments"), REPO_TOOLS, ids=[t for t, _ in REPO_TOOLS])
@pytest.mark.parametrize("blank", ["", "  "], ids=["empty", "whitespace"])
async def test_a_blank_repo_path_is_an_error_and_does_not_mean_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    arguments: dict[str, Any],
    blank: str,
) -> None:
    """It was the empty string that resolved to the working directory (Path("") is "."); a
    whitespace-only value was a protocol error. Both are pinned."""
    monkeypatch.chdir(tmp_path)

    with patch("trelix_mcp.server.Indexer") as indexer:
        text = await error_text(tool, **arguments, repo_path=blank)

    assert text == f"repo_path must not be empty or whitespace (got '{blank}'); {ROOT_HINT}"
    assert not (tmp_path / ".trelix").exists()
    indexer.assert_not_called()


@pytest.mark.parametrize(("tool", "arguments"), REPO_TOOLS, ids=[t for t, _ in REPO_TOOLS])
async def test_a_repo_path_that_does_not_exist_is_a_tool_error_not_a_protocol_error(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    missing = tmp_path / "gone" / "repo"

    text = await error_text(tool, **arguments, repo_path=str(missing))

    assert text == f"repo_path does not exist: '{missing}'; {ROOT_HINT}"
    assert not (tmp_path / "gone").exists()


@pytest.mark.parametrize(("tool", "arguments"), REPO_TOOLS, ids=[t for t, _ in REPO_TOOLS])
async def test_a_file_as_repo_path_is_told_to_pass_its_directory(
    a_file: str, tool: str, arguments: dict[str, Any]
) -> None:
    text = await error_text(tool, **arguments, repo_path=a_file)

    assert text == (
        f"repo_path is not a directory: '{a_file}'; pass the repository root, not a file in it."
    )
    assert sorted(p.name for p in Path(a_file).parent.iterdir()) == ["module.py"]


async def test_a_long_value_is_cut_in_the_message(tmp_path: Path) -> None:
    missing = tmp_path / ("x" * 250)

    text = await error_text("search_code", query="q", repo_path=str(missing))

    assert text == f"repo_path does not exist: '{str(missing)[:200]}...'; {ROOT_HINT}"


async def test_a_name_the_filesystem_rejects_is_told_it_does_not_exist(tmp_path: Path) -> None:
    """A component over 255 bytes cannot be stat-ed. On Python 3.12 and 3.13 Path.exists() raised
    OSError, which FastMCP masked into `Error calling tool 'search_code': [Errno 63] File name too
    long: '<the whole path>'`, naming no argument and cutting nothing; 3.14's pathlib answers
    False. Every version now gives the documented message (the 250-character case above stops
    just short of this boundary)."""
    missing = tmp_path / ("x" * 256)

    text = await error_text("search_code", query="q", repo_path=str(missing))

    assert text == f"repo_path does not exist: '{str(missing)[:200]}...'; {ROOT_HINT}"


# ---------------------------------------------------------------------------
# Blank text arguments
# ---------------------------------------------------------------------------

BLANK_TEXT_CASES = [
    (
        "search_code",
        {"query": ""},
        "query must not be empty or whitespace (got ''); pass the text to search for.",
    ),
    (
        "search_code",
        {"query": "   "},
        "query must not be empty or whitespace (got '   '); pass the text to search for.",
    ),
    (
        "graph_search_mcp",
        {"query": ""},
        "query must not be empty or whitespace (got ''); pass the text to search for.",
    ),
    (
        "ask_agent",
        {"query": " "},
        "query must not be empty or whitespace (got ' '); pass the question to answer.",
    ),
    (
        "get_symbol",
        {"qualified_name": ""},
        "qualified_name must not be empty or whitespace (got ''); pass the symbol's qualified "
        "name, e.g. MyClass.my_method.",
    ),
    (
        "blast_radius",
        {"symbol_name": "\t"},
        "symbol_name must not be empty or whitespace (got '\t'); pass the name or qualified name "
        "of the symbol to analyse.",
    ),
    (
        "agent_clear_session",
        {"session_id": ""},
        "session_id must not be empty or whitespace (got ''); pass the session_id to delete (see "
        "agent_list_sessions).",
    ),
    (
        "ask_agent",
        {"query": "what", "session_id": "  "},
        "session_id must not be empty or whitespace (got '  '); pass the session_id a previous "
        "answer returned, or omit it for a new session.",
    ),
    (
        "ask_agent",
        {"query": "what", "session_id": ""},
        "session_id must not be empty or whitespace (got ''); pass the session_id a previous "
        "answer returned, or omit it for a new session.",
    ),
]


def blank_case_id(tool: str, arguments: dict[str, Any]) -> str:
    """`<tool>-<argument>-<length>` for the blank argument, which is the last one given."""
    name, value = list(arguments.items())[-1]
    return f"{tool}-{name}-{len(value)}"


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    BLANK_TEXT_CASES,
    ids=[blank_case_id(t, a) for t, a, _ in BLANK_TEXT_CASES],
)
async def test_a_blank_argument_is_an_error_before_anything_is_opened(
    indexed: str, tool: str, arguments: dict[str, Any], expected: str
) -> None:
    with (
        patch("trelix_mcp.server.Retriever") as retriever,
        patch("trelix_mcp.server.Database") as database,
        patch("trelix_mcp.server.AgentLoop") as agent_loop,
    ):
        text = await error_text(tool, **arguments, repo_path=indexed)

    assert text == expected
    retriever.assert_not_called()
    database.assert_not_called()
    agent_loop.assert_not_called()


async def test_federation_search_all_rejects_a_blank_query_before_reading_the_registry() -> None:
    with patch("trelix_mcp.server.RepoRegistry") as registry:
        text = await error_text("federation_search_all", query="")

    assert text == "query must not be empty or whitespace (got ''); pass the text to search for."
    registry.load.assert_not_called()


# ---------------------------------------------------------------------------
# federation_add_repo and federation_remove_repo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            {"alias": " ", "path": "{indexed}"},
            "alias must not be empty or whitespace (got ' '); pass a short unique name for the "
            "repo, e.g. auth-service.",
        ),
        (
            {"alias": "svc", "path": ""},
            f"path must not be empty or whitespace (got ''); {ROOT_HINT}",
        ),
        (
            {"alias": "svc", "path": "services/auth"},
            f"path must be an absolute path (got 'services/auth'); {ROOT_HINT}",
        ),
        (
            {"alias": "svc", "path": "{missing}"},
            f"path does not exist: '{{missing}}'; {ROOT_HINT}",
        ),
        (
            {"alias": "svc", "path": "{a_file}"},
            "path is not a directory: '{a_file}'; pass the repository root, not a file in it.",
        ),
        (
            {"alias": "svc", "path": "{indexed}", "weight": -1.5},
            "weight must be a positive number (got -1.5); 1.0 is the default, and a higher value "
            "ranks that repo's results higher.",
        ),
        (
            {"alias": "svc", "path": "{indexed}", "weight": 0.0},
            "weight must be a positive number (got 0.0); 1.0 is the default, and a higher value "
            "ranks that repo's results higher.",
        ),
    ],
    ids=["blank-alias", "blank-path", "relative-path", "missing-path", "file-path", "-1.5", "0"],
)
async def test_federation_add_repo_rejects_a_bad_alias_path_or_weight_without_touching_the_registry(
    tmp_path: Path, indexed: str, a_file: str, arguments: dict[str, Any], expected: str
) -> None:
    places = {"indexed": indexed, "a_file": a_file, "missing": str(tmp_path / "missing")}
    arguments = {k: v.format(**places) if isinstance(v, str) else v for k, v in arguments.items()}

    with patch("trelix_mcp.server.RepoRegistry") as registry:
        text = await error_text("federation_add_repo", **arguments)

    assert text == expected.format(**places)
    registry.load.assert_not_called()


def test_an_infinite_weight_is_rejected_too(indexed: str) -> None:
    """JSON cannot carry infinity, so this one is driven directly."""
    with pytest.raises(ToolError, match=r"^weight must be a positive number \(got inf\); "):
        srv.federation_add_repo(alias="svc", path=indexed, weight=math.inf)


async def test_federation_remove_repo_rejects_a_blank_alias_without_touching_the_registry() -> None:
    with patch("trelix_mcp.server.RepoRegistry") as registry:
        text = await error_text("federation_remove_repo", alias="")

    assert text == (
        "alias must not be empty or whitespace (got ''); pass the alias to unregister (see "
        "federation_list_repos)."
    )
    registry.load.assert_not_called()


# ---------------------------------------------------------------------------
# config_path on the four federation tools
# ---------------------------------------------------------------------------

FEDERATION_TOOLS = [
    ("federation_list_repos", {}),
    ("federation_add_repo", {"alias": "svc", "path": "{indexed}"}),
    ("federation_remove_repo", {"alias": "svc"}),
    ("federation_search_all", {"query": "q"}),
]
CONFIG_PATH_HINT = (
    "pass a path inside ~/.config/trelix or <cwd>/.trelix, or omit it for the default registry."
)


@pytest.mark.parametrize(
    ("tool", "arguments"), FEDERATION_TOOLS, ids=[t for t, _ in FEDERATION_TOOLS]
)
@pytest.mark.parametrize("blank", ["", " "], ids=["empty", "whitespace"])
async def test_a_blank_config_path_is_an_error_and_does_not_mean_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indexed: str,
    tool: str,
    arguments: dict[str, Any],
    blank: str,
) -> None:
    """A blank config_path used to resolve to the server's working directory and be refused with a
    200 `error` dict naming the allowed roots and that directory, which the caller never passed."""
    monkeypatch.chdir(tmp_path)
    arguments = {k: v.format(indexed=indexed) for k, v in arguments.items()}

    with patch("trelix_mcp.server.RepoRegistry") as registry:
        text = await error_text(tool, **arguments, config_path=blank)

    assert text == (
        f"config_path must not be empty or whitespace (got '{blank}'); {CONFIG_PATH_HINT}"
    )
    assert str(tmp_path) not in text
    registry.load.assert_not_called()


# ---------------------------------------------------------------------------
# What FastMCP rejects itself, before a tool runs: a wrong type, a value outside a Literal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "argument", "valid"),
    [
        ("search_code", {"query": "q", "k": "abc"}, "k", "Input should be a valid integer"),
        ("search_code", {"query": "q", "cursor": 1.5}, "cursor", "Input should be a valid integer"),
        (
            "graph_search_mcp",
            {"query": "q", "detail": "verbose"},
            "detail",
            "Input should be 'concise' or 'detailed'",
        ),
        (
            "index_codebase",
            {"provider": "bogus"},
            "provider",
            "Input should be 'local', 'openai', 'azure', 'voyage' or 'local-code'",
        ),
    ],
    ids=["k-str", "cursor-float", "detail-unknown", "provider-unknown"],
)
async def test_a_wrong_type_or_unknown_choice_is_an_error_naming_the_argument(
    indexed: str, tool: str, arguments: dict[str, Any], argument: str, valid: str
) -> None:
    """FastMCP's own argument validation: an `isError` result whose text names the argument and
    the valid values (its wording is pydantic's, so only those two facts are pinned)."""
    text = await error_text(tool, **arguments, repo_path=indexed)

    assert f"\n{argument}\n" in text
    assert valid in text
