"""`k`, `limit`, `cursor` and `detail` on the list-returning MCP tools.

* `TRELIX_MCP_MAX_K` and `TRELIX_MCP_MAX_RESULT_CHARS` are read on every call, blank means the
  default, and an unusable value is a clear error (at start-up, and as a tool error on a call).
* `k` and `limit` clamp to 1..TRELIX_MCP_MAX_K; a negative cursor is an error result.
* `detail="concise"` drops `body` and carries a one-line `signature`.

Every expected value is a literal written here; the tools are driven directly (their plain
return values) or through an in-process `fastmcp.Client` (what reaches a client).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trelix_mcp.server as srv
from budget_support import REPO, SIGNATURE, call, hit, session, texts
from fastmcp.exceptions import ToolError
from trelix_mcp.budget import BudgetConfigError, limits_from_env

# ---------------------------------------------------------------------------
# The two environment variables
# ---------------------------------------------------------------------------


def test_the_defaults_are_fifty_and_fifteen_thousand() -> None:
    limits = limits_from_env({})
    assert (limits.max_k, limits.max_result_chars) == (50, 15000)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"TRELIX_MCP_MAX_K": "7"}, (7, 15000)),
        ({"TRELIX_MCP_MAX_K": "  "}, (50, 15000)),
        ({"TRELIX_MCP_MAX_RESULT_CHARS": "0"}, (50, 0)),
        ({"TRELIX_MCP_MAX_RESULT_CHARS": ""}, (50, 15000)),
        ({"TRELIX_MCP_MAX_K": "3", "TRELIX_MCP_MAX_RESULT_CHARS": "2500"}, (3, 2500)),
    ],
    ids=["max-k", "blank-max-k", "chars-off", "blank-chars", "both"],
)
def test_the_variables_are_read(environ: dict[str, str], expected: tuple[int, int]) -> None:
    limits = limits_from_env(environ)
    assert (limits.max_k, limits.max_result_chars) == expected


@pytest.mark.parametrize(
    ("name", "value", "minimum"),
    [
        ("TRELIX_MCP_MAX_K", "many", 1),
        ("TRELIX_MCP_MAX_K", "0", 1),
        ("TRELIX_MCP_MAX_K", "-5", 1),
        ("TRELIX_MCP_MAX_K", "2.5", 1),
        ("TRELIX_MCP_MAX_RESULT_CHARS", "lots", 0),
        ("TRELIX_MCP_MAX_RESULT_CHARS", "-1", 0),
    ],
)
def test_an_unusable_value_is_a_config_error(name: str, value: str, minimum: int) -> None:
    with pytest.raises(BudgetConfigError) as caught:
        limits_from_env({name: value})
    assert str(caught.value) == f"{name} must be an integer of at least {minimum}, got {value!r}"


def test_a_tool_call_with_an_unusable_limit_is_a_tool_error(
    monkeypatch: pytest.MonkeyPatch, backends: SimpleNamespace
) -> None:
    monkeypatch.setenv("TRELIX_MCP_MAX_K", "many")
    with pytest.raises(ToolError, match="TRELIX_MCP_MAX_K must be an integer of at least 1"):
        srv.search_code("q", REPO)


# ---------------------------------------------------------------------------
# k and limit clamp; a negative cursor is an error
# ---------------------------------------------------------------------------


def test_k_is_clamped(backends: SimpleNamespace) -> None:
    backends.hits = [hit(i) for i in range(120)]

    page = srv.search_code("q", REPO, k=500)
    assert (page["page_size"], len(page["results"]), page["next_cursor"]) == (50, 50, 50)
    assert srv.search_code("q", REPO, k=0)["page_size"] == 1
    assert len(srv.search_code("q", REPO, k=-7)["results"]) == 1

    assert len(srv.graph_search_mcp("q", REPO, k=500)) == 50
    assert backends.graph_max_results == 50
    assert len(srv.graph_search_mcp("q", REPO, k=0)) == 1

    federated = srv.federation_search_all("q", k=500)
    assert (federated["page_size"], len(federated["results"])) == (50, 50)
    assert len(srv.federation_search_all("q", k=-1)["results"]) == 1


def test_the_clamp_follows_trelix_mcp_max_k(
    monkeypatch: pytest.MonkeyPatch, backends: SimpleNamespace
) -> None:
    backends.hits = [hit(i) for i in range(20)]
    monkeypatch.setenv("TRELIX_MCP_MAX_K", "5")
    page = srv.search_code("q", REPO, k=10)
    assert (page["page_size"], len(page["results"]), page["next_cursor"]) == (5, 5, 5)


def test_agent_list_sessions_limit_is_clamped(sessions: SimpleNamespace) -> None:
    sessions.items = [session(i) for i in range(80)]

    response = srv.agent_list_sessions(REPO, limit=500)
    assert (response["page_size"], response["count"]) == (50, 50)
    sessions.database.list_agent_sessions.assert_called_once_with(limit=50)
    assert srv.agent_list_sessions(REPO, limit=0)["page_size"] == 1


async def test_blast_radius_limit_defaults_to_100_and_is_clamped_to_1_and_500(
    monkeypatch: pytest.MonkeyPatch, dependents_repo: Path
) -> None:
    repo = str(dependents_repo)
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", "0")

    default = await call("blast_radius", symbol_name="target.run", repo_path=repo)
    assert len(json.loads(texts(default)[0])) == 100
    one = await call("blast_radius", symbol_name="target.run", repo_path=repo, limit=0)
    assert len(json.loads(texts(one)[0])) == 1
    many = await call("blast_radius", symbol_name="target.run", repo_path=repo, limit=10_000)
    assert len(json.loads(texts(many)[0])) == 500
    assert many.meta["trelix"] == {"total_available": 520, "omitted": 20}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [("search_code", {"query": "q", "repo_path": REPO}), ("federation_search_all", {"query": "q"})],
)
async def test_negative_cursor_is_iserror(
    backends: SimpleNamespace, tool: str, arguments: dict[str, str]
) -> None:
    result = await call(tool, cursor=-1, **arguments)

    assert result.is_error is True
    assert texts(result)[0].endswith(
        "cursor must be 0 or greater, got -1. Use 0 for the first page, then pass the "
        "next_cursor value from the previous response."
    )
    backends.retriever.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# detail
# ---------------------------------------------------------------------------


def test_detailed_rows_are_the_rows_the_tools_always_returned(backends: SimpleNamespace) -> None:
    backends.hits = [hit(7, body="b" * 1000)]

    assert srv.search_code("q", REPO)["results"] == [
        {
            "file": "src/pkg/module_007.py",
            "symbol": "pkg.module_007.handler",
            "kind": "function",
            "lines": "1-10",
            "score": 0.5,
            "source": "repo-a:vector",
            "body": "b" * 800,
            "language": "python",
        }
    ]
    assert srv.graph_search_mcp("q", REPO) == [
        {
            "file": "src/pkg/module_007.py",
            "symbol": "pkg.module_007.handler",
            "kind": "function",
            "score": 0.5,
            "source": "repo-a:vector",
            "body": "b" * 600,
        }
    ]
    row = srv.federation_search_all("q")["results"][0]
    assert (row["repo"], row["body"]) == ("repo-a", "b" * 800)


@pytest.mark.parametrize(
    ("fetch", "keys"),
    [
        (
            lambda: srv.search_code("q", REPO, detail="concise")["results"][0],
            {"file", "symbol", "kind", "lines", "score", "source", "signature", "language"},
        ),
        (
            lambda: srv.graph_search_mcp("q", REPO, detail="concise")[0],
            {"file", "symbol", "kind", "score", "source", "signature"},
        ),
        (
            lambda: srv.federation_search_all("q", detail="concise")["results"][0],
            {"repo", "file", "symbol", "kind", "score", "source", "signature", "language"},
        ),
    ],
    ids=["search_code", "graph_search_mcp", "federation_search_all"],
)
def test_concise_rows_drop_the_body_and_carry_a_one_line_signature(
    backends: SimpleNamespace, fetch: Any, keys: set[str]
) -> None:
    backends.hits = [hit(1, body="b" * 1000, signature=f"{SIGNATURE}\n    '''doc'''")]

    row = fetch()

    assert set(row) == keys
    assert row["signature"] == SIGNATURE


def test_a_blank_signature_falls_back_to_the_first_line_of_the_body(
    backends: SimpleNamespace,
) -> None:
    backends.hits = [hit(1, body="\n  first line\nsecond line", signature="   ")]
    assert srv.search_code("q", REPO, detail="concise")["results"][0]["signature"] == "first line"


def test_a_very_long_signature_line_is_cut_to_200_characters(backends: SimpleNamespace) -> None:
    backends.hits = [hit(1, signature="s" * 500)]
    assert srv.search_code("q", REPO, detail="concise")["results"][0]["signature"] == "s" * 200
