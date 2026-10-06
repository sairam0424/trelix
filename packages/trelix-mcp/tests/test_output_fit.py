"""Fit to budget: a list result stays inside TRELIX_MCP_MAX_RESULT_CHARS, as a client is sent it.

The budget counts both copies of the JSON a client receives (the text block, with its quotes
escaped, and `structuredContent`) and a fixed reserve for the rest of the response, so the whole
response is at most twice the budget. `sent` in budget_support measures that.

A cursor-paged envelope (`search_code`, `federation_search_all`) drops its tail and keeps
`next_cursor` continuous. A bare array (`blast_radius`, `graph_search_mcp`) keeps `content[0]` as
its JSON array and adds a note block and `_meta.trelix`, so a cut can never go unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import trelix_mcp.server as srv
from budget_support import REPO, call, compact, hit, sent, session, texts

# ---------------------------------------------------------------------------
# A cursor-paged envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["b" * 300, "é" * 300], ids=["ascii", "non-ascii"])
def test_a_page_over_budget_drops_its_tail_and_says_so(
    monkeypatch: pytest.MonkeyPatch, backends: SimpleNamespace, body: str
) -> None:
    backends.hits = [hit(i, body=body) for i in range(30)]
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", "2000")

    page = srv.search_code("q", REPO, k=20)

    text = compact(page)
    kept = len(page["results"])
    assert sent(text) <= 4000
    assert 0 < kept < 20
    assert page["truncated"] is True
    assert page["omitted"] == 20 - kept
    assert page["next_cursor"] == kept
    assert (page["total_available"], page["page_size"]) == (30, 20)
    # Maximal: the smallest budget that holds this page keeps the same rows, one less keeps one
    # fewer. (Non-ASCII text is counted as FastMCP sends it, not as ASCII escapes.)
    budget = (sent(text) + 1) // 2
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", str(budget))
    assert len(srv.search_code("q", REPO, k=20)["results"]) == kept
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", str(budget - 1))
    assert len(srv.search_code("q", REPO, k=20)["results"]) == kept - 1


def test_a_page_within_budget_is_reported_as_not_truncated(backends: SimpleNamespace) -> None:
    backends.hits = [hit(i) for i in range(5)]
    page = srv.search_code("q", REPO)
    assert (page["truncated"], page["omitted"], page["page_size"]) == (False, 0, 10)
    assert page["next_cursor"] is None


def test_the_default_budget_is_fifteen_thousand_characters(backends: SimpleNamespace) -> None:
    backends.hits = [hit(i, body="b" * 800) for i in range(30)]
    page = srv.search_code("q", REPO, k=30)
    assert page["truncated"] is True
    assert len(page["results"]) == 14
    assert len(compact(page)) <= 15000
    assert sent(compact(page)) <= 30000


def test_a_budget_of_zero_turns_fitting_off(
    monkeypatch: pytest.MonkeyPatch, backends: SimpleNamespace
) -> None:
    backends.hits = [hit(i, body="b" * 800) for i in range(30)]
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", "0")
    page = srv.search_code("q", REPO, k=30)
    assert (len(page["results"]), page["truncated"]) == (30, False)
    assert len(compact(page)) > 15000


def test_one_result_is_always_kept_so_paging_advances(
    monkeypatch: pytest.MonkeyPatch, backends: SimpleNamespace
) -> None:
    backends.hits = [hit(i) for i in range(5)]
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", "1")
    page = srv.search_code("q", REPO, k=5)
    assert (len(page["results"]), page["omitted"], page["next_cursor"]) == (1, 4, 1)


@pytest.mark.parametrize("tool", ["search_code", "federation_search_all"])
def test_paging_through_a_truncated_result_set_loses_and_repeats_nothing(
    monkeypatch: pytest.MonkeyPatch, backends: SimpleNamespace, tool: str
) -> None:
    backends.hits = [hit(i, body="b" * 300) for i in range(40)]
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", "1800")
    fetch = {
        "search_code": lambda cursor: srv.search_code("q", REPO, k=10, cursor=cursor),
        "federation_search_all": lambda cursor: srv.federation_search_all("q", k=10, cursor=cursor),
    }[tool]

    seen: list[str] = []
    truncated_pages = 0
    cursor = 0
    for _ in range(60):
        page = fetch(cursor)
        assert sent(compact(page)) <= 3600
        seen += [row["symbol"] for row in page["results"]]
        truncated_pages += page["truncated"]
        if page["next_cursor"] is None:
            break
        assert page["next_cursor"] == cursor + len(page["results"])
        cursor = page["next_cursor"]

    assert seen == [f"pkg.module_{i:03d}.handler" for i in range(40)]
    assert truncated_pages >= 2


def test_agent_list_sessions_drops_the_oldest_sessions_over_budget(
    monkeypatch: pytest.MonkeyPatch, sessions: SimpleNamespace
) -> None:
    sessions.items = [session(i, query="q" * 150) for i in range(50)]
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", "5000")

    response = srv.agent_list_sessions(REPO)

    assert sent(compact(response)) <= 10000
    assert (response["truncated"], response["count"]) == (True, len(response["sessions"]))
    assert response["omitted"] == 50 - response["count"]
    assert response["sessions"][0] == sessions.items[0]


def test_agent_list_sessions_cuts_a_long_query_and_says_so(sessions: SimpleNamespace) -> None:
    """A session's latest prompt is unbounded, so the listing shows the first 300 characters."""
    sessions.items = [session(0, query="q" * 5_000), session(1, query="short")]

    response = srv.agent_list_sessions(REPO)

    long_row, short_row = response["sessions"]
    assert long_row == {**sessions.items[0], "query": "q" * 300, "query_truncated": True}
    assert short_row == sessions.items[1]
    assert response["truncated"] is False
    assert sessions.items[0]["query"] == "q" * 5_000


@pytest.mark.parametrize(
    ("length", "is_cut"), [(299, False), (300, False), (301, True)], ids=["under", "at", "over"]
)
def test_agent_list_sessions_cuts_a_query_only_past_three_hundred_characters(
    sessions: SimpleNamespace, length: int, is_cut: bool
) -> None:
    sessions.items = [session(0, query="q" * length)]

    row = srv.agent_list_sessions(REPO)["sessions"][0]

    assert (len(row["query"]), row.get("query_truncated", False)) == (min(length, 300), is_cut)
    assert ("query_truncated" in row) is is_cut


# ---------------------------------------------------------------------------
# A bare array
# ---------------------------------------------------------------------------


async def test_a_cut_bare_array_keeps_the_array_first_and_says_so(dependents_repo: Path) -> None:
    result = await call("blast_radius", symbol_name="target.run", repo_path=str(dependents_repo))

    first, note = result.content
    entries = json.loads(first.text)
    assert len(entries) == 100
    assert entries[0] == {
        "file": "src/callers/caller_module_0000.py",
        "symbol": "callers.m0.uses",
        "kind": "function",
        "line_start": 5,
        "language": "python",
    }
    assert note.text == (
        "Truncated: 100 of 520 dependents returned, 420 omitted. The list is bounded by the "
        "limit argument (at most 500) and by TRELIX_MCP_MAX_RESULT_CHARS; raise either to see more."
    )
    assert result.meta["trelix"] == {"total_available": 520, "omitted": 420}
    assert result.structured_content == {"result": entries}
    assert result.meta["fastmcp"] == {"wrap_result": True}
    assert result.is_error is False


async def test_a_bare_array_over_budget_counts_the_note_in_the_budget(
    monkeypatch: pytest.MonkeyPatch, dependents_repo: Path
) -> None:
    arguments = {"symbol_name": "target.run", "repo_path": str(dependents_repo), "limit": 500}
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", "3000")
    result = await call("blast_radius", **arguments)

    array, note = texts(result)
    entries = json.loads(array)
    kept = len(entries)
    assert sent(array, note) <= 6000
    assert 0 < kept < 500
    assert result.meta["trelix"] == {"total_available": 520, "omitted": 520 - kept}
    assert result.structured_content == {"result": entries}
    # Maximal with the note counted: the smallest budget that holds this response keeps the same
    # rows, one less keeps one fewer.
    budget = (sent(array, note) + 1) // 2
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", str(budget))
    assert len(json.loads(texts(await call("blast_radius", **arguments))[0])) == kept
    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", str(budget - 1))
    assert len(json.loads(texts(await call("blast_radius", **arguments))[0])) == kept - 1


async def test_a_bare_array_that_fits_is_returned_untouched(small_repo: Path) -> None:
    result = await call("blast_radius", symbol_name="target.run", repo_path=str(small_repo))

    assert len(texts(result)) == 1
    assert len(json.loads(texts(result)[0])) == 3
    assert "trelix" not in (result.meta or {})
    assert result.structured_content == {"result": json.loads(texts(result)[0])}


async def test_a_bare_array_that_exactly_fits_is_not_charged_for_a_note(
    monkeypatch: pytest.MonkeyPatch, small_repo: Path
) -> None:
    arguments = {"symbol_name": "target.run", "repo_path": str(small_repo)}
    array = texts(await call("blast_radius", **arguments))[0]

    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", str((sent(array) + 1) // 2))
    fits = await call("blast_radius", **arguments)
    assert texts(fits) == [array]
    assert "trelix" not in (fits.meta or {})

    monkeypatch.setenv("TRELIX_MCP_MAX_RESULT_CHARS", str((sent(array) + 1) // 2 - 1))
    cut = await call("blast_radius", **arguments)
    assert len(json.loads(texts(cut)[0])) == 2
    assert cut.meta["trelix"] == {"total_available": 3, "omitted": 1}


async def test_graph_search_over_budget_reports_what_it_left_out(
    backends: SimpleNamespace,
) -> None:
    backends.hits = [hit(i, body="b" * 700) for i in range(50)]

    result = await call("graph_search_mcp", query="q", repo_path=REPO, k=50)

    first, note = result.content
    kept = len(json.loads(first.text))
    assert 0 < kept < 50
    assert note.text == (
        f"Truncated: {kept} of 50 results returned, {50 - kept} omitted. "
        "The list is bounded by TRELIX_MCP_MAX_RESULT_CHARS, not by k; "
        'use detail="concise" for shorter results, or raise it, to see more.'
    )
    assert result.meta["trelix"] == {"total_available": 50, "omitted": 50 - kept}


async def test_graph_search_concise_is_the_remedy_its_note_names(backends: SimpleNamespace) -> None:
    backends.hits = [hit(i, body="b" * 700) for i in range(50)]

    result = await call("graph_search_mcp", query="q", repo_path=REPO, k=50, detail="concise")

    assert len(result.content) == 1
    assert len(json.loads(result.content[0].text)) == 50
    assert "trelix" not in (result.meta or {})
