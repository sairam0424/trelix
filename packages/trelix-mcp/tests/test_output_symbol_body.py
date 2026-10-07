"""`get_symbol` cuts long bodies, and it and the resource readers close their database.

A connection left open keeps a file handle (and, in WAL mode, a -wal/-shm pair) alive for as long
as the server process runs. A spy records every `Database` the code opens; afterwards each one
must refuse a query.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import trelix_mcp.resources as resources
import trelix_mcp.server as srv
from budget_support import assert_all_closed, seed, spy_on_database
from fastmcp.exceptions import ToolError


@pytest.fixture
def long_body_repo(tmp_path: Path) -> Path:
    return seed(tmp_path, 0, target_body="x" * 25_000)


@pytest.mark.parametrize(
    ("max_body_chars", "body_length", "truncated"),
    [
        (25_000, 25_000, False),
        (24_999, 24_999, True),
        (0, 25_000, False),
        (100, 100, True),
        (30_000, 25_000, False),
    ],
)
def test_get_symbol_cuts_a_long_body(
    long_body_repo: Path, max_body_chars: int, body_length: int, truncated: bool
) -> None:
    symbol = srv.get_symbol("target.run", str(long_body_repo), max_body_chars=max_body_chars)
    assert (len(symbol["body"]), symbol["body_truncated"]) == (body_length, truncated)


def test_get_symbol_cuts_at_twenty_thousand_by_default(long_body_repo: Path) -> None:
    symbol = srv.get_symbol("target.run", str(long_body_repo))
    assert (len(symbol["body"]), symbol["body_truncated"]) == (20_000, True)
    assert set(symbol) == {
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
        "body_truncated",
    }


def test_a_negative_max_body_chars_is_a_tool_error(long_body_repo: Path) -> None:
    with pytest.raises(ToolError, match="max_body_chars must be 0 .no limit. or greater, got -1"):
        srv.get_symbol("target.run", str(long_body_repo), max_body_chars=-1)


def test_get_symbol_closes_database(monkeypatch: pytest.MonkeyPatch, long_body_repo: Path) -> None:
    opened = spy_on_database(monkeypatch, srv)

    assert srv.get_symbol("target.run", str(long_body_repo)) is not None
    # An unknown symbol travels as a ToolResult (a `null` text block), not as a bare None.
    assert srv.get_symbol("no.such.symbol", str(long_body_repo)).structured_content == {
        "result": None
    }

    assert_all_closed(opened, 2)


def test_get_symbol_closes_database_when_a_query_fails(
    monkeypatch: pytest.MonkeyPatch, long_body_repo: Path
) -> None:
    opened = spy_on_database(monkeypatch, srv, failing=True)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        srv.get_symbol("target.run", str(long_body_repo))

    assert_all_closed(opened, 1)


RESOURCE_READERS: dict[str, Any] = {
    "index_stats": lambda repo: resources.get_index_stats(repo),
    "repo_manifest": lambda repo: resources.get_repo_manifest(repo),
    "symbol_source": lambda repo: resources.get_symbol_source(repo, "target.run"),
}


@pytest.mark.parametrize(
    "read",
    [*RESOURCE_READERS.values(), lambda repo: resources.get_symbol_source(repo, "no.such.symbol")],
    ids=[*RESOURCE_READERS, "symbol_source_missing"],
)
def test_resource_readers_close_their_database(
    monkeypatch: pytest.MonkeyPatch, small_repo: Path, read: Any
) -> None:
    opened = spy_on_database(monkeypatch, resources)

    assert isinstance(json.loads(read(str(small_repo))), dict)

    assert_all_closed(opened, 1)


@pytest.mark.parametrize("read", RESOURCE_READERS.values(), ids=list(RESOURCE_READERS))
def test_resource_readers_close_their_database_when_a_query_fails(
    monkeypatch: pytest.MonkeyPatch, small_repo: Path, read: Any
) -> None:
    opened = spy_on_database(monkeypatch, resources, failing=True)

    assert json.loads(read(str(small_repo)))["error"] == "disk I/O error"

    assert_all_closed(opened, 1)
