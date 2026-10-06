"""Fakes and helpers shared by the output-limit tests (test_output_*.py).

Every expected value in those tests is a literal written in the test; nothing here supplies one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trelix_mcp.server as srv
from fastmcp import Client

from trelix.core.config import IndexConfig
from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import Database

REPO = "/fake/repo"
SIGNATURE = "def handler(a, b):"


def hit(index: int, *, body: str = "b", signature: str = SIGNATURE) -> SimpleNamespace:
    """A search result shaped like trelix's SearchResult, as far as the tools read it."""
    return SimpleNamespace(
        file=SimpleNamespace(
            rel_path=f"src/pkg/module_{index:03d}.py", language=SimpleNamespace(value="python")
        ),
        symbol=SimpleNamespace(
            qualified_name=f"pkg.module_{index:03d}.handler",
            kind=SimpleNamespace(value="function"),
            line_start=1,
            line_end=10,
            body=body,
            signature=signature,
        ),
        chunk=SimpleNamespace(symbol_id=index),
        score=0.5,
        source="repo-a:vector",
    )


def session(index: int, query: str = "where is the retry logic") -> dict[str, Any]:
    return {
        "session_id": f"00000000-0000-0000-0000-{index:012d}",
        "created_at": "2026-10-01T10:00:00",
        "last_active_at": "2026-10-01T10:05:00",
        "query": query,
        "turn_count": 2,
    }


def seed(
    repo: Path, dependents: int, target_body: str = "pass", caller_dir: str = "src/callers"
) -> Path:
    """A real index: symbol `target.run` plus `dependents` callers, one file each."""
    db = Database(IndexConfig(repo_path=str(repo)).db_path_absolute)

    def add(rel_path: str, qualified_name: str, body: str) -> int:
        file_id = db.upsert_file(
            IndexedFile(
                path=f"/repo/{rel_path}",
                rel_path=rel_path,
                language=Language.PYTHON,
                hash=f"h-{rel_path}",
                size_bytes=10,
            )
        )
        return db.insert_symbol(
            Symbol(
                file_id=file_id,
                name=qualified_name.split(".")[-1],
                qualified_name=qualified_name,
                kind=SymbolKind.FUNCTION,
                line_start=5,
                line_end=9,
                signature="def f():",
                body=body,
            )
        )

    target = add("src/target.py", "target.run", target_body)
    edges = [
        CallEdge(
            caller_id=add(f"{caller_dir}/caller_module_{i:04d}.py", f"callers.m{i}.uses", "pass"),
            callee_id=target,
            callee_name="run",
            line=3,
        )
        for i in range(dependents)
    ]
    db.insert_call_edges(edges)
    db._conn.commit()
    db.close()
    return repo


async def call(tool: str, **arguments: Any) -> Any:
    """The whole CallToolResult of one tool call through an in-process client."""
    async with Client(srv.mcp) as client:
        return await client.call_tool_mcp(tool, arguments)


def texts(result: Any) -> list[str]:
    return [block.text for block in result.content if block.type == "text"]


def structured(result: Any) -> Any:
    """The structured content, unwrapped the way FastMCP wraps a bare array."""
    wrapped = (result.meta or {}).get("fastmcp", {}).get("wrap_result")
    return result.structured_content["result"] if wrapped else result.structured_content


def compact(value: Any) -> str:
    """The JSON text FastMCP sends for a result: compact, non-ASCII kept."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def sent(text: str, note: str = "") -> int:
    """What a client is sent for JSON `text` (and a `note` block), as the budget counts it.

    The text block carries `text` as a JSON string (quotes and backslashes escaped),
    `structuredContent` carries it as it is, and 400 characters cover the CallToolResult around
    them (measured at 213 to 358). A response fits a budget B when this is at most 2 * B.
    """
    total = len(text) + len(json.dumps(text, ensure_ascii=False)) + 400
    if note:
        total += len(json.dumps(note, ensure_ascii=False))
    return total


class FailingConnection:
    """A sqlite3 connection whose every query fails; `close` still reaches the real one."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self.real = real

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    def close(self) -> None:
        self.real.close()


def spy_on_database(
    monkeypatch: pytest.MonkeyPatch, module: Any, *, failing: bool = False
) -> list[Database]:
    """Record every Database `module` opens from now on; with `failing`, each one's queries fail."""
    opened: list[Database] = []

    class Spy(Database):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if failing:
                self._conn = FailingConnection(self._conn)  # type: ignore[assignment]
            opened.append(self)

    monkeypatch.setattr(module, "Database", Spy)
    return opened


def assert_all_closed(databases: list[Database], expected_count: int) -> None:
    assert len(databases) == expected_count
    for database in databases:
        connection = getattr(database._conn, "real", database._conn)
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
