"""Every tool annotated readOnlyHint=True leaves the index database byte-for-byte alone.

The hint is a promise clients may act on (skip a confirmation, run the tool in parallel), so
it is claimed only where this file has watched a real call do nothing to the file. A real
temporary repository is indexed, each tool runs through an in-process `fastmcp.Client`, and
the SHA-256 of `.trelix/index.db` and its `-wal` and `-shm` files is taken before and after.

The claim is scoped: an index already at the current schema and telemetry off (the fixture
pins it, so an operator's environment cannot change the result), and it is about the index
database, not every file. What falls outside it is recorded as tests at the bottom, so the
docs that mention it stay true: two cases that change the database, and the trace file that
`search_code` writes beside it (the database stays identical).

Both snapshots are taken with every connection closed (the cached Retriever is dropped and
the garbage collector run): an open SQLite connection in WAL mode creates empty `-wal` and
`-shm` files and a closing one checkpoints into the main file, and either would look like a
write. The embedder and the tokenizer are small deterministic stand-ins (the real tokenizer
downloads its vocabulary file on a cold cache, and tests use no network); the index, the
SQLite file and the tool code are real. `build_knowledge_graph` is the control: it rewrites
graph metadata, and the same measurement must see it.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import trelix_mcp.server as srv
from fastmcp import Client

from trelix.embedder.base import BaseEmbedder

# Tools annotated readOnlyHint=True. The test below fails when the annotations and this list
# disagree, so a tool cannot be marked read-only without being added (and measured) here.
READ_ONLY_TOOLS = ("search_code", "get_symbol", "blast_radius")


class _HashEmbedder(BaseEmbedder):
    """Deterministic 8-dimensional vectors from a SHA-256 of the text. No model, no network."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i] / 255.0 for i in range(8)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return 8


class _ByteEncoding:
    """tiktoken stand-in: one token per UTF-8 byte. The chunker only decodes to split a symbol
    over its token budget, and the two-file repository has none."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode())


def _fingerprint(repo: Path) -> dict[str, str | None]:
    """SHA-256 of the index file and its -wal and -shm files; None for a file that is absent."""
    db_file = repo / ".trelix" / "index.db"
    result: dict[str, str | None] = {}
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(db_file) + suffix)
        result[suffix or "db"] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
    return result


def _files_under_dot_trelix(repo: Path) -> list[str]:
    """Every file under .trelix, as sorted paths relative to it."""
    root = repo / ".trelix"
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _close_every_connection() -> None:
    srv._retriever_cache.clear()
    gc.collect()


def _sql(repo: Path, statement: str) -> Any:
    """Run one statement on the index with a short-lived connection; return its first value."""
    connection = sqlite3.connect(repo / ".trelix" / "index.db")
    try:
        row = connection.execute(statement).fetchone()
        connection.commit()
        return None if row is None else row[0]
    finally:
        connection.close()


@pytest.fixture
def indexed_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real two-file Python repository, indexed, with no database connection left open."""
    monkeypatch.setenv("TRELIX_TELEMETRY_ENABLED", "false")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text(
        "def target():\n    return 1\n\n\ndef helper():\n    return target()\n"
    )
    (repo / "b.py").write_text("from a import target\n\n\ndef user():\n    return target()\n")
    embedder = _HashEmbedder()
    monkeypatch.setattr("tiktoken.get_encoding", lambda _name: _ByteEncoding())
    monkeypatch.setattr("trelix.indexing.indexer.make_embedder", lambda _config: embedder)
    monkeypatch.setattr("trelix.retrieval.retriever.make_embedder", lambda _config: embedder)
    srv.index_codebase(str(repo))
    _close_every_connection()
    return repo


def _arguments(tool: str, repo: Path) -> dict[str, str]:
    return {
        "search_code": {"query": "target function", "repo_path": str(repo)},
        "get_symbol": {"qualified_name": "target", "repo_path": str(repo)},
        "blast_radius": {"symbol_name": "target", "repo_path": str(repo)},
        "build_knowledge_graph": {"repo_path": str(repo)},
    }[tool]


async def _call(tool: str, repo: Path) -> Any:
    async with Client(srv.mcp) as client:
        result = await client.call_tool(tool, _arguments(tool, repo))
    return result.structured_content


@pytest.mark.asyncio
async def test_the_tools_annotated_read_only_are_the_ones_measured_here() -> None:
    async with Client(srv.mcp) as client:
        tools = await client.list_tools_mcp()

    annotated = {tool.name for tool in tools.tools if tool.annotations.read_only_hint}

    assert annotated == set(READ_ONLY_TOOLS)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", READ_ONLY_TOOLS)
async def test_readonly_tools_do_not_modify_index_db(tool: str, indexed_repo: Path) -> None:
    before = _fingerprint(indexed_repo)
    assert before["db"] is not None
    assert before["-wal"] is None
    assert before["-shm"] is None

    structured = await _call(tool, indexed_repo)
    _close_every_connection()

    assert _fingerprint(indexed_repo) == before
    assert structured  # the tool really ran against the index and answered


@pytest.mark.asyncio
async def test_the_measurement_sees_a_tool_that_does_write(indexed_repo: Path) -> None:
    """Control: build_knowledge_graph saves graph metadata, so the fingerprint must change."""
    before = _fingerprint(indexed_repo)

    structured = await _call("build_knowledge_graph", indexed_repo)
    _close_every_connection()

    assert structured["node_count"] == 3
    assert _fingerprint(indexed_repo) != before


@pytest.mark.asyncio
async def test_the_read_only_tools_return_real_answers_from_the_index(indexed_repo: Path) -> None:
    """Not vacuous: each read-only tool answered from the seeded repository."""
    found = await _call("search_code", indexed_repo)
    symbol = await _call("get_symbol", indexed_repo)
    radius = await _call("blast_radius", indexed_repo)

    assert "target" in {hit["symbol"] for hit in found["results"]}
    assert "next_cursor" in found  # the key the instructions tell a model to pass back
    assert symbol["result"]["qualified_name"] == "target"
    assert {entry["symbol"] for entry in radius["result"]} == {"helper", "user"}


@pytest.mark.asyncio
async def test_known_exception_search_code_records_a_telemetry_row_when_telemetry_is_on(
    indexed_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TRELIX_TELEMETRY_ENABLED=true: each search_code inserts one query_telemetry row."""
    assert _sql(indexed_repo, "SELECT COUNT(*) FROM query_telemetry") == 0
    before = _fingerprint(indexed_repo)
    monkeypatch.setenv("TRELIX_TELEMETRY_ENABLED", "true")

    structured = await _call("search_code", indexed_repo)
    _close_every_connection()

    assert structured
    assert _fingerprint(indexed_repo) != before
    assert _sql(indexed_repo, "SELECT COUNT(*) FROM query_telemetry") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", READ_ONLY_TOOLS)
async def test_known_exception_the_first_open_of_an_older_schema_index_migrates_it(
    tool: str, indexed_repo: Path
) -> None:
    """An index stamped by an older trelix (user_version 0) is upgraded by the first open."""
    _sql(indexed_repo, "PRAGMA user_version = 0")
    before = _fingerprint(indexed_repo)

    structured = await _call(tool, indexed_repo)
    _close_every_connection()

    assert structured
    assert _fingerprint(indexed_repo) != before
    assert _sql(indexed_repo, "PRAGMA user_version") > 0


@pytest.mark.asyncio
async def test_known_exception_search_code_writes_a_trace_file_and_leaves_the_database_alone(
    indexed_repo: Path,
) -> None:
    """Every search_code writes one JSON trace of the query to .trelix/debug/ (telemetry is off)."""
    files_before = _files_under_dot_trelix(indexed_repo)
    database_before = _fingerprint(indexed_repo)

    structured = await _call("search_code", indexed_repo)
    _close_every_connection()

    assert structured
    assert _fingerprint(indexed_repo) == database_before
    created = sorted(set(_files_under_dot_trelix(indexed_repo)) - set(files_before))
    assert len(created) == 1
    assert created[0].startswith("debug/")
    assert created[0].endswith(".json")
    trace = json.loads((indexed_repo / ".trelix" / created[0]).read_text())
    assert trace["query"] == "target function"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("get_symbol", "blast_radius"))
async def test_get_symbol_and_blast_radius_create_no_file_under_dot_trelix(
    tool: str, indexed_repo: Path
) -> None:
    """The trace file is search_code's alone: the other two read-only tools add nothing."""
    files_before = _files_under_dot_trelix(indexed_repo)

    structured = await _call(tool, indexed_repo)
    _close_every_connection()

    assert structured
    assert _files_under_dot_trelix(indexed_repo) == files_before
