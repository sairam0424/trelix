"""Tests for multi-granularity sub-symbol indexing."""
from __future__ import annotations

from pathlib import Path

from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.indexing.multi_granularity import Granularity, SubSymbolChunk
from trelix.store.db import Database

_PROCESS_BODY = (
    "def process(data):\n"
    "    result = []\n"
    "    for item in data:\n"
    "        result.append(item)\n"
    "    return result\n"
)


def _make_db_with_symbol(tmp_path: Path) -> tuple[Database, int]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=50
        )
    )
    sid = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="process",
            qualified_name="process",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=20,
            signature="def process(data)",
            body=_PROCESS_BODY,
        )
    )
    return db, sid


class TestSubSymbolChunk:
    def test_dataclass_fields(self) -> None:
        chunk = SubSymbolChunk(
            parent_symbol_id=1,
            granularity=Granularity.BLOCK,
            chunk_text="for item in data:\n    result.append(item)",
            line_start=3,
            line_end=4,
            token_count=10,
        )
        assert chunk.granularity == Granularity.BLOCK
        assert chunk.line_start == 3

    def test_granularity_values(self) -> None:
        assert Granularity.FUNCTION == "function"
        assert Granularity.BLOCK == "block"
        assert Granularity.STATEMENT == "statement"


class TestSubChunksDB:
    def test_insert_and_retrieve(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        chunks = [
            SubSymbolChunk(sid, Granularity.BLOCK, "for item in data: ...", 3, 4, 8),
            SubSymbolChunk(sid, Granularity.STATEMENT, "result = []", 2, 2, 4),
        ]
        ids = db.insert_sub_chunks(chunks)
        assert len(ids) == 2
        result = db.get_sub_chunks_for_symbol(sid)
        assert len(result) == 2

    def test_filter_by_granularity(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        chunks = [
            SubSymbolChunk(sid, Granularity.BLOCK, "block text", 3, 5, 6),
            SubSymbolChunk(sid, Granularity.STATEMENT, "stmt text", 2, 2, 3),
        ]
        db.insert_sub_chunks(chunks)
        blocks = db.get_sub_chunks_for_symbol(sid, granularity="block")
        assert len(blocks) == 1
        assert blocks[0].granularity == Granularity.BLOCK

    def test_empty_for_unknown_symbol(self, tmp_path: Path) -> None:
        db, _ = _make_db_with_symbol(tmp_path)
        result = db.get_sub_chunks_for_symbol(9999)
        assert result == []
