"""Tests for query telemetry recording and expansion observability (v2.4)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trelix.retrieval.query_expansion import ExpandResult, MultiQueryExpander
from trelix.retrieval.telemetry import TelemetryWriter
from trelix.store.db import Database


# ---------------------------------------------------------------------------
# ExpandResult / MultiQueryExpander tests (Task 2)
# ---------------------------------------------------------------------------


def test_expand_result_dataclass() -> None:
    """ExpandResult carries queries, llm_used, and elapsed_ms."""
    r = ExpandResult(queries=["a", "b"], llm_used=True, elapsed_ms=50.0)
    assert r.queries == ["a", "b"]
    assert r.llm_used is True
    assert r.elapsed_ms == 50.0


def test_multi_query_expander_no_llm_returns_expand_result() -> None:
    """Without LLM config, expand() returns ExpandResult with llm_used=False."""
    expander = MultiQueryExpander(llm_config=None, n=2)
    result = expander.expand("authentication flow")
    assert isinstance(result, ExpandResult)
    assert result.queries == ["authentication flow"]
    assert result.llm_used is False
    assert result.elapsed_ms >= 0.0


def test_multi_query_expander_returns_original_on_failure() -> None:
    """When LLM raises, expand() returns original query with llm_used=False."""
    from unittest.mock import patch

    expander = MultiQueryExpander(llm_config=MagicMock(), n=2)
    with patch.object(expander, "_get_client", side_effect=Exception("LLM unavailable")):
        result = expander.expand("login function")
    assert result.queries == ["login function"]
    assert result.llm_used is False


# ---------------------------------------------------------------------------
# v2.4 expansion observability tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    return Database(db_path)


def test_query_telemetry_expansion_columns_exist(db: Database) -> None:
    """query_telemetry table has all 3 expansion columns after init."""
    cols = {
        row[1]
        for row in db._conn.execute("PRAGMA table_info(query_telemetry)").fetchall()
    }
    assert "expansion_used" in cols
    assert "expansion_variants" in cols
    assert "expansion_elapsed_ms" in cols


def test_insert_query_telemetry_with_expansion(db: Database) -> None:
    """insert_query_telemetry stores expansion info when provided."""
    row_id = db.insert_query_telemetry(
        query="how does auth work",
        intent="function_lookup",
        elapsed_ms=42.5,
        result_count=10,
        leg_sizes={"vector": 10, "bm25": 8},
        expansion_used=True,
        expansion_variants=3,
        expansion_elapsed_ms=120.0,
    )
    assert row_id > 0
    row = db._conn.execute(
        "SELECT expansion_used, expansion_variants, expansion_elapsed_ms "
        "FROM query_telemetry WHERE id=?",
        (row_id,),
    ).fetchone()
    assert row[0] == 1        # expansion_used = True stored as 1
    assert row[1] == 3        # expansion_variants
    assert row[2] == pytest.approx(120.0)


def test_insert_query_telemetry_without_expansion(db: Database) -> None:
    """insert_query_telemetry stores NULL when expansion not provided."""
    row_id = db.insert_query_telemetry(
        query="find login function",
        intent="",
        elapsed_ms=10.0,
        result_count=5,
    )
    row = db._conn.execute(
        "SELECT expansion_used, expansion_variants, expansion_elapsed_ms "
        "FROM query_telemetry WHERE id=?",
        (row_id,),
    ).fetchone()
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None


def test_insert_query_telemetry_expansion_used_false(db: Database) -> None:
    """expansion_used=False (LLM unavailable) is stored as 0, not NULL."""
    row_id = db.insert_query_telemetry(
        query="parse config",
        intent="",
        elapsed_ms=5.0,
        result_count=3,
        expansion_used=False,
        expansion_variants=1,  # only original query used
        expansion_elapsed_ms=0.0,
    )
    row = db._conn.execute(
        "SELECT expansion_used FROM query_telemetry WHERE id=?",
        (row_id,),
    ).fetchone()
    assert row[0] == 0   # False stored as 0


class TestTelemetryWriter:
    def test_record_inserts_row(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        writer = TelemetryWriter(db, enabled=True)
        mock_ctx = MagicMock()
        mock_ctx.query = "how does auth work"
        mock_ctx.intent = "code_search"
        mock_ctx.results = [MagicMock(), MagicMock()]
        writer.record(mock_ctx, elapsed_ms=42.5)
        rows = db.get_recent_telemetry(limit=10)
        assert len(rows) == 1
        assert rows[0]["query"] == "how does auth work"
        assert rows[0]["elapsed_ms"] == pytest.approx(42.5)
        assert rows[0]["result_count"] == 2

    def test_record_noop_when_disabled(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        writer = TelemetryWriter(db, enabled=False)
        mock_ctx = MagicMock()
        mock_ctx.query = "test"
        mock_ctx.results = []
        writer.record(mock_ctx, elapsed_ms=10.0)
        rows = db.get_recent_telemetry(limit=10)
        assert len(rows) == 0

    def test_record_never_raises(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        writer = TelemetryWriter(db, enabled=True)
        # Malformed context should not crash
        writer.record(MagicMock(spec=[]), elapsed_ms=0.0)

    def test_telemetry_disabled_by_default(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path))
        assert cfg.telemetry_enabled is False


def test_telemetry_writer_records_expansion(tmp_path) -> None:
    """TelemetryWriter.record() persists expansion metadata when provided."""
    from trelix.store.db import Database
    from unittest.mock import MagicMock

    db = Database(tmp_path / "test.db")
    writer = TelemetryWriter(db, enabled=True)

    context = MagicMock()
    context.query = "find auth code"
    context.intent = "function_lookup"
    context.results = [MagicMock() for _ in range(5)]

    expand_result = ExpandResult(queries=["find auth code", "locate login"], llm_used=True, elapsed_ms=88.5)
    writer.record(context, elapsed_ms=200.0, expansion_result=expand_result)

    row = db._conn.execute(
        "SELECT expansion_used, expansion_variants, expansion_elapsed_ms "
        "FROM query_telemetry ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == 1           # llm_used=True
    assert row[1] == 2           # 2 queries in result
    assert row[2] == pytest.approx(88.5)


def test_telemetry_writer_no_expansion_stores_null(tmp_path) -> None:
    """TelemetryWriter.record() stores NULL when expansion_result=None."""
    from trelix.store.db import Database
    from unittest.mock import MagicMock

    db = Database(tmp_path / "test.db")
    writer = TelemetryWriter(db, enabled=True)

    context = MagicMock()
    context.query = "hash password"
    context.intent = ""
    context.results = []

    writer.record(context, elapsed_ms=15.0)

    row = db._conn.execute(
        "SELECT expansion_used, expansion_variants FROM query_telemetry ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] is None
    assert row[1] is None
