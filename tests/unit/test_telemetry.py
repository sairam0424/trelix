"""Tests for query telemetry recording."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trelix.retrieval.telemetry import TelemetryWriter
from trelix.store.db import Database


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
