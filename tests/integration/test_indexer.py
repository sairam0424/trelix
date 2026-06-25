"""
Integration test for the Indexer orchestrator (Phase 8).

Creates a mini repo with Python + TypeScript files, runs the full 4-phase
pipeline with the local (no-API-key) embedder, and asserts core invariants.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mini_repo(tmp_path: Path) -> Path:
    """Create a minimal repo with a Python file and a TypeScript file."""

    # Python file: 2 functions + 1 class
    py_src = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            \"\"\"Return the sum of two integers.\"\"\"
            return a + b


        def multiply(a: int, b: int) -> int:
            \"\"\"Return the product of two integers.\"\"\"
            return a * b


        class Calculator:
            \"\"\"Simple calculator that uses add and multiply.\"\"\"

            def __init__(self) -> None:
                self.history: list[int] = []

            def compute(self, a: int, b: int) -> int:
                result = add(a, b)
                self.history.append(result)
                return result
    """)
    (tmp_path / "calc.py").write_text(py_src, encoding="utf-8")

    # TypeScript file: 1 interface + 1 function
    ts_src = textwrap.dedent("""\
        interface Vector {
            x: number;
            y: number;
        }

        function magnitude(v: Vector): number {
            return Math.sqrt(v.x * v.x + v.y * v.y);
        }
    """)
    (tmp_path / "vector.ts").write_text(ts_src, encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_config(repo_path: Path) -> IndexConfig:
    """Build an IndexConfig using the local (no-API-key) embedder."""
    embedder_cfg = EmbedderConfig(provider="local")
    return IndexConfig(
        repo_path=str(repo_path),
        incremental=False,
        parse_workers=2,
        embedder=embedder_cfg,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_indexer_full_pipeline(mini_repo: Path) -> None:
    """
    Run the full 4-phase indexing pipeline and assert core statistics.
    """
    config = _make_config(mini_repo)
    stats = Indexer(config, quiet=True).index()

    # At least both files must be indexed
    assert stats["files_indexed"] >= 2, (
        f"Expected at least 2 files_indexed, got {stats['files_indexed']}"
    )

    # Symbols must have been extracted from at least one file
    assert stats["symbols_extracted"] > 0, (
        f"Expected symbols_extracted > 0, got {stats['symbols_extracted']}"
    )

    # Chunks must have been embedded
    assert stats["chunks_embedded"] > 0, (
        f"Expected chunks_embedded > 0, got {stats['chunks_embedded']}"
    )

    # No errors should have occurred
    assert stats["errors"] == 0, (
        f"Expected errors == 0, got {stats['errors']}"
    )

    # DB file must exist at the configured path
    db_path = config.db_path_absolute
    assert db_path.exists(), f"DB file not found at {db_path}"


def test_indexer_db_has_content(mini_repo: Path) -> None:
    """Verify the DB actually contains records after indexing."""
    from trelix.store.db import Database

    config = _make_config(mini_repo)
    Indexer(config, quiet=True).index()

    db = Database(config.db_path_absolute)
    try:
        files = db._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        symbols = db._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        chunks = db._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        db.close()

    assert files >= 2, f"Expected >= 2 file rows, got {files}"
    assert symbols > 0, f"Expected > 0 symbol rows, got {symbols}"
    assert chunks > 0, f"Expected > 0 chunk rows, got {chunks}"


def test_indexer_progress_callback(mini_repo: Path) -> None:
    """Verify the progress_callback is invoked with plausible data."""
    config = _make_config(mini_repo)
    events: list[dict] = []

    def on_progress(event: dict) -> None:
        events.append(event)

    Indexer(config, quiet=True, progress_callback=on_progress).index()

    assert len(events) > 0, "Expected at least one progress callback event"

    # All events must have the required keys
    for evt in events:
        assert "phase" in evt
        assert "phase_label" in evt
        assert "progress" in evt
        assert "stats" in evt
        assert 0.0 <= evt["progress"] <= 1.0, (
            f"progress out of range: {evt['progress']}"
        )

    # Final event must be at progress = 1.0
    last = events[-1]
    assert last["progress"] == 1.0, (
        f"Final progress event should be 1.0, got {last['progress']}"
    )


def test_indexer_incremental_skips_unchanged(mini_repo: Path) -> None:
    """Second run with incremental=True must skip all unchanged files."""
    embedder_cfg = EmbedderConfig(provider="local")
    config = IndexConfig(
        repo_path=str(mini_repo),
        incremental=True,
        parse_workers=2,
        embedder=embedder_cfg,
    )

    # First run: index everything
    stats1 = Indexer(config, quiet=True).index()
    assert stats1["files_indexed"] >= 2

    # Second run: nothing changed → nothing to index
    stats2 = Indexer(config, quiet=True).index()
    assert stats2["files_found"] >= 2
    # All files should be skipped (hash unchanged)
    assert stats2["files_indexed"] == 0, (
        f"Expected 0 files_indexed on second run, got {stats2['files_indexed']}"
    )


def test_index_file_single_update(mini_repo: Path) -> None:
    """index_file() must re-index a single modified file and return status ok."""
    config = _make_config(mini_repo)
    indexer = Indexer(config, quiet=True)

    # Full index first
    indexer.index()

    # Modify the Python file
    py_file = mini_repo / "calc.py"
    original = py_file.read_text()
    py_file.write_text(original + "\n\ndef subtract(a: int, b: int) -> int:\n    return a - b\n")

    result = indexer.index_file(str(py_file))

    assert result["status"] == "ok", f"index_file returned error: {result}"
    assert result["symbols_updated"] > 0, (
        f"Expected symbols_updated > 0 after file modification, got {result}"
    )
