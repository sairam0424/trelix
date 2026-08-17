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
    assert stats["errors"] == 0, f"Expected errors == 0, got {stats['errors']}"

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
        assert 0.0 <= evt["progress"] <= 1.0, f"progress out of range: {evt['progress']}"

    # Final event must be at progress = 1.0
    last = events[-1]
    assert last["progress"] == 1.0, f"Final progress event should be 1.0, got {last['progress']}"


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


def test_a_real_index_run_records_provenance(mini_repo: Path) -> None:
    """`index()` must persist what it was built from, not just the embeddings.

    Wiring test rather than a unit test: capture happens at the top of `index()` before
    it routes to the streaming pipeline, and the write happens at the tail of whichever
    pipeline ran. Both halves can be individually correct while the pair is never
    invoked, which is exactly the failure a unit test on the module would miss.
    """
    from trelix.store.db import Database
    from trelix.store.provenance import read_provenance

    config = _make_config(mini_repo)
    Indexer(config, quiet=True).index()

    with Database(config.db_path_absolute) as db:
        provenance = read_provenance(db)

    assert not provenance.is_empty, "index() completed without recording any provenance"
    assert provenance.trelix_version is not None
    assert provenance.indexed_at is not None
    assert provenance.embedder_provider == "local"
    # The walk config is what makes a later drift check trustworthy; an index that omits
    # it silently downgrades `trelix stats --drift` to unverifiable.
    assert provenance.walk_config is not None


def test_drift_is_clean_immediately_after_indexing(mini_repo: Path) -> None:
    """The strongest available check that drift and the indexer agree on 'unchanged'.

    They must share both the hash function and the ignore filters. If either diverged,
    every file would read as stale (or as new) the instant an index finished — a report
    that is permanently wrong in a way no aggregate would reveal.
    """
    from trelix.store.db import Database
    from trelix.store.provenance import compute_drift

    config = _make_config(mini_repo)
    Indexer(config, quiet=True).index()

    with Database(config.db_path_absolute) as db:
        report = compute_drift(config, db)

    assert report.is_clean, (
        f"fresh index already reports drift — stale={report.stale} new={report.new} "
        f"missing={report.missing}"
    )
    assert report.unchanged_count > 0
    assert report.missing_is_trustworthy, (
        "an index written by this run must record a comparable walk config"
    )


def test_editing_a_file_after_indexing_shows_up_as_drift(mini_repo: Path) -> None:
    """The end-to-end path the feature exists for: index, edit, detect."""
    from trelix.store.db import Database
    from trelix.store.provenance import compute_drift

    config = _make_config(mini_repo)
    Indexer(config, quiet=True).index()

    py_file = mini_repo / "calc.py"
    py_file.write_text(py_file.read_text() + "\n\ndef divide(a, b):\n    return a / b\n")

    with Database(config.db_path_absolute) as db:
        report = compute_drift(config, db)

    assert "calc.py" in report.stale, f"edited file not detected as stale: {report.stale}"
    assert report.missing == (), "editing a file must not make anything look deleted"


class TestPruneGuardAgainstARealIndexRun:
    """The prune guard must survive a REAL `Indexer.index()` immediately before it.

    Every one of the 131 unit tests covering `--prune` passed against a guard that could
    not fire, because their `_FakeIndexer.index()` never calls `_record_provenance()` and
    their fixture writes provenance *before* the run. That inverts the production ordering
    and makes three of the five guard conditions unfalsifiable.

    The real ordering: `index()` writes provenance at the END of the run, and the CLI prunes
    after that. So a drift check that reads provenance itself compares the current walk
    config against a copy the same run just wrote — it can only ever report "unchanged".

    Reproduced end to end before the fix: index three files, re-run with one file edited and
    `vendored/` newly ignored, and the plan came back with 2 candidates and ZERO refusals for
    files that were still on disk. On the trelix repository that shape is the 35 files under
    `packages/`.

    These use a real Indexer with the local embedder, which is why they live here rather
    than in tests/unit — the defect is in the wiring, and only a real run exposes it.
    """

    @staticmethod
    def _snapshot_then_index(config, repo: Path):  # type: ignore[no-untyped-def]
        """Mirror the CLI: read provenance BEFORE the run, index, then plan."""
        from trelix.store.db import Database
        from trelix.store.provenance import compute_drift, plan_prune, read_provenance

        with Database(config.db_path_absolute) as db:
            before = read_provenance(db)

        Indexer(config, quiet=True).index()

        with Database(config.db_path_absolute) as db:
            report = compute_drift(config, db, provenance=before)
            plan = plan_prune(report, before, indexed_count=len(db.get_all_file_rel_paths()))
        return before, plan

    def test_a_newly_ignored_directory_is_refused_not_pruned(self, tmp_path: Path) -> None:
        """The exact defect. One edited file is the whole precondition.

        With nothing changed, `index()` returns early at `if not to_parse` and never reaches
        the provenance write, so the stale record survives and the guard fires by accident.
        Editing one file removes that accident, which is why this test edits one.
        """
        import os

        repo = tmp_path / "repo"
        (repo / "keep").mkdir(parents=True)
        (repo / "vendored").mkdir()
        (repo / "keep" / "a.py").write_text("def a(): pass\n")
        (repo / "vendored" / "b.py").write_text("def b(): pass\n")
        (repo / "vendored" / "c.py").write_text("def c(): pass\n")

        Indexer(_make_config(repo), quiet=True).index()

        # Now ignore `vendored/`, and edit a file so the second run reaches the provenance
        # write. Both vendored files remain ON DISK.
        os.environ["TRELIX_WALKER_EXTRA_IGNORE_DIRS"] = '[".git", ".trelix", "vendored"]'
        try:
            (repo / "keep" / "a.py").write_text("def a():\n    return 1\n")
            before, plan = self._snapshot_then_index(_make_config(repo), repo)
        finally:
            os.environ.pop("TRELIX_WALKER_EXTRA_IGNORE_DIRS", None)

        assert before.walk_config is not None, "pass 1 recorded no walk config"
        assert (repo / "vendored" / "b.py").exists()
        assert set(plan.candidates) == {"vendored/b.py", "vendored/c.py"}, plan.candidates
        assert plan.is_refused, (
            "the guard did not fire: it would delete index rows for "
            f"{list(plan.candidates)}, which are still on disk"
        )
        assert any("walk settings changed" in r for r in plan.refusals), plan.refusals

    def test_a_genuinely_deleted_file_is_still_prunable(self, tmp_path: Path) -> None:
        """The guard must not refuse everything — that would be a different bug.

        Without this, the test above is satisfied by a guard hard-wired to refuse.
        """
        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "keep.py").write_text("def keep(): pass\n")
        (repo / "src" / "gone.py").write_text("def gone(): pass\n")

        config = _make_config(repo)
        Indexer(config, quiet=True).index()

        (repo / "src" / "gone.py").unlink()
        (repo / "src" / "keep.py").write_text("def keep():\n    return 1\n")

        _before, plan = self._snapshot_then_index(_make_config(repo), repo)

        assert plan.candidates == ("src/gone.py",), plan.candidates
        assert not plan.is_refused, f"a real deletion was refused: {plan.refusals}"
