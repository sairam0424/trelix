"""`trelix index --prune` must refuse far more often than it acts.

WHY THIS FEATURE WAS WITHHELD. A prune keyed on "indexed paths the walk did not yield"
reads a truncated or differently-configured walk as a set of deletions and destroys
embeddings that cost real money to recompute. Both innocent explanations are measured
facts on this repository, not hypotheticals:

* a walk run without `scripts/self-index.sh`'s environment reported 35 of 467 indexed
  files as absent when every one was present — `packages` is in the default
  `extra_ignore_dirs` for .NET NuGet output (provenance.py:_WALK_FIELDS);
* one line in `workspace-vscode/.gitignore` governs 74% of this project's own index
  (walker.py:11-13), so an edit to the ignore chain alone can make most of the index
  look deleted.

So the guard is a conjunction of four independent facts, and this file pins each one
failing on its own — a guard that is right three times out of four still deletes the
index. It also pins the two things that make a wrong decision survivable: the default is
a dry run, and a prune above a share of the index refuses whatever the other four say.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from trelix import __version__
from trelix.core.config import IndexConfig
from trelix.core.models import Chunk, IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import Database
from trelix.store.provenance import (
    _PRUNE_MIN_CANDIDATES_FOR_CAP,
    DriftReport,
    IndexProvenance,
    PrunePlan,
    capture_provenance,
    plan_prune,
    write_provenance,
)

runner = CliRunner()


def _flat(output: str) -> str:
    """Output with whitespace collapsed. Rich wraps at 80 columns off a terminal, so a
    phrase under test can otherwise straddle a newline."""
    return " ".join(output.split())


def _trustworthy_report(**overrides: object) -> DriftReport:
    """A report whose `missing` every guard has cleared — the only prunable shape."""
    defaults: dict[str, object] = {
        "missing": ("gone.py",),
        "unchanged_count": 9,
        "walk_was_complete": True,
        "walk_config_comparable": True,
        "walk_config_diff": (),
    }
    defaults.update(overrides)
    return DriftReport(**defaults)  # type: ignore[arg-type]


def _current_provenance() -> IndexProvenance:
    return IndexProvenance(trelix_version=__version__, walk_config="{}")


class TestEachGuardRefusesOnItsOwn:
    def test_a_clear_report_is_prunable(self) -> None:
        plan = plan_prune(_trustworthy_report(), _current_provenance(), indexed_count=10)
        assert not plan.is_refused, plan.refusals
        assert plan.candidates == ("gone.py",)

    def test_an_incomplete_walk_refuses_and_names_the_path(self) -> None:
        report = _trustworthy_report(walk_was_complete=False, incomplete_paths=("packages",))
        plan = plan_prune(report, _current_provenance(), indexed_count=10)
        assert plan.is_refused
        assert any("packages" in r for r in plan.refusals)

    def test_an_index_without_a_recorded_walk_config_refuses(self) -> None:
        """The live index's exact state: `index_metadata` holds only embedding_dimension."""
        report = _trustworthy_report(walk_config_comparable=False)
        plan = plan_prune(report, IndexProvenance(trelix_version=__version__), indexed_count=10)
        assert plan.is_refused
        # The message must name the fix, or the user's next move is to force it.
        assert any("trelix index" in r for r in plan.refusals)

    def test_changed_walk_settings_refuse_and_name_the_setting(self) -> None:
        report = _trustworthy_report(
            walk_config_diff=(("extra_ignore_dirs", "['a']", "['a', 'packages']"),)
        )
        plan = plan_prune(report, _current_provenance(), indexed_count=10)
        assert plan.is_refused
        assert any("extra_ignore_dirs" in r for r in plan.refusals)

    def test_a_version_mismatch_refuses(self) -> None:
        """v3.1.2 changed which files are indexable without any config field moving."""
        stale = IndexProvenance(trelix_version="3.0.0", walk_config="{}")
        plan = plan_prune(_trustworthy_report(), stale, indexed_count=10)
        assert plan.is_refused
        assert any("3.0.0" in r and __version__ in r for r in plan.refusals)

    def test_an_unrecorded_version_refuses_rather_than_being_treated_as_current(self) -> None:
        plan = plan_prune(
            _trustworthy_report(), IndexProvenance(walk_config="{}"), indexed_count=10
        )
        assert plan.is_refused

    def test_no_refusal_implies_the_whole_conjunction_held(self) -> None:
        """Pins the guard as a conjunction: no single flag may license a prune."""
        report = _trustworthy_report()
        plan = plan_prune(report, _current_provenance(), indexed_count=10)
        assert not plan.is_refused
        assert report.missing_is_trustworthy
        assert report.walk_was_complete
        assert report.walk_config_comparable
        assert not report.walk_config_changed


class TestTheRefusalCap:
    def test_a_large_share_of_the_index_refuses_even_when_every_guard_passed(self) -> None:
        """The catastrophic shape: a whole subtree stops being reachable."""
        report = _trustworthy_report(missing=tuple(f"pkg/f{i}.py" for i in range(60)))
        plan = plan_prune(report, _current_provenance(), indexed_count=467)
        assert plan.is_refused
        assert any("%" in r for r in plan.refusals)

    def test_a_small_absolute_count_is_allowed_however_small_the_index(self) -> None:
        """10% of a 20-file repo is 2 files — an ordinary commit, not a catastrophe."""
        report = _trustworthy_report(missing=("a.py", "b.py", "c.py"))
        plan = plan_prune(report, _current_provenance(), indexed_count=5)
        assert not plan.is_refused, plan.refusals
        assert plan.fraction_of_index > 0.5

    def test_the_cap_is_raisable_but_only_explicitly(self) -> None:
        report = _trustworthy_report(missing=tuple(f"pkg/f{i}.py" for i in range(60)))
        assert plan_prune(report, _current_provenance(), indexed_count=467).is_refused
        raised = plan_prune(report, _current_provenance(), indexed_count=467, max_fraction=0.5)
        assert not raised.is_refused, raised.refusals

    def test_exactly_at_the_cap_is_allowed(self) -> None:
        report = _trustworthy_report(missing=tuple(f"f{i}.py" for i in range(10)))
        plan = plan_prune(report, _current_provenance(), indexed_count=100, max_fraction=0.10)
        assert not plan.is_refused, plan.refusals

    def test_the_cap_floor_is_a_named_constant_not_a_magic_number(self) -> None:
        assert _PRUNE_MIN_CANDIDATES_FOR_CAP >= 1

    def test_an_empty_index_does_not_divide_by_zero(self) -> None:
        plan = PrunePlan(candidates=("a.py",), indexed_count=0)
        assert plan.fraction_of_index == 0.0


# ---------------------------------------------------------------------------
# End-to-end through the CLI, against a real index on disk
# ---------------------------------------------------------------------------


class _RecordingVectorStore:
    """Stands in for the vector store so deleted chunk ids are observable."""

    def __init__(self) -> None:
        self.deleted: list[int] = []

    def delete_batch(self, chunk_ids: list[int]) -> None:
        self.deleted.extend(chunk_ids)


def _seed_index(repo: Path, *, present: list[str], deleted: list[str]) -> Database:
    """Write files to disk and rows to a real index; `deleted` gets rows but no file."""
    for name in present:
        (repo / name).write_text(f"def {name[:-3]}():\n    return 1\n", encoding="utf-8")

    config = IndexConfig(repo_path=str(repo))
    db = Database(config.db_path_absolute)
    for name in present + deleted:
        path = repo / name
        db.upsert_file(
            IndexedFile(
                path=str(path),
                rel_path=name,
                language=Language.PYTHON,
                # Deliberately not the on-disk hash: `stale` must not block a prune, only
                # `missing` is at stake here.
                hash="0" * 64,
                size_bytes=1,
            )
        )
    return db


def _write_current_provenance(repo: Path, db: Database) -> None:
    """Record provenance exactly as a full `trelix index` would."""
    write_provenance(db, capture_provenance(IndexConfig(repo_path=str(repo))))


@pytest.fixture
def fake_indexer(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Replace Indexer with one that opens the seeded DB and embeds nothing.

    The real Indexer loads an embedding model in __init__, which a unit test cannot pay
    for. Everything under test here — the drift walk, the guard, the delete — runs on the
    real Database and the real walker.
    """

    def _install(db: Database, store: _RecordingVectorStore) -> None:
        class _FakeIndexer:
            def __init__(self, config: IndexConfig, **_kwargs: object) -> None:
                self.config = config
                self.db = db
                self.vector_store = store

            def index(self) -> dict[str, object]:
                return {"files_found": 0, "files_indexed": 0}

        import trelix.indexing.indexer as indexer_mod

        monkeypatch.setattr(indexer_mod, "Indexer", _FakeIndexer)

    return _install


class TestThePruneCliRefusesOnTheCurrentIndexShape:
    def test_no_provenance_refuses_and_says_a_full_index_records_it(
        self, tmp_path: Path, fake_indexer
    ) -> None:  # type: ignore[no-untyped-def]
        """The shipping index at 01d148d: index_metadata holds embedding_dimension only."""
        db = _seed_index(tmp_path, present=["a.py", "b.py"], deleted=["gone.py"])
        # No write_provenance() call — this is an index that predates provenance.
        assert "provenance.walk_config" not in dict(
            db._conn.execute("SELECT key, value FROM index_metadata")
        )
        fake_indexer(db, _RecordingVectorStore())

        from trelix.cli.main import app

        result = runner.invoke(app, ["index", str(tmp_path), "--prune", "--yes"])

        assert result.exit_code != 0, result.output
        assert "trelix index" in _flat(result.output)
        assert db.get_file_hash("gone.py") is not None, "refused prune still deleted the row"


class TestThePruneCliActsOnlyWithConsent:
    def test_the_default_is_a_dry_run(self, tmp_path: Path, fake_indexer) -> None:  # type: ignore[no-untyped-def]
        db = _seed_index(tmp_path, present=["a.py", "b.py"], deleted=["gone.py"])
        _write_current_provenance(tmp_path, db)
        store = _RecordingVectorStore()
        fake_indexer(db, store)

        from trelix.cli.main import app

        result = runner.invoke(app, ["index", str(tmp_path), "--prune"])

        assert result.exit_code == 0, result.output
        assert "gone.py" in _flat(result.output)
        assert db.get_file_hash("gone.py") is not None, "a dry run deleted a row"
        assert store.deleted == []

    def test_yes_deletes_the_row_and_its_vectors(self, tmp_path: Path, fake_indexer) -> None:  # type: ignore[no-untyped-def]
        db = _seed_index(tmp_path, present=["a.py", "b.py"], deleted=["gone.py"])
        file_id = db._conn.execute("SELECT id FROM files WHERE rel_path = 'gone.py'").fetchone()[0]
        symbol_id = db.insert_symbol(
            Symbol(
                file_id=file_id,
                name="gone",
                qualified_name="gone",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=2,
                signature="def gone()",
                body="def gone():\n    return 1\n",
            )
        )
        chunk_id = db.insert_chunk(Chunk(symbol_id=symbol_id, chunk_text="x", token_count=1))
        _write_current_provenance(tmp_path, db)
        store = _RecordingVectorStore()
        fake_indexer(db, store)

        from trelix.cli.main import app

        result = runner.invoke(app, ["index", str(tmp_path), "--prune", "--yes"])

        assert result.exit_code == 0, result.output
        assert db.get_file_hash("gone.py") is None, "prune did not remove the file row"
        assert db.get_file_hash("a.py") is not None, "prune removed a present file"
        # The file-summary sentinel `-(file_id)` is deleted in the same call. Summary
        # vectors live in `chunk_embeddings` under a negative chunk_id, so they are not in
        # get_chunk_ids_for_file() and a chunk-only delete left them behind — which
        # verify-index.sh's "summary vectors == summaries" gate caught the first time a real
        # --prune removed a file (482 vectors against 481 summaries).
        assert store.deleted == [chunk_id, -file_id], "paid-for vectors were orphaned, not deleted"

    def test_yes_without_prune_is_an_error_rather_than_a_no_op(self, tmp_path: Path) -> None:
        """A user who typed --yes consented to something; silently ignoring it is worse."""
        (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

        from trelix.cli.main import app

        result = runner.invoke(app, ["index", str(tmp_path), "--yes"])

        assert result.exit_code != 0
        assert "--prune" in _flat(result.output)


class TestTheGuardSeesRealWalkConfigDrift:
    def test_a_changed_ignore_rule_refuses_against_a_real_walk(
        self, tmp_path: Path, fake_indexer, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # type: ignore[no-untyped-def]
        """The 35-phantom-deletion shape, reproduced end to end.

        Provenance is recorded with the default ignore list, then the walk is re-run with
        `packages` no longer ignored — the recorded and current configs differ, so every
        indexed path the new walk misses must stay put.
        """
        db = _seed_index(tmp_path, present=["a.py"], deleted=["gone.py"])
        _write_current_provenance(tmp_path, db)
        recorded = json.loads(
            db.get_index_metadata("provenance.walk_config") or "{}"  # type: ignore[arg-type]
        )
        assert "extra_ignore_dirs" in recorded, "walk config shape moved — update this test"

        monkeypatch.setenv("TRELIX_WALKER_EXTRA_IGNORE_DIRS", '["a-directory-that-moved"]')
        fake_indexer(db, _RecordingVectorStore())

        from trelix.cli.main import app

        result = runner.invoke(app, ["index", str(tmp_path), "--prune", "--yes"])

        assert result.exit_code != 0, result.output
        assert "extra_ignore_dirs" in _flat(result.output)
        assert db.get_file_hash("gone.py") is not None


class TestRenderingIsMarkupSafe:
    def test_a_bracketed_path_survives_the_plan_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A directory named "old[1]" must not be eaten by Rich, or read as an error."""
        from trelix.cli import main as cli_main

        recorder = Console(record=True, width=400, no_color=True, legacy_windows=False)
        monkeypatch.setattr(cli_main, "console", recorder)
        plan = PrunePlan(candidates=("old[/1]/gone.py",), indexed_count=10)
        cli_main._print_prune_plan(plan, will_act=False)
        assert "old[/1]/gone.py" in recorder.export_text()
