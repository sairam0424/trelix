"""The next step a `--prune` refusal names must be one that clears it.

WHY. After a trelix upgrade the first `trelix index --prune` is refused, and before this
file existed the refusal said "Re-index with this version, then prune" or "A full
`trelix index <repo>` writes provenance". Measured on real runs over an unchanged tree
(the normal state right after an upgrade): a plain `trelix index` prints "Nothing to index
— all files up to date." and returns BEFORE it writes provenance, so following the advice
left the index exactly as uncomparable as it was and the refusal came back on every attempt
until some file happened to change. What clears it is
`TRELIX_INCREMENTAL=false trelix index <repo>`, which re-parses every file (reaching the
provenance write) and re-embeds nothing that is unchanged. For an index whose re-index would
be refused again (a walk config recorded with no history, as 3.1.2 did; an older
digest scheme; an `unrecorded` member; several members) that re-index is itself a dead end,
so EVERY refusal of that index names the rebuild instead. That is proven on real runs in
tests/integration/test_prune_refusal_steps_real_runs.py; THIS file pins the words, because
the words are what a user acts on.

The guards are not touched and are pinned, one per guard, in test_prune.py. Every refusal
below is still a refusal: this file only reads what the refusal SAYS.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trelix import __version__
from trelix.core.config import IndexConfig
from trelix.core.models import IndexedFile, Language
from trelix.store.db import Database
from trelix.store.provenance import (
    DriftReport,
    IndexProvenance,
    capture_provenance,
    is_walk_config_comparable,
    plan_prune,
    write_provenance,
)

runner = CliRunner()

# Literal on purpose: the command a user types is the contract, and it must not follow the
# constant that happens to build the sentence.
_REINDEX_COMMAND = "`TRELIX_INCREMENTAL=false trelix index <repo>`"
_REBUILD_COMMAND = "Delete `<repo>/.trelix/index.db` and run `trelix index <repo>`"
# The whole rebuild sentence a refusal ends with: the command, what it costs, and that the
# prune comes after. A refusal that names only the command leaves out two of the three.
_REBUILD_STEP = (
    "Delete `<repo>/.trelix/index.db` and run `trelix index <repo>` "
    "(this re-embeds everything), then prune again."
)
# Why a walk config with no history is not sent to the re-index that would fix "no record".
_WHY_NOT_A_REINDEX = (
    "A re-index would not clear this: it adds this run beside an `unrecorded` member "
    "standing for the rows an earlier run wrote, and that is refused again."
)
# The two qualifiers the re-index step carries. The first keeps a user from running it under
# a changed environment, which would add a second walk config to the history (a rebuild
# then); the second says the re-index certifies the walk as it is now.
_ENVIRONMENT_QUALIFIER = "with the environment and `--provider` that built the index"
_DISK_CAVEAT = "check the files listed against the disk before `--yes`"
_ONE_WALK_CONFIG = ("0" * 16,)
_TWO_WALK_CONFIGS = ("0" * 16, "1" * 16)
# What 3.1.2 recorded: a walk config under the old digest scheme, and no history at all.
_OLD_SCHEME_WALK_CONFIG = '{"gitignore_chain": "1 file(s) [.], sha256:0123456789abcdef"}'
# The digest member `_parse_history` falls back to for a history it cannot read.
_UNRECORDED_MEMBER = "unrecorded"


def _flat(output: str) -> str:
    """Rich wraps at 80 columns off a terminal, so a phrase can straddle a newline."""
    return " ".join(output.split())


def _report(**overrides: object) -> DriftReport:
    """A report whose `missing` every report-level guard has cleared."""
    defaults: dict[str, object] = {
        "missing": ("gone.py",),
        "unchanged_count": 9,
        "walk_was_complete": True,
        "walk_config_comparable": True,
        "walk_config_diff": (),
    }
    defaults.update(overrides)
    return DriftReport(**defaults)  # type: ignore[arg-type]


class TestTheVersionRefusalNamesTheStepThatWorks:
    def test_it_names_both_versions_and_the_command(self) -> None:
        stale = IndexProvenance(
            trelix_version="3.0.0", walk_config="{}", walk_config_history=_ONE_WALK_CONFIG
        )
        plan = plan_prune(_report(), stale, indexed_count=10)

        assert len(plan.refusals) == 1, plan.refusals  # the version guard, and only it
        refusal = plan.refusals[0]
        assert "built by trelix 3.0.0" in refusal
        assert f"and this is {__version__}" in refusal
        assert _REINDEX_COMMAND in refusal

    def test_it_does_not_send_people_to_a_plain_index(self) -> None:
        stale = IndexProvenance(
            trelix_version="3.0.0", walk_config="{}", walk_config_history=_ONE_WALK_CONFIG
        )
        refusal = plan_prune(_report(), stale, indexed_count=10).refusals[0]

        # The advice this replaced: it sent people to a command that changes nothing.
        assert "Re-index with this version" not in refusal

    def test_it_says_why_a_release_can_matter(self) -> None:
        """The reason the guard compares versions at all, in the user's words."""
        stale = IndexProvenance(
            trelix_version="3.0.0", walk_config="{}", walk_config_history=_ONE_WALK_CONFIG
        )
        refusal = plan_prune(_report(), stale, indexed_count=10).refusals[0]

        assert "a release can change which files are walked" in refusal
        assert "3.1.2" in refusal

    @pytest.mark.parametrize(
        "provenance",
        [
            IndexProvenance(
                trelix_version="3.0.0", walk_config="{}", walk_config_history=_ONE_WALK_CONFIG
            ),
            IndexProvenance(walk_config="{}", walk_config_history=_ONE_WALK_CONFIG),
        ],
        ids=["an-older-version", "an-unrecorded-version"],
    )
    def test_the_step_carries_the_qualifiers_the_docs_give_it(
        self, provenance: IndexProvenance
    ) -> None:
        """The upgrade case passes the version guard by re-stamping the version, not by
        proving anything about the rows, so the refusal says what the step does not do
        (measured: a row for an on-disk file the walk does not yield is listed, and `--yes`
        deletes it). It also says whose environment to run it under."""
        refusal = plan_prune(_report(), provenance, indexed_count=10).refusals[0]

        assert _ENVIRONMENT_QUALIFIER in refusal, refusal
        assert _DISK_CAVEAT in refusal, refusal

    def test_a_changed_walk_beside_the_version_refusal_names_the_original_environment(
        self,
    ) -> None:
        """Both refusals fire at once. Run under the CHANGED environment, the step adds a
        second walk config to the history (measured on a real CLI run: "2 different walk
        configurations" on the next prune, recoverable only by a rebuild), so the version
        refusal has to say the step belongs to the environment that built the index."""
        stale = IndexProvenance(
            trelix_version="3.0.0", walk_config="{}", walk_config_history=_ONE_WALK_CONFIG
        )
        changed = _report(walk_config_diff=(("extra_ignore_dirs", "['.git']", "['.git', 'v']"),))
        plan = plan_prune(changed, stale, indexed_count=10)

        assert len(plan.refusals) == 2, plan.refusals  # the changed walk, and the version
        walk = [r for r in plan.refusals if "the walk settings changed" in r]
        version = [r for r in plan.refusals if "built by trelix 3.0.0" in r]
        assert len(walk) == 1 and len(version) == 1, plan.refusals
        assert _ENVIRONMENT_QUALIFIER in version[0], version[0]

    def test_an_unrecorded_version_is_still_named_as_unrecorded(self) -> None:
        unrecorded = IndexProvenance(walk_config="{}", walk_config_history=_ONE_WALK_CONFIG)
        refusal = plan_prune(_report(), unrecorded, indexed_count=10).refusals[0]

        assert "built by trelix (unrecorded)" in refusal
        assert _REINDEX_COMMAND in refusal

    def test_an_index_from_before_the_history_is_told_to_rebuild(self) -> None:
        """A walk config but no history: a re-index would add `unrecorded` beside this run's
        digest and be refused again, so the version refusal must not name it either."""
        old = IndexProvenance(trelix_version="3.1.2", walk_config="{}")
        plan = plan_prune(_report(), old, indexed_count=10)

        version = [r for r in plan.refusals if "built by trelix 3.1.2" in r]
        assert len(version) == 1, plan.refusals
        assert _REBUILD_STEP in version[0]
        assert _REINDEX_COMMAND not in version[0]

    def test_rows_from_several_walk_configs_are_told_to_rebuild(self) -> None:
        """The history only grows, so a re-index cannot bring two members back to one."""
        several = IndexProvenance(
            trelix_version="3.0.0", walk_config="{}", walk_config_history=_TWO_WALK_CONFIGS
        )
        plan = plan_prune(_report(), several, indexed_count=10)

        assert len(plan.refusals) == 2, plan.refusals  # several configs, and the version
        version = [r for r in plan.refusals if "built by trelix 3.0.0" in r]
        assert len(version) == 1, plan.refusals
        assert _REBUILD_STEP in version[0]
        for refusal in plan.refusals:
            assert _REINDEX_COMMAND not in refusal, refusal


class TestTheWalkConfigRefusalIsWordedForItsCause:
    def test_no_record_says_so_and_names_the_command(self) -> None:
        provenance = IndexProvenance(trelix_version=__version__)
        plan = plan_prune(_report(walk_config_comparable=False), provenance, indexed_count=10)

        no_record = [r for r in plan.refusals if "this index records no walk config" in r]
        assert len(no_record) == 1, plan.refusals
        refusal = no_record[0]
        assert _REINDEX_COMMAND in refusal
        assert "A full `trelix index <repo>` writes provenance" not in refusal
        # With nothing to compare, the step certifies the walk as of now: say so.
        assert "check the files listed against the disk before `--yes`" in refusal

    def test_a_record_under_an_older_digest_scheme_is_not_called_unrecorded(self) -> None:
        """A record IS there; it cannot be compared. Saying "records no walk config" sent
        people looking for a record that exists. Its history names it, so the re-index
        would write a different digest beside it (measured): the step is the rebuild."""
        provenance = IndexProvenance(
            trelix_version=__version__,
            walk_config=_OLD_SCHEME_WALK_CONFIG,
            walk_config_history=_ONE_WALK_CONFIG,
        )
        # The handcrafted record is a true "older scheme" shape, per the real classifier.
        assert is_walk_config_comparable(provenance) is False

        plan = plan_prune(_report(walk_config_comparable=False), provenance, indexed_count=10)

        assert len(plan.refusals) == 1, plan.refusals
        refusal = plan.refusals[0]
        assert "records no walk config" not in refusal
        assert "cannot compare (written under an older `.gitignore` digest scheme" in refusal
        assert _REBUILD_STEP in refusal
        assert _REINDEX_COMMAND not in refusal
        # A rebuild writes every row under this walk, so the "as of now" caveat (for a
        # re-index over an index that recorded nothing) does not apply.
        assert "against the disk" not in refusal

    def test_the_3_1_2_shape_is_told_to_rebuild_by_every_refusal(self) -> None:
        """What 3.1.2 really recorded: the old-scheme walk config and NO history. A re-index
        would end in "2 different walk configurations" (proven on a real run in
        tests/integration/test_prune_refusal_steps_real_runs.py), so none of the three
        refusals may name it."""
        provenance = IndexProvenance(trelix_version="3.1.2", walk_config=_OLD_SCHEME_WALK_CONFIG)
        assert is_walk_config_comparable(provenance) is False

        plan = plan_prune(_report(walk_config_comparable=False), provenance, indexed_count=10)

        assert len(plan.refusals) == 3, plan.refusals  # cannot compare, no history, version
        for refusal in plan.refusals:
            assert _REBUILD_STEP in refusal, refusal
            assert _REINDEX_COMMAND not in refusal, refusal
            assert "against the disk" not in refusal, refusal
        # The cause is worded for THIS shape (the one 3.1.2 wrote): a walk config IS
        # recorded, it only cannot be compared, so nothing may say it records none.
        assert not any("records no walk config" in r for r in plan.refusals), plan.refusals
        cannot_compare = [
            r
            for r in plan.refusals
            if "cannot compare (written under an older `.gitignore` digest scheme" in r
        ]
        assert len(cannot_compare) == 1, plan.refusals

    @pytest.mark.parametrize(
        "provenance",
        [
            IndexProvenance(
                trelix_version="3.0.0", walk_config="{}", walk_config_history=(_UNRECORDED_MEMBER,)
            ),
            IndexProvenance(trelix_version="3.0.0", walk_config_history=_TWO_WALK_CONFIGS),
        ],
        ids=["an-unrecorded-member", "several-members-and-no-walk-config"],
    )
    def test_a_history_a_reindex_cannot_shrink_is_rebuilt_whatever_else_is_recorded(
        self, provenance: IndexProvenance
    ) -> None:
        """A re-index adds a member and never removes one. An `unrecorded` member (a history
        row that could not be read, measured: the re-index leaves two members) and several
        members stay refused, so the refusals must agree on the rebuild."""
        report = _report(walk_config_comparable=is_walk_config_comparable(provenance))
        plan = plan_prune(report, provenance, indexed_count=10)

        naming_a_step = [r for r in plan.refusals if _REBUILD_COMMAND in r or _REINDEX_COMMAND in r]
        assert naming_a_step, plan.refusals
        for refusal in naming_a_step:
            assert _REBUILD_STEP in refusal, refusal
            assert _REINDEX_COMMAND not in refusal, refusal


class TestTheHistoryRefusalNamesTheStepThatClearsIt:
    def test_with_nothing_recorded_the_reindex_also_starts_the_history(self) -> None:
        """An index with no provenance at all: the same forced re-index that fixes the walk
        config records a one-member history, so it is the step for this refusal too."""
        provenance = IndexProvenance(trelix_version=__version__)
        plan = plan_prune(_report(walk_config_comparable=False), provenance, indexed_count=10)

        history = [r for r in plan.refusals if "does not record which walk configurations" in r]
        assert len(history) == 1, plan.refusals
        assert _REINDEX_COMMAND in history[0]
        assert _REBUILD_COMMAND not in history[0]
        assert "would not clear this" not in history[0]

    def test_with_a_walk_config_but_no_history_only_a_rebuild_clears_it(self) -> None:
        """An index from before the history was recorded. A re-index would add this run's
        digest beside an `unrecorded` member — two members, refused again (measured on a real
        run) — so naming the re-index here would be a second dead end."""
        provenance = IndexProvenance(trelix_version=__version__, walk_config="{}")
        plan = plan_prune(_report(), provenance, indexed_count=10)

        assert len(plan.refusals) == 1, plan.refusals
        refusal = plan.refusals[0]
        assert "does not record which walk configurations" in refusal
        assert _REBUILD_STEP in refusal
        assert _WHY_NOT_A_REINDEX in refusal
        assert _REINDEX_COMMAND not in refusal


def _index_with_a_deleted_file(repo: Path, *, trelix_version: str | None) -> Database:
    """`a.py` on disk and indexed; `gone.py` indexed and not on disk.

    `trelix_version` None leaves the index without any provenance (one written by a trelix
    that recorded none); otherwise provenance is written as a full index would, then
    stamped as written by that version — the only thing an upgrade changes.
    """
    (repo / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    config = IndexConfig(repo_path=str(repo))
    db = Database(config.db_path_absolute)
    for name in ("a.py", "gone.py"):
        db.upsert_file(
            IndexedFile(
                path=str(repo / name),
                rel_path=name,
                language=Language.PYTHON,
                hash="0" * 64,
                size_bytes=1,
            )
        )
    if trelix_version is not None:
        write_provenance(db, replace(capture_provenance(config), trelix_version=trelix_version))
    return db


@pytest.fixture
def no_op_indexer(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """An Indexer whose run finds nothing to do and so writes NOTHING.

    That is what the real one does over an unchanged tree — the state right after an
    upgrade: `Indexer.index()` returns at `if not to_parse` before its provenance write
    (tests/integration/test_indexer.py proves it on a real run). A fake that recorded
    provenance here would hide the dead end this file exists for.
    """

    def _install(db: Database) -> None:
        class _NoOpIndexer:
            def __init__(self, config: IndexConfig, **_kwargs: object) -> None:
                self.config = config
                self.db = db
                self.vector_store = None

            def index(self) -> dict[str, object]:
                return {"files_found": 0, "files_indexed": 0}

        import trelix.indexing.indexer as indexer_mod

        monkeypatch.setattr(indexer_mod, "Indexer", _NoOpIndexer)

    return _install


class TestTheStepReachesTheUserThroughTheCli:
    def test_a_version_only_refusal_prints_the_command_and_deletes_nothing(
        self, tmp_path: Path, no_op_indexer
    ) -> None:  # type: ignore[no-untyped-def]
        """The upgrade case: every recorded setting matches, only the stamp differs."""
        with _index_with_a_deleted_file(tmp_path, trelix_version="3.0.0") as db:
            no_op_indexer(db)

            from trelix.cli.main import app

            result = runner.invoke(app, ["index", str(tmp_path), "--prune", "--yes"])

            out = _flat(result.output)
            assert result.exit_code == 1, result.output
            assert "Prune refused" in out
            assert "built by trelix 3.0.0" in out
            assert _REINDEX_COMMAND in out
            assert db.get_file_hash("gone.py") is not None, "a refused prune deleted a row"

    def test_an_index_with_no_provenance_prints_the_command_not_the_dead_end(
        self, tmp_path: Path, no_op_indexer
    ) -> None:  # type: ignore[no-untyped-def]
        with _index_with_a_deleted_file(tmp_path, trelix_version=None) as db:
            no_op_indexer(db)

            from trelix.cli.main import app

            result = runner.invoke(app, ["index", str(tmp_path), "--prune", "--yes"])

            out = _flat(result.output)
            assert result.exit_code == 1, result.output
            assert _REINDEX_COMMAND in out
            assert "check the files listed against the disk before `--yes`" in out
            assert "A full `trelix index <repo>` writes provenance" not in out
            assert "Re-index with this version" not in out
            assert db.get_file_hash("gone.py") is not None, "a refused prune deleted a row"


class TestTheDriftReportIsWordedForItsCause:
    def test_an_incomparable_record_is_not_only_called_unrecorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`walk_config_comparable` is False both for an index that recorded no walk config
        and for one recorded under an older `.gitignore` digest scheme. "Predates walk-config
        recording" alone sent the second kind of user looking for a record that exists (the
        `--prune` refusal above was corrected for the same reason)."""
        from rich.console import Console

        from trelix.cli import main as cli_main

        # Wide, because rich wraps at 80 columns off a terminal and a phrase can straddle it.
        recorder = Console(record=True, width=400, no_color=True, legacy_windows=False)
        monkeypatch.setattr(cli_main, "console", recorder)
        cli_main._print_drift(_report(walk_config_comparable=False))
        out = recorder.export_text()

        assert "walk config cannot be compared" in out
        assert "older `.gitignore` digest scheme" in out
