"""The step a `--prune` refusal names, proven on real index runs.

WHY. After a trelix upgrade the first `trelix index --prune` was refused with "Re-index with
this version, then prune". Right after an upgrade the tree is unchanged, and a plain
`trelix index` over an unchanged tree returns at `if not to_parse` BEFORE it writes
provenance, so that advice changed nothing and the refusal came back on every attempt. The
refusals now name one step, decided from the shape of the record:

  * `TRELIX_INCREMENTAL=false trelix index <repo>` re-parses every file, reaches the
    provenance write, and costs nothing over an unchanged tree;
  * a rebuild, for an index a re-index would leave refused again.

tests/unit/test_prune_refusal_steps.py pins the words. THIS file pins that the words are
true: the plan-level tests run the real Indexer and read the plan the CLI would print, and
`TestThePruneCommandOnARealIndex` runs `trelix index --prune --yes` itself (CliRunner, the
real Indexer with the local embedder) once per state, asserting the exit code, the command the
user is told to type and the rows that remain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.core.models import IndexedFile, Language
from trelix.indexing.indexer import Indexer
from trelix.store.db import Database
from trelix.store.provenance import (
    IndexProvenance,
    PrunePlan,
    compute_drift,
    plan_prune,
    read_provenance,
)

runner = CliRunner()

# Literal on purpose: the words a user acts on are the contract.
_FORCED_REINDEX = "TRELIX_INCREMENTAL=false"
_REINDEX_COMMAND = "`TRELIX_INCREMENTAL=false trelix index <repo>`"
_REBUILD = "Delete `<repo>/.trelix/index.db`"
_OLD_SCHEME_CHAIN = "1 file(s) [.], sha256:0123456789abcdef"
_FORCED_ENV = {"TRELIX_INCREMENTAL": "false"}


def _cli_config(repo_path: Path) -> IndexConfig:
    """What `trelix index <repo>` builds: only the environment decides `incremental`."""
    return IndexConfig(
        repo_path=str(repo_path),
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
    )


def _snapshot_then_index(repo: Path) -> tuple[IndexProvenance, PrunePlan]:
    """Mirror the CLI: read provenance BEFORE the run, index, then plan."""
    config = _cli_config(repo)
    with Database(config.db_path_absolute) as db:
        before = read_provenance(db)

    Indexer(config, quiet=True).index()

    with Database(config.db_path_absolute) as db:
        report = compute_drift(config, db, provenance=before)
        plan = plan_prune(report, before, indexed_count=len(db.get_all_file_rel_paths()))
    return before, plan


def _run_forced_reindex(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """`TRELIX_INCREMENTAL=false trelix index <repo>`: the environment is the spelling."""
    monkeypatch.setenv("TRELIX_INCREMENTAL", "false")
    try:
        config = _cli_config(repo)
        assert config.incremental is False
        return Indexer(config, quiet=True).index()
    finally:
        monkeypatch.delenv("TRELIX_INCREMENTAL")


class TestTheStepClearsWhatItIsNamedFor:
    def test_a_version_only_refusal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The upgrade case: an upgrade changes the version stamp and nothing else.

        A plain `trelix index` leaves the refusal in place and records nothing (which is why
        the words have to name something else); the forced re-index clears it, embeds
        nothing, and licenses only the file that is really gone.
        """
        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "keep.py").write_text("def keep(): pass\n")
        (repo / "src" / "gone.py").write_text("def gone(): pass\n")
        config = _cli_config(repo)
        Indexer(config, quiet=True).index()

        with Database(config.db_path_absolute) as db:
            db.set_index_metadata("provenance.trelix_version", "0.0.1")
        (repo / "src" / "gone.py").unlink()

        # 1. The old advice: a plain index. Refused before, refused after, nothing recorded.
        assert _cli_config(repo).incremental is True
        before, plan = _snapshot_then_index(repo)
        assert before.trelix_version == "0.0.1"
        assert plan.is_refused and any("0.0.1" in r for r in plan.refusals), plan.refusals
        with Database(config.db_path_absolute) as db:
            assert read_provenance(db).trelix_version == "0.0.1", (
                "a plain index over an unchanged tree recorded provenance: the dead end this "
                "file exists for is gone, and the refusal's words need another look"
            )

        # 2. The named step.
        stats = _run_forced_reindex(repo, monkeypatch)
        assert stats["chunks_embedded"] == 0, "the named step re-embedded an unchanged tree"
        with Database(config.db_path_absolute) as db:
            assert read_provenance(db).trelix_version != "0.0.1"

        # 3. Prune again: the deleted file is now licensed, and only it.
        _before, plan = _snapshot_then_index(repo)
        assert plan.candidates == ("src/gone.py",), plan.candidates
        assert not plan.is_refused, f"the named step did not clear the refusal: {plan.refusals}"

    def test_and_with_file_summaries_it_summarises_only_a_file_that_changed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With file summaries enabled the step still costs nothing on an unchanged tree.

        `_insert_one` builds a summary request only for a file with a changed symbol, so the
        docs' "embeds nothing that is unchanged" holds for the summary leg too, and a file
        that did change is summarised again (as a plain index would). The summarizer here
        counts calls; it never reaches a model.
        """
        calls: list[str] = []

        class _CountingSummarizer:
            def summarize(self, rel_path: str, symbols: object, language: object) -> str:
                calls.append(rel_path)
                return "canary summary"

        monkeypatch.setenv("TRELIX_FILE_SUMMARIES_ENABLED", "true")
        monkeypatch.setattr(
            Indexer, "_build_file_summarizer", lambda _self, _c: _CountingSummarizer()
        )
        repo = tmp_path / "repo"
        repo.mkdir()
        for name in ("a", "b", "c"):
            (repo / f"{name}.py").write_text(f"def {name}(): pass\n")
        Indexer(_cli_config(repo), quiet=True).index()
        assert sorted(calls) == ["a.py", "b.py", "c.py"], "the control run summarised nothing"

        calls.clear()
        stats = _run_forced_reindex(repo, monkeypatch)
        assert stats["files_indexed"] == 3, "the forced run did not re-parse every file"
        assert calls == [], f"the named step requested summaries for unchanged files: {calls}"

        (repo / "b.py").write_text("def b(): return 2\n")
        _run_forced_reindex(repo, monkeypatch)
        assert calls == ["b.py"], (
            f"the step summarised something other than the changed file: {calls}"
        )


def _delete_index(repo: Path) -> None:
    """The rebuild's first half: delete `<repo>/.trelix/index.db` (and its WAL files)."""
    db_path = _cli_config(repo).db_path_absolute
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def _strip_record(repo: Path) -> None:
    """An index written before any provenance was recorded: no `provenance.*` row at all."""
    with Database(_cli_config(repo).db_path_absolute) as db:
        for key in list(db.get_index_metadata_with_prefix("provenance.")):
            db.delete_index_metadata(f"provenance.{key}")


def _record_under_the_old_scheme(repo: Path, *, history_names_it: bool) -> None:
    """A walk config recorded under the old `.gitignore` digest scheme.

    3.1.2 wrote this with NO history (3.1.3 added the history and the digest scheme together,
    so no released version writes the other shape). `history_names_it` gives the record the
    one-member history a later scheme change would leave: the digest of the old record.
    """
    with Database(_cli_config(repo).db_path_absolute) as db:
        record = json.loads(read_provenance(db).walk_config or "{}")
        record["gitignore_chain"] = _OLD_SCHEME_CHAIN
        text = json.dumps(record, sort_keys=True)
        db.set_index_metadata("provenance.walk_config", text)
        if history_names_it:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            db.set_index_metadata("provenance.walk_config_digests", json.dumps([digest]))
        else:
            db.delete_index_metadata("provenance.walk_config_digests")
        db.set_index_metadata("provenance.trelix_version", "3.1.2")


class TestTheStepIsDecidedFromTheShapeOfTheRecord:
    @pytest.mark.parametrize(
        ("history_names_it", "refusal_count"),
        [(False, 3), (True, 2)],
        ids=["3.1.2-no-history", "old-scheme-with-its-own-history"],
    )
    def test_an_old_scheme_record_is_told_to_rebuild_by_every_refusal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        history_names_it: bool,
        refusal_count: int,
    ) -> None:
        """A record the re-index would end up beside: the re-index is a dead end, and no
        refusal names it.

        A re-index adds this run's digest beside the old record's (or beside an `unrecorded`
        member when there is no history), which refuses again ("2 different walk
        configurations"). That is reproduced here first, so the words are not merely
        claiming it, and then the rebuild the refusals name is run.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "keep.py").write_text("def keep(): pass\n")
        (repo / "gone.py").write_text("def gone(): pass\n")
        Indexer(_cli_config(repo), quiet=True).index()
        (repo / "gone.py").unlink()
        _record_under_the_old_scheme(repo, history_names_it=history_names_it)

        before, plan = _snapshot_then_index(repo)

        assert before.walk_config, before
        assert len(before.walk_config_history) == (1 if history_names_it else 0), before
        assert plan.candidates == ("gone.py",), plan.candidates
        assert len(plan.refusals) == refusal_count, plan.refusals
        for refusal in plan.refusals:
            assert _REBUILD in refusal, refusal
            assert _FORCED_REINDEX not in refusal, refusal
        # The cause is worded for this shape: a walk config IS recorded (3.1.2 wrote one),
        # it only cannot be compared, so no refusal may say the index records none.
        assert not any("records no walk config" in r for r in plan.refusals), plan.refusals
        cannot_compare = [
            r
            for r in plan.refusals
            if "cannot compare (written under an older `.gitignore` digest scheme" in r
        ]
        assert len(cannot_compare) == 1, plan.refusals

        # The dead end the refusals no longer send anyone to.
        _run_forced_reindex(repo, monkeypatch)
        _before, plan = _snapshot_then_index(repo)
        assert any("2 different walk configurations" in r for r in plan.refusals), plan.refusals

        # The step they do name.
        _delete_index(repo)
        Indexer(_cli_config(repo), quiet=True).index()
        _before, plan = _snapshot_then_index(repo)
        assert plan.refusals == (), f"the rebuild did not clear the refusals: {plan.refusals}"

    def test_an_index_with_no_record_is_certified_as_of_the_step_not_as_of_its_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KNOWN LIMIT, pinned so the refusal's words cannot drift from it.

        An index that recorded nothing has no walk config to compare, so after the step the
        prune compares against the walk as it is NOW. A file an older trelix indexed that
        this walk no longer yields (here: a nested `.gitignore` the older trelix did not
        read) is listed as a candidate although it is on disk, and `--yes` would delete its
        rows. That is why the refusal says to check the listed files against the disk; the
        gap itself is a decision for the owner, not something this test blesses. If it is
        ever closed, this test should flip.
        """
        repo = tmp_path / "repo"
        (repo / "sub").mkdir(parents=True)
        (repo / "sub" / "x.py").write_text("def x(): pass\n")
        (repo / "sub" / "y.py").write_text("def y(): pass\n")
        (repo / "keep.py").write_text("def keep(): pass\n")
        Indexer(_cli_config(repo), quiet=True).index()
        (repo / "sub" / ".gitignore").write_text("x.py\n")
        _strip_record(repo)

        before, plan = _snapshot_then_index(repo)

        assert before.walk_config is None and before.walk_config_history == (), before
        assert plan.candidates == ("sub/x.py",), plan.candidates
        assert len(plan.refusals) == 3, plan.refusals
        no_record = [r for r in plan.refusals if "records no walk config" in r]
        assert len(no_record) == 1, plan.refusals
        assert "check the files listed against the disk before `--yes`" in no_record[0]

        _run_forced_reindex(repo, monkeypatch)
        _before, plan = _snapshot_then_index(repo)

        assert (repo / "sub" / "x.py").is_file()
        assert plan.candidates == ("sub/x.py",), plan.candidates
        assert not plan.is_refused, plan.refusals

    def test_a_version_only_refusal_is_certified_as_of_the_step_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KNOWN LIMIT of the upgrade case, pinned so the refusal's words cannot drift from it.

        The version guard is passed by re-stamping the version, not by proving the rows were
        written under this walk. A row an older trelix wrote for a file this walk does not
        yield (`notes.xyz`, a language trelix does not index) is a candidate after the step
        although the file is on disk, and `--yes` would delete its row. The 10% cap and the
        10-candidate floor are the only backstops left, so the refusal says to check the
        listed files against the disk. The gap itself is a decision for the owner; if it is
        ever closed, this test should flip.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "keep.py").write_text("def keep(): pass\n")
        (repo / "notes.xyz").write_text("not a language trelix indexes\n")
        config = _cli_config(repo)
        Indexer(config, quiet=True).index()
        with Database(config.db_path_absolute) as db:
            db.upsert_file(
                IndexedFile(
                    path=str(repo / "notes.xyz"),
                    rel_path="notes.xyz",
                    language=Language.PYTHON,
                    hash="0" * 64,
                    size_bytes=1,
                )
            )
            db.set_index_metadata("provenance.trelix_version", "0.0.1")

        _before, plan = _snapshot_then_index(repo)

        assert plan.candidates == ("notes.xyz",), plan.candidates
        version = [r for r in plan.refusals if "built by trelix 0.0.1" in r]
        assert len(version) == 1, plan.refusals
        assert "check the files listed against the disk before `--yes`" in version[0]

        _run_forced_reindex(repo, monkeypatch)
        _before, plan = _snapshot_then_index(repo)

        assert (repo / "notes.xyz").is_file()
        assert plan.candidates == ("notes.xyz",), plan.candidates
        assert not plan.is_refused, plan.refusals


def _cli(repo: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    """`trelix index <repo> --provider local <args>`: the real command and the real Indexer.

    Returns the exit code and the output with its line wraps undone (rich wraps at 80 columns
    off a terminal, so a phrase can straddle a newline).
    """
    result = runner.invoke(
        app,
        ["index", str(repo), "--provider", "local", *args],
        env={"TRELIX_PARSE_WORKERS": "2", **(env or {})},
    )
    return result.exit_code, " ".join(result.output.split())


def _indexed(repo: Path) -> list[str]:
    """The `rel_path` of every file the index holds a row for."""
    with Database(_cli_config(repo).db_path_absolute) as db:
        return sorted(db.get_all_file_rel_paths())


def _stamp_version(repo: Path, version: str) -> None:
    """What an upgrade changes about an index: the version that wrote it, and nothing else."""
    with Database(_cli_config(repo).db_path_absolute) as db:
        db.set_index_metadata("provenance.trelix_version", version)


def _recorded_version(repo: Path) -> str | None:
    with Database(_cli_config(repo).db_path_absolute) as db:
        return read_provenance(db).trelix_version


def _indexed_repo_with_a_deleted_file(tmp_path: Path) -> Path:
    """`keep.py` and `gone.py` indexed by the real command, then `gone.py` deleted."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text("def keep(): pass\n")
    (repo / "gone.py").write_text("def gone(): pass\n")
    code, out = _cli(repo)
    assert code == 0, out
    assert _indexed(repo) == ["gone.py", "keep.py"]
    (repo / "gone.py").unlink()
    return repo


class TestThePruneCommandOnARealIndex:
    """`trelix index --prune --yes` through CliRunner and the real Indexer, one test per state.

    The plan-level tests above mirror the CLI by hand. These do not: the snapshot the guards
    read, the exit code, the command printed and the rows left behind all come from the command.
    """

    def test_the_upgrade_refusal_names_the_step_and_the_step_clears_it(
        self, tmp_path: Path
    ) -> None:
        repo = _indexed_repo_with_a_deleted_file(tmp_path)
        _stamp_version(repo, "0.0.1")

        # The old advice: a plain index. Refused, nothing deleted, and nothing recorded.
        code, out = _cli(repo, "--prune", "--yes")
        assert code == 1, out
        assert "built by trelix 0.0.1" in out
        assert _REINDEX_COMMAND in out
        assert _indexed(repo) == ["gone.py", "keep.py"], "a refused prune deleted a row"
        assert _recorded_version(repo) == "0.0.1", "a plain index recorded provenance"

        # The named step, then the prune as a second command.
        code, out = _cli(repo, env=_FORCED_ENV)
        assert code == 0, out
        assert _recorded_version(repo) != "0.0.1", "the named step did not re-stamp the index"
        code, out = _cli(repo, "--prune", "--yes")
        assert code == 0, out
        assert "Pruned 1 file(s)" in out
        assert _indexed(repo) == ["keep.py"]

    def test_the_step_and_the_prune_in_one_command_is_still_refused(self, tmp_path: Path) -> None:
        """The guards read the record as it stood BEFORE the run. A run that re-indexes cannot
        vouch for the prune beside it, which is why the step and the prune are two commands."""
        repo = _indexed_repo_with_a_deleted_file(tmp_path)
        _stamp_version(repo, "0.0.1")

        code, out = _cli(repo, "--prune", "--yes", env=_FORCED_ENV)

        assert code == 1, out
        assert "built by trelix 0.0.1" in out
        assert _indexed(repo) == ["gone.py", "keep.py"], "the run licensed its own prune"
        assert _recorded_version(repo) != "0.0.1", "the run never re-indexed: this proves nothing"

    def test_an_index_with_no_record_is_refused_and_the_step_clears_it(
        self, tmp_path: Path
    ) -> None:
        repo = _indexed_repo_with_a_deleted_file(tmp_path)
        _strip_record(repo)

        code, out = _cli(repo, "--prune", "--yes")
        assert code == 1, out
        assert "records no walk config" in out
        assert _REINDEX_COMMAND in out
        assert "check the files listed against the disk before `--yes`" in out
        assert _indexed(repo) == ["gone.py", "keep.py"], "a refused prune deleted a row"

        code, out = _cli(repo, env=_FORCED_ENV)
        assert code == 0, out
        code, out = _cli(repo, "--prune", "--yes")
        assert code == 0, out
        assert "Pruned 1 file(s)" in out
        assert _indexed(repo) == ["keep.py"]

    def test_the_3_1_2_shape_is_told_to_rebuild_and_the_rebuild_clears_it(
        self, tmp_path: Path
    ) -> None:
        repo = _indexed_repo_with_a_deleted_file(tmp_path)
        _record_under_the_old_scheme(repo, history_names_it=False)

        code, out = _cli(repo, "--prune", "--yes")
        assert code == 1, out
        assert _REBUILD in out
        assert _FORCED_REINDEX not in out
        assert _indexed(repo) == ["gone.py", "keep.py"], "a refused prune deleted a row"

        # The step named: delete the index, index again. Then a file comes and goes, and the
        # prune that was refused for this index removes exactly it.
        _delete_index(repo)
        code, out = _cli(repo)
        assert code == 0, out
        (repo / "later.py").write_text("def later(): pass\n")
        code, out = _cli(repo)
        assert code == 0, out
        (repo / "later.py").unlink()
        code, out = _cli(repo, "--prune", "--yes")
        assert code == 0, out
        assert "Pruned 1 file(s)" in out
        assert _indexed(repo) == ["keep.py"]

    def test_a_real_walk_change_stays_refused_after_the_step(self, tmp_path: Path) -> None:
        """`vendored/` is indexed, then ignored. The step rewrites the record under the narrowed
        walk but leaves two walk configs in the history, so the present file's row stays."""
        repo = tmp_path / "repo"
        (repo / "keep").mkdir(parents=True)
        (repo / "vendored").mkdir()
        (repo / "keep" / "a.py").write_text("def a(): pass\n")
        (repo / "vendored" / "b.py").write_text("def b(): pass\n")
        assert _cli(repo)[0] == 0
        narrowed = {"TRELIX_WALKER_EXTRA_IGNORE_DIRS": '[".git", ".trelix", "vendored"]'}

        code, out = _cli(repo, "--prune", "--yes", env=narrowed)
        assert code == 1, out
        assert "the walk settings changed" in out

        # The step, run under the changed walk: it cannot launder it.
        assert _cli(repo, env={**narrowed, **_FORCED_ENV})[0] == 0
        code, out = _cli(repo, "--prune", "--yes", env=narrowed)
        assert code == 1, out
        assert "2 different walk configurations" in out
        assert "vendored/b.py" in _indexed(repo), "a present file's row was deleted"
        assert (repo / "vendored" / "b.py").is_file()
