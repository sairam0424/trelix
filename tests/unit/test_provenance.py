"""Tests for index provenance and worktree drift.

The load-bearing behaviour here is not "does it count files" — it is "does it refuse to
present a count that has an innocent explanation". `missing` is derived from indexed
paths the walk did not yield, and acting on it deletes embeddings that cost money to
recompute, so every path that can inflate it must be detectable.

Measured on this repository while building the feature: a drift check run without
`scripts/self-index.sh`'s environment reported 35 files under `packages/` as deleted when
every one was present, because `extra_ignore_dirs` REPLACES the default 30-entry list
(which contains "packages" for .NET NuGet output) rather than extending it.
"""

from __future__ import annotations

import json

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.core.models import IndexedFile, Language
from trelix.store.db import Database
from trelix.store.provenance import (
    DriftReport,
    IndexProvenance,
    _walk_config_json,
    capture_provenance,
    commits_since,
    compute_drift,
    read_provenance,
    walk_config_differences,
    write_provenance,
)


def _config(repo, **walker_overrides):  # type: ignore[no-untyped-def]
    """Build a config, overriding walker settings at construction rather than after.

    Constructed rather than mutated so the overrides go through pydantic validation — a
    post-hoc `object.__setattr__` would let a test pass a shape the real config could
    never hold, and the walk-config comparison exists precisely to catch shape drift.
    """
    if not walker_overrides:
        return IndexConfig(repo_path=str(repo))
    return IndexConfig(repo_path=str(repo), walker=WalkerConfig(**walker_overrides))


class TestIndexMetadataHelpers:
    def test_round_trips_a_value(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            db.set_index_metadata("a.b", "1")
            assert db.get_index_metadata("a.b") == "1"

    def test_an_absent_key_is_none_not_an_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            assert db.get_index_metadata("nope") is None

    def test_deleting_an_absent_key_is_not_an_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            db.delete_index_metadata("nope")  # must not raise

    def test_prefix_query_strips_the_prefix(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            db.set_index_metadata("p.one", "1")
            db.set_index_metadata("p.two", "2")
            assert db.get_index_metadata_with_prefix("p.") == {"one": "1", "two": "2"}

    def test_prefix_query_excludes_other_keys(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            db.set_index_metadata("p.one", "1")
            db.set_index_metadata("embedding_dimension", "3072")
            assert db.get_index_metadata_with_prefix("p.") == {"one": "1"}

    def test_an_underscore_in_the_prefix_is_not_a_wildcard(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """`_` is a single-char wildcard in SQL LIKE — unescaped, `a_` matches `ab`."""
        with Database(tmp_path / "index.db") as db:
            db.set_index_metadata("a_x", "right")
            db.set_index_metadata("abx", "wrong")
            assert db.get_index_metadata_with_prefix("a_") == {"x": "right"}

    def test_a_percent_in_the_prefix_is_not_a_wildcard(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            db.set_index_metadata("a%x", "right")
            db.set_index_metadata("azzzx", "wrong")
            assert db.get_index_metadata_with_prefix("a%") == {"x": "right"}

    def test_the_dimension_helpers_still_work_through_the_generic_pair(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """They were refactored to delegate; the dimension guard calls them by name."""
        with Database(tmp_path / "index.db") as db:
            assert db.get_embedding_dimension() is None
            db.set_embedding_dimension(3072)
            assert db.get_embedding_dimension() == 3072
            db.delete_embedding_dimension_key()
            assert db.get_embedding_dimension() is None


class TestProvenanceRoundTrip:
    def test_an_index_without_provenance_reads_as_empty(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            assert read_provenance(db).is_empty

    def test_a_written_record_reads_back_identically(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = IndexProvenance(
            git_commit="abc123",
            git_branch="main",
            git_dirty=True,
            indexed_at="2026-08-17T00:00:00+00:00",
            trelix_version="3.1.2",
            embedder_provider="azure",
            embedder_model="text-embedding-3-large",
            walk_config='{"languages": ["python"]}',
        )
        with Database(tmp_path / "index.db") as db:
            write_provenance(db, original)
            assert read_provenance(db) == original

    def test_dirty_false_survives_the_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A bool stored as a string is the classic place False becomes True."""
        with Database(tmp_path / "index.db") as db:
            write_provenance(db, IndexProvenance(git_dirty=False))
            assert read_provenance(db).git_dirty is False

    def test_dirty_unknown_stays_unknown(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """None is 'could not tell', which must not collapse into False."""
        with Database(tmp_path / "index.db") as db:
            write_provenance(db, IndexProvenance(git_dirty=None, git_commit="x"))
            assert read_provenance(db).git_dirty is None

    def test_a_none_field_clears_a_previous_value(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Re-indexing from a non-git copy must not leave the old commit readable."""
        with Database(tmp_path / "index.db") as db:
            write_provenance(db, IndexProvenance(git_commit="old", git_branch="main"))
            write_provenance(db, IndexProvenance(git_commit=None, git_branch="main"))
            assert read_provenance(db).git_commit is None

    def test_provenance_does_not_disturb_the_dimension_key(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with Database(tmp_path / "index.db") as db:
            db.set_embedding_dimension(3072)
            write_provenance(db, IndexProvenance(git_commit="abc"))
            assert db.get_embedding_dimension() == 3072


class TestWalkConfigFingerprint:
    def test_it_records_the_fields_that_decide_which_files_are_walked(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        payload = json.loads(_walk_config_json(_config(tmp_path)) or "{}")
        for field in (
            "languages",
            "extra_ignore_dirs",
            "extra_ignore_extensions",
            "extra_ignore_filenames",
            "max_file_size_bytes",
            "respect_gitignore",
        ):
            assert field in payload, f"{field} changes which files are indexed"

    def test_ordering_does_not_count_as_a_difference(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The walker turns each list into a set, so order cannot change the walk."""
        a = _config(tmp_path, extra_ignore_dirs=["x", "y"])
        b = _config(tmp_path, extra_ignore_dirs=["y", "x"])
        assert _walk_config_json(a) == _walk_config_json(b)

    def test_a_changed_ignore_list_is_reported_and_named(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """This is the 35-phantom-deletion case that motivated the field."""
        indexed_with = _config(tmp_path, extra_ignore_dirs=["node_modules"])
        checked_with = _config(tmp_path, extra_ignore_dirs=["node_modules", "packages"])

        recorded = IndexProvenance(walk_config=_walk_config_json(indexed_with))
        diff = walk_config_differences(recorded, checked_with)

        assert "extra_ignore_dirs" in diff, "a changed ignore list must be detected"
        before, after = diff["extra_ignore_dirs"]
        assert "packages" not in before
        assert "packages" in after

    def test_an_identical_config_reports_no_difference(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        cfg = _config(tmp_path)
        recorded = IndexProvenance(walk_config=_walk_config_json(cfg))
        assert walk_config_differences(recorded, cfg) == {}

    def test_an_index_predating_the_field_is_not_reported_as_matching(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Empty diff plus walk_config=None means 'unknown', not 'same'."""
        recorded = IndexProvenance(walk_config=None)
        assert walk_config_differences(recorded, _config(tmp_path)) == {}
        assert recorded.walk_config is None


class TestCommitsSince:
    def test_a_missing_commit_is_unknown_not_zero(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """0 would read as 'index is current' — the opposite of what is known."""
        assert commits_since(_config(tmp_path), None) is None

    def test_a_non_git_directory_is_unknown_not_zero(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert commits_since(_config(tmp_path), "deadbeef") is None


class TestCaptureNeverRaises:
    def test_a_non_git_directory_still_yields_a_record(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An index of a plain directory must still record version and embedder."""
        provenance = capture_provenance(_config(tmp_path))
        assert provenance.git_commit is None
        assert provenance.trelix_version is not None
        assert provenance.indexed_at is not None
        assert not provenance.is_empty


class TestDriftClassification:
    def _write(self, path, text):  # type: ignore[no-untyped-def]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_an_unchanged_file_is_counted_unchanged(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from trelix.indexing.walker import FileWalker

        self._write(tmp_path / "a.py", "def f(): pass\n")
        cfg = _config(tmp_path)
        walked = next(iter(FileWalker(cfg).walk()))

        with Database(tmp_path / ".trelix" / "index.db") as db:
            db.upsert_file(walked)
            db._conn.commit()
            report = compute_drift(cfg, db)

        assert report.unchanged_count == 1
        assert report.is_clean

    def test_an_edited_file_is_stale(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from trelix.indexing.walker import FileWalker

        target = self._write(tmp_path / "a.py", "def f(): pass\n")
        cfg = _config(tmp_path)
        walked = next(iter(FileWalker(cfg).walk()))

        with Database(tmp_path / ".trelix" / "index.db") as db:
            db.upsert_file(walked)
            db._conn.commit()
            target.write_text("def f(): return 1\n")
            report = compute_drift(cfg, db)

        assert report.stale == ("a.py",)
        assert not report.is_clean

    def test_an_unindexed_file_is_new(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        self._write(tmp_path / "a.py", "def f(): pass\n")
        cfg = _config(tmp_path)

        with Database(tmp_path / ".trelix" / "index.db") as db:
            report = compute_drift(cfg, db)

        assert report.new == ("a.py",)

    def test_a_deleted_file_is_missing(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        cfg = _config(tmp_path)
        with Database(tmp_path / ".trelix" / "index.db") as db:
            db.upsert_file(
                IndexedFile(
                    path=str(tmp_path / "gone.py"),
                    rel_path="gone.py",
                    language=Language.PYTHON,
                    hash="h",
                    size_bytes=1,
                )
            )
            db._conn.commit()
            report = compute_drift(cfg, db)

        assert report.missing == ("gone.py",)

    def test_drift_uses_the_walkers_own_hash(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A different hash function would report every file stale.

        Pinned because the value of this feature collapses entirely if the two disagree:
        a permanently-100%-stale report is indistinguishable from a broken index.
        """
        from trelix.indexing.walker import FileWalker

        self._write(tmp_path / "a.py", "x = 1\n")
        cfg = _config(tmp_path)
        walked = next(iter(FileWalker(cfg).walk()))

        with Database(tmp_path / ".trelix" / "index.db") as db:
            db.upsert_file(walked)
            db._conn.commit()
            assert compute_drift(cfg, db).stale == ()


class TestMissingIsNotPresentedAsTrustworthy:
    def test_a_clean_comparable_walk_is_trustworthy(self) -> None:
        report = DriftReport(walk_was_complete=True, walk_config_comparable=True)
        assert report.missing_is_trustworthy

    def test_an_incomplete_walk_is_not(self) -> None:
        """Files under an unreadable directory land in `missing` while being present."""
        report = DriftReport(missing=("a.py",), walk_was_complete=False, incomplete_paths=("sub",))
        assert not report.missing_is_trustworthy

    def test_a_changed_walk_config_is_not(self) -> None:
        report = DriftReport(
            missing=("a.py",),
            walk_config_diff=(("extra_ignore_dirs", "['x']", "['x', 'packages']"),),
        )
        assert report.walk_config_changed
        assert not report.missing_is_trustworthy

    def test_an_unrecordable_walk_config_is_not(self) -> None:
        """An index predating the field cannot be verified, so it is not trusted."""
        report = DriftReport(missing=("a.py",), walk_config_comparable=False)
        assert not report.missing_is_trustworthy

    def test_drifted_count_sums_all_three_states(self) -> None:
        report = DriftReport(stale=("a",), new=("b", "c"), missing=("d",))
        assert report.drifted_count == 4


class TestWalkFieldsIsComplete:
    """Every `WalkerConfig` setting that changes WHICH files are walked must be recorded.

    An unrecorded field is a silent false-TRUE in `missing_is_trustworthy`: the drift
    check compares against a different walk than the one that built the index, so present
    files report as `missing` while the report says the count can be trusted. A future
    `--prune` reading that deletes embeddings that cost money to recompute.

    `follow_symlinks` was exactly that — 7 fields on the config, 6 recorded.
    """

    def test_walk_fields_covers_every_walker_setting(self) -> None:
        from trelix.core.config import WalkerConfig
        from trelix.store.provenance import _WALK_FIELDS

        # Nothing in WalkerConfig is walk-irrelevant today. If that changes, add the
        # exemption here WITH a reason rather than quietly shrinking the fingerprint.
        walk_irrelevant: set[str] = set()

        missing = set(WalkerConfig.model_fields) - set(_WALK_FIELDS) - walk_irrelevant
        assert not missing, (
            f"{sorted(missing)} affect which files are walked but are not recorded in "
            "provenance, so a drift check cannot tell that the walk differed. Either add "
            "them to _WALK_FIELDS or list them in walk_irrelevant with a justification."
        )

    def test_recorded_fields_all_exist_on_the_config(self) -> None:
        """A renamed field would silently stop being fingerprinted."""
        from trelix.core.config import WalkerConfig
        from trelix.store.provenance import _WALK_FIELDS

        unknown = set(_WALK_FIELDS) - set(WalkerConfig.model_fields)
        assert not unknown, f"{sorted(unknown)} are recorded but no longer exist"

    def test_a_follow_symlinks_difference_is_now_detected(self) -> None:
        """The concrete false-TRUE this closed."""
        from trelix.store.provenance import (
            IndexProvenance,
            _walk_config_json,
            walk_config_differences,
        )

        indexed_with = _config(".", follow_symlinks=True)
        checked_with = _config(".", follow_symlinks=False)

        recorded = IndexProvenance(walk_config=_walk_config_json(indexed_with))
        diff = walk_config_differences(recorded, checked_with)

        assert "follow_symlinks" in diff, (
            "a follow_symlinks change no longer registers, so missing_is_trustworthy "
            "would wrongly return True"
        )
