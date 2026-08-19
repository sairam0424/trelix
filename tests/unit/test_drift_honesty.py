"""`trelix stats --drift` must not print a number it has already disclaimed.

Two separate false-confidence bugs are pinned here, both in the same blast radius: a
future `--prune` that reads `missing` deletes embeddings that cost money to recompute.

1. **The headline total and the remedy ignored `missing_is_trustworthy`.** Measured on
   this repository at bc91d7a::

       Changed 49 | Not indexed 15 | Indexed but not found 35 | Unchanged 383
       99 file(s) have drifted. Run `trelix index` to rebuild, ...

   The "not found" 35 are every file under `packages/` and all 35 exist on disk —
   `packages` is in the default `extra_ignore_dirs` (core/config.py:98) for .NET NuGet
   output, so the walk cannot reach this repo's own monorepo packages. The honesty
   warning DID print immediately above, and then the total folded the 35 in anyway and
   prescribed a `trelix index` that would restore none of them.

2. **`.gitignore` CONTENTS were not fingerprinted.** `_WALK_FIELDS` recorded
   `respect_gitignore` as a bool only, so editing ignore rules alone left
   `missing_is_trustworthy` at True while files legitimately dropped out of the walk.
   Not hypothetical at this scale: nested-`.gitignore` support alone moved this index by
   74% (walker.py:11-13).
"""

from __future__ import annotations

import pytest
from rich.console import Console

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.core.models import IndexedFile, Language
from trelix.store.db import Database
from trelix.store.provenance import (
    DriftReport,
    IndexProvenance,
    _walk_config_json,
    compute_drift,
    walk_config_differences,
)


def _config(repo, **walker_overrides):  # type: ignore[no-untyped-def]
    """Build a config, pushing walker overrides through pydantic validation.

    Mirrors tests/unit/test_provenance.py deliberately: a post-hoc `__setattr__` would
    let a test assert against a config shape the real loader could never produce, and
    the whole point of the walk-config fingerprint is to catch shape drift.
    """
    if not walker_overrides:
        return IndexConfig(repo_path=str(repo))
    return IndexConfig(repo_path=str(repo), walker=WalkerConfig(**walker_overrides))


def _rendered(report: DriftReport, monkeypatch: pytest.MonkeyPatch) -> str:
    """`_print_drift(report)` as plain text.

    Width is pinned wide because rich falls back to 80 columns off a terminal, and an
    80-column drift table ellipsizes the sentence under it — a width-driven failure
    would look exactly like the defect under test.
    """
    from trelix.cli import main as cli_main

    recorder = Console(record=True, width=400, no_color=True, legacy_windows=False)
    monkeypatch.setattr(cli_main, "console", recorder)
    cli_main._print_drift(report)
    return recorder.export_text()


def _untrustworthy(**overrides) -> DriftReport:  # type: ignore[no-untyped-def]
    """The bc91d7a shape, shrunk: 2 real drifts and 3 unverifiable "not found"."""
    defaults: dict[str, object] = {
        "stale": ("a.py",),
        "new": ("b.py",),
        "missing": ("packages/x.py", "packages/y.py", "packages/z.py"),
        "unchanged_count": 383,
        "walk_config_comparable": False,
    }
    defaults.update(overrides)
    return DriftReport(**defaults)  # type: ignore[arg-type]


class TestTheHeadlineTotalIsGatedOnTrustworthiness:
    def test_unverifiable_missing_is_not_folded_into_the_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 99-vs-64 bug: `drifted_count` sums all three states unconditionally."""
        report = _untrustworthy()
        assert report.drifted_count == 5
        assert not report.missing_is_trustworthy

        out = _rendered(report, monkeypatch)

        assert "5 file(s) have drifted" not in out, (
            "the total still folds in 3 `missing` files the report has just disclaimed"
        )
        assert "2 file(s) have drifted" in out

    def test_an_incomplete_walk_also_gates_the_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unreadable directories inflate `missing` exactly like a changed ignore rule."""
        report = _untrustworthy(
            walk_config_comparable=True,
            walk_was_complete=False,
            incomplete_paths=("packages",),
        )
        out = _rendered(report, monkeypatch)

        assert "2 file(s) have drifted" in out
        assert "5 file(s) have drifted" not in out

    def test_a_changed_walk_config_also_gates_the_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _untrustworthy(
            walk_config_comparable=True,
            walk_config_diff=(("extra_ignore_dirs", "['x']", "['x', 'packages']"),),
        )
        out = _rendered(report, monkeypatch)

        assert "2 file(s) have drifted" in out
        assert "5 file(s) have drifted" not in out

    def test_a_trustworthy_report_still_counts_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gating must not cost the honest case its number. A genuinely deleted file IS
        drift, and `trelix index` genuinely does fix it."""
        report = _untrustworthy(walk_config_comparable=True)
        assert report.missing_is_trustworthy

        out = _rendered(report, monkeypatch)

        assert "5 file(s) have drifted" in out
        assert "Run `trelix index` to rebuild" in out


class TestTheRemedyDoesNotPromiseTheImpossible:
    def test_it_does_not_claim_trelix_index_restores_unreachable_files(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare `trelix index` reuses the same walk that could not reach them, so it
        has nothing to re-index — the advice was not merely optimistic, it was void."""
        out = _rendered(_untrustworthy(), monkeypatch)

        assert "NOT restore" in out, (
            "the remedy still implies `trelix index` fixes the unreachable files"
        )
        assert "3" in out, "the excluded count should still be stated, not hidden"

    def test_the_unverified_state_is_labelled_in_the_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row and the total must agree; a bare 'Indexed but not found 3' next to a
        total of 2 reads as an arithmetic bug rather than a deliberate exclusion."""
        out = _rendered(_untrustworthy(), monkeypatch)
        assert "Indexed but not found (unverified)" in out

    def test_a_trustworthy_row_is_not_labelled_unverified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _rendered(_untrustworthy(walk_config_comparable=True), monkeypatch)
        assert "Indexed but not found" in out
        assert "unverified" not in out


class TestGitignoreContentsAreFingerprinted:
    """`respect_gitignore=True` on both sides says nothing about WHAT was ignored."""

    def _repo(self, tmp_path, gitignore: str):  # type: ignore[no-untyped-def]
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "sub").mkdir(exist_ok=True)
        (tmp_path / "sub" / "b.py").write_text("y = 2\n")
        (tmp_path / ".gitignore").write_text(gitignore)
        return tmp_path

    def test_editing_gitignore_contents_alone_is_detected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The false-TRUE: same config object, same bool, different set of walked files."""
        repo = self._repo(tmp_path, "nothing_here\n")
        recorded = IndexProvenance(walk_config=_walk_config_json(_config(repo)))

        (repo / ".gitignore").write_text("sub/\n")
        diff = walk_config_differences(recorded, _config(repo))

        assert diff, (
            "a .gitignore edit that removes sub/b.py from the walk is invisible to the "
            "fingerprint, so missing_is_trustworthy stays True and a --prune would "
            "delete a present file"
        )

    def test_a_nested_gitignore_is_part_of_the_fingerprint(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Nested `.gitignore` files are 74% of this project's own index (walker.py:13),
        so a root-only digest would miss the majority of the ignore surface."""
        repo = self._repo(tmp_path, "nothing_here\n")
        recorded = IndexProvenance(walk_config=_walk_config_json(_config(repo)))

        (repo / "sub" / ".gitignore").write_text("b.py\n")
        assert walk_config_differences(recorded, _config(repo))

    def test_the_digest_is_stable_across_runs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An unstable digest would report every drift check as untrustworthy, which
        disables the feature just as thoroughly as a false TRUE enables the bug."""
        repo = self._repo(tmp_path, "sub/\n#comment\n")
        (repo / "sub" / ".gitignore").write_text("b.py\n")

        first = _walk_config_json(_config(repo))
        second = _walk_config_json(_config(repo))

        assert first == second

    def test_the_digest_does_not_depend_on_the_repo_location(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Paths must enter the digest repo-relative. An absolute path would make the
        fingerprint differ between a checkout and its copy, or between CI and a laptop,
        for two byte-identical trees."""
        import json

        one = self._repo(tmp_path / "one", "sub/\n")
        two = self._repo(tmp_path / "two", "sub/\n")

        assert json.loads(_walk_config_json(_config(one))) == json.loads(
            _walk_config_json(_config(two))
        )

    def test_no_gitignore_is_distinguishable_from_an_empty_one(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        bare = tmp_path / "bare"
        bare.mkdir()
        (bare / "a.py").write_text("x = 1\n")
        recorded = IndexProvenance(walk_config=_walk_config_json(_config(bare)))

        (bare / ".gitignore").write_text("")
        assert walk_config_differences(recorded, _config(bare))

    def test_the_digest_is_omitted_when_gitignore_is_not_respected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """With `respect_gitignore=False` the contents cannot change the walk, so
        digesting them would raise a warning that has no consequence — and a spurious
        untrustworthy verdict trains users to ignore the real one."""
        repo = self._repo(tmp_path, "sub/\n")
        cfg = _config(repo, respect_gitignore=False)
        recorded = IndexProvenance(walk_config=_walk_config_json(cfg))

        (repo / ".gitignore").write_text("a.py\n")
        assert not walk_config_differences(recorded, _config(repo, respect_gitignore=False))

    def test_the_digest_covers_exactly_the_chain_the_walk_consults(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The digest must fingerprint the walker's own view, not the filesystem's.

        A `rglob(".gitignore")` would sweep in files under pruned directories — on this
        repository that is 33 `.gitignore` files against the 3 the walk actually gives
        authority to (`node_modules`, `.pytest_cache`, worktree copies). Fingerprinting
        those would report drift for edits that cannot change a single indexed file.
        """
        from trelix.indexing.walker import FileWalker
        from trelix.store.provenance import _DIGEST_SCHEME, _gitignore_digest

        repo = self._repo(tmp_path, "pruned/\n")
        (repo / "sub" / ".gitignore").write_text("nothing\n")
        (repo / "pruned").mkdir()
        (repo / "pruned" / ".gitignore").write_text("invisible\n")
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / ".gitignore").write_text("also_invisible\n")

        cfg = _config(repo)
        walker = FileWalker(cfg)
        list(walker.walk())
        consulted = {d for d, spec in walker._spec_cache.items() if spec is not None}

        assert consulted == {repo, repo / "sub"}
        digest = _gitignore_digest(cfg)
        assert digest is not None
        # Membership, not only the count: two chains of the same size over different
        # directories are a different walk, and the count cannot tell them apart.
        assert digest.startswith(f"{_DIGEST_SCHEME}:{len(consulted)} file(s) [., sub],"), (
            f"digest covered a different set of .gitignore files than the walk reads: "
            f"{digest} vs {sorted(str(d) for d in consulted)}"
        )
        assert "pruned" not in digest and "node_modules" not in digest

    def test_an_edit_under_a_pruned_directory_does_not_report_drift(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An ignore file the walk never reads cannot change which files are indexable, so
        warning about it would be crying wolf."""
        from trelix.store.provenance import _gitignore_digest

        repo = self._repo(tmp_path, "nothing_here\n")
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / ".gitignore").write_text("a\n")
        before = _gitignore_digest(_config(repo))

        (repo / "node_modules" / ".gitignore").write_text("completely different\n")
        assert _gitignore_digest(_config(repo)) == before

    def test_a_gitignore_edit_makes_missing_untrustworthy_end_to_end(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The whole point, through `compute_drift`: b.py is present the entire time."""
        from trelix.indexing.walker import FileWalker
        from trelix.store.provenance import write_provenance

        repo = self._repo(tmp_path, "nothing_here\n")
        cfg = _config(repo)

        with Database(repo / ".trelix" / "index.db") as db:
            for walked in FileWalker(cfg).walk():
                db.upsert_file(walked)
            write_provenance(db, IndexProvenance(walk_config=_walk_config_json(cfg)))
            db._conn.commit()

            (repo / ".gitignore").write_text("sub/\n")
            report = compute_drift(cfg, db)

        assert "sub/b.py" in report.missing
        assert (repo / "sub" / "b.py").is_file(), "the file never left disk"
        assert not report.missing_is_trustworthy, (
            "a --prune reading this would delete the embedding for a present file"
        )


class TestMoreThanOneWalkConfigMakesMissingUnverified:
    """The reporting half of `AUD-01`. `--prune` refuses on it; the count must disclaim it.

    Fixing only `plan_prune` would leave `trelix stats --drift` announcing those same rows
    as deletions with no "unverified" marker — the identical false confidence, one command
    away from the destructive one, and the number a human reads before typing `--yes`.
    """

    def test_a_second_walk_config_makes_the_missing_count_unverified(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from trelix.indexing.walker import FileWalker
        from trelix.store.provenance import read_provenance, write_provenance

        repo = tmp_path
        (repo / "keep").mkdir()
        (repo / "vendored").mkdir()
        (repo / "keep" / "a.py").write_text("x = 1\n")
        (repo / "vendored" / "b.py").write_text("y = 2\n")

        wide = _config(repo)
        narrow = _config(repo, extra_ignore_dirs=[".git", ".trelix", "vendored"])

        with Database(repo / ".trelix" / "index.db") as db:
            for walked in FileWalker(wide).walk():
                db.upsert_file(walked)
            write_provenance(db, IndexProvenance(walk_config=_walk_config_json(wide)))
            # The narrowing re-index: no row is deleted, and the record now describes the
            # narrow walk. Both the snapshot and the current config are `narrow` from here.
            write_provenance(db, IndexProvenance(walk_config=_walk_config_json(narrow)))
            db._conn.commit()

            report = compute_drift(narrow, db, provenance=read_provenance(db))

        assert "vendored/b.py" in report.missing
        assert (repo / "vendored" / "b.py").is_file(), "the file never left disk"
        assert report.walk_config_comparable, "not the comparability guard's doing"
        assert not report.walk_config_changed, "not the changed-settings guard's doing"
        assert not report.missing_is_trustworthy, (
            "the count is presented as verified while a present file sits in `missing` — "
            "one walk_config row was read as describing every row in the index"
        )
        assert report.actionable_drifted_count == len(report.stale) + len(report.new)


class TestTheDigestSchemeIsVersioned:
    """`DOG-12` leg 6: changing WHAT the digest hashes must not read as "the walk narrowed".

    Every index already on disk carries a digest computed the old way. Without a scheme
    tag the first run after such a change reports `walk_config_changed = True` for every
    existing user — a false positive that says "your ignore rules moved" when nothing moved
    but trelix's own arithmetic, and that trains users to force past the one refusal that
    matters. There is in-tree precedent: v3.1.2 started reading nested `.gitignore` chains
    and moved this project's own index by 74% without a single config field changing.

    "Recorded under an older scheme, cannot compare" is the honest verdict, and it is the
    one that keeps `--prune` refusing (`missing_is_trustworthy` is False either way).
    """

    def _record_under_the_old_scheme(self, cfg) -> IndexProvenance:  # type: ignore[no-untyped-def]
        """The record a pre-fix trelix would have written: a digest with no scheme tag."""
        import json

        from trelix.store.provenance import _DIGEST_SCHEME, _GITIGNORE_KEY

        payload = json.loads(_walk_config_json(cfg) or "{}")
        assert payload[_GITIGNORE_KEY].startswith(f"{_DIGEST_SCHEME}:")
        payload[_GITIGNORE_KEY] = payload[_GITIGNORE_KEY].split(":", 1)[1]
        return IndexProvenance(walk_config=json.dumps(payload, sort_keys=True))

    def test_the_current_digest_carries_its_scheme(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from trelix.store.provenance import _DIGEST_SCHEME, _gitignore_digest

        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / ".gitignore").write_text("nothing\n")
        digest = _gitignore_digest(_config(tmp_path))
        assert digest is not None
        assert digest.startswith(f"{_DIGEST_SCHEME}:")

    def test_an_older_scheme_is_incomparable_rather_than_changed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from trelix.store.provenance import is_walk_config_comparable

        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / ".gitignore").write_text("nothing\n")
        cfg = _config(tmp_path)
        recorded = self._record_under_the_old_scheme(cfg)

        assert not is_walk_config_comparable(recorded)
        assert walk_config_differences(recorded, cfg) == {}, (
            "an old-scheme record was diffed against a new-scheme one, so every existing "
            "index reports a walk-config change that never happened"
        )

    def test_compute_drift_reports_it_as_incomparable_end_to_end(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / ".gitignore").write_text("nothing\n")
        cfg = _config(tmp_path)
        recorded = self._record_under_the_old_scheme(cfg)

        with Database(tmp_path / ".trelix" / "index.db") as db:
            report = compute_drift(cfg, db, provenance=recorded)

        assert not report.walk_config_comparable
        assert not report.walk_config_changed, "'cannot compare' was rendered as 'changed'"
        assert not report.missing_is_trustworthy

    def test_an_index_that_never_consulted_gitignore_stays_comparable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """With `respect_gitignore=False` there is no digest to tag, so the old and new
        schemes record the same thing — refusing to compare those would be crying wolf."""
        from trelix.store.provenance import is_walk_config_comparable

        cfg = _config(tmp_path, respect_gitignore=False)
        assert is_walk_config_comparable(IndexProvenance(walk_config=_walk_config_json(cfg)))


class TestTheDigestNamesWhatChanged:
    """`DOG-12`: "N file(s)" can read identically on both sides while membership changed.

    The project's own documented backup ritual is `cp -a .trelix .trelix.bak-<name>`. The
    default ignore list contains `.trelix` and nothing matching `.trelix.bak-*`, so the
    backup puts a copy of `.trelix/.gitignore` INTO the walk and moves the digest — and the
    remediation trelix prints ("re-run with the environment that built the index") cannot
    work, because the poisoning is a filesystem fact, not an environment one. A comparison
    that shows only two hex strings and a count cannot tell the user that.
    """

    def test_a_trelix_backup_sibling_is_named_in_the_comparison(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / ".gitignore").write_text("nothing\n")
        cfg = _config(tmp_path)
        recorded = IndexProvenance(walk_config=_walk_config_json(cfg))

        # `cp -a .trelix .trelix.bak-selfindex`, as scripts/self-index.sh's own safety step
        # prescribes. `.trelix` is ignored by default; its copy is not.
        backup = tmp_path / ".trelix.bak-selfindex"
        backup.mkdir()
        (backup / ".gitignore").write_text("*\n")

        diff = walk_config_differences(recorded, cfg)

        assert "gitignore_chain" in diff, "a copy of .trelix changed the walk unnoticed"
        _before, after = diff["gitignore_chain"]
        assert ".trelix.bak-selfindex" in str(after), (
            "the comparison names no member, so a user cannot tell a backup directory from "
            "a real ignore-rule change and the printed remediation cannot help them: "
            f"{after}"
        )


class TestUnreadableStateDoesNotBreakTheDigest:
    def test_a_repo_path_that_does_not_exist_still_serialises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """`_walk_config_json` runs inside `capture_provenance`, which must never raise —
        provenance is additive context and losing it must not fail an index."""
        cfg = IndexConfig.model_construct(repo_path=str(tmp_path / "gone"))
        object.__setattr__(cfg, "walker", WalkerConfig())
        assert _walk_config_json(cfg) is not None


def test_a_deleted_file_is_still_reported_as_drift(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Guard against over-correcting: when the walk is verifiably comparable, a real
    deletion must keep counting, or gating the total would silently hide deletions."""
    repo = tmp_path
    (repo / "a.py").write_text("x = 1\n")
    cfg = _config(repo)

    with Database(repo / ".trelix" / "index.db") as db:
        from trelix.store.provenance import write_provenance

        db.upsert_file(
            IndexedFile(
                path=str(repo / "gone.py"),
                rel_path="gone.py",
                language=Language.PYTHON,
                hash="h",
                size_bytes=1,
            )
        )
        write_provenance(db, IndexProvenance(walk_config=_walk_config_json(cfg)))
        db._conn.commit()
        report = compute_drift(cfg, db)

    assert report.missing == ("gone.py",)
    assert report.missing_is_trustworthy
    assert report.actionable_drifted_count == report.drifted_count
