"""Contract tests pinning every member of `walker._STORE_PATH_SEGMENTS`.

`_STORE_PATH_SEGMENTS` is the veto that stops the conditional-ignore tier
(`_CONDITIONAL_IGNORE_DIRS` — `packages/` and `bin/`) from reclassifying a directory as
first-party source when it lives inside a package store. Package stores contain complete
copies of other projects, workspace manifests included, so a `pnpm-workspace.yaml` found
inside one proves nothing about the repo being indexed and would re-admit the exact
duplicate-content shape nested-`.gitignore` support was added to remove in v3.1.2.

MUTATIONS these tests must fail on (deleting `".pnp"` was a MEASURED survivor of 294 tests
-- test_walker + test_walker_containment + test_walker_filters + test_conditional_ignore_dirs
+ test_detect_language_contract + test_prune + test_drift_honesty + test_dry_run all passed
with it gone, i.e. a Yarn PnP store's vendored `packages/` could be pulled into the index in
silence):

  * deleting ANY member of `walker._STORE_PATH_SEGMENTS`
  * adding a member without recording it here
  * dropping the case-fold in `_under_store_path` (`part.lower() in _STORE_PATH_SEGMENTS`)
  * making `_under_store_path` check only the immediate parent rather than every ancestor
    segment (the fixtures nest the store segment two levels above `packages/`)
  * removing the `if self._under_store_path(parent): return {}` guard from
    `_classify_conditional_dirs` altogether

Why the table below is written out by hand instead of being read from walker.py: an
assertion that loops over `_STORE_PATH_SEGMENTS` checks LESS the moment a member is
deleted, and still passes. That is exactly how 14 of 43 `EXTENSION_MAP` entries became
deletable with a green suite. The expected table here is a literal, and every
parametrisation is driven from the literal, so a deletion fails and an ADDITION fails the
set-equality check until someone records the new segment.

Nothing here asserts on `_under_store_path` or `_classify_conditional_dirs` directly. Every
behavioural test asserts on the exact set of `rel_path`s `FileWalker.walk()` yields, so a
data-only pin that stopped being read would fail too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.indexing.walker import _STORE_PATH_SEGMENTS, FileWalker

# ---------------------------------------------------------------------------
# The contract. Hand-written literals, never read back out of walker.py.
# ---------------------------------------------------------------------------

# Store segments that the SHIPPED `extra_ignore_dirs` ALSO excludes unconditionally. For
# these two the store rule is belt-and-braces under the default config: the traversal never
# descends, so the exclusion cannot be attributed to `_STORE_PATH_SEGMENTS` unless
# `extra_ignore_dirs` is narrowed first. They are still load-bearing, because
# `extra_ignore_dirs` is user-configurable (TRELIX_WALKER_EXTRA_IGNORE_DIRS).
SHADOWED_BY_UNCONDITIONAL_TIER: tuple[str, ...] = (
    "node_modules",
    ".git",
)

# Store segments that NOTHING else excludes. The walk really does descend into these under
# the shipped defaults, so the store rule is the only thing holding their vendored
# `packages/` out of the index.
NOT_OTHERWISE_IGNORED: tuple[str, ...] = (
    ".pnpm-store",
    ".yarn",
    ".npm",
    ".pnp",
)

EXPECTED_STORE_SEGMENTS: tuple[str, ...] = (
    "node_modules",
    ".pnpm-store",
    ".yarn",
    ".npm",
    ".pnp",
    ".git",
)

# A directory name deliberately absent from the table above. It is the discrimination
# control: the same workspace shape under this name MUST be indexed, which is the only way
# the exclusions asserted below could otherwise pass by construction.
CONTROL_DIR = "firstparty"


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _build_workspace(parent: Path) -> None:
    """A minimal *proven* JS workspace: a root manifest with a `workspaces` key.

    `package.json` sits beside `packages/`, which is what `_classify_conditional_dirs`
    accepts as positive evidence. `mod.py` inside is the indexable file whose presence in
    the walk answers the question.
    """
    (parent / "packages" / "app").mkdir(parents=True)
    (parent / "package.json").write_text(
        json.dumps({"name": "w", "workspaces": ["packages/*"]}), encoding="utf-8"
    )
    (parent / "packages" / "app" / "mod.py").write_text("x = 1\n", encoding="utf-8")


def _walk_rel_paths(repo: Path, *, extra_ignore_dirs: list[str] | None = None) -> set[str]:
    """Walk with the conditional tier ENFORCING, since that is the tier the veto guards.

    With `index_conditional_dirs=False` a vetoed `packages/` is excluded by the
    unconditional tier regardless, and the veto only changes whether a WARNING is logged.
    """
    walker_kwargs: dict[str, object] = {}
    if extra_ignore_dirs is not None:
        walker_kwargs["extra_ignore_dirs"] = extra_ignore_dirs
    config = IndexConfig(repo_path=str(repo), walker=WalkerConfig(**walker_kwargs))  # type: ignore[arg-type]
    return {
        Path(f.rel_path).as_posix() for f in FileWalker(config, index_conditional_dirs=True).walk()
    }


class TestStorePathSegmentsTable:
    def test_expected_table_is_exactly_the_store_path_segments(self) -> None:
        """MUTATION: delete or add any `_STORE_PATH_SEGMENTS` member (e.g. drop `".pnp"`).

        Set equality both ways, not a count: deleting `.pnp` while adding `.foo` keeps the
        length identical and must still fail.
        """
        expected = set(EXPECTED_STORE_SEGMENTS)
        assert len(expected) == len(EXPECTED_STORE_SEGMENTS), (
            "EXPECTED_STORE_SEGMENTS has duplicate entries"
        )
        assert set(_STORE_PATH_SEGMENTS) == expected, (
            "_STORE_PATH_SEGMENTS no longer matches the recorded contract. Missing from the "
            f"constant: {sorted(expected - set(_STORE_PATH_SEGMENTS))}; present in the "
            f"constant but not recorded here: "
            f"{sorted(set(_STORE_PATH_SEGMENTS) - expected)}. Record the new segment in "
            "EXPECTED_STORE_SEGMENTS and in exactly one of "
            "SHADOWED_BY_UNCONDITIONAL_TIER / NOT_OTHERWISE_IGNORED rather than loosening "
            "this assertion."
        )

    def test_sub_tables_partition_the_expected_table(self) -> None:
        """Bookkeeping guard: a newly added segment must be classified, not just listed.

        Without this, someone could append to EXPECTED_STORE_SEGMENTS and leave the new
        segment out of both behavioural parametrisations below.
        """
        shadowed = set(SHADOWED_BY_UNCONDITIONAL_TIER)
        unshadowed = set(NOT_OTHERWISE_IGNORED)
        assert shadowed & unshadowed == set(), "a segment is in both sub-tables"
        assert shadowed | unshadowed == set(EXPECTED_STORE_SEGMENTS)

    def test_shadow_classification_matches_the_shipped_ignore_list(self) -> None:
        """Discrimination precondition for `test_store_segment_vetoes_under_shipped_ignore_dirs`.

        That test walks with the SHIPPED `extra_ignore_dirs`. If any of the four
        NOT_OTHERWISE_IGNORED names were ever added to `extra_ignore_dirs`, the walk would
        stop descending into it and the test would pass by construction — proving nothing
        about `_STORE_PATH_SEGMENTS`. This fails first, and names the reason.
        """
        shipped = set(WalkerConfig().extra_ignore_dirs)
        assert set(EXPECTED_STORE_SEGMENTS) & shipped == set(SHADOWED_BY_UNCONDITIONAL_TIER), (
            "the store segments that WalkerConfig.extra_ignore_dirs also excludes have "
            "changed. Move the affected name between SHADOWED_BY_UNCONDITIONAL_TIER and "
            "NOT_OTHERWISE_IGNORED; do not leave it in the shipped-default parametrisation, "
            "where the unconditional tier would answer for it."
        )
        assert CONTROL_DIR not in shipped, (
            f"the control directory {CONTROL_DIR!r} is now in extra_ignore_dirs, so the "
            "control arm of every behavioural test below has stopped discriminating"
        )


class TestStorePathSegmentsBehaviour:
    @pytest.mark.parametrize("segment", EXPECTED_STORE_SEGMENTS)
    def test_store_segment_vetoes_a_proven_workspace(self, segment: str, tmp_path: Path) -> None:
        """MUTATION: delete `segment` from `_STORE_PATH_SEGMENTS`.

        Also fails if `_under_store_path` is narrowed to the immediate parent — the store
        segment here is `packages/`'s GRANDparent.

        `extra_ignore_dirs` is narrowed to just `packages` so that no store segment is also
        excluded by the unconditional tier; every one of the six is then judged by
        `_STORE_PATH_SEGMENTS` alone, through the same mechanism.

        The exact-set assertion carries both preconditions:
          * `<segment>/inner/package.json` IS yielded — the walk really descended INTO the
            store directory, so the missing `mod.py` is the veto and not a traversal skip.
          * the `firstparty/` control IS indexed — the fixture can still admit a workspace,
            so the exclusion is about the segment name and nothing else.
        """
        _build_workspace(tmp_path / segment / "inner")
        _build_workspace(tmp_path / CONTROL_DIR / "inner")

        assert _walk_rel_paths(tmp_path, extra_ignore_dirs=["packages"]) == {
            f"{segment}/inner/package.json",
            f"{CONTROL_DIR}/inner/package.json",
            f"{CONTROL_DIR}/inner/packages/app/mod.py",
        }

    @pytest.mark.parametrize("segment", NOT_OTHERWISE_IGNORED)
    # Named for `extra_ignore_dirs`, NOT the shipped default config: the walk helper
    # always passes index_conditional_dirs=True, so the default WalkerConfig is not what
    # is under test here. The previous name invited a reader to believe otherwise.
    def test_store_segment_vetoes_under_shipped_ignore_dirs(
        self, segment: str, tmp_path: Path
    ) -> None:
        """MUTATION: delete `segment` from `_STORE_PATH_SEGMENTS`.

        The same claim under trelix's SHIPPED `extra_ignore_dirs`, i.e. the configuration
        real indexes run with. Restricted to the four segments nothing else excludes, so
        the veto is provably the only thing holding these out — the claim walker.py makes
        about `.pnpm-store` in particular. See
        `test_shadow_classification_matches_the_shipped_ignore_list`, which fails first if
        that stops being true.
        """
        _build_workspace(tmp_path / segment / "inner")
        _build_workspace(tmp_path / CONTROL_DIR / "inner")

        assert _walk_rel_paths(tmp_path) == {
            f"{segment}/inner/package.json",
            f"{CONTROL_DIR}/inner/package.json",
            f"{CONTROL_DIR}/inner/packages/app/mod.py",
        }

    @pytest.mark.parametrize("segment", EXPECTED_STORE_SEGMENTS)
    def test_store_segment_veto_is_case_insensitive(self, segment: str, tmp_path: Path) -> None:
        """MUTATION: drop `.lower()` from `part.lower() in _STORE_PATH_SEGMENTS`.

        A case-insensitive filesystem makes `.PNPM-STORE` the same directory as
        `.pnpm-store`, so an exact-case comparison lets a real store through the veto. This
        is the same defect that `_DOTNET_MARKER_FILES` records having been measured.
        """
        upper = segment.upper()
        assert upper != segment, (
            f"{segment!r} has no distinct uppercase form, so this parametrisation cannot "
            "discriminate a dropped case-fold"
        )
        _build_workspace(tmp_path / upper / "inner")
        _build_workspace(tmp_path / CONTROL_DIR / "inner")

        assert _walk_rel_paths(tmp_path, extra_ignore_dirs=["packages"]) == {
            f"{upper}/inner/package.json",
            f"{CONTROL_DIR}/inner/package.json",
            f"{CONTROL_DIR}/inner/packages/app/mod.py",
        }
