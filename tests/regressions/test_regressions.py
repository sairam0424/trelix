"""Restore each shipped defect and require the tests written for it to FAIL.

WHAT THIS FILE MEASURES THAT NOTHING ELSE IN THIS REPO CAN
---------------------------------------------------------
Mutation testing mutates PRODUCTION source. cargo-mutants' own documentation states the
limit plainly -- "the diff is only matched against the code under test, not the test code"
-- and mutmut and cosmic-ray share it. So no mutation score in this repository can tell
whether a TEST still detects the defect it was written for, and every vacuity fix landed
across five rounds of work here has been a test-side change. Weaken
``assert _PHANTOM_ID not in batched`` to ``assert True`` and the mutation score does not
move by one mutant.

This file is the instrument for that question. It re-applies the defect and demands the
failure. A weakened, deleted, renamed, vacuous or silently-deselected test shows up here as
a survivor.

THE ASSERTIONS LIVE IN harness.evaluate, ON PURPOSE
--------------------------------------------------
Not for tidiness -- so that they can be shown to fail. ``test_the_kill_check_itself_can_fail``
and ``test_the_remainder_check_itself_can_fail`` re-evaluate a REAL, already-collected run
against a deliberately wrong manifest entry and require ``evaluate`` to object. Without
those two, a green result here would mean only "evaluate returned", which is also what a
gutted ``evaluate`` does. They reuse cached run results, so they cost no extra child runs.

WHY EVERY NODE ID IS WRITTEN OUT IN REGRESSIONS.toml RATHER THAN DISCOVERED
--------------------------------------------------------------------------
``must_fail`` is an explicit table compared BOTH WAYS against what the patched run
reported: the listed nodes must fail, and every other node in the same file must still
pass. Deriving the kill set from "whatever failed" would make this file agree with any
outcome, which is the shape of a test that cannot fail.

COST, AND THE MARKER
--------------------
Each defect costs one throwaway ``git worktree`` and one child pytest. Measured on this
tree at 32.1s and 61.9s on two runs of the same 22 tests -- the spread is machine load, and
both are quoted rather than the flattering one. Either way it is far over the 4.0s file-cost
threshold, so it carries ``slow`` via
``tests/conftest.py::SLOW_FILES`` -- the same treatment ``test_marker_taxonomy.py`` gets
for the same reason, and with the same consequence stated plainly: this file does NOT run
in a ``-m "not slow"`` inner loop. It is a config-and-suite guard, so a full run is the
right place for it.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. Weaken any assertion in a mapped test (e.g. change ``assert _PHANTOM_ID not in batched``
   to ``assert True`` in tests/unit/test_sparse_padding_contamination.py)
   -> ``test_restoring_the_defect_makes_its_tests_fail[splade-padding-3.2.0]`` fails with
      that node id listed as a survivor. This is the whole point of the file, and it is the
      one mutation no mutation-testing tool in this class can express.
2. Rename or delete a mapped test
   -> ``test_every_mapped_node_id_still_collects`` fails (anti-dark).
3. Edit ``src/`` so an anti-fix patch no longer applies
   -> ``harness.PatchDidNotApply``, a hard error. Never a skip.
4. Delete a ``[[defect]]`` entry, or add one and forget the patch file
   -> ``test_the_manifest_covers_exactly_the_verified_defects`` or
      ``test_every_patch_file_on_disk_is_claimed_by_the_manifest`` fails.
5. Remove an assertion from ``harness.evaluate``
   -> ``test_the_kill_check_itself_can_fail`` /
      ``test_the_remainder_check_itself_can_fail`` fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.regressions import harness

# Loaded at import so the parametrization ids are the manifest ids. A malformed manifest
# raises here and errors collection of this file, which is the intended loudness: a
# manifest that cannot be parsed must not degrade into "0 defects checked, all green".
DEFECTS = harness.load_defects()

# The literal, third statement of the taxonomy -- independent of both the manifest and the
# patches directory, so set-equality against it catches a defect being quietly dropped as
# well as one being added without review. Each id was verified by running its mapped tests
# under its patch; none is here on the strength of the CHANGELOG.
EXPECTED_DEFECT_IDS = frozenset(
    {
        "splade-padding-3.2.0",
        "nomic-code-document-prefix-3.2.0",
        "nomic-code-query-prompt-name-3.2.0",
        "bge-code-foreign-prompt-3.2.0",
        "nomic-code-einops-undeclared-3.2.0",
    }
)

_RESULT_CACHE: dict[str, harness.RunResult] = {}


def _missing_requirements(defect: harness.Defect) -> list[str]:
    return [name for name in defect.requires if importlib.util.find_spec(name) is None]


def _result_for(defect: harness.Defect) -> harness.RunResult:
    """One child run per defect, memoized across the tests in this file.

    Deliberately NOT a pytest fixture keyed on the parametrized defect: the control tests
    below re-evaluate a *real* result against a wrong manifest entry, and they must reuse
    the very same measurement the passing test used rather than take a fresh one.
    """
    if defect.id not in _RESULT_CACHE:
        _RESULT_CACHE[defect.id] = harness.run_under_patch(defect)
    return _RESULT_CACHE[defect.id]


def _skip_if_out_of_reach(defect: harness.Defect) -> None:
    if harness.running_as_child():
        pytest.skip(
            "refusing to run inside a child pytest spawned by this harness; "
            "tests/regressions must not recurse"
        )
    missing = _missing_requirements(defect)
    if missing:
        pytest.skip(
            f"{defect.id} cannot be measured here: {missing} not importable. The mapped "
            f"tests would not discriminate without them -- this is NOT a pass, it is out "
            f"of reach. Install the extra that provides {missing} to make it meaningful."
        )
    reason = harness.unavailable_reason()
    if reason is not None:
        pytest.skip(f"{defect.id} cannot be measured here: {reason}")


# ---------------------------------------------------------------------------
# Manifest integrity. These are cheap and run without spawning anything.
# ---------------------------------------------------------------------------


def test_the_manifest_covers_exactly_the_verified_defects() -> None:
    """Set equality both ways against the literal above.

    One direction catches a defect entry being deleted -- which would remove a guard while
    leaving the file green. The other catches one being added without a row in the literal,
    which is the review gate: an unverified entry has to be noticed.
    """
    present = {defect.id for defect in DEFECTS}
    assert present - EXPECTED_DEFECT_IDS == frozenset(), (
        f"manifest defines defects absent from this file's literal set: "
        f"{sorted(present - EXPECTED_DEFECT_IDS)}. Verify the new entry (patch applies, "
        f"mapped tests fail under it) and then add its id here."
    )
    assert EXPECTED_DEFECT_IDS - present == frozenset(), (
        f"defects have vanished from REGRESSIONS.toml: "
        f"{sorted(EXPECTED_DEFECT_IDS - present)}. Each one is a shipped fix that is now "
        f"unguarded against a test-side regression."
    )


def test_every_patch_file_on_disk_is_claimed_by_the_manifest() -> None:
    """Both ways again: no orphan patches, no missing patches.

    An orphaned ``.patch`` is a defect somebody stopped checking; a manifest entry whose
    patch file is gone is a guard that cannot run.
    """
    on_disk = {
        path.relative_to(harness.REGRESSIONS_DIR).as_posix()
        for path in harness.PATCHES_DIR.glob("*.patch")
    }
    assert on_disk, (
        f"no .patch files under {harness.PATCHES_DIR}; every assertion in this file would "
        f"be vacuous"
    )
    claimed = {defect.patch for defect in DEFECTS}
    assert on_disk - claimed == set(), (
        f"patch files no manifest entry references: {sorted(on_disk - claimed)}"
    )
    assert claimed - on_disk == set(), (
        f"manifest entries whose patch file is missing: {sorted(claimed - on_disk)}"
    )


@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.id)
def test_no_anti_fix_patch_touches_a_test_file(defect: harness.Defect) -> None:
    """A patch that edits the test it is checked against is circular.

    ``git apply --numstat`` also proves the patch PARSES, so a corrupted patch fails here
    rather than surfacing later as an empty file list that reads like "changes nothing".
    """
    targets = harness.patch_target_files(defect)
    offenders = [path for path in targets if path.startswith("tests/")]
    assert not offenders, (
        f"{defect.id}: its patch edits {offenders}. An anti-fix must restore the DEFECT, "
        f"never adjust the test that is supposed to catch it."
    )
    assert targets, f"{defect.id}: patch names no files"


# 600s, not the suite's 60s default. Collecting tests/unit/test_sparse_padding_contamination.py
# imports torch, which measured ~10s cold on this machine and is the kind of thing that is
# several times slower on a cold CI runner. The default would turn a slow import into a
# spurious red; the point of this test is drift in node ids, not wall clock.
@pytest.mark.timeout(600)
@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.id)
def test_every_mapped_node_id_still_collects(defect: harness.Defect) -> None:
    """ANTI-DARK. A mapped node that no longer collects must fail loudly, not vanish.

    Renaming a test, moving it between classes, or dropping it silently removes it from
    the kill set: the patched run simply never reports it, ``must_fail`` shrinks to nothing
    it can check, and this whole directory goes quiet while staying green. That is exactly
    how the ``integration`` marker went dark for a release -- registered, documented as
    *the* credential-free run, and carried by nobody.

    Deselected ids are checked too: a ``deselect`` entry pointing at a node that no longer
    exists means the exclusion is stale and its reason is no longer true of anything.
    """
    if harness.running_as_child():
        pytest.skip("not run inside a child pytest spawned by this harness")
    collected = harness.collect_node_ids(defect.mapped_files)
    assert collected, (
        f"{defect.id}: {list(defect.mapped_files)} collected NOTHING, so no node id in "
        f"this entry can be verified"
    )
    for label, node_ids in (("must_fail", defect.must_fail), ("deselect", defect.deselect)):
        gone = sorted(node_id for node_id in node_ids if node_id not in collected)
        assert not gone, (
            f"{defect.id}: {label} names node id(s) that no longer collect: {gone}. "
            f"Renamed or deleted tests must be re-pointed in REGRESSIONS.toml in the SAME "
            f"change, or the defect they guard becomes unguarded silently."
        )


# ---------------------------------------------------------------------------
# The measurement. One throwaway worktree and one child pytest per defect.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(600)
@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.id)
def test_restoring_the_defect_makes_its_tests_fail(defect: harness.Defect) -> None:
    """The three assertions, on a real patched run. See ``harness.evaluate``.

    A stale patch raises :class:`harness.PatchDidNotApply` from inside
    :func:`harness.run_under_patch` and is NOT caught here. That is deliberate and it is
    the single most important design decision in this directory: a harness that skips on a
    patch which no longer applies looks identical to one that verified everything.
    """
    _skip_if_out_of_reach(defect)
    result = _result_for(defect)
    harness.evaluate(defect, result)


@pytest.mark.timeout(600)
def test_the_kill_check_itself_can_fail() -> None:
    """CONTROL: ``evaluate`` must object when a node that PASSED is listed as a kill.

    Reuses the bge-code run rather than taking a new measurement, so this asserts about
    the same numbers the test above accepted. If ``evaluate``'s survivor check were
    deleted, every defect above would still report green and only this test would notice.
    """
    defect = next(d for d in DEFECTS if d.id == "bge-code-foreign-prompt-3.2.0")
    _skip_if_out_of_reach(defect)
    result = _result_for(defect)

    # This node passes under the patch -- it is the AST detector's own control, which the
    # bge-code prompt cannot affect. Claiming it as a kill must be rejected.
    known_passing = (
        "tests/unit/test_provider_prompt_provenance.py::TestNoPromptIsSharedBetweenProviders"
        "::test_the_ast_detector_ignores_comments_but_not_code"
    )
    assert result.outcomes.get(known_passing) == "passed", (
        f"this control assumes {known_passing} passes under the bge-code anti-fix; it "
        f"reported {result.outcomes.get(known_passing)!r}, so pick another node"
    )
    wrong = harness.Defect(
        id=defect.id + "::control",
        changelog=defect.changelog,
        summary=defect.summary,
        patch=defect.patch,
        must_fail=(*defect.must_fail, known_passing),
    )
    with pytest.raises(AssertionError, match="DID NOT FAIL"):
        harness.evaluate(wrong, result)


@pytest.mark.timeout(600)
def test_the_remainder_check_itself_can_fail() -> None:
    """CONTROL: ``evaluate`` must object when a real failure is left out of ``must_fail``.

    Under-specifying the kill set is the subtler error, because the run still goes red in
    the right places. Dropping three of bge-code's four kills leaves them in the remainder,
    where they must be reported as collateral damage.
    """
    defect = next(d for d in DEFECTS if d.id == "bge-code-foreign-prompt-3.2.0")
    _skip_if_out_of_reach(defect)
    result = _result_for(defect)
    assert len(defect.must_fail) >= 2, (
        "this control needs a defect with more than one kill so that truncating the list "
        "leaves a real failure in the remainder"
    )
    truncated = harness.Defect(
        id=defect.id + "::control",
        changelog=defect.changelog,
        summary=defect.summary,
        patch=defect.patch,
        must_fail=defect.must_fail[:1],
    )
    with pytest.raises(AssertionError, match="also broke tests it does not claim to"):
        harness.evaluate(truncated, result)


@pytest.mark.timeout(600)
def test_a_patch_that_no_longer_applies_is_an_error_and_not_a_skip(
    tmp_path: Path,
) -> None:
    """CONTROL on assertion (1): staleness must be LOUD.

    A deliberately impossible patch -- real headers, context lines that have never existed
    in ``sparse.py`` -- driven through the real ``git apply`` path rather than trusting that
    the raise site exists. ``Defect.patch`` is joined onto ``REGRESSIONS_DIR``, and an
    absolute path wins that join, so the fixture never writes into the repo tree.

    If this ever starts passing by SKIPPING instead of raising, the harness has acquired
    the one failure mode this whole directory was built to remove: a stale patch means the
    source moved and nobody re-derived the anti-fix, so the defect it names is unguarded
    while the suite stays green.
    """
    if harness.running_as_child():
        pytest.skip("not run inside a child pytest spawned by this harness")
    reason = harness.unavailable_reason()
    if reason is not None:
        pytest.skip(f"cannot exercise git apply here: {reason}")

    stale = tmp_path / "control_stale.patch"
    stale.write_text(
        "diff --git a/src/trelix/embedder/sparse.py b/src/trelix/embedder/sparse.py\n"
        "--- a/src/trelix/embedder/sparse.py\n"
        "+++ b/src/trelix/embedder/sparse.py\n"
        "@@ -164,3 +164,1 @@ class SparseEmbedder:\n"
        "-                this line has never existed in this file\n"
        "-                nor has this one\n"
        "-                nor this\n"
        "+                agg = 0\n",
        encoding="utf-8",
    )
    defect = harness.Defect(
        id="control-stale-patch",
        changelog="n/a",
        summary="a patch whose context does not exist",
        patch=str(stale),
        must_fail=(
            "tests/unit/test_sparse_padding_contamination.py"
            "::TestPaddingDoesNotLeakIntoTheStoredVector::test_the_phantom_token_never_appears",
        ),
    )
    with pytest.raises(harness.PatchDidNotApply):
        harness.run_under_patch(defect)


def test_a_manifest_entry_with_an_unknown_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONTROL on the parser: ``must_faill`` must not become an empty kill set.

    A typo'd key is how a defect entry silently stops asserting anything, so the loader
    refuses unknown keys instead of ignoring them. Driven through the real loader with
    ``MANIFEST_PATH`` redirected, and restored by ``monkeypatch`` rather than by hand.
    """
    bad = tmp_path / "manifest.toml"
    bad.write_text(
        '[[defect]]\nid = "x"\nchangelog = "0"\nsummary = "s"\n'
        'patch = "patches/splade-padding.patch"\nmust_fail = ["tests/unit/a.py::b"]\n'
        'must_faill = ["typo"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(harness, "MANIFEST_PATH", bad)
    with pytest.raises(AssertionError, match="unknown keys"):
        harness.load_defects()


def test_a_manifest_entry_with_an_empty_kill_set_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONTROL: ``must_fail = []`` must be refused, not accepted as "nothing to check".

    An entry with no kills would satisfy every assertion in ``evaluate`` -- there would be
    no survivors because there would be no candidates. That is the purest form of the
    green-when-vacuous shape, so it is rejected at parse time.
    """
    bad = tmp_path / "manifest.toml"
    bad.write_text(
        '[[defect]]\nid = "x"\nchangelog = "0"\nsummary = "s"\n'
        'patch = "patches/splade-padding.patch"\nmust_fail = []\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(harness, "MANIFEST_PATH", bad)
    with pytest.raises(AssertionError, match="empty must_fail"):
        harness.load_defects()
