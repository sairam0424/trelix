"""The guard on the marker taxonomy: registered, applied, and actually selecting.

Why this file is non-negotiable
------------------------------
``--strict-markers`` validates ``@pytest.mark.<name>``. It does NOT validate ``-m``
expressions. Measured on this tree, before this file existed::

    $ pytest tests/unit/test_multi_watcher.py --collect-only -q -m "not integrationn"
    ...
    8 tests collected in 0.16s
    EXIT=0

A one-character typo in the deselection expression silently ran everything and
exited 0. That is how ``integration`` came to be registered, documented in
CONTRIBUTING.md as *the* credential-free run, carried by no test, and deselecting
nothing while driving live Azure/Bedrock calls. Nothing pytest ships catches it,
so the only guard available is a test that pins, for every registered marker:

  * something carries it (a marker carried by nobody reads like protection and
    is not), and
  * excluding it actually removes those tests.

``test_a_misspelled_marker_expression_deselects_nothing`` is this file's own
CONTROL: it runs the same probe with a deliberately misspelled name and asserts
the OPPOSITE outcome. Without it, ``_collect()`` returning a constant would make
every assertion here pass for the wrong reason.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. Delete any ``item.add_marker(...)`` line from ``tests/conftest.py``
   -> the corresponding ``test_marker_selects_and_deselects_its_carrier_file``
      case fails (selected drops to 0 / deselected drops to 0).
2. Add a name to ``markers`` in ``pyproject.toml`` without applying it
   -> ``test_registered_markers_are_exactly_the_measured_set`` fails.
3. Remove a name from ``markers`` while ``tests/conftest.py`` still applies it
   -> same test fails, and ``--strict-markers`` also errors the run.
4. Add a module-scope ``pytest.importorskip`` to a file that is not in
   ``REQUIRES_EXTRA_FILES``
   -> ``test_requires_extra_is_exactly_the_module_scope_importorskip_files`` fails.
5. Delete ``strict = true`` or ``asyncio_default_fixture_loop_scope`` from
   ``pyproject.toml``
   -> ``test_pyproject_pins_the_strictness_umbrella_and_the_asyncio_loop_scope`` fails.
6. Rename or delete any file named in a taxonomy table
   -> ``test_no_taxonomy_table_entry_points_at_a_missing_file`` fails.
7. Drop a coverage.json at the repo root
   -> ``test_no_coverage_json_outside_scratch_pad`` fails.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests import conftest as taxonomy

_TESTS_DIR = Path(taxonomy.__file__).parent
_ROOT = _TESTS_DIR.parent
_PYPROJECT = _ROOT / "pyproject.toml"

# The literal. Written out here rather than derived from pyproject or from
# tests/conftest.py, so that this set is a third, independent statement of the
# taxonomy that both of those must agree with.
EXPECTED_MARKERS = frozenset(
    {
        "api",
        "cli",
        "integration",
        "parser",
        "requires_extra",
        "requires_network",
        "requires_weights",
        "security",
        "slow",
    }
)

# marker -> (a file in which EVERY collected test carries it, module whose absence
# legitimately empties that file). Measured collection counts on this tree, for the
# record only -- the assertions below never hard-code them, so adding a test to one
# of these files does not break this file:
#   integration/test_recall.py 14, unit/test_cli_watch_all_signals.py 2,
#   unit/test_cli_serve_exposure_warning.py 10, unit/test_api_graph.py 11,
#   unit/test_parser_go.py 21, unit/test_network_is_blocked.py 2,
#   unit/test_multi_watcher_filtering.py 15.
CARRIER_FILES: dict[str, tuple[str, str | None]] = {
    "integration": ("tests/integration/test_recall.py", None),
    "requires_network": ("tests/integration/test_recall.py", None),
    "slow": ("tests/unit/test_cli_watch_all_signals.py", None),
    "cli": ("tests/unit/test_cli_serve_exposure_warning.py", None),
    "api": ("tests/unit/test_api_graph.py", None),
    "parser": ("tests/unit/test_parser_go.py", None),
    "security": ("tests/unit/test_network_is_blocked.py", None),
    "requires_extra": ("tests/unit/test_multi_watcher_filtering.py", "watchfiles"),
}

# requires_weights is node-level, so it has no whole-file carrier; it gets its own
# set-equality test over the two files that hold it.
REQUIRES_WEIGHTS_FILES = (
    "tests/unit/test_embedder_bge.py",
    "tests/unit/test_sparse_padding_contamination.py",
)
EXPECTED_REQUIRES_WEIGHTS_NODE_IDS = frozenset(
    {
        "tests/unit/test_embedder_bge.py::TestBGECodePooling"
        "::test_constructed_class_pools_the_way_the_model_was_published",
        "tests/unit/test_embedder_bge.py::TestBGECodePooling"
        "::test_two_queries_differing_after_token_0_get_different_embeddings",
        "tests/unit/test_sparse_padding_contamination.py::test_real_weights_agree_alone_and_batched",
    }
)

_MODULE_SCOPE_IMPORTORSKIP = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]* = )?pytest\.importorskip\(", re.MULTILINE
)


def _collect(paths: tuple[str, ...], marker_expr: str | None = None) -> frozenset[str]:
    """Return the node ids a fresh pytest collects for ``paths`` under ``-m marker_expr``.

    A child process, not this session's ``request.session.items``: under any partial
    run (``-k``, ``-m``, a single file) ``session.items`` holds only what THIS
    invocation selected, so an assertion built on it would report green while
    measuring nothing. Round 4 of this suite's history lost exactly that bet.
    """
    argv = [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "--collect-only",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    if marker_expr is not None:
        argv += ["-m", marker_expr]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # tests/integration/test_cli.py resolves the `trelix` console script at import
    # time; the venv's bin must be reachable or collecting tests/integration errors.
    bin_dir = str(Path(sys.executable).parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(argv, cwd=_ROOT, capture_output=True, text=True, timeout=120)
    # 0 = collected something, 5 = collected nothing (a legitimate outcome for
    # `-m "not <marker>"`). Anything else -- 2 interrupted by a collection error,
    # 4 usage error -- means the probe itself is broken and its empty result must
    # not be read as "the marker deselected everything".
    assert proc.returncode in (0, 5), (
        f"pytest --collect-only {paths} -m {marker_expr!r} exited {proc.returncode}; "
        f"this probe is broken, not measuring.\nstdout tail:\n{proc.stdout[-3000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    return frozenset(
        line.strip()
        for line in proc.stdout.splitlines()
        if "::" in line and line.startswith("tests/")
    )


def _registered_marker_names() -> frozenset[str]:
    with _PYPROJECT.open("rb") as handle:
        ini = tomllib.load(handle)["tool"]["pytest"]["ini_options"]
    return frozenset(entry.split(":", 1)[0].strip() for entry in ini["markers"])


def test_registered_markers_are_exactly_the_measured_set() -> None:
    """pyproject's `markers` must equal the literal above, both directions.

    A name registered but never applied is the failure this whole taxonomy exists
    to prevent, and a name applied but not registered is a hard collection error
    under --strict-markers. Set equality both ways is the only assertion that
    catches both.
    """
    registered = _registered_marker_names()
    assert registered - EXPECTED_MARKERS == frozenset(), (
        f"registered in pyproject but not in this test's literal set: "
        f"{sorted(registered - EXPECTED_MARKERS)}. If the marker is real, add it here "
        f"AND give it a carrier in CARRIER_FILES; if nobody carries it, delete it from "
        f"pyproject rather than leaving protection that does not protect."
    )
    assert EXPECTED_MARKERS - registered == frozenset(), (
        f"expected but missing from pyproject's markers: {sorted(EXPECTED_MARKERS - registered)}"
    )


def test_every_registered_marker_has_a_carrier_probe() -> None:
    """No marker may escape the selection probe below.

    Without this, adding a marker to EXPECTED_MARKERS and to pyproject while
    forgetting CARRIER_FILES would leave it completely unverified -- registered,
    possibly carried by nobody, and green.
    """
    probed = frozenset(CARRIER_FILES) | {"requires_weights"}
    assert probed == EXPECTED_MARKERS, (
        f"markers with no selection probe: {sorted(EXPECTED_MARKERS - probed)}; "
        f"probes for unregistered markers: {sorted(probed - EXPECTED_MARKERS)}"
    )


@pytest.mark.parametrize(
    ("marker", "path", "gate_module"),
    [(marker, path, gate) for marker, (path, gate) in sorted(CARRIER_FILES.items())],
    ids=sorted(CARRIER_FILES),
)
def test_marker_selects_and_deselects_its_carrier_file(
    marker: str, path: str, gate_module: str | None
) -> None:
    """`-m <marker>` must select all of `path`, and `-m "not <marker>"` none of it.

    The deselection half is the point: a marker that deselects 0 tests is the
    "registered but carried by nobody" failure, and it is exactly what shipped
    once already.
    """
    total = _collect((path,))
    if not total:
        if gate_module is not None and importlib.util.find_spec(gate_module) is None:
            pytest.skip(
                f"{path} collected 0 tests because the optional module {gate_module!r} "
                f"is not installed -- this probe cannot discriminate here. Install the "
                f"extra to make it meaningful; it is NOT passing, it is skipped."
            )
        pytest.fail(
            f"{path} collected 0 tests, so it cannot prove anything about {marker!r}. "
            f"Pick a different carrier file in CARRIER_FILES."
        )

    selected = _collect((path,), marker)
    assert selected == total, (
        f"-m {marker!r} selected {len(selected)} of {len(total)} tests in {path}; "
        f"tests/conftest.py is not applying {marker!r} to all of them. Missing: "
        f"{sorted(total - selected)[:5]}"
    )

    remaining = _collect((path,), f"not {marker}")
    assert remaining == frozenset(), (
        f'-m "not {marker}" still collected {len(remaining)} tests from {path}, so '
        f"the marker deselects nothing there: {sorted(remaining)[:5]}"
    )


def test_a_misspelled_marker_expression_deselects_nothing() -> None:
    """CONTROL for the test above: a typo'd name must select EVERYTHING.

    This reads differently from every assertion above, which is the whole point.
    If `_collect` were broken -- wrong cwd, wrong python, stdout parsed wrongly --
    it would return the empty set here too and this test would fail, instead of
    the survivor-shaped green that a broken probe otherwise produces.

    It also pins the hazard itself: pytest offers no strictness flag that catches
    a misspelled `-m` name, so the taxonomy's correctness can only ever be
    asserted, never delegated.
    """
    path = "tests/unit/test_multi_watcher.py"
    total = _collect((path,))
    assert total, f"{path} collected nothing; this control cannot discriminate"

    typo = _collect((path,), "not integrationn")
    assert typo == total, (
        "pytest appears to have gained validation of -m expressions: "
        f'-m "not integrationn" now deselects {len(total - typo)} tests. If so, that '
        "is good news -- record the pytest version and relax this control."
    )

    correct = _collect((path,), "not requires_extra")
    assert correct == frozenset(), (
        "the correctly-spelled name failed to deselect, so the comparison above is "
        "not measuring what it claims"
    )


def test_requires_weights_is_carried_by_exactly_these_three_tests() -> None:
    """Node-level marker: explicit id set, compared both ways.

    File-level marking would be wrong here and the wrongness is the reason this
    test spells the ids out: both files also hold the portable fake-model proofs
    that must keep running on a runner with no HuggingFace cache.
    """
    selected = _collect(REQUIRES_WEIGHTS_FILES, "requires_weights")
    assert selected - EXPECTED_REQUIRES_WEIGHTS_NODE_IDS == frozenset(), (
        f"newly marked requires_weights: {sorted(selected - EXPECTED_REQUIRES_WEIGHTS_NODE_IDS)}"
    )
    assert EXPECTED_REQUIRES_WEIGHTS_NODE_IDS - selected == frozenset(), (
        f"no longer marked requires_weights: "
        f"{sorted(EXPECTED_REQUIRES_WEIGHTS_NODE_IDS - selected)}"
    )

    total = _collect(REQUIRES_WEIGHTS_FILES)
    assert len(total) > len(selected), (
        "the two weight-loading files collected nothing beyond the marked tests, so "
        "the node-level rule is indistinguishable from a file-level one -- the "
        "portable fake-model tests these files exist to protect have gone missing"
    )


def test_requires_extra_is_exactly_the_module_scope_importorskip_files() -> None:
    """Re-derive the table from the sources; a drifting table must fail, not mislead.

    "Whole file skips without the extra" IS "module-scope importorskip", so the
    table is mechanically checkable and there is no excuse for it being a
    judgement call. Adding one without updating tests/conftest.py fails here.
    """
    derived = set()
    for source in sorted(_TESTS_DIR.rglob("test_*.py")):
        if _MODULE_SCOPE_IMPORTORSKIP.search(source.read_text(encoding="utf-8")):
            derived.add(source.relative_to(_TESTS_DIR).as_posix())

    assert derived, (
        "found no module-scope pytest.importorskip anywhere under tests/, which "
        "means the regex stopped matching rather than that the suite changed"
    )
    table = set(taxonomy.REQUIRES_EXTRA_FILES)
    assert derived - table == set(), (
        f"files skip wholesale on a missing extra but carry no requires_extra marker: "
        f"{sorted(derived - table)}"
    )
    assert table - derived == set(), (
        f"listed as requires_extra but no module-scope importorskip: {sorted(table - derived)}"
    )


def test_no_taxonomy_table_entry_points_at_a_missing_file() -> None:
    """A renamed file must fail loudly instead of quietly un-marking itself."""
    listed = (
        set(taxonomy.SLOW_FILES)
        | set(taxonomy.SECURITY_FILES)
        | set(taxonomy.REQUIRES_EXTRA_FILES)
        | set(taxonomy.API_EXTRA_FILES)
        | set(taxonomy.PARSER_EXTRA_FILES)
        | {path for path, _name in taxonomy.REQUIRES_WEIGHTS_NODES}
    )
    assert len(listed) >= 40, (
        f"only {len(listed)} paths across every taxonomy table; the tables have been "
        f"emptied and every membership assertion below is vacuous"
    )
    missing = sorted(rel for rel in listed if not (_TESTS_DIR / rel).is_file())
    assert missing == [], f"taxonomy tables name files that do not exist: {missing}"


def test_pyproject_pins_the_strictness_umbrella_and_the_asyncio_loop_scope() -> None:
    """`strict` and `asyncio_default_fixture_loop_scope` must stay set explicitly.

    `strict` is what makes an unknown ini key an error rather than a warning
    nobody reads, and what makes an xfail that starts passing fail the run.
    Pinning the asyncio loop scope is what keeps `strict_config` from turning
    pytest-asyncio's "unset" warning into a hard error at every invocation.
    """
    with _PYPROJECT.open("rb") as handle:
        ini = tomllib.load(handle)["tool"]["pytest"]["ini_options"]
    assert ini.get("strict") is True, (
        "pyproject [tool.pytest.ini_options] no longer sets strict = true; "
        "strict_config, strict_markers, strict_xfail and strict_parametrization_ids "
        "are all off again"
    )
    assert ini.get("asyncio_default_fixture_loop_scope") == "function", (
        'asyncio_default_fixture_loop_scope must stay pinned to "function"; '
        "leaving it unset makes pytest-asyncio warn, and strict_config makes that warning fatal"
    )
    assert "--strict-markers" in ini.get("addopts", ""), (
        "--strict-markers dropped from addopts; keep it explicit so the flag survives "
        "someone removing the umbrella"
    )


def test_no_coverage_json_outside_scratch_pad() -> None:
    """A stray coverage.json at the repo root is stale evidence, and it has already lied.

    An untracked, gitignored coverage.json sat at the repo root long enough for two
    separate audit agents to read it as current and report a phantom finding from it.
    Coverage artefacts belong under scratch-pad/ where nobody mistakes them for the
    measurement of the run they are looking at.
    """
    strays = sorted(
        path.relative_to(_ROOT).as_posix()
        for path in _ROOT.rglob("coverage.json")
        if "scratch-pad" not in path.parts
        and ".venv" not in path.parts
        and "node_modules" not in path.parts
    )
    assert strays == [], (
        f"stale coverage artefact(s) present: {strays}. These are untracked and "
        f"gitignored, so remove them with `rm` -- NOT `git rm` -- or write them under "
        f"scratch-pad/ in the first place (`--cov-report=json:scratch-pad/coverage.json`)."
    )
