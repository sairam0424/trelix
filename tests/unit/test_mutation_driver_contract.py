"""Contract tests for scripts/mutation.py and its checked-in survivor baseline.

WHY THIS FILE EXISTS AT ALL
---------------------------
`scripts/` is not under `testpaths`, so the driver itself is never collected. Its
failure mode is not a crash -- it is REPORTING A NUMBER IT DID NOT MEASURE: a scope
pattern that matches nothing yields a perfect score, a renamed SCOPE key orphans its
ceiling, and a baseline written under one mutmut config compared against a run under
another is two different measurements wearing one name. None of that raises.

WHY NOT ONE OF THESE ASSERTS RUNS mutmut
----------------------------------------
mutmut is deliberately absent from `[project.optional-dependencies].dev`, so CI's unit
job does not install it -- and this suite has twice been green locally and red on all
four CI legs from an assertion stronger than the environment guarantees. Every test
here reads only the driver's declarative tables, the repo's own files, and the
checked-in JSON. `test_the_driver_imports_without_mutmut_installed` is the one test
that touches the question of mutmut's presence, and it SKIPS LOUDLY, naming what was
out of reach, rather than asserting either way.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DRIVER_PATH = _ROOT / "scripts" / "mutation.py"


def _load_driver() -> object:
    """Import scripts/mutation.py by path.

    By path rather than `import scripts.mutation`: `scripts/` has no `__init__.py`, so
    the dotted form depends on implicit-namespace-package resolution against
    pyproject's `pythonpath = ["src", "."]`. A file loader depends on nothing.
    """
    spec = importlib.util.spec_from_file_location("_trelix_mutation_driver", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # MUST be registered in sys.modules BEFORE exec_module: scripts/mutation.py defines
    # a `@dataclass(frozen=True)` class, and dataclasses' `_is_type` helper (since it
    # started resolving stringified/deferred annotations) does
    # `sys.modules.get(cls.__module__).__dict__` while the class body is executing. An
    # unregistered module makes that `.get(...)` return None, and the class definition
    # itself raises `AttributeError: 'NoneType' object has no attribute '__dict__'` --
    # which is a collection error for THIS file, which aborts collection of the WHOLE
    # `tests/unit` directory (pytest treats one collection error as fatal), not merely
    # a failure of this module's own tests. Measured: `pytest tests/unit --collect-only`
    # goes from 4102 tests collected to "Interrupted: 1 error during collection" with
    # this line missing, and back to a clean collection with it present.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_driver()

# ---------------------------------------------------------------------------
# The literals. Written out here, NOT imported from the driver and NOT derived from
# the filesystem, so that this file is an independent second statement of the contract
# that the driver has to agree with.
# ---------------------------------------------------------------------------
EXPECTED_SCOPE_KEYS = frozenset(
    {
        "compression.extractive",
        "eval.ndcg",
        "graph",
        "indexing.chunker",
        "indexing.parser",
        "indexing.walker",
        "retrieval.bm25",
        "retrieval.fusion",
        "store.db",
        "store.dimension_guard",
        "store.vector",
    }
)

# One representative real source file per scope key. The point is not to re-list the
# globs; it is to prove each glob MATCHES SOMETHING THAT EXISTS. A pattern matching
# nothing is the single failure that turns this whole instrument into a rubber stamp:
# mutmut then reports "0 files mutated" and a flawless score.
EXPECTED_SCOPE_WITNESS = {
    "compression.extractive": "src/trelix/compression/extractive.py",
    "eval.ndcg": "src/trelix/eval/ndcg.py",
    "graph": "src/trelix/graph/builder.py",
    "indexing.chunker": "src/trelix/indexing/chunker.py",
    "indexing.parser": "src/trelix/indexing/parser/extractors/python.py",
    "indexing.walker": "src/trelix/indexing/walker.py",
    "retrieval.bm25": "src/trelix/retrieval/bm25.py",
    "retrieval.fusion": "src/trelix/retrieval/fusion.py",
    "store.db": "src/trelix/store/db.py",
    "store.dimension_guard": "src/trelix/store/dimension_guard.py",
    "store.vector": "src/trelix/store/vector.py",
}

# A file that is NOT in each scope, so the glob is shown to DISCRIMINATE rather than
# just to match. `indexing.parser`'s pattern is a subtree glob and fnmatch's "*"
# crosses "/", so without this half `src/trelix/indexing/parser/*` and
# `src/trelix/indexing/*` would be indistinguishable here.
EXPECTED_SCOPE_NON_WITNESS = {
    "compression.extractive": "src/trelix/eval/ndcg.py",
    "eval.ndcg": "src/trelix/retrieval/bm25.py",
    "graph": "src/trelix/store/db.py",
    "indexing.chunker": "src/trelix/indexing/walker.py",
    "indexing.parser": "src/trelix/indexing/chunker.py",
    "indexing.walker": "src/trelix/indexing/chunker.py",
    "retrieval.bm25": "src/trelix/retrieval/fusion.py",
    "retrieval.fusion": "src/trelix/retrieval/bm25.py",
    "store.db": "src/trelix/store/vector.py",
    "store.dimension_guard": "src/trelix/store/db.py",
    "store.vector": "src/trelix/store/db.py",
}

# mutmut/__main__.py:82. The verdicts whose classification decides whether a mutant is
# a survivor, a coverage gap, or unmeasurable. Transcribed, never imported.
EXPECTED_STATUS_BY_EXIT_CODE = {
    None: "not_checked",
    0: "survived",
    1: "killed",
    3: "killed",
    5: "no_tests",
    33: "no_tests",
    34: "skipped",
    36: "timeout",
    37: "caught_by_type_check",
    -11: "segfault",
    -9: "segfault",
}


class TestScopeTable:
    def test_scope_keys_are_exactly_the_expected_set(self) -> None:
        """MUTATION: rename or delete any key of scripts/mutation.py's SCOPE.

        Set equality both ways. A renamed key silently orphans that module's recorded
        ceiling -- `--check` then reports "measured now, but the baseline has no
        measurement to ratchet against" for the new name while the old name's ceiling
        rots in the JSON, and nobody is watching either.
        """
        actual = frozenset(driver.SCOPE)
        assert actual - EXPECTED_SCOPE_KEYS == frozenset(), (
            f"SCOPE has keys this contract does not know about: "
            f"{sorted(actual - EXPECTED_SCOPE_KEYS)}. Add them here AND to "
            f"EXPECTED_SCOPE_WITNESS/EXPECTED_SCOPE_NON_WITNESS."
        )
        assert EXPECTED_SCOPE_KEYS - actual == frozenset(), (
            f"SCOPE lost keys: {sorted(EXPECTED_SCOPE_KEYS - actual)}"
        )

    def test_every_scope_key_has_both_a_witness_and_a_non_witness(self) -> None:
        """MUTATION: delete an entry from EXPECTED_SCOPE_WITNESS.

        Without this, adding a SCOPE key and forgetting its witness would leave that
        scope's glob completely unverified -- present, possibly matching nothing, and
        green. This is the guard on the guard.
        """
        assert frozenset(EXPECTED_SCOPE_WITNESS) == EXPECTED_SCOPE_KEYS
        assert frozenset(EXPECTED_SCOPE_NON_WITNESS) == EXPECTED_SCOPE_KEYS

    @pytest.mark.parametrize("module", sorted(EXPECTED_SCOPE_KEYS))
    def test_scope_pattern_matches_a_real_file_and_excludes_another(self, module: str) -> None:
        """MUTATION: typo any SCOPE pattern, e.g. `parser/*` -> `parsers/*`.

        Both halves matter. The witness half catches a pattern that matches nothing,
        which is what makes mutmut report a perfect score over an empty set. The
        non-witness half catches a pattern that has been widened past its module, which
        would make one module's ceiling silently absorb another's mutants.

        fnmatch, not glob, and not pathlib.match: this is the exact predicate mutmut
        applies in Config._should_include_for_mutation (configuration.py:216).
        """
        patterns = driver.SCOPE[module]
        witness = EXPECTED_SCOPE_WITNESS[module]
        non_witness = EXPECTED_SCOPE_NON_WITNESS[module]

        assert (_ROOT / witness).is_file(), (
            f"the witness file {witness} no longer exists, so this test cannot tell a "
            f"working pattern from a broken one. Pick another file in {module}."
        )
        assert (_ROOT / non_witness).is_file(), (
            f"the non-witness file {non_witness} no longer exists; this half of the "
            f"assertion has stopped discriminating"
        )
        assert any(fnmatch.fnmatch(witness, p) for p in patterns), (
            f"{module}: none of {list(patterns)} matches {witness}. mutmut would "
            f"report '0 files mutated' and a flawless score."
        )
        assert not any(fnmatch.fnmatch(non_witness, p) for p in patterns), (
            f"{module}: {list(patterns)} also matches {non_witness}, which belongs to "
            f"another scope key"
        )

    @pytest.mark.parametrize("module", sorted(EXPECTED_SCOPE_KEYS))
    def test_scope_pattern_is_shaped_the_way_mutmut_requires(self, module: str) -> None:
        """MUTATION: drop the trailing `*` from `src/trelix/graph/*`.

        mutmut's _load_config (configuration.py:119) accepts only patterns ending in
        ".py" or "*" and merely WARNS about the rest -- and a warning in a run whose
        stdout is a spinner is a warning nobody reads. A bare directory pattern
        matches no file, so the scope is empty and the score is perfect.
        """
        for pattern in driver.SCOPE[module]:
            assert pattern.endswith((".py", "*")), (
                f"{module}: pattern {pattern!r} ends in neither '.py' nor '*'; mutmut "
                f"only warns about this and then mutates nothing"
            )

    def test_excluded_scopes_do_not_overlap_the_measured_scope(self) -> None:
        """MUTATION: add "graph" to EXCLUDED_SCOPES.

        EXCLUDED_SCOPES is documentation of a hard limit (torch model construction
        segfaults 245 of 317 mutants). A key appearing in both tables would mean the
        driver measures something it also declares unmeasurable.
        """
        overlap = frozenset(driver.EXCLUDED_SCOPES) & frozenset(driver.SCOPE)
        assert overlap == frozenset(), f"declared both measured and excluded: {sorted(overlap)}"
        assert driver.EXCLUDED_SCOPES, (
            "EXCLUDED_SCOPES is empty; the torch-segfault limit has stopped being "
            "recorded and the next reader will treat the scope as complete"
        )


class TestVerdictClassification:
    def test_exit_code_statuses_match_mutmuts_own_table(self) -> None:
        """MUTATION: in scripts/mutation.py, map exit code 0 to "killed".

        Exit code 0 means the tests PASSED with the mutant applied, i.e. the mutant
        survived. Getting this one backwards inverts every number in the baseline
        while every assertion about "files mutated" still passes.
        """
        for exit_code, status in EXPECTED_STATUS_BY_EXIT_CODE.items():
            assert driver.STATUS_BY_EXIT_CODE[exit_code] == status, (
                f"exit code {exit_code} classified as "
                f"{driver.STATUS_BY_EXIT_CODE[exit_code]!r}, expected {status!r}"
            )

    def test_unknown_exit_codes_are_suspicious_not_killed(self) -> None:
        """MUTATION: set UNKNOWN_STATUS to "killed".

        An exit code the table does not know about must never be counted as a kill;
        that is how a crashing runner turns into a perfect score.
        """
        assert driver.UNKNOWN_STATUS == "suspicious"

    def test_ceilings_are_kept_for_survived_and_no_tests_separately(self) -> None:
        """MUTATION: CEILING_KEYS = ("survived",).

        `survived` and `no_tests` have different fixes -- a test that ran and did not
        notice, versus no test reaching the code at all -- so folding them together
        loses the actionable half. Dropping `no_tests` entirely would let a module's
        unreachable-code count grow without limit, which matters MORE here because
        mutate_only_covered_lines is off (see the next test).
        """
        assert driver.CEILING_KEYS == ("survived", "no_tests")


class TestPinnedMutmutConfig:
    def test_covered_lines_filter_is_off_with_its_reason_recorded(self) -> None:
        """NON-DISCRIMINATING COMPANION for the flag; a REAL assertion for the reason.

        No mutation of the driver makes the boolean itself wrong -- either value is a
        legitimate configuration. What must not happen silently is the value changing
        without the measured reason chain (numpy ImportError -> beartype circular
        import -> torch SIGSEGV -> lancedb pyo3 SetLoggerError) going with it, because
        that chain is the only reason anyone would know not to just flip it back.
        """
        assert driver.MUTMUT_CONFIG["mutate_only_covered_lines"] is False
        source = _DRIVER_PATH.read_text(encoding="utf-8")
        evidence_strings = (
            "Segmentation fault",
            "SetLoggerError",
            "cannot load module more than once",
        )
        for evidence in evidence_strings:
            assert evidence in source, (
                f"the measured reason {evidence!r} for disabling "
                f"mutate_only_covered_lines is no longer written down in the driver"
            )

    def test_do_not_mutate_patterns_cover_raises_and_logging(self) -> None:
        """MUTATION: empty do_not_mutate_patterns.

        Mutating an exception TYPE swaps one failure for another, so a test asserting
        "this raises" passes either way; log lines are asserted on nowhere. Both
        classes are pure noise that buries the real survivors.
        """
        assert driver.MUTMUT_CONFIG["do_not_mutate_patterns"] == [r"raise \w+", r"logger\.\w+"]

    def test_max_stack_depth_is_pinned_to_a_finite_value(self) -> None:
        """MUTATION: max_stack_depth = -1.

        -1 is mutmut's "unlimited", which attributes a mutant to every test anywhere
        below it on the stack. The recorded counts are only comparable against a run
        at the SAME depth, which is why the value is written into the baseline JSON.
        """
        depth = driver.MUTMUT_CONFIG["max_stack_depth"]
        assert isinstance(depth, int)
        assert depth > 0, "max_stack_depth must be finite and recorded, not -1"

    def test_test_selection_is_the_whole_unit_suite(self) -> None:
        """MUTATION: pytest_add_cli_args_test_selection = ["tests/unit/test_ndcg_known_answers.py"].

        A mutant "survives" only if NO test kills it. Narrowing the selection is the
        cheapest way to manufacture survivors, and it leaves no trace in the output.
        """
        assert driver.MUTMUT_CONFIG["pytest_add_cli_args_test_selection"] == ["tests/unit"]

    def test_every_deselected_file_still_exists(self) -> None:
        """MUTATION: rename one entry of DESELECTED_FILES.

        A `--ignore` naming a path that no longer exists does not error -- it silently
        ignores nothing. For these four that means the OTel global-provider crash
        ("Overriding of current MeterProvider is not allowed" -> "Failed to run clean
        test") comes back and every mutant is abandoned as `not checked`.
        """
        assert driver.DESELECTED_FILES, "the deselection list is empty"
        missing = [path for path in driver.DESELECTED_FILES if not (_ROOT / path).is_file()]
        assert missing == [], f"--ignore names files that do not exist: {missing}"

    def test_deselection_is_a_handful_of_files_not_a_wholesale_narrowing(self) -> None:
        """MUTATION: add every slow test file to DESELECTED_FILES.

        There is no principled threshold here, so this pins the ORDER OF MAGNITUDE:
        the deselection exists for tests that cannot run twice in one process, and
        that is a handful. If it ever needs to be dozens, the right response is to
        stop calling the result a survivor count.
        """
        assert len(driver.DESELECTED_FILES) <= 8, (
            f"{len(driver.DESELECTED_FILES)} files removed from the kill set; a "
            f"survivor count measured against a suite this narrowed is not a survivor "
            f"count"
        )

    def test_effective_config_splices_the_deselections_into_pytest_args(self) -> None:
        """MUTATION: make effective_mutmut_config return MUTMUT_CONFIG unchanged.

        The deselections only take effect because they are spliced into
        pytest_add_cli_args at render time. If that splice is dropped, the config
        written into the throwaway tree and the config recorded in the baseline are
        both wrong in the same direction, so they still agree with each other.
        """
        config = driver.effective_mutmut_config(["src/trelix/eval/ndcg.py"])
        args = config["pytest_add_cli_args"]
        assert config["only_mutate"] == ["src/trelix/eval/ndcg.py"]
        for path in driver.DESELECTED_FILES:
            assert "--ignore" in args and path in args, (
                f"{path} is in DESELECTED_FILES but not in the rendered pytest args"
            )


class TestCheckedInBaseline:
    """The JSON is the artifact everything else exists to protect."""

    @staticmethod
    def _baseline() -> dict:
        path = _ROOT / "scripts" / "mutation_baseline.json"
        assert path.is_file(), (
            f"{path} is missing. It is the checked-in survivor baseline; without it "
            f"`--check` has nothing to ratchet against."
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_baseline_module_keys_match_the_scope_table(self) -> None:
        """MUTATION: delete a module entry from scripts/mutation_baseline.json.

        Both directions. A scope with no baseline entry cannot be ratcheted; a
        baseline entry with no scope is a ceiling for a module that is no longer
        measured, and it will sit there looking like coverage.
        """
        modules = frozenset(self._baseline()["modules"])
        assert modules - EXPECTED_SCOPE_KEYS == frozenset(), (
            f"baseline names modules outside SCOPE: "
            f"{sorted(modules - EXPECTED_SCOPE_KEYS)}. A ceiling for a module nobody "
            f"measures any more reads as coverage."
        )
        assert EXPECTED_SCOPE_KEYS - modules == frozenset(), (
            f"SCOPE keys absent from the baseline: {sorted(EXPECTED_SCOPE_KEYS - modules)}"
        )

    def test_at_least_one_module_is_actually_measured(self) -> None:
        """MUTATION: set every module's "measured" to false.

        An all-unmeasured baseline satisfies every structural assertion above while
        recording nothing. This is the anti-vacuous guard on the file itself.
        """
        measured = sorted(
            name for name, rec in self._baseline()["modules"].items() if rec.get("measured")
        )
        assert measured, (
            "no module in the baseline is marked measured, so every ceiling below is "
            "absent and `--check` can never fail"
        )

    def test_unmeasured_modules_carry_no_ceilings(self) -> None:
        """MUTATION: give an unmeasured module `"ceilings": {"survived": 0}`.

        A ceiling of 0 on a module nobody measured is a perfect score for free, and it
        is indistinguishable from a real one once it is in the file. Absence has to
        stay absence.
        """
        for name, rec in sorted(self._baseline()["modules"].items()):
            if not rec.get("measured"):
                assert "ceilings" not in rec, (
                    f"{name} is not measured but carries ceilings {rec['ceilings']}; "
                    f"that is a free perfect score"
                )
                assert "counts" not in rec, f"{name} is not measured but carries counts"

    def test_measured_modules_carry_non_vacuous_evidence(self) -> None:
        """MUTATION: record a measured module with files_mutated = 0.

        Every measured entry has to prove it measured something: files mutated,
        mutants generated, and a sha256 for each source file, so the tree the numbers
        came from is never in doubt. Two handed-down figures for this repo were both
        from an earlier tree.
        """
        for name, rec in sorted(self._baseline()["modules"].items()):
            if not rec.get("measured"):
                continue
            assert rec["files_mutated"] > 0, f"{name}: measured with 0 files mutated"
            assert rec["total_mutants"] > 0, f"{name}: measured with 0 mutants"
            assert rec["source_sha256"], f"{name}: no per-file sha256 recorded"
            for key in driver.CEILING_KEYS:
                ceiling = rec["ceilings"][key]
                assert isinstance(ceiling, int) and ceiling >= 0, (
                    f"{name}: {key} ceiling {ceiling!r} is not a count"
                )
            assert sum(rec["counts"].values()) == rec["total_mutants"], (
                f"{name}: per-status counts do not sum to total_mutants"
            )
            assert rec["counts"].get("not_checked", 0) < rec["total_mutants"], (
                f"{name}: every mutant is 'not checked', so mutmut ran none of them"
            )

    def test_baseline_records_the_config_it_was_measured_under(self) -> None:
        """MUTATION: change max_stack_depth in the driver without re-measuring.

        A count is only comparable against a run under the same config. Recording the
        config alongside it turns "these two numbers disagree" from a mystery into a
        diff. Comparing to the DRIVER's live table is the point: this fails the moment
        the driver changes and the baseline is not refreshed.
        """
        baseline = self._baseline()
        assert baseline["schema"] == 1
        recorded = baseline["mutmut_config"]
        for key, value in driver.MUTMUT_CONFIG.items():
            assert recorded[key] == value, (
                f"baseline was measured with {key}={recorded.get(key)!r} but the driver "
                f"now says {value!r}. Re-run `--update-baseline` or revert the driver; "
                f"do not compare the two."
            )

    def test_baseline_states_that_no_ratio_is_gated_on(self) -> None:
        """NON-DISCRIMINATING COMPANION -- it pins a string, and I am saying so.

        No mutation of the driver's logic makes this fail. It is here because the one
        thing every future reader will reach for is a mutation-score percentage, and
        the file has to say in its own body that there is deliberately none: Stryker
        ships break:null, Google rejected the aggregate at 16.9M mutants, and 77% of
        one scope here is unmeasurable.
        """
        gate = self._baseline()["gate"]
        assert "never a score ratio" in gate, (
            f"the baseline's own `gate` field no longer says what it does and does not "
            f"gate on; it currently reads {gate!r}"
        )


def test_the_driver_imports_without_mutmut_installed() -> None:
    """The driver must be importable, and `--list-scope` runnable, with no mutmut.

    CI's unit job installs [local,otel,sso,dev] and mutmut is in none of them, so an
    import-time `import mutmut` would make this whole file a collection error there
    while staying green in a venv where someone sideloaded it. That is the exact shape
    of the two commits that were green locally and red on all four CI legs.

    SKIPS LOUDLY where it cannot discriminate: if mutmut IS importable in this
    interpreter the subprocess would succeed for the wrong reason, so the test says so
    instead of passing.
    """
    if importlib.util.find_spec("mutmut") is not None:
        pytest.skip(
            "mutmut is importable in this interpreter, so this test cannot prove the "
            "driver works without it. Re-run in an interpreter with no mutmut (which "
            "is what CI's unit job has)."
        )
    proc = subprocess.run(
        [sys.executable, str(_DRIVER_PATH), "--list-scope"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"`mutation.py --list-scope` exited {proc.returncode} with no mutmut "
        f"installed.\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    for module in sorted(EXPECTED_SCOPE_KEYS):
        assert module in proc.stdout, f"--list-scope did not print {module}"
