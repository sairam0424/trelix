"""`make eval-suite` must run `trelix eval-suite` with an arm and an output path, and nothing else.

WHY THIS EXISTS. The target is the documented way to run one arm of a committed suite, and
nothing in CI executes it (it clones a repository and builds an index). What it runs can only
be pinned by reading the Makefile: a recipe that drifted to `trelix eval` would run the LIVE
planner against the current checkout and print numbers that look like a suite run; a default
arm name would make two runs of the same default indistinguishable from an experiment; a
results default outside `.trelix/` could be committed by accident, and a committed results
file is exactly what a rebaseline PR exists to prevent happening silently.

The parser is copied from `test_makefile_eval_full_freezes_planner.py` on purpose: the two
files pin different targets, and a shared helper would couple them.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MAKEFILE = _ROOT / "Makefile"
_TARGET = "eval-suite"


def _recipe(target: str) -> list[str]:
    """The tab-indented recipe lines of `target`, in order."""
    lines = _MAKEFILE.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith(f"{target}:"):
            inside = True
            continue
        if inside:
            if line.startswith("\t"):
                out.append(line[1:])
            elif line.strip() == "":
                continue
            else:
                break
    return out


def test_the_recipe_parser_found_a_real_recipe() -> None:
    """Precondition for the assertions below, which look for a substring.

    Named fixture: the ``eval-suite`` target in ``Makefile``. If ``_recipe`` returned an empty
    list (renamed target, spaces instead of tabs) a substring assertion would fail for the
    wrong reason and a substring ABSENCE assertion would pass vacuously.

    NON-DISCRIMINATING COMPANION: no change to the command under test fails this test.
    """
    recipe = _recipe(_TARGET)
    assert recipe, f"no recipe found for {_TARGET}: in {_MAKEFILE}"
    assert any(line.lstrip("@").startswith("trelix eval-suite ") for line in recipe), recipe


def test_eval_suite_runs_the_suite_command_with_an_arm_and_an_out_and_never_trelix_eval() -> None:
    """MUTATIONS THAT MUST FAIL THIS TEST:
    Makefile, ``eval-suite``: ``trelix eval-suite ...`` replaced by ``trelix eval ...`` (the live
    planner against the current checkout); ``--arm '$(EVAL_ARM)'`` or ``--out '$(EVAL_RESULTS)'``
    dropped (the command would refuse, or write nowhere).
    """
    recipe = _recipe(_TARGET)
    command = [line for line in recipe if line.lstrip("@").startswith("trelix eval-suite ")]
    assert len(command) == 1, command
    assert "'$(EVAL_SUITE)'" in command[0]
    assert "--arm '$(EVAL_ARM)'" in command[0]
    assert "--out '$(EVAL_RESULTS)'" in command[0]
    assert not any(line.lstrip("@").startswith("trelix eval ") for line in recipe), recipe


def test_the_results_default_cannot_be_committed_by_accident() -> None:
    """MUTATION THAT MUST FAIL THIS TEST:
    Makefile: ``EVAL_RESULTS ?= .trelix/eval-suite/results.json`` ->
    ``EVAL_RESULTS ?= results.json`` (repo root, not gitignored).

    A results file is a run artifact. Committing one would hand the next reader a baseline
    nobody re-baselined on purpose.
    """
    text = _MAKEFILE.read_text(encoding="utf-8")
    assert "EVAL_RESULTS ?= .trelix/eval-suite/results.json" in text
    gitignore = (_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".trelix/" in [line.strip() for line in gitignore]


def test_both_variables_are_required_by_a_guard() -> None:
    """MUTATION THAT MUST FAIL THIS TEST:
    Makefile: either ``test -n`` guard removed. An unset ``EVAL_ARM`` would then reach the CLI
    as an empty option, and an unset ``EVAL_SUITE`` as an empty path.
    """
    recipe = _recipe(_TARGET)
    assert any(line.startswith("@test -n '$(EVAL_SUITE)' ||") for line in recipe), recipe
    assert any(line.startswith("@test -n '$(EVAL_ARM)' ||") for line in recipe), recipe
    assert "EVAL_SUITE ?=\n" in _MAKEFILE.read_text(encoding="utf-8")
    assert "EVAL_ARM ?=\n" in _MAKEFILE.read_text(encoding="utf-8")


def test_eval_suite_is_declared_phony() -> None:
    """MUTATION THAT MUST FAIL THIS TEST:
    Makefile: remove ``eval-suite`` from the ``.PHONY`` list. ``make`` would then skip the
    target on any tree that happens to contain a file named ``eval-suite``.
    """
    phony = [
        line
        for line in _MAKEFILE.read_text(encoding="utf-8").splitlines()
        if line.startswith(".PHONY:")
    ]
    assert len(phony) == 1, phony
    assert _TARGET in phony[0].split()
