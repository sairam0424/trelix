"""`make eval-full` must freeze the query planner, or its numbers are not comparable.

WHY THIS EXISTS. ``eval/README.md`` documents the instrument's noise floor, measured on
this golden set: nDCG@10 run-to-run **sd 0.02202** with a live planner (both caches off,
n=5) and **sd 0.02872** through the shipped CLI (n=3), because **0 of 54 plans reproduce
byte-for-byte** at temperature=0.0. Replayed from frozen plans the same pipeline reported
**sd exactly 0.000000** over six runs. ``make eval-full`` is the command the repo tells
people to run for a full self-eval, and it invoked ``trelix eval`` with no
``--plan-cache-file`` — so every A/B smaller than ~0.03 nDCG@10 taken through that target
measured the planner, not the change under test.

The flag's machinery (record once, replay with no planner LLM call, RAISE on a query the
file does not cover) is covered by ``tests/unit/test_planner_determinism.py`` and
``tests/unit/test_planner_llm_call_count.py``. This file pins only that the documented
entry point actually passes it, which no other test can see: the target spends embedding
and planner API calls, so nothing in CI executes it.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MAKEFILE = _ROOT / "Makefile"
_TARGET = "eval-full"


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
    """Precondition (rule 4) for the assertions below, which look for a substring.

    Named fixture: the ``eval-full`` target in ``Makefile``. If ``_recipe`` returned an
    empty list — renamed target, spaces instead of tabs — a substring assertion would
    fail for the wrong reason and a substring ABSENCE assertion would pass vacuously.

    NON-DISCRIMINATING COMPANION: no change to the flags under test fails this test.
    """
    recipe = _recipe(_TARGET)
    assert recipe, f"no recipe found for {_TARGET}: in {_MAKEFILE}"
    assert any(line.lstrip("@").startswith("trelix eval ") for line in recipe), recipe


def test_eval_full_passes_plan_cache_file_to_trelix_eval() -> None:
    """MUTATION THAT MUST FAIL THIS TEST:
    Makefile, ``eval-full``: drop ``--plan-cache-file '$(EVAL_PLAN_CACHE)'`` from the
    ``trelix eval`` line. That is the state this test was written against, and it made
    every measurement taken through the target carry an sd of 0.022-0.029 nDCG@10.
    """
    command = [line for line in _recipe(_TARGET) if line.lstrip("@").startswith("trelix eval ")]
    assert len(command) == 1, command
    assert "--plan-cache-file" in command[0]
    assert "--golden eval/golden.jsonl" in command[0]


def test_the_plan_cache_default_cannot_be_committed_by_accident() -> None:
    """MUTATION THAT MUST FAIL THIS TEST:
    Makefile: ``EVAL_PLAN_CACHE ?= .trelix/eval-plan-cache.jsonl`` ->
    ``EVAL_PLAN_CACHE ?= eval-plan-cache.jsonl`` (repo root, not gitignored).

    A recorded plan cache is a 54-line snapshot of one golden set against one index.
    Committing it would hand the next reader a file that silently RAISES on any query it
    does not cover — and pairing it with an edited golden set is exactly the half-frozen
    run the freeze exists to prevent.
    """
    text = _MAKEFILE.read_text(encoding="utf-8")
    assert "EVAL_PLAN_CACHE ?= .trelix/eval-plan-cache.jsonl" in text
    gitignore = (_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".trelix/" in [line.strip() for line in gitignore]


def test_eval_full_is_declared_phony() -> None:
    """MUTATION THAT MUST FAIL THIS TEST:
    Makefile: remove ``eval-full`` from the ``.PHONY`` list. ``make`` would then skip the
    target entirely on any tree that happens to contain a file named ``eval-full``.
    """
    phony = [
        line
        for line in _MAKEFILE.read_text(encoding="utf-8").splitlines()
        if line.startswith(".PHONY:")
    ]
    assert len(phony) == 1, phony
    assert _TARGET in phony[0].split()
