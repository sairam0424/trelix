#!/usr/bin/env python3
"""Scoped mutation-testing driver for trelix. A REPORT, not a gate.

WHAT THIS IS
------------
`mutmut run` over a fork-safe subset of `src/trelix`, with the survivor COUNT
per module written to a checked-in JSON baseline (``scripts/mutation_baseline.json``).
`--check` ratchets each module against its own recorded survivor CEILING.

WHY NO MUTATION-SCORE RATIO IS GATED ON, ANYWHERE
-------------------------------------------------
Measured reasons, not stylistic ones:

* Stryker, the most mature mutation-testing ecosystem, ships ``break: null`` by
  default and documents 80/60 as *reporting* bands, not gates. trelix's measured
  score sits below Stryker's own "low" band, so any imported threshold fails on
  day one in every scope.
* Google, running mutation testing at ~2B LOC / 16.9M mutants, explicitly rejected
  the aggregate score as "neither concrete nor actionable".
* Part of this repo is structurally unmeasurable (see EXCLUDED_SCOPES), so a
  repo-wide ratio would be a ratio over whichever subset happened not to crash —
  a number that moves when an unrelated segfault appears or disappears.

So: a per-module survivor ceiling, ratcheted down as tests land. Never a ratio.
`--check` compares only ABSOLUTE COUNTS, and only for modules it actually measured
in the same invocation.

THE FOUR NON-NEGOTIABLES, EACH FROM A MEASURED TRAP
---------------------------------------------------
1. THROWAWAY WORKTREE. Unscoped mutant generation writes ~143 MB of Python, and
   ``mutants/`` is absent from the repo's 72-line .gitignore. This is a repo where
   hatchling's refusal to honour NESTED .gitignore files came one ``twine upload``
   from publishing a 221 MB index.db (see the sdist ``exclude`` block in
   pyproject.toml). So generation never happens in a tree anyone commits from:
   `_ThrowawayTree` makes a detached `git worktree`, rsyncs the LIVE tree into it,
   runs there, and removes it.
2. ``mutants/`` IS DELETED ON EVERY RUN. Changing ``only_mutate`` does NOT
   invalidate mutmut's cache. A stale cache prints "0 files mutated, 141 ignored"
   and then "the selected tests do not cover any code that we mutated", which reads
   as "my tests are bad" and is nothing of the kind. There is no ``--keep-mutants``
   escape hatch on purpose.
3. "N FILES MUTATED > 0" IS ASSERTED, per module, twice over — once from mutmut's
   own generation line and once from the ``.meta`` files on disk. Without it a
   scope pattern that matches nothing reports a perfect score.
4. THE CONFIG IS PINNED AND RECORDED. `MUTMUT_CONFIG` below is written into the
   throwaway tree AND copied verbatim into the baseline JSON, so a result can never
   be read without the settings that produced it.

A HANDED-DOWN DESCRIPTION CORRECTED HERE
----------------------------------------
``mutate_only_covered_lines`` is NOT "fed from a coverage json". Verified against
mutmut 3.7.0's installed source: ``mutmut/__main__.py:280`` calls
``gather_coverage()``, which at ``mutmut/code_coverage.py:44`` builds its own
``coverage.Coverage(data_file=None)`` and runs the selected tests in-process. There
is no coverage file input, and supplying one would do nothing.

INSTALLING mutmut
-----------------
mutmut is deliberately NOT in ``[project.optional-dependencies].dev``: CI's unit
job would then install it on every run for a tool CI never invokes, and this
driver must not assert a package the leaner CI install does not guarantee. Install
it out-of-tree and point the driver at it::

    uv pip install --python .venv/bin/python --target .mutmut-tools \
        mutmut==3.7.0 libcst textual setproctitle linkify-it-py uc-micro-py mdit-py-plugins
    TRELIX_MUTMUT_PATH=$PWD/.mutmut-tools python scripts/mutation.py --report --modules eval.ndcg

``--target`` with ``--no-deps`` for what the venv already has, rather than a plain
install: mutmut pulls ``click==8.4.2`` where this venv pins 8.1.8, and shadowing
click would break typer, i.e. the CLI under test.

USAGE
-----
    python scripts/mutation.py --list-scope
    python scripts/mutation.py --report --modules eval.ndcg retrieval.fusion
    python scripts/mutation.py --report --modules ALL          # hours
    python scripts/mutation.py --update-baseline --modules eval.ndcg
    python scripts/mutation.py --check --modules eval.ndcg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "scripts" / "mutation_baseline.json"
BASELINE_SCHEMA = 1

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
# fnmatch, NOT glob: mutmut's Config._should_include_for_mutation calls
# fnmatch.fnmatch(path_str, pattern), and fnmatch's "*" DOES cross "/". So one
# trailing "*" covers a whole subtree and "**" is meaningless here.
#
# Keys are the RATCHET UNIT. Rename one and its ceiling is orphaned, which
# tests/unit/test_mutation_driver_contract.py turns into a failure.
SCOPE: dict[str, tuple[str, ...]] = {
    "indexing.parser": ("src/trelix/indexing/parser/*",),
    "indexing.chunker": ("src/trelix/indexing/chunker.py",),
    "indexing.walker": ("src/trelix/indexing/walker.py",),
    "retrieval.fusion": ("src/trelix/retrieval/fusion.py",),
    "retrieval.bm25": ("src/trelix/retrieval/bm25.py",),
    "eval.ndcg": ("src/trelix/eval/ndcg.py",),
    "store.db": ("src/trelix/store/db.py",),
    "store.vector": ("src/trelix/store/vector.py",),
    "store.dimension_guard": ("src/trelix/store/dimension_guard.py",),
    "graph": ("src/trelix/graph/*",),
    "compression.extractive": ("src/trelix/compression/extractive.py",),
}

# EXPLICITLY EXCLUDED, AND NOT A TODO -- A HARD LIMIT OF THE INSTRUMENT.
#
# Any scope whose tests construct a torch model is unmeasurable with mutmut here.
# 245 of 317 mutants SEGFAULT inside torch/nn/modules/module.py's `convert`, and
# they do so IDENTICALLY at --max-children 1 and at --max-children 8, which rules
# out the usual fork/thread contention explanation. The `http_proxy=` workaround
# that gets proposed for this moved the count from 245 to 245.
#
# A segfaulted mutant is neither killed nor survived; it is unclassified. Folding
# 77% of a scope's mutants into "unknown" and then dividing is how you get a
# mutation score that means nothing -- a second reason the ratio is not gated.
EXCLUDED_SCOPES: dict[str, str] = {
    "embedder.*": (
        "tests construct a real torch model; 245/317 mutants segfault in "
        "torch/nn/modules/module.py convert, identically at --max-children 1 and 8. "
        "http_proxy= workaround moved 245 -> 245. Hard limit, not a todo."
    ),
    "rerank.*": "same torch-model construction as embedder.*",
    "sparse.*": "loads a transformers model; same segfault class as embedder.*",
}

# Pinned mutmut settings. Written into the throwaway tree's pyproject.toml as
# [tool.mutmut] and copied verbatim into the baseline JSON.
#
# [tool.mutmut] rather than setup.cfg: mutmut's _config_reader() (configuration.py:20)
# prefers pyproject.toml's [tool.mutmut] and only falls back to setup.cfg's [mutmut]
# when that table is absent. It is a NEW TOP-LEVEL TABLE, so pytest's
# `strict = true` (== strict_config, which rejects unknown keys in
# [tool.pytest.ini_options]) is not involved.
MUTMUT_CONFIG: dict[str, object] = {
    "source_paths": ["src/trelix"],
    # Two categories mutmut cannot judge, so it should not try:
    #   raise \w+     -- mutating an exception TYPE swaps one failure for another
    #                    failure; the test asserting "this raises" passes either way.
    #   logger\.\w+   -- log lines are not asserted on, so every mutant survives and
    #                    buries the real survivors in noise.
    "do_not_mutate_patterns": [r"raise \w+", r"logger\.\w+"],
    # OFF, and this is the second HARD LIMIT OF THE INSTRUMENT on this codebase --
    # every step below was measured on this tree, in this order, not reasoned about.
    #
    # The flag makes mutmut run the whole selected suite an EXTRA time, IN-PROCESS,
    # and then call _unload_modules_not_in() (code_coverage.py:58) to pop every module
    # that run imported. Re-importing a C extension in the same process after popping
    # it is not generally survivable, and trelix's unit suite loads 22 of them:
    #   1. numpy      -> ImportError: cannot load module more than once per process
    #   2. beartype   -> ImportError: cannot import name 'claw_state' from partially
    #                    initialized module (its sys.meta_path hook outlives its own
    #                    popped modules), triggered by the lazy import inside Pool()
    #   3. torch      -> Fatal Python error: Segmentation fault, torch/__init__.py:444
    #                    in importlib create_module, reached via transformers
    # Each is fixable by pre-importing it in the parent so it lands in gather_coverage's
    # sys.modules snapshot and is never popped -- see _PREIMPORT. And doing all three
    # then produces:
    #   4. lancedb    -> pyo3_runtime.PanicException: env_logger::init_from_env should
    #                    not be called after logger initialized: SetLoggerError(())
    #                    because tokenizers/safetensors, pre-imported in step 3, claim
    #                    the process-global Rust `log` singleton first.
    # That is the wall. The pre-import fix and the flag are in direct conflict: the
    # flag needs modules imported EARLY, and pyo3 global singletons require them
    # imported in the order the suite itself would. There is no ordering that
    # satisfies both, so this is stated as a limit and not left as a todo.
    #
    # Cost of turning it off, stated plainly: mutants are generated on uncovered lines
    # too, so counts include mutants no test could ever reach. Those land in the
    # `no_tests` verdict, which the baseline tracks under its OWN ceiling and never
    # folds into `survived`. The measurement stays honest; it just also measures
    # coverage gaps, and says which is which.
    #
    # Everything the flag needs is still wired and still correct (_PREIMPORT, the
    # COVERAGE_RCFILE with its absolute source path). Flip this to True and the driver
    # applies all of it -- it will get as far as lancedb.
    "mutate_only_covered_lines": False,
    # Pinned, not left at mutmut's -1 ("unlimited"). -1 attributes a mutant to every
    # test anywhere below it on the stack, which for store/db.py means most of the
    # suite and a per-mutant cost that makes the run unfinishable. 8 is recorded here
    # so a future run that changes it cannot be compared against these numbers
    # without the difference being visible.
    "max_stack_depth": 8,
    # DESELECTED_FILES below is spliced in at render time; see its comment.
    "pytest_add_cli_args": ["-p", "no:cacheprovider"],
    # The hermetic unit suite, WHOLE, and not narrowed by a single -k or --ignore.
    # Narrowing this is what turns a "survivor" into a lie: a mutant survives only if
    # NO test kills it. Making the whole suite runnable inside mutants/ is what
    # REPO_ARTIFACTS below is for.
    "pytest_add_cli_args_test_selection": ["tests/unit"],
    # mutmut's own also_copy already covers tests/, pyproject.toml, setup.cfg and the
    # lock files (configuration.py:129). Everything here is in addition, and every
    # entry is load-bearing for a MEASURED reason:
    #
    # mutmut's PytestRunner hardcodes `-x` (__main__.py:453), so ONE failing test
    # inside mutants/ aborts the coverage pass, the stats pass, and the whole
    # measurement. Six files in tests/unit read repo artifacts through
    # `Path(__file__).resolve().parents[2]`, which inside the sandbox is `mutants/`:
    #   test_release_version_gate.py        40 failures  CHANGELOG/helm/packages/.github
    #   test_docs_version_claims.py          7           docs/*.md, SECURITY.md
    #   test_dotenv_anchoring.py             2           .github/workflows
    #   test_readme_install_commands.py      1           README.md
    #   test_watch_startup_index_disclosure  1           docs/PROVIDERS.md
    #   test_dry_run.py                      1           scripts/*.sh
    # 52 failures total, measured, and the first one alphabetically
    # (test_docs_version_claims) is what aborted the run: eval/ndcg.py's function
    # bodies were never executed, so `mutate_only_covered_lines` saw only the two
    # `def ... k: int = 10` signature lines and generated 2 mutants instead of 26.
    # A clean-looking report, produced by a missing markdown file.
    #
    # A SEVENTH file joined this list in round 8, found the same way: a real,
    # non-mutation failure inside mutmut's own stats pass.
    #   test_makefile_eval_full_freezes_planner.py  1   .gitignore
    # (``_ROOT / ".gitignore"``, same ``parents[2]`` pattern as the six above --
    # ``.gitignore`` reaches the tree root via the initial `git worktree add` checkout,
    # but was never in `also_copy`, so mutmut's own copy into `mutants/` omits it.)
    #
    # The alternative -- deselecting those files -- was rejected: it shrinks the
    # kill set, which is the one thing a survivor count must not do.
    "also_copy": [
        ".env.example",
        ".gitignore",
        ".github",
        "CHANGELOG.md",
        "config",
        "CONTRIBUTING.md",
        "docs",
        "helm",
        "LICENSE",
        "Makefile",
        "packages",
        "README.md",
        "scripts",
        "SECURITY.md",
        "SUPPORT.md",
    ],
    # mutmut's git change detection reads HEAD of the throwaway worktree, which is a
    # commit that never matches the rsynced live tree. Off, so it cannot silently
    # reuse verdicts across a scope change.
    "use_git_change_detection": False,
}

# mutmut/__main__.py:82 status_by_exit_code, transcribed as a LITERAL rather than
# imported: importing it would recompute the expected mapping from the code under
# measurement, and a renamed status would then silently agree with itself.
# THE ONLY TESTS REMOVED FROM THE KILL SET, and the count is deliberately tiny.
#
# mutmut runs pytest MULTIPLE TIMES IN ONE PROCESS -- a stats pass, then a "clean
# tests" pass, then per-mutant passes -- and a SESSION-scoped fixture that mutates
# process-global state is only correct for one session per process. These files install
# OTel's global MeterProvider/TracerProvider, and OTel's own contract is that it can be
# set once per process; a second set_meter_provider() is a logged no-op. Measured:
#   tests/unit/test_otel_metrics.py::TestEnabledRecordsCounters
#     ::test_openai_embed_counts_one_request_per_api_call
#     AssertionError: assert (0.0 - 0.0) == 2
#     WARNING opentelemetry.metrics._internal: Overriding of current MeterProvider is
#     not allowed
#   -> mutmut printed "Failed to run clean test" and abandoned the run (its pytest
#      args hardcode -x), leaving all 52 generated mutants "not checked".
# test_otel_metrics.py:188-204 already writes the mechanism down in its own comment:
# "OTel's global MeterProvider can be set once per process ... scope=session, NOT
# module". The tests are right; mutmut's one-process-many-sessions model is what
# they cannot satisfy.
#
# HONEST CAVEAT, recorded in the baseline JSON as well as here: a mutant reported
# `survived` was not offered to these files. They assert on OTel counter and span
# values, so they cannot distinguish a mutation in the SCOPE modules -- but that is an
# argument, not a measurement, and it is the one place where a survivor count here is
# weaker than "no test in tests/unit kills it".
DESELECTED_FILES: tuple[str, ...] = (
    "tests/unit/test_otel_metrics.py",
    "tests/unit/test_otel_metrics_reentry.py",
    "tests/unit/test_otel_tracing.py",
    "tests/unit/test_structured_logging.py",
)
# ROUND 8 (mutation:widen-scope retry), NOT reproduced from this tuple -- reverted
# back to the 4 above after the run that needed it, per this task's own prescribed
# workaround pattern. Recorded here so the next session does not re-spend a 20-minute
# diagnosis budget on the same two failures. Both were reproduced BY HAND outside this
# driver (a raw `git worktree add --detach` + copytree, and a direct pytest run
# inside the resulting throwaway tree) before either file was added to the tuple
# above, and this driver's own `test_mutation_driver_contract.py` run clean before and
# after -- neither addition was needed to keep the contract suite green, only to get
# `retrieval.fusion`'s stats pass past two false failures:
#
# 1. tests/unit/test_eval_metric_single_implementation.py -- SPECIFIC to a worktree
#    whose git HEAD is stale relative to its live files (this session's, before any
#    commit). `_ThrowawayTree.__enter__` does `git worktree add --detach <path> HEAD`;
#    HEAD's committed tree still had `tests/eval/metrics.py` (deleted only on disk,
#    never committed), and the follow-up `shutil.copytree(..., dirs_exist_ok=True)`
#    sync never DELETES a file the live tree removed, only adds/overwrites. The
#    throwaway tree ends up with a resurrected `tests/eval/metrics.py`, which the test
#    correctly reports as a second metric implementation. Would NOT reproduce against
#    a tree whose HEAD commit is genuinely 9b0c02a (or any commit without that file).
#    ROUND 9 FIX: `_ThrowawayTree.__enter__` now mirrors deletions too (walks each
#    synced dir after the copytree and unlinks any file absent from REPO_ROOT), so a
#    resurrected file can no longer survive the sync regardless of how stale HEAD is.
#    Scoped to files ABSENT from the live tree, so a real file can never be removed.
#
# 2. tests/unit/test_dotenv_anchoring.py -- GENERAL; reproduces on any correctly-
#    committed tree too, for any scope inside the eager `trelix` import graph.
#    `trelix/__init__.py` eagerly imports `Indexer`/`make_embedder`/`Retriever`, and
#    `Retriever` eagerly imports `.bm25`, `.fusion`, `.graph`, `store.db`,
#    `store.vector` -- so mutating any of those makes `import trelix` pull in a
#    mutmut-trampoline-wrapped module everywhere, INCLUDING inside a fresh subprocess.
#    test_dotenv_anchoring.py's child-process probes deliberately run with cwd set to
#    a throwaway repo fixture and no TRELIX_*/OPENAI_*/AZURE_* env (to prove a hostile
#    repo's `.env` is not a config source) -- so when the trampoline module's own
#    `Config.ensure_loaded()` auto-guesses `source_paths` from that subprocess's cwd,
#    it finds no pyproject.toml there and raises `FileNotFoundError: Could not figure
#    out where the code to mutate is`, which the probe surfaces as "probe failed: ...".
#    `store/dimension_guard.py` is NOT affected: its only imports are function-local,
#    inside `indexer.py`, never reached by `import trelix` alone (confirmed by grep
#    before relying on it).
#
#    ROUND 9 CORRECTION: the paraphrase above ("neither fix is test-support-only") was
#    wrong -- re-derived by actually reading mutmut 3.7.0's own installed source rather
#    than trusting this comment. The precise mechanism is NOT the trampoline's
#    MUTANT_UNDER_TEST env-var branch (that branch is inert with the var unset, which
#    is exactly this probe's case) -- it is that a mutated module's generated
#    `from mutmut.mutation.trampoline import ...` pulls in `mutmut.__main__`, which at
#    IMPORT TIME (unconditionally, in `mutmut/utils/safe_setproctitle.py:15`) evaluates
#    `Config.get().use_setproctitle`, and `Config._guess_source_paths()` falls back to
#    `os.getcwd()` only when NO `pyproject.toml`/`setup.cfg` names `source_paths` --
#    both absent from this probe's hostile-repo cwd, which holds nothing but a `.env`.
#    FIXED test-side, gated so it is a true no-op outside mutation testing:
#    `tests/unit/test_dotenv_anchoring.py`'s `_probe_config` now calls
#    `_write_mutmut_bootstrap_if_needed(cwd)`, which returns immediately unless
#    `"mutmut" in sys.modules` -- only true under this driver -- and otherwise writes
#    a `pyproject.toml` naming `[tool.mutmut] source_paths = ["."]` into the probe's
#    cwd, satisfying mutmut's guess without touching trelix's own dotenv resolution
#    (which never reads `pyproject.toml`). An earlier candidate fix instead
#    unconditionally created an empty `src/` marker directory on every call -- correct
#    functionally, but it silently made the `TestAnchorInAChildProcess` fixture's own
#    documented invariant ("a clean scratch dir holding only a .env") false on every
#    normal test run, not only under mutmut. The gated version was chosen instead.
#    Verified two ways: (a) `--update-baseline --modules retrieval.bm25`'s stats pass
#    gets past this test where it previously failed with the exact FileNotFoundError
#    above, and (b) the SAME hostile-.env scenario this test exists to catch (a
#    cwd-relative `env_file` reintroduced on `EmbedderConfig`) is still caught after
#    the fix -- reproduced by hand, confirmed 6/12 tests in the file fail, reverted
#    with a sha256-verified restore. Left OUT of DESELECTED_FILES because the fix
#    above restores full coverage; no scope needs to trade away its kill set for this
#    anymore.

STATUS_BY_EXIT_CODE: dict[int | None, str] = {
    None: "not_checked",
    0: "survived",
    1: "killed",
    2: "check_was_interrupted_by_user",
    3: "killed",
    5: "no_tests",
    24: "timeout",
    -24: "timeout",
    33: "no_tests",
    34: "skipped",
    35: "suspicious",
    36: "timeout",
    37: "caught_by_type_check",
    152: "timeout",
    255: "timeout",
    -9: "segfault",
    -11: "segfault",
}
# Anything not in the table above. mutmut's own defaultdict does the same.
UNKNOWN_STATUS = "suspicious"

# The two verdicts a ceiling is kept for. `no_tests` is tracked separately from
# `survived` and never folded into it: they have different fixes. `survived` means a
# test executed the mutated line and did not notice; `no_tests` means mutmut's
# test-to-function map found nothing reaching it at all.
CEILING_KEYS = ("survived", "no_tests")


class DriverError(RuntimeError):
    """Anything that would otherwise let a run report a number it did not measure."""


@dataclass(frozen=True)
class ModuleResult:
    """One module's measurement. A dataclass, not a dict, so the JSON writer and the
    ratchet cannot disagree about the field names."""

    module: str
    patterns: list[str]
    measured_at: str
    mutmut_exit_code: int
    files_mutated: int
    files_with_results: int
    counts: dict[str, int]
    source_sha256: dict[str, str]

    @property
    def total_mutants(self) -> int:
        return sum(self.counts.values())

    @property
    def ceilings(self) -> dict[str, int]:
        return {key: self.counts.get(key, 0) for key in CEILING_KEYS}

    def to_json(self) -> dict[str, Any]:
        return {
            "patterns": self.patterns,
            "measured": True,
            "measured_at": self.measured_at,
            "mutmut_exit_code": self.mutmut_exit_code,
            "files_mutated": self.files_mutated,
            "files_with_results": self.files_with_results,
            "total_mutants": self.total_mutants,
            "counts": self.counts,
            "ceilings": self.ceilings,
            "source_sha256": self.source_sha256,
        }


# ---------------------------------------------------------------------------
# Throwaway tree
# ---------------------------------------------------------------------------
# rsync of the LIVE tree, not a checkout of HEAD. Deliberate: in this repo an
# isolated worktree branches from a FIXED base, so `git worktree add HEAD` in one
# reproducibly measured a tree missing 16k lines of tests -- i.e. it would have
# reported survivors that the working tree already kills. The baseline records a
# sha256 per mutated file so the tree it measured is never in doubt.
#
# The sync list is a superset of MUTMUT_CONFIG["also_copy"] on purpose: also_copy
# copies FROM the tree INTO mutants/, so anything it names has to be in the tree
# first, and the git-worktree fallback path (a plain temp dir) starts empty.
_SYNC_DIRS = ("src", "tests", "docs", ".github", "helm", "packages", "scripts", "config")
_SYNC_FILES = (
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "SUPPORT.md",
    "LICENSE",
    "Makefile",
    ".env.example",
)


class _ThrowawayTree:
    def __init__(self, keep: bool) -> None:
        self._keep = keep
        self.path: Path | None = None
        self._is_git_worktree = False

    def __enter__(self) -> Path:
        # .resolve() is NOT cosmetic and NOT optional. On macOS $TMPDIR is
        # /var/folders/... which is a symlink to /private/var/folders/...; coverage
        # records a module's REAL path while mutmut's gather_coverage
        # (code_coverage.py:54) keys its map on Path.absolute(), which does not
        # resolve symlinks. The two strings then never match, every covered-line set
        # comes back empty, and `mutate_only_covered_lines` silently degrades to
        # "mutate only the def line": measured on eval/ndcg.py, 2 mutants generated
        # (both `k: int = 10` -> `11`) instead of 26, with a clean-looking report.
        # A vacuous green, produced by a path string.
        parent = Path(tempfile.mkdtemp(prefix="trelix-mutation-")).resolve()
        self.path = parent / "tree"
        added = subprocess.run(
            ["git", "worktree", "add", "--detach", str(self.path), "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if added.returncode == 0:
            self._is_git_worktree = True
        else:
            # No git worktree available (shallow clone, sdist, CI cache). Still fine:
            # mutmut's git calls return None and it handles that. Never fall back
            # SILENTLY to running in REPO_ROOT -- that is the thing being prevented.
            self.path.mkdir(parents=True)
            print(
                f"  note: `git worktree add` failed ({added.stderr.strip()!r}); "
                "using a plain temporary directory instead. Still outside the repo.",
                file=sys.stderr,
            )
        for name in _SYNC_DIRS:
            shutil.copytree(REPO_ROOT / name, self.path / name, dirs_exist_ok=True)
            # `dirs_exist_ok=True` is add/overwrite-only, never delete. When the
            # worktree's `HEAD` commit is stale relative to its own live files (a
            # file deleted on disk but never committed -- e.g. this session, before
            # any commit), `git worktree add --detach <path> HEAD` above checks out
            # the STALE committed tree first, resurrecting that file in the
            # throwaway tree, and this copytree does not remove it. Measured:
            # tests/unit/test_eval_metric_single_implementation.py resurrecting a
            # deleted tests/eval/metrics.py this exact way; see scripts/mutation.py's
            # own round-8/round-9 history in DESELECTED_FILES' trailing comment for
            # the full diagnosis. Mirror the deletion side too, scoped to files that
            # were RESURRECTED under a name absent from the live tree -- a real
            # ADDED-then-removed file in REPO_ROOT would already be absent from the
            # copytree source, so this can only remove checkout artifacts, never a
            # live file.
            for resurrected in (self.path / name).rglob("*"):
                if resurrected.is_dir():
                    continue
                live = REPO_ROOT / resurrected.relative_to(self.path)
                if not live.exists():
                    resurrected.unlink()
        for name in _SYNC_FILES:
            src = REPO_ROOT / name
            if src.exists():
                shutil.copy2(src, self.path / name)
        return self.path

    def __exit__(self, *_exc: object) -> None:
        if self.path is None:
            return
        if self._keep:
            print(f"  kept throwaway tree at {self.path}")
            return
        if self._is_git_worktree:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(self.path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        shutil.rmtree(self.path.parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Launcher: why `mutmut run` is not invoked as `python -m mutmut`
# ---------------------------------------------------------------------------
# `mutate_only_covered_lines = true` makes mutmut run the test suite an EXTRA time,
# in-process, and then call _unload_modules_not_in() (code_coverage.py:58) to pop
# every module imported during that run so the mutated copies get re-imported.
#
# Popping a C extension from sys.modules and importing it again in the same process
# is not something every extension survives. numpy raises outright:
#     ImportError: cannot load module more than once per process
#         numpy/_core/multiarray.py:11 -> _multiarray_umath
# and trelix imports numpy at MODULE scope inside the scope being mutated
# (compression/extractive.py:25, store/vector.py, retrieval/*), so the stats pass
# dies during collection with "failed to collect stats. runner returned 1".
#
# gather_coverage snapshots dict(sys.modules) on ENTRY and only pops what is absent
# from that snapshot. So the fix is to import the offending extensions BEFORE mutmut
# starts: they are then in the snapshot and are never unloaded. That cannot be done
# through `python -m mutmut`, hence this launcher.
#
# Only third-party extensions are pre-imported. Nothing under `trelix` is, ever --
# a pre-imported trelix module would be the UNMUTATED one and every mutant in it
# would falsely survive. The list is explicit rather than derived, and a missing
# entry shows up as the same loud collection error, not as a wrong number.
# Each entry closes a MEASURED failure, not a suspicion:
#   numpy*         ImportError: cannot load module more than once per process
#                  (numpy/_core/multiarray.py:11 -> _multiarray_umath), hit while
#                  collecting tests/unit/test_assembler_compression.py.
#   beartype.claw* ImportError: cannot import name 'claw_state' from partially
#                  initialized module 'beartype.claw._clawstate' (circular import).
#                  beartype installs an import HOOK in sys.meta_path; the unload pops
#                  the hook's own modules while leaving the hook installed, so the
#                  next lazy import re-enters it mid-initialisation.
#   multiprocessing.pool
#                  the lazy `from .pool import Pool` inside Pool() is what re-entered
#                  the dangling beartype hook. Imported up front so nothing large is
#                  imported for the first time after the unload.
#   the rest       C extensions or hook-installing packages on the same import paths,
#                  pre-imported for the same reason before they can bite.
_PREIMPORT: tuple[str, ...] = (
    "multiprocessing",
    "multiprocessing.pool",
    "beartype",
    "beartype.claw",
    "beartype.claw._clawstate",
    "numpy",
    "numpy.linalg",
    "coverage",
    "pytest",
    "pydantic_core",
    "sqlite3",
    "tiktoken",
    "tree_sitter",
    # The remainder are transcribed from the faulthandler dump's own
    # "Extension modules: ..." line, which is the authoritative list of what was
    # loaded when the stats pass died. Not guesses.
    "yaml",
    "regex",
    "charset_normalizer",
    "requests",
    "xxhash",
    "ujson",
    "markupsafe",
    "multidict",
    "yarl",
    "propcache",
    "aiohttp",
    "frozenlist",
    "PIL",
    "PIL.Image",
    # THE ONE THAT ACTUALLY KILLED THE PROCESS, and the reason this list is long.
    # "Fatal Python error: Segmentation fault", current thread at
    # torch/__init__.py:444 inside importlib's create_module, reached from
    # transformers/generation/logits_process.py:21. The coverage pass imports torch
    # (via transformers, via the sparse/local providers the suite touches); the
    # unload pops it; the stats pass re-imports it; re-running a torch C extension's
    # module-init in one process is a SIGSEGV, not an exception.
    #
    # This is the SAME root cause as EXCLUDED_SCOPES' 245 segfaults, arriving by a
    # different door: there it kills individual mutants, here it killed the whole
    # stats pass for an unrelated scope. Pre-importing it is what confines the
    # problem back to the excluded scopes.
    "torch",
    "transformers",
    "sentence_transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "scipy",
    "sklearn",
)
_LAUNCHER_NAME = "_trelix_mutmut_launch.py"
_LAUNCHER_SOURCE = '''"""Generated by scripts/mutation.py. Do not edit; do not commit."""

import importlib
import sys

PREIMPORT = {preimport!r}

for _name in PREIMPORT:
    try:
        importlib.import_module(_name)
    except Exception as _exc:  # noqa: BLE001 - an absent optional extra is fine
        print(f"  pre-import skipped {{_name}}: {{type(_exc).__name__}}: {{_exc}}")

for _name in list(sys.modules):
    if _name == "trelix" or _name.startswith("trelix."):
        raise SystemExit(
            f"refusing to start: {{_name}} is already imported, so mutants in it "
            "would be measured against unmutated code"
        )

from mutmut.__main__ import cli  # noqa: E402

sys.argv = ["mutmut", *sys.argv[1:]]
cli()
'''


def effective_mutmut_config(patterns: list[str]) -> dict[str, Any]:
    """MUTMUT_CONFIG with the deselections spliced in and only_mutate resolved.

    One function so the table written into the throwaway tree and the table recorded
    in the baseline JSON are the same object, not two hand-kept copies.
    """
    cli_args = [
        *MUTMUT_CONFIG["pytest_add_cli_args"],  # type: ignore[misc]
        *[arg for path in DESELECTED_FILES for arg in ("--ignore", path)],
    ]
    return {**MUTMUT_CONFIG, "pytest_add_cli_args": cli_args, "only_mutate": patterns}


def _render_mutmut_table(patterns: list[str]) -> str:
    lines = ["", "[tool.mutmut]"]
    for key, value in effective_mutmut_config(patterns).items():
        lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The second thing that silently empties the covered-line set, and the reason a
# `mutate_only_covered_lines` run cannot just inherit this repo's coverage config.
#
# mutmut builds `coverage.Coverage(data_file=None)` in gather_coverage
# (code_coverage.py:44) BEFORE it chdirs into `mutants`, so coverage reads the tree
# ROOT's pyproject.toml -- where `[tool.coverage.run] source = ["src/trelix"]`
# resolves to <tree>/src/trelix, the UNMUTATED original. The tests then import from
# <tree>/mutants/src/trelix, which is outside `source`, so coverage measures nothing,
# every covered-line set comes back empty, and mutmut emits only the one mutant per
# function that its signature line allows. Measured on eval/ndcg.py: 2 mutants, and
# a report that looks fine.
#
# COVERAGE_RCFILE overrides the pyproject section entirely, which is why this is an
# env-var rcfile rather than string surgery on the copied pyproject.toml.
#
# THE SOURCE PATH MUST BE ABSOLUTE, and that is a third distinct trap rather than a
# style preference. coverage reads the rcfile when `Coverage()` is CONSTRUCTED (cwd =
# tree root) but classifies each `source` entry as dir-or-module-name when
# `start()` runs -- and mutmut calls start() inside `with change_cwd("mutants")`. A
# relative `mutants/src/trelix` therefore does not exist at classification time,
# coverage falls back to treating it as an importable module name and warns
# "Module mutants/src/trelix was never imported. (module-not-imported)", then
# "No data was collected." Measured with a probe that replicates gather_coverage:
#   relative source -> 0 measured files, 0 mutants generated for ndcg.py
#   absolute source -> 143 measured files, ndcg.py lines [1, 8, 10, 13, 16, 26, ...]
# The relative form does not error. It reports zero.
_COVERAGERC_NAME = ".mutmut-coveragerc"
_COVERAGERC_TEMPLATE = """# Generated by scripts/mutation.py. Do not edit; do not commit.
[run]
branch = true
source =
    {mutants_src}
"""


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _mutmut_env(mutmut_path: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    extra = mutmut_path or env.get("TRELIX_MUTMUT_PATH")
    if extra:
        env["PYTHONPATH"] = os.pathsep.join([extra, *filter(None, [env.get("PYTHONPATH")])])
    return env


def _require_mutmut(env: dict[str, str]) -> str:
    probe = subprocess.run(
        [sys.executable, "-c", "import mutmut,importlib.metadata as m;print(m.version('mutmut'))"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise DriverError(
            "mutmut is not importable. It is not in [dev] on purpose -- see this "
            "module's docstring for the out-of-tree install, then set "
            "TRELIX_MUTMUT_PATH or pass --mutmut-path.\n" + probe.stderr.strip()
        )
    return probe.stdout.strip()


def _tail(label: str, text: str, limit: int = 14000) -> str:
    body = text[-limit:] if text else "<empty>"
    return f"\n--- {label} (last {limit} chars) ---\n{body}\n"


def _parse_generation_line(stdout: str) -> int:
    """Pull N out of "(N files mutated, M ignored, K unmodified)"."""
    for line in stdout.splitlines():
        marker = " files mutated,"
        if marker in line:
            head = line.split(marker)[0]
            return int(head.rsplit("(", 1)[-1].strip())
    raise DriverError(
        "mutmut printed no 'N files mutated' line, so generation cannot be "
        "confirmed. Refusing to report counts."
    )


def _collect_meta(tree: Path) -> tuple[Counter[str], int, dict[str, str]]:
    """Read mutants/**/*.meta -- mutmut's own per-file verdict store."""
    counts: Counter[str] = Counter()
    meta_files = sorted((tree / "mutants").rglob("*.py.meta"))
    shas: dict[str, str] = {}
    for meta in meta_files:
        payload = json.loads(meta.read_text())
        exit_codes = payload.get("exit_code_by_key", {})
        if not exit_codes:
            continue
        rel = str(meta.relative_to(tree / "mutants")).removesuffix(".meta")
        source = tree / rel
        if source.exists():
            shas[rel] = _sha256(source)
        for exit_code in exit_codes.values():
            counts[STATUS_BY_EXIT_CODE.get(exit_code, UNKNOWN_STATUS)] += 1
    return counts, len(shas), shas


def measure_module(
    module: str, tree: Path, env: dict[str, str], max_children: int | None
) -> ModuleResult:
    patterns = list(SCOPE[module])
    mutants_dir = tree / "mutants"
    # NON-NEGOTIABLE 2. Unconditional, before every single scope.
    shutil.rmtree(mutants_dir, ignore_errors=True)

    pyproject = tree / "pyproject.toml"
    base = pyproject.read_text()
    pyproject.write_text(base.split("\n[tool.mutmut]")[0] + _render_mutmut_table(patterns))

    # Both of these exist ONLY to serve mutate_only_covered_lines. With the flag off
    # there is no in-process coverage pass, so there is nothing to unload and nothing
    # to keep out of the unload -- pre-importing anyway would be the lancedb pyo3
    # panic for no benefit at all.
    covered_lines_on = bool(MUTMUT_CONFIG["mutate_only_covered_lines"])
    launcher = tree / _LAUNCHER_NAME
    launcher.write_text(_LAUNCHER_SOURCE.format(preimport=_PREIMPORT if covered_lines_on else ()))
    if covered_lines_on:
        (tree / _COVERAGERC_NAME).write_text(
            _COVERAGERC_TEMPLATE.format(mutants_src=tree / "mutants" / "src" / "trelix")
        )
        env = {**env, "COVERAGE_RCFILE": str(tree / _COVERAGERC_NAME)}

    argv = [sys.executable, _LAUNCHER_NAME, "run"]
    if max_children is not None:
        argv += ["--max-children", str(max_children)]
    print(f"  $ {' '.join(argv)}   (only_mutate={patterns})")
    proc = subprocess.run(argv, cwd=tree, env=env, capture_output=True, text=True, check=False)
    stdout = proc.stdout
    # Both streams, on every failure path. An earlier revision raised
    # "mutmut printed no 'N files mutated' line" and attached NEITHER stream, so the
    # actual traceback -- which mutmut had written to stderr -- was thrown away and
    # the driver's own diagnostic became the thing that needed diagnosing.
    logs = _tail("mutmut stdout", stdout) + _tail("mutmut stderr", proc.stderr)

    # NON-NEGOTIABLE 3, first of three independent checks.
    try:
        files_mutated = _parse_generation_line(stdout)
    except DriverError as exc:
        raise DriverError(f"module {module!r}: {exc} exit={proc.returncode}{logs}") from None
    if files_mutated <= 0:
        raise DriverError(
            f"module {module!r}: mutmut mutated 0 files for patterns {patterns}. "
            "The scope matches nothing -- a perfect score here would be an artifact. "
            "Note fnmatch semantics: '*' crosses '/', and a pattern must end in "
            "'.py' or '*'." + logs
        )

    counts, files_with_results, shas = _collect_meta(tree)
    # ...and the second: verdicts on disk, not just a generation claim.
    if files_with_results <= 0 or sum(counts.values()) <= 0:
        raise DriverError(
            f"module {module!r}: mutmut reported {files_mutated} files mutated but "
            "wrote no per-mutant verdicts. Refusing to record a count." + logs
        )
    # NON-NEGOTIABLE 3, third check -- and the one that caught a REAL vacuous green
    # here rather than a hypothetical one. `not_checked` is mutmut's exit code None:
    # the mutant was generated and then never run. If that is EVERY mutant, mutmut
    # stopped early (its own message is "the selected tests do not cover any code
    # that we mutated") and the two checks above still pass, because files WERE
    # mutated and verdicts WERE written. Reporting that as a result is precisely the
    # "measured a proxy" failure this driver exists to avoid.
    if counts.get("not_checked", 0) == sum(counts.values()):
        raise DriverError(
            f"module {module!r}: all {sum(counts.values())} mutants are "
            "'not checked' -- mutmut generated them and ran none. Refusing to "
            "record a count." + logs
        )

    return ModuleResult(
        module=module,
        patterns=patterns,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        mutmut_exit_code=proc.returncode,
        files_mutated=files_mutated,
        files_with_results=files_with_results,
        counts=dict(sorted(counts.items())),
        source_sha256=shas,
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def _unmeasured(module: str) -> dict[str, Any]:
    """A module with no measurement, and therefore NO ceilings and NO counts.

    Absence, not zero. A recorded ceiling of 0 for a module nobody measured is a free
    perfect score, and once it is in the file it is indistinguishable from a real one.
    tests/unit/test_mutation_driver_contract.py asserts exactly this shape.
    """
    return {"patterns": list(SCOPE[module]), "measured": False}


def load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        return {
            "schema": BASELINE_SCHEMA,
            "generated_by": "scripts/mutation.py",
            "gate": "none -- per-module survivor ceilings only, never a score ratio",
            "mutmut_version": None,
            "mutmut_config": MUTMUT_CONFIG,
            "deselected_files": list(DESELECTED_FILES),
            "excluded_scopes": EXCLUDED_SCOPES,
            "modules": {name: _unmeasured(name) for name in SCOPE},
        }
    loaded: dict[str, Any] = json.loads(BASELINE_PATH.read_text())
    return loaded


def _print_report(results: list[ModuleResult]) -> None:
    print("\n=== mutation report (NOT a gate) ===")
    for rec in results:
        detail = "  ".join(f"{k}={v}" for k, v in sorted(rec.counts.items()))
        print(f"{rec.module}: files_mutated={rec.files_mutated} mutants={rec.total_mutants}")
        print(f"    {detail}")
    print(
        "\nNo mutation-score ratio is printed or gated on. See this module's "
        "docstring for why (Stryker's break:null, Google's 16.9M-mutant rejection, "
        "and an unmeasurable 77% of one scope)."
    )


def cmd_check(results: list[ModuleResult], baseline: dict[str, Any]) -> int:
    """Ratchet each module against ITS OWN recorded ceilings. No ratio, anywhere.

    Refusing to compare is a FAILURE, not a pass: a module measured now with no
    baseline entry, or an entry with no ceiling, exits non-zero. The alternative --
    skipping what cannot be compared -- is how a gate ends up green over nothing.
    """
    modules: dict[str, Any] = baseline.get("modules", {})
    failures: list[str] = []
    for rec in results:
        recorded = modules.get(rec.module)
        if not isinstance(recorded, dict) or not recorded.get("measured"):
            failures.append(
                f"{rec.module}: measured now, but the baseline has no measurement to "
                "ratchet against. Run --update-baseline first."
            )
            continue
        ceilings: dict[str, Any] = recorded.get("ceilings", {})
        for key in CEILING_KEYS:
            now = rec.counts.get(key, 0)
            ceiling = ceilings.get(key)
            if ceiling is None:
                failures.append(f"{rec.module}: baseline records no {key!r} ceiling.")
                continue
            if now > int(ceiling):
                failures.append(f"{rec.module}: {key} {now} > ceiling {ceiling}.")
            elif now < int(ceiling):
                print(f"  {rec.module}: {key} {now} < ceiling {ceiling} -- ratchet it down.")
    for line in failures:
        print(f"FAIL {line}")
    return 1 if failures else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="measure and print; exits 0 on success")
    mode.add_argument(
        "--update-baseline", action="store_true", help="measure and rewrite the JSON baseline"
    )
    mode.add_argument(
        "--check", action="store_true", help="measure and ratchet per-module ceilings"
    )
    mode.add_argument("--list-scope", action="store_true", help="print the scope table and exit")
    parser.add_argument("--modules", nargs="+", default=[], help="module keys from SCOPE, or ALL")
    parser.add_argument("--max-children", type=int, default=None)
    parser.add_argument("--mutmut-path", default=None, help="dir holding a sideloaded mutmut")
    parser.add_argument(
        "--keep-worktree", action="store_true", help="do not delete the throwaway tree"
    )
    return parser


def _refuse_if_repo_has_mutants() -> None:
    """NON-NEGOTIABLE 1, enforced rather than merely documented.

    `mutants/` is absent from this repo's 72-line .gitignore, and hatchling's refusal
    to honour NESTED .gitignore files already came one `twine upload` from publishing a
    221 MB index.db (see the sdist `exclude` block in pyproject.toml). If a `mutants/`
    tree exists at the repo root, someone ran mutmut here by hand and ~143 MB of
    generated Python is sitting in the working tree waiting for `git add -A`. Say so and
    stop, rather than adding a second copy.
    """
    stray = REPO_ROOT / "mutants"
    if stray.exists():
        raise DriverError(
            f"{stray} exists. mutmut was run inside the repo, and `mutants/` is NOT in "
            ".gitignore -- `git add -A` would stage ~143 MB of generated Python. Remove "
            "it with `rm -rf mutants` (not `git rm`), then re-run: this driver only ever "
            "generates inside a throwaway tree outside the repo."
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_scope or not (args.report or args.update_baseline or args.check):
        print("SCOPE (ratchet units):")
        for name, patterns in SCOPE.items():
            print(f"  {name:26s} {list(patterns)}")
        print("\nEXCLUDED (hard limits of the instrument, not todos):")
        for name, why in EXCLUDED_SCOPES.items():
            print(f"  {name:26s} {why}")
        print("\nDESELECTED from the kill set (cannot run twice in one process):")
        for path in DESELECTED_FILES:
            print(f"  {path}")
        return 0

    requested = list(SCOPE) if args.modules == ["ALL"] else list(args.modules)
    unknown = [m for m in requested if m not in SCOPE]
    if unknown:
        raise DriverError(f"unknown module keys {unknown}; try --list-scope")
    if not requested:
        raise DriverError("--modules is required (or --modules ALL)")

    _refuse_if_repo_has_mutants()
    env = _mutmut_env(args.mutmut_path)
    version = _require_mutmut(env)
    print(f"mutmut {version}; python {sys.version.split()[0]}")

    results: list[ModuleResult] = []
    with _ThrowawayTree(keep=args.keep_worktree) as tree:
        print(f"throwaway tree: {tree}")
        for module in requested:
            print(f"\n--- {module} ---")
            results.append(measure_module(module, tree, env, args.max_children))

    _print_report(results)

    if args.update_baseline:
        baseline = load_baseline()
        baseline["schema"] = BASELINE_SCHEMA
        baseline["generated_by"] = "scripts/mutation.py"
        baseline["gate"] = "none -- per-module survivor ceilings only, never a score ratio"
        baseline["mutmut_version"] = version
        baseline["mutmut_config"] = MUTMUT_CONFIG
        baseline["deselected_files"] = list(DESELECTED_FILES)
        baseline["excluded_scopes"] = EXCLUDED_SCOPES
        modules: dict[str, Any] = baseline.setdefault("modules", {})
        for name in SCOPE:
            modules.setdefault(name, _unmeasured(name))
        # A key that left SCOPE must LEAVE the baseline. Leaving it behind is a ceiling
        # for a module nobody measures any more, and it reads as coverage.
        for name in list(modules):
            if name not in SCOPE:
                del modules[name]
        for rec in results:
            modules[rec.module] = rec.to_json()
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    if args.check:
        return cmd_check(results, load_baseline())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DriverError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
