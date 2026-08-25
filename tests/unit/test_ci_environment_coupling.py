"""The guard on assertions that are stronger than CI's environment guarantees.

WHY THIS FILE EXISTS
--------------------
Two commits in a row were green locally and red on all four unit legs, both from
the same cause: an assertion validated in a rich development venv that the leaner
CI install does not satisfy.

  1. ``assert sorted(k for k in os.environ if k.upper().startswith("AZURE_")) == []``
     -- GitHub's ubuntu runners preinstall the Azure CLI, which exports
     ``AZURE_EXTENSION_DIR``. The assertion had claimed a whole third-party
     namespace the scrub never claimed. (Now fixed; see
     ``tests/unit/test_env_isolation_covers_config_aliases.py``.)
  2. A meta-test that required all eight provider SDKs to be importable, with a
     1,200-literal floor measured over all eight. CI's unit job installs NONE of
     seven of them; ``openai`` alone yields 22. (Now fixed; see
     ``tests/unit/test_env_isolation_covers_sdk_env.py``.)

Both were LOUD. The quieter direction is a test whose discriminating precondition
cannot hold under the leaner install, so it reports green while asserting nothing.
This file guards both directions with four static rules, each carrying its own
control so the rule itself cannot go vacuous.

WHAT THIS GUARD CANNOT SEE, stated up front so nobody mistakes it for coverage
-----------------------------------------------------------------------------
* It is STATIC. It reads sources and configuration; it never runs the suite under
  a leaner install. A coupling that only manifests at runtime -- a value read from
  an env var this file does not know about, a package whose ABSENCE changes a
  computed number rather than an import -- is invisible here. Reproducing a job's
  install set for real is a separate exercise
  (``scratch-pad/leanenv/ci_job_blocker.py`` does it with a ``sys.meta_path``
  wrapper); this file is the cheap always-on half.
* The absent-package table is derived from what an extra DECLARES, so a package
  that reaches CI only transitively is classified by an explicit hand-written
  table with the chain written down. If one of those chains breaks upstream (see
  ``PRESENT_ANYWAY``, where three entries depend on a floor as loose as
  ``fastmcp>=3.4.0,<4``), this file keeps saying "present" until someone re-derives
  it. That is the known soft spot.
* Rule 2 (env-namespace prefixes) matches a STRING LITERAL in a comprehension over
  ``os.environ``. A prefix built at runtime, or the same claim written as
  ``assert not [k for k in list(os.environ) if ...]`` through an alias this file
  does not recognise, is not matched.
* A NEW CI JOB LEG that runs ``tests/unit`` under a different install set was
  considered and deliberately NOT added: CI's unit job IS the lean environment for
  ``tests/unit``, so a second lean leg would be measuring a configuration nobody
  ships. What would add signal is running the suite under the *integration* job's
  thinner set -- but no job does that, so a green there would guard nothing real.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
import warnings
from pathlib import Path

import pytest
import yaml


def _parse(source: str) -> ast.Module:
    """``ast.parse`` without inheriting other files' SyntaxWarnings.

    Several test files contain regex literals with invalid escapes inside
    docstrings (tests/unit/test_git_linker.py:513 among them); compiling them here
    re-emits a DeprecationWarning that belongs to those files, not to this run.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        return ast.parse(source)


_TESTS_DIR = Path(__file__).resolve().parents[1]
_ROOT = _TESTS_DIR.parent
_UNIT_DIR = _TESTS_DIR / "unit"
_PYPROJECT = _ROOT / "pyproject.toml"
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"


# ===========================================================================
# 1. The anchor: what each CI job actually installs.
# ===========================================================================

# Verbatim from .github/workflows/ci.yml, written out as literals here rather
# than read from it, so this file is an independent second statement that the
# workflow must agree with. Every table below is only valid FOR THESE SETS, which
# is why a workflow edit has to fail here first.
#
# Note the two steps per job, not one: `test` and `lint` additionally install the
# three adapter packages. `dev` now declares uvicorn and watchfiles directly (see
# pyproject.toml's [dev] extra), so the unit job has them on that ground alone; it is
# no longer contingent on packages/trelix-mcp's own dependency graph (mcp>=1.24 ->
# uvicorn>=0.31.1; fastmcp>=3.4 -> fastmcp-slim[client,server] -> watchfiles>=1.0.0),
# though that graph still supplies them too and both floors are compatible. The
# `integration` job also installs `dev`, so it now has both as well -- a behaviour
# change from before this fix, when it had neither.
CI_JOB_PIP_INSTALLS: dict[str, tuple[str, ...]] = {
    "lint": (
        'pip install -e ".[local,sso,dev]"',
        "pip install -e packages/trelix-langchain",
        "pip install -e packages/trelix-llama-index",
        "pip install -e packages/trelix-mcp",
    ),
    "type-check-extras": ('pip install -e ".[local,sso,voyage,vertex,watch,dev]"',),
    "test": (
        'pip install -e ".[local,otel,sso,bge-code,nomic-code,dev,graph-viz]"',
        "pip install -e packages/trelix-langchain",
        "pip install -e packages/trelix-llama-index",
        "pip install -e packages/trelix-mcp",
    ),
    "integration": ('pip install -e ".[local,otel,sso,dev]"',),
}

#: The job whose install set ``tests/unit`` runs under (4 Python legs).
UNIT_JOB = "test"

#: Extras that job's `pip install -e ".[...]"` names. Parsed out below and
#: compared, so this cannot fall behind the line above.
#
# `graph-viz` added for the identical reason uvicorn/watchfiles were declared
# directly: tests/unit/test_graph_visualizer_escaping.py (SEC-04) does a
# module-scope `pytest.importorskip("pyvis.network")`, and pyvis was declared by
# no extra this job installed, so the whole file skipped on every CI run and
# SEC-04's XSS-prevention assertions never executed. See ci.yml's comment on this
# job's install step for why `graph-viz` and not `dev` (three other jobs install
# `dev` and never run tests/unit).
UNIT_JOB_EXTRAS = frozenset({"local", "otel", "sso", "bge-code", "nomic-code", "dev", "graph-viz"})


def _workflow_pip_installs() -> dict[str, tuple[str, ...]]:
    spec = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, ...]] = {}
    for job, body in spec["jobs"].items():
        cmds = sorted(
            line.strip()
            for step in body.get("steps", [])
            for line in (step.get("run") or "").splitlines()
            if line.strip().startswith("pip install")
        )
        if cmds:
            out[job] = tuple(cmds)
    return out


def test_ci_yml_pip_install_lines_are_exactly_these() -> None:
    """Every job's install set, pinned both ways against the workflow.

    This is the anchor for the whole file: the absent-package table below is a
    property of these lines, so an edit to them must fail HERE, loudly, rather
    than silently invalidating the tables.

    MUTATION that must make this fail: add or remove any extra in any
    ``pip install -e ".[...]"`` line in .github/workflows/ci.yml, or delete one of
    the ``pip install -e packages/...`` steps.
    """
    observed = _workflow_pip_installs()
    expected = {job: tuple(sorted(cmds)) for job, cmds in CI_JOB_PIP_INSTALLS.items()}

    assert set(observed) == set(expected), (
        f"jobs with pip-install steps changed: only in ci.yml "
        f"{sorted(set(observed) - set(expected))}; only here "
        f"{sorted(set(expected) - set(observed))}"
    )
    for job in sorted(expected):
        assert observed[job] == expected[job], (
            f"job {job!r} install steps drifted.\n  ci.yml: {observed[job]}\n  here:   "
            f"{expected[job]}\nRe-derive UNIT_JOB_ABSENT_IMPORTS / PRESENT_ANYWAY before "
            f"updating this literal."
        )


def test_the_pinned_unit_job_extras_match_its_pip_line() -> None:
    """``UNIT_JOB_EXTRAS`` is the parsed form of the unit job's install line.

    Kept separate from the test above so a failure says WHICH representation
    drifted. Parsed from ``CI_JOB_PIP_INSTALLS`` (already tied to ci.yml) rather
    than re-read from the workflow.

    MUTATION that must make this fail: drop ``"otel"`` from ``UNIT_JOB_EXTRAS``.
    """
    line = next(c for c in CI_JOB_PIP_INSTALLS[UNIT_JOB] if '".[' in c)
    match = re.search(r"\.\[([^\]]+)\]", line)
    assert match is not None, f"could not parse extras out of {line!r}"
    parsed = frozenset(part.strip() for part in match.group(1).split(","))
    assert parsed == UNIT_JOB_EXTRAS, (
        f"parsed {sorted(parsed)} but UNIT_JOB_EXTRAS says {sorted(UNIT_JOB_EXTRAS)}"
    )


# ===========================================================================
# 2. Which top-level import names the unit job does NOT have.
# ===========================================================================

# Distribution name -> the top-level import name(s) it provides. Only the
# distributions pyproject's extras DECLARE are listed; transitive ones are handled
# by the two tables below. Hand-written because there is no offline way to map a
# distribution to its import name without the package installed, and asking the
# installed venv would make this file's answer a property of the venv -- which is
# the exact mistake this file guards.
#
# Keys are pip-normalised (lower-case, `-` not `_`); values are the IMPORT names,
# which are case-sensitive and often differ (`pyjwt` -> `jwt`, `flagembedding` ->
# `FlagEmbedding`, `pyinstaller` -> `PyInstaller`).
DECLARED_DIST_TO_IMPORTS: dict[str, tuple[str, ...]] = {
    "flagembedding": ("FlagEmbedding",),
    "anthropic": ("anthropic",),
    "boto3": ("boto3",),
    "build": ("build",),
    "hypothesis": ("hypothesis",),
    "cohere": ("cohere",),
    "einops": ("einops",),
    "fastapi": ("fastapi",),
    "google-genai": ("google.genai",),
    "httpx2": ("httpx2",),
    "lancedb": ("lancedb",),
    "litellm": ("litellm",),
    "mypy": ("mypy",),
    "networkx": ("networkx",),
    "opentelemetry-api": ("opentelemetry",),
    "opentelemetry-exporter-otlp-proto-http": ("opentelemetry",),
    "opentelemetry-sdk": ("opentelemetry",),
    "opentelemetry-util-genai": ("opentelemetry",),
    "pyarrow": ("pyarrow",),
    "pyinstaller": ("PyInstaller",),
    "pyjwt": ("jwt",),
    "pytest": ("pytest",),
    "pytest-asyncio": ("pytest_asyncio",),
    "pytest-cov": ("pytest_cov",),
    "pytest-socket": ("pytest_socket",),
    "pytest-timeout": ("pytest_timeout",),
    "pyvis": ("pyvis",),
    "qdrant-client": ("qdrant_client",),
    "ragatouille": ("ragatouille",),
    "ruff": ("ruff",),
    "semgrep": ("semgrep",),
    "sentence-transformers": ("sentence_transformers",),
    "torch": ("torch",),
    "transformers": ("transformers",),
    "twine": ("twine",),
    "uvicorn": ("uvicorn",),
    "voyageai": ("voyageai",),
    "watchdog": ("watchdog",),
    "watchfiles": ("watchfiles",),
}

# Declared by an extra the unit job OMITS, yet importable there anyway. Each entry
# names the chain, because each is a dependency of something else and can break
# upstream without any change to this repo.
PRESENT_ANYWAY: dict[str, str] = {
    # [sparse] declares both; [local]'s sentence-transformers requires
    # torch>=1.11.0 and transformers>=4.41.0, so [sparse] adds no install weight.
    "torch": "sentence-transformers requires torch>=1.11.0 ([local])",
    "transformers": "sentence-transformers requires transformers>=4.41.0 ([local])",
    # [lance] declares pyarrow; [bge-code]'s FlagEmbedding requires datasets, which
    # requires pyarrow>=21.
    "pyarrow": "FlagEmbedding -> datasets -> pyarrow ([bge-code])",
    # [serve] declares both; [dev] declares fastapi directly. uvicorn and watchfiles
    # (also declared by [serve]/[watch] respectively) no longer need an entry here at
    # all: [dev] now declares both of THEM directly too (as of this fix), so they are
    # already inside `_declared_imports_for(UNIT_JOB_EXTRAS)` and the derivation's own
    # subtraction removes them from `declared_absent` before this table is even
    # consulted -- unlike before this fix, when they were only reachable via
    # packages/trelix-mcp's dependency graph and had to be excused here as SOFT.
    "fastapi": "declared directly by [dev]",
    # [binary] declares both; [dev] declares pyinstaller and [local]
    # sentence-transformers.
    "PyInstaller": "declared directly by [dev]",
    "sentence_transformers": "declared directly by [local]",
    # [knowledge-graph] declares networkx, which is a base dependency.
    "networkx": "base dependency of the project",
}

# Absent, but declared by no extra at all -- so the derivation below cannot reach
# them. Listed explicitly because tests DO import them.
ABSENT_TRANSITIVELY: dict[str, str] = {
    "botocore": "transitive of boto3 ([bedrock]); absent because boto3 is",
}

#: The answer. Top-level import names a test running in CI's unit job may NOT
#: import without a guard.
UNIT_JOB_ABSENT_IMPORTS = frozenset(
    {
        "anthropic",
        "boto3",
        "botocore",
        "cohere",
        "google.genai",
        "lancedb",
        "litellm",
        "qdrant_client",
        "ragatouille",
        "semgrep",
        "voyageai",
        "watchdog",
    }
)


def _extras() -> dict[str, list[str]]:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def _declared_imports_for(extras: set[str]) -> set[str]:
    """Import names the given extras DECLARE, following ``trelix[...]`` aggregates."""
    todo = set(extras)
    seen: set[str] = set()
    names: set[str] = set()
    table = _extras()
    while todo:
        extra = todo.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for req in table[extra]:
            spec = req.split(";")[0].strip()
            dist = re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0].strip()
            key = dist.lower().replace("_", "-")
            if key == "trelix":
                inner = re.search(r"\[([^\]]+)\]", spec)
                if inner:
                    todo |= {p.strip() for p in inner.group(1).split(",")}
                continue
            assert key in DECLARED_DIST_TO_IMPORTS, (
                f"pyproject's [{extra}] extra declares {dist!r}, which has no entry in "
                f"DECLARED_DIST_TO_IMPORTS. Add it (key pip-normalised, value the import "
                f"name) so the absent-package derivation below can classify it."
            )
            names |= set(DECLARED_DIST_TO_IMPORTS[key])
    return names


def test_absent_table_is_exactly_what_the_omitted_extras_declare() -> None:
    """Re-derive the absent set from pyproject; a drifting table must fail.

    The derivation: import names declared by the extras the unit job OMITS, minus
    those the unit job gets anyway (``PRESENT_ANYWAY``), plus the ones no extra
    declares (``ABSENT_TRANSITIVELY``). Set-equality both ways, so a NEW extra with
    a new package fails here until somebody classifies it -- which is the whole
    point: the classification is the thinking, and it must not be skippable.

    MUTATION that must make this fail: remove ``"cohere"`` from
    ``UNIT_JOB_ABSENT_IMPORTS``, or add an extra to ``UNIT_JOB_EXTRAS``.
    """
    all_extras = set(_extras())
    omitted = all_extras - set(UNIT_JOB_EXTRAS)
    assert omitted, "the unit job installs every extra; this derivation is vacuous"

    declared_absent = _declared_imports_for(omitted) - _declared_imports_for(set(UNIT_JOB_EXTRAS))
    derived = (declared_absent - set(PRESENT_ANYWAY)) | set(ABSENT_TRANSITIVELY)

    assert derived - UNIT_JOB_ABSENT_IMPORTS == set(), (
        f"declared by an omitted extra and unclassified: "
        f"{sorted(derived - UNIT_JOB_ABSENT_IMPORTS)}. Either add it to "
        f"UNIT_JOB_ABSENT_IMPORTS, or to PRESENT_ANYWAY with the dependency chain that "
        f"puts it in the unit job."
    )
    assert UNIT_JOB_ABSENT_IMPORTS - derived == set(), (
        f"listed as absent but no omitted extra declares it (and it is not in "
        f"ABSENT_TRANSITIVELY): {sorted(UNIT_JOB_ABSENT_IMPORTS - derived)}"
    )


def test_present_anyway_entries_are_not_also_claimed_absent() -> None:
    """The two tables must not overlap, or the derivation above is self-cancelling.

    MUTATION that must make this fail: add ``"torch"`` to
    ``UNIT_JOB_ABSENT_IMPORTS`` while leaving it in ``PRESENT_ANYWAY``.
    """
    overlap = sorted(set(PRESENT_ANYWAY) & UNIT_JOB_ABSENT_IMPORTS)
    assert overlap == [], f"claimed both present and absent in the unit job: {overlap}"


# ===========================================================================
# 3. RULE 1 -- no unguarded import of a package the unit job lacks.
# ===========================================================================

# The one site where a module-scope ``sys.modules`` injection is the guard, rather
# than importorskip or find_spec. Explicit, with the mechanism named, because the
# AST rule below cannot recognise it: tests/unit/test_watcher.py calls
# ``_inject_fake_watchdog()`` at module scope, which registers a stub package under
# ``sys.modules["watchdog"]``, so the later ``import watchdog.observers`` resolves
# from the cache and never reaches the (absent) real package. Verified by running
# the file with watchdog hidden at sys.meta_path: all tests pass.
STUB_GUARDED_IMPORT_SITES: dict[str, str] = {
    "unit/test_watcher.py": "watchdog",
}


def _guarded_roots(node: ast.AST) -> set[str]:
    """Roots named by ``pytest.importorskip(...)`` / ``find_spec(...)`` inside *node*."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute) and func.attr in {"importorskip", "find_spec"}:
            if sub.args and isinstance(sub.args[0], ast.Constant):
                value = sub.args[0].value
                if isinstance(value, str):
                    out.add(value.split(".")[0])
    return out


def _unguarded_absent_imports(source: str, absent: frozenset[str]) -> list[tuple[int, str]]:
    """(lineno, name) for every import of an *absent* root with no guard in scope.

    "In scope" = the same function body, or module scope, contains an
    ``importorskip``/``find_spec`` naming that root. Roots are compared at the top
    level, so ``google.genai`` is matched by its first segment as well as whole.
    """
    tree = _parse(source)
    roots = {name.split(".")[0] for name in absent}
    hits: list[tuple[int, str]] = []

    def visit(node: ast.AST, guards: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = guards
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = guards | _guarded_roots(child)
            if isinstance(child, ast.Import):
                for alias in child.names:
                    root = alias.name.split(".")[0]
                    if root in roots and root not in inner:
                        hits.append((child.lineno, f"import {alias.name}"))
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                root = child.module.split(".")[0]
                if root in roots and root not in inner:
                    hits.append((child.lineno, f"from {child.module} import ..."))
            visit(child, inner)

    visit(tree, _guarded_roots(tree))
    return hits


def test_no_unit_test_imports_a_ci_absent_package_without_a_guard() -> None:
    """RULE 1. ``import anthropic`` in a unit test is red on all four CI legs.

    Direction (a): loud, and the cheapest one to prevent. A test that needs an
    absent SDK must reach it through ``pytest.importorskip``, which skips with a
    reason naming what was out of reach.

    MUTATION that must make this fail: add a bare ``import cohere`` to any file
    under tests/unit.
    """
    offenders: list[str] = []
    for path in sorted(_UNIT_DIR.rglob("test_*.py")):
        key = path.relative_to(_TESTS_DIR).as_posix()
        absent = UNIT_JOB_ABSENT_IMPORTS
        stubbed = STUB_GUARDED_IMPORT_SITES.get(key)
        if stubbed is not None:
            absent = frozenset(n for n in absent if n.split(".")[0] != stubbed)
        for lineno, what in _unguarded_absent_imports(path.read_text(encoding="utf-8"), absent):
            offenders.append(f"{key}:{lineno}  {what}")
    assert offenders == [], (
        "these unit tests import a package CI's unit job does not install "
        f"({CI_JOB_PIP_INSTALLS[UNIT_JOB][0]}):\n  " + "\n  ".join(offenders)
    )


def test_rule_1_flags_an_unguarded_import_and_clears_a_guarded_one() -> None:
    """CONTROL for RULE 1, in both directions.

    Without this, a broken ``_unguarded_absent_imports`` -- wrong node types, an
    exception swallowed, the guard test scoped wrongly -- would return ``[]`` for
    everything and the rule above would be a green no-op. Two samples, so it also
    pins that the guard is actually honoured rather than the rule matching nothing.

    The absent set here is a LOCAL literal, not ``UNIT_JOB_ABSENT_IMPORTS``: this
    control is about the matcher, and reading the real table would make one edit to
    that table kill both this test and
    ``test_absent_table_is_exactly_what_the_omitted_extras_declare`` -- a coupled
    measurement in which the kill is not attributable.
    """
    absent = frozenset({"zzz_probe_pkg"})

    unguarded = "import zzz_probe_pkg\n\n\ndef test_x():\n    assert zzz_probe_pkg\n"
    assert [w for _, w in _unguarded_absent_imports(unguarded, absent)] == [
        "import zzz_probe_pkg"
    ], "RULE 1 no longer sees a bare module-scope import"

    from_form = "def test_x():\n    from zzz_probe_pkg.sub import thing\n\n    assert thing\n"
    assert [w for _, w in _unguarded_absent_imports(from_form, absent)] == [
        "from zzz_probe_pkg.sub import ..."
    ], "RULE 1 no longer sees the `from X import` form"

    guarded = (
        "import pytest\n\n\ndef test_x():\n"
        '    pytest.importorskip("zzz_probe_pkg")\n'
        "    import zzz_probe_pkg\n"
        "    assert zzz_probe_pkg\n"
    )
    assert _unguarded_absent_imports(guarded, absent) == [], (
        "RULE 1 flags an import that IS guarded by importorskip in the same function"
    )


def test_stub_guarded_sites_still_exist_and_still_import_what_they_claim() -> None:
    """A stale exemption is worse than none: it silences the rule for a real hit.

    MUTATION that must make this fail: point ``STUB_GUARDED_IMPORT_SITES`` at a
    renamed file, or at a file that no longer imports the stubbed package.
    """
    for key, stubbed in sorted(STUB_GUARDED_IMPORT_SITES.items()):
        path = _TESTS_DIR / key
        assert path.is_file(), f"STUB_GUARDED_IMPORT_SITES names a missing file: {key}"
        source = path.read_text(encoding="utf-8")
        hits = _unguarded_absent_imports(source, frozenset({stubbed}))
        assert hits, (
            f"{key} no longer has an unguarded `import {stubbed}`, so this exemption is "
            f"stale and is now hiding the rule from whatever else the file imports. "
            f"Delete the entry."
        )
        assert f'sys.modules["{stubbed}"]' in source or f"sys.modules['{stubbed}']" in source, (
            f"{key} is exempted on the grounds that it injects a stub under "
            f"sys.modules[{stubbed!r}], and that injection is gone"
        )


# ===========================================================================
# 4. RULE 2 -- no assertion over a third-party env namespace by prefix.
# ===========================================================================

#: Prefixes trelix owns, mirroring ``tests/_env_isolation.SCRUB_PREFIXES``. An
#: assertion over one of these is a claim about trelix's own namespace and is fine.
OWNED_ENV_PREFIXES = ("TRELIX_",)


def _environ_prefix_literals(source: str) -> list[tuple[int, tuple[str, ...]]]:
    """(lineno, literals) for each ``os.environ`` iteration filtered by a literal prefix."""

    def is_environ(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            return isinstance(node.value, ast.Name) and node.value.id == "os"
        return isinstance(node, ast.Name) and node.id == "environ"

    out: list[tuple[int, tuple[str, ...]]] = []
    for node in ast.walk(_parse(source)):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            generators: list[ast.AST] = list(node.generators)
        elif isinstance(node, ast.For):
            generators = [node]
        else:
            continue
        for gen in generators:
            iterated = getattr(gen, "iter", None)
            if iterated is None:
                continue
            # `os.environ`, or a one-call wrapper like list(os.environ).
            targets = [iterated]
            if isinstance(iterated, ast.Call):
                targets = list(iterated.args)
            if not any(is_environ(t) for t in targets):
                continue
            for cond in getattr(gen, "ifs", []):
                for sub in ast.walk(cond):
                    if not (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "startswith"
                    ):
                        continue
                    literals: list[str] = []
                    for arg in sub.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            literals.append(arg.value)
                        elif isinstance(arg, ast.Tuple):
                            literals += [
                                e.value
                                for e in arg.elts
                                if isinstance(e, ast.Constant) and isinstance(e.value, str)
                            ]
                    if literals:
                        out.append((sub.lineno, tuple(literals)))
    return out


def test_no_test_claims_a_third_party_env_namespace_by_prefix() -> None:
    """RULE 2. The exact shape that turned all four legs red.

    A prefix filter over ``os.environ`` with a literal that trelix does not own is
    a claim about somebody else's namespace. ``AZURE_EXTENSION_DIR`` is the counter-
    example that already happened; ``JAVA_HOME_11_X64``, ``ANDROID_NDK_ROOT``,
    ``DOTNET_ROOT``, ``CONDA``, ``RUNNER_TEMP`` and ~40 more are waiting behind it.

    A test that needs breadth should assert over a DERIVED name set (what config.py
    declares) rather than a namespace -- which is what
    tests/unit/test_env_isolation_covers_config_aliases.py now does.

    MUTATION that must make this fail: restore
    ``assert sorted(k for k in os.environ if k.upper().startswith("AZURE_")) == []``
    anywhere under tests/.
    """
    offenders: list[str] = []
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        key = path.relative_to(_TESTS_DIR).as_posix()
        for lineno, literals in _environ_prefix_literals(path.read_text(encoding="utf-8")):
            unowned = [lit for lit in literals if not lit.startswith(OWNED_ENV_PREFIXES)]
            if unowned:
                offenders.append(f"{key}:{lineno}  startswith({unowned!r})")
    assert offenders == [], (
        "these iterate os.environ filtered by a namespace trelix does not own; a CI "
        "runner sets names in those namespaces that no trelix code claims:\n  "
        + "\n  ".join(offenders)
    )


def test_rule_2_flags_the_assertion_that_actually_broke_ci() -> None:
    """CONTROL for RULE 2, using the original line verbatim.

    The rule above reports ZERO on this tree, which is indistinguishable from a
    broken matcher. This feeds it the code that was reverted out of
    tests/unit/test_env_isolation_covers_config_aliases.py and requires a hit, plus
    an owned-prefix sample that must NOT be flagged.
    """
    broke_ci = (
        "import os\n\n\ndef test_x():\n"
        '    assert sorted(k for k in os.environ if k.upper().startswith("AZURE_")) == []\n'
    )
    found = _environ_prefix_literals(broke_ci)
    assert [lits for _, lits in found] == [("AZURE_",)], (
        f"RULE 2 no longer recognises the assertion that turned four legs red: {found}"
    )

    owned = (
        "import os\n\n\ndef test_x():\n"
        '    assert [k for k in os.environ if k.startswith("TRELIX_")] == []\n'
    )
    hits = _environ_prefix_literals(owned)
    assert [lits for _, lits in hits] == [("TRELIX_",)], "RULE 2 stopped seeing the owned case"
    assert all(lit.startswith(OWNED_ENV_PREFIXES) for _, lits in hits for lit in lits), (
        "the owned-prefix sample would now be reported as an offender"
    )


# ===========================================================================
# 5. RULE 3 -- every WHOLESALE module-level skip is declared.
# ===========================================================================

# tests/unit/test_marker_taxonomy.py derives `requires_extra` from a regex for
# module-scope ``pytest.importorskip``. That criterion is mechanical and correct
# for what it names, and it is BLIND to the other construct that skips a whole
# module: ``pytest.skip(..., allow_module_level=True)``. Measured on this tree,
# tests/unit/test_assembler_backcompat_golden.py takes 177 assertions dark that way
# and the run still exits 0 -- with the file's own docstring saying "Silently
# skipping is how this suite went dark the first time".
#
# file -> (what it needs, is it guaranteed by CI's unit job?)
ALLOW_MODULE_LEVEL_SKIPS: dict[str, tuple[str, bool]] = {
    "unit/test_assembler_backcompat_golden.py": ("the v2.12.0 git tag", True),
    # tests/integration is credential-gated and does not run in the unit job.
    "integration/test_llm_e2e.py": ("TRELIX_LIVE_LLM_TESTS + live credentials", False),
}


def _allow_module_level_skip_lines(source: str) -> list[int]:
    out: list[int] = []
    for node in ast.walk(_parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skip"
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "allow_module_level" and isinstance(kw.value, ast.Constant):
                if kw.value.value is True:
                    out.append(node.lineno)
    return out


def test_every_allow_module_level_skip_is_declared() -> None:
    """RULE 3. A whole file may not go dark without being written down.

    Set equality BOTH ways: a new ``allow_module_level=True`` fails until declared,
    and a declaration for a file that no longer has one fails as stale. This is the
    gap in the marker taxonomy's importorskip-only derivation, not a duplicate of it.

    MUTATION that must make this fail: add
    ``pytest.skip("x", allow_module_level=True)`` to any file under tests/, or
    delete the ``unit/test_assembler_backcompat_golden.py`` entry above.
    """
    derived: dict[str, list[int]] = {}
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        lines = _allow_module_level_skip_lines(path.read_text(encoding="utf-8"))
        if lines:
            derived[path.relative_to(_TESTS_DIR).as_posix()] = lines

    assert derived, (
        "found no allow_module_level skip anywhere under tests/, which means this "
        "detector stopped matching rather than that the suite changed -- "
        "tests/unit/test_assembler_backcompat_golden.py has one"
    )
    table = set(ALLOW_MODULE_LEVEL_SKIPS)
    assert set(derived) - table == set(), (
        f"these files can skip WHOLESALE and are undeclared: "
        f"{sorted(set(derived) - table)}. Note the marker taxonomy does NOT see this "
        f"construct, so the file also carries no requires_extra marker."
    )
    assert table - set(derived) == set(), (
        f"declared but no allow_module_level skip found: {sorted(table - set(derived))}"
    )


def test_no_module_scope_pytestmark_skips_a_whole_file() -> None:
    """The third wholesale construct, pinned at zero with a control.

    ``pytestmark = pytest.mark.skipif(...)`` at module scope skips every test in the
    file and is invisible to BOTH the taxonomy's importorskip regex and RULE 3.
    Nothing under tests/ uses it today; this keeps it that way rather than
    discovering the fourth mechanism after it costs a cycle.

    MUTATION that must make this fail: add
    ``pytestmark = pytest.mark.skipif(True, reason="x")`` to any file under tests/.
    """

    def module_scope_pytestmarks(source: str) -> list[int]:
        return [
            stmt.lineno
            for stmt in _parse(source).body
            if isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets)
        ]

    # Control first: the detector must see one when there is one.
    sample = 'import pytest\npytestmark = pytest.mark.skipif(True, reason="x")\n'
    assert module_scope_pytestmarks(sample) == [2], "the pytestmark detector is broken"

    offenders = [
        f"{path.relative_to(_TESTS_DIR).as_posix()}:{lineno}"
        for path in sorted(_TESTS_DIR.rglob("test_*.py"))
        for lineno in module_scope_pytestmarks(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "module-scope `pytestmark` skips every test in the file and is seen by neither "
        f"the marker taxonomy nor RULE 3: {offenders}"
    )


# ===========================================================================
# 6. The one environment condition that is worth failing on, not skipping.
# ===========================================================================


def test_the_backcompat_golden_baseline_is_reachable_in_this_checkout() -> None:
    """177 assertions hang off one git tag; a shallow checkout silences them.

    tests/unit/test_assembler_backcompat_golden.py builds its baseline from
    ``git show v2.12.0:src/trelix/retrieval/assembler.py`` and, when that fails,
    skips at module level. Measured with a git shim that fails only that lookup:
    "1 skipped", EXIT=0, 177 assertions gone and nothing red. ci.yml carries
    ``fetch-depth: 0`` + ``fetch-tags: true`` for exactly this reason -- but a future
    depth reduction would be green here and blind there.

    Conditioned rather than absolute (rule 10): a source tree with no ``.git`` --
    an sdist, a vendored copy -- genuinely cannot read a tag, so it SKIPS naming
    what was out of reach. A git checkout, which is what CI and every dev tree is,
    must have the tag.

    MUTATION that must make this fail: point ``_REF`` at a tag that does not exist.
    """
    _REF = "v2.12.0"
    _PATH = "src/trelix/retrieval/assembler.py"

    inside = subprocess.run(  # noqa: S603
        ["git", "-C", str(_ROOT), "rev-parse", "--is-inside-work-tree"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip(
            f"{_ROOT} is not a git work tree, so the {_REF} baseline is out of reach here "
            f"and tests/unit/test_assembler_backcompat_golden.py legitimately skips. "
            f"This is NOT a pass."
        )

    probe = subprocess.run(  # noqa: S603
        ["git", "-C", str(_ROOT), "cat-file", "-e", f"{_REF}:{_PATH}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, (
        f"this IS a git checkout but {_REF}:{_PATH} is unreachable "
        f"({probe.stderr.strip()}). tests/unit/test_assembler_backcompat_golden.py will "
        f"skip at module level and 177 assertions will report nothing while the run "
        f"exits 0. Fix the checkout: actions/checkout needs fetch-depth: 0 and "
        f"fetch-tags: true."
    )
