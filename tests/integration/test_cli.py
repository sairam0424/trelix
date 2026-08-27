"""
Integration tests for the trelix CLI (Phase 14).

Each test runs the `trelix` binary via subprocess against a small temporary
Python repo to verify the full end-to-end CLI wiring.

Because the CLI runs as a SUBPROCESS, none of the in-process isolation in
``tests/integration/conftest.py`` reaches it — monkeypatch cannot touch a
spawned process's config resolution. Child isolation is therefore done here, by
``_env()`` (scrubs inherited ``TRELIX_*``, pins the settings the tests depend on)
and ``_clean_cwd()`` (spawns from a directory with no ``.env``). See those two
helpers for the failure mode this prevents.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_VENV = _REPO_ROOT / ".venv"


def _resolve_trelix_bin() -> str:
    """
    Locate the ``trelix`` console script: interpreter's own bin dir, then the
    repo-relative venv, then PATH.

    This used to be hardcoded to ``<repo>/.venv/bin/trelix``, which only exists
    for local uv/venv development — a CI job that does ``pip install -e .`` into
    the interpreter installs the script onto PATH instead, and every test here
    would have died with FileNotFoundError. Raising (rather than skipping) on a
    genuine miss is deliberate: a silent module-level skip would let this file
    report "all green" while executing nothing.
    """
    # WHY sys.executable IS TRIED FIRST, and it is a bug fix rather than a reordering.
    # The two later probes are both ambient: _VENV is derived from __file__ and
    # shutil.which reads PATH. Both fail in the same ordinary situation --
    # `pytest tests/integration --collect-only` in a shell where the venv's bin was
    # never put on PATH (activation is a convenience, not a requirement), and any
    # git worktree that shares the primary clone's venv. Reproduced: RuntimeError at
    # COLLECTION time, strictly worse than a test failure because --collect-only
    # cannot even enumerate this file. Path(sys.executable).parent is the bin dir of
    # the interpreter ACTUALLY running these tests, derived from the running process
    # rather than the filesystem layout or the shell's configuration. The two ambient
    # probes are KEPT as fallbacks, not replaced: script-on-PATH-but-not-beside-the-
    # interpreter is legitimate (pipx, a wrapper shim, Windows Scripts vs bin).
    # TEST-SIDE resolution only -- nothing in src/ is involved.
    interpreter_bin = Path(sys.executable).parent
    for candidate in (interpreter_bin / "trelix", interpreter_bin / "trelix.exe"):
        if candidate.is_file():
            return str(candidate)

    venv_bin = _VENV / "bin" / "trelix"
    if venv_bin.exists():
        return str(venv_bin)
    on_path = shutil.which("trelix")
    if on_path:
        return on_path
    raise RuntimeError(
        f"`trelix` console script not found beside the running interpreter "
        f"({interpreter_bin}), at {venv_bin}, or on PATH. "
        "Install the package first (e.g. `pip install -e .`) — these tests drive "
        "the real CLI as a subprocess and cannot run without it."
    )


TRELIX_BIN = _resolve_trelix_bin()

# Settings the CLI subprocess must see. ``tests/integration/conftest.py``
# neutralizes feature flags with monkeypatch, but monkeypatch is IN-PROCESS
# ONLY — it cannot reach a `trelix` binary spawned via subprocess. So the child
# env has to be scrubbed and pinned explicitly here.
_CHILD_SETTINGS: dict[str, str] = {
    # The load-bearing one: a developer .env with an API-backed provider (e.g.
    # TRELIX_EMBEDDER_PROVIDER=azure, 3072-dim) makes every `search`/`query`
    # subcommand die with "Embedding dimension mismatch: index was built with
    # 384-dim vectors but the current provider 'azure' produces 3072-dim".
    "TRELIX_EMBEDDER_PROVIDER": "local",
    # Silence sentence-transformers / torch progress output in tests.
    "TOKENIZERS_PARALLELISM": "false",
}


@functools.cache
def _clean_cwd() -> Path:
    """
    An empty directory to spawn the CLI from (created once per test session).

    trelix's config classes use ``env_file=".env"``, which pydantic-settings
    resolves relative to the *process* cwd. Running the child from the repo root
    therefore loads the developer's ./.env. Scrubbing TRELIX_* out of the child
    env is not sufficient on its own, because the .env-file source reads the
    file directly rather than through os.environ — so we also start the child in
    a directory that has no .env at all. Belt and braces: either mechanism alone
    would fix today's failures, together they stop any future .env key leaking.
    """
    return Path(tempfile.mkdtemp(prefix="trelix_cli_clean_cwd_"))


def _env() -> dict[str, str]:
    """
    Build a scrubbed subprocess environment using the project venv.

    Every ``TRELIX_*`` variable inherited from the parent is dropped so the run
    is reproducible regardless of the developer's shell exports or .env, then
    the handful of settings the tests actually depend on are pinned.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRELIX_")}
    if _VENV.is_dir():
        # Local venv development: put the venv first so the child resolves its
        # own interpreter. Skipped when there is no venv (CI installs onto PATH).
        env["PATH"] = str(_VENV / "bin") + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(_VENV)
    env.update(_CHILD_SETTINGS)
    return env


def _run(*args: str) -> subprocess.CompletedProcess:
    """
    Run `trelix <args>` from a clean cwd inside the venv, capturing output.

    Every subcommand under test takes its target repo as an explicit path
    argument, so the child never needs to run *inside* the repo.
    """
    return subprocess.run(
        [TRELIX_BIN, *args],
        capture_output=True,
        text=True,
        env=_env(),
        cwd=str(_clean_cwd()),
    )


# ---------------------------------------------------------------------------
# Fixture: minimal Python repo
# ---------------------------------------------------------------------------


def _write_mini_repo(root: Path) -> Path:
    """Write a minimal Python + TypeScript repo under ``root`` and return it."""
    py_src = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            \"\"\"Return the sum.\"\"\"
            return a + b


        def multiply(a: int, b: int) -> int:
            \"\"\"Return the product.\"\"\"
            return a * b


        class Calculator:
            \"\"\"Simple calculator.\"\"\"

            def compute(self, a: int, b: int) -> int:
                return add(a, b)
    """)
    (root / "calc.py").write_text(py_src, encoding="utf-8")

    ts_src = textwrap.dedent("""\
        function greet(name: string): string {
            return `Hello, ${name}!`;
        }
    """)
    (root / "greet.ts").write_text(ts_src, encoding="utf-8")

    return root


@pytest.fixture()
def mini_repo(tmp_path: Path) -> Path:
    """A fresh, NOT-yet-indexed mini repo — for the tests that index it themselves."""
    return _write_mini_repo(tmp_path)


@pytest.fixture(scope="session")
def indexed_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    A mini repo indexed ONCE by the real `trelix index` CLI, shared by the tests
    that only need *an* index to query against.

    Every one of those tests used to shell out to `trelix index` as setup, and
    each of those subprocesses paid the full sentence-transformers model load.
    Twelve index runs dominated this file's ~212 s runtime. Indexing once cuts
    that to three (this fixture plus the two tests that assert on indexing
    itself) without changing a single assertion: the subcommands under test are
    still exercised against an index built end-to-end by the real binary.
    """
    repo = _write_mini_repo(tmp_path_factory.mktemp("trelix_cli_indexed_repo"))
    result = _run("index", str(repo), "--provider", "local")
    assert result.returncode == 0, (
        f"shared index build failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_help_exits_zero() -> None:
    """trelix --help must exit 0."""
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "trelix" in result.stdout.lower()


def test_index_exits_zero(mini_repo: Path) -> None:
    """trelix index <repo> --provider local must exit 0."""
    result = _run("index", str(mini_repo), "--provider", "local")
    assert result.returncode == 0, (
        f"trelix index failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_index_creates_db(mini_repo: Path) -> None:
    """After indexing, .trelix/index.db must exist."""
    _run("index", str(mini_repo), "--provider", "local")
    db_path = mini_repo / ".trelix" / "index.db"
    assert db_path.exists(), f"DB not found at {db_path}"


def test_search_exits_zero(indexed_repo: Path) -> None:
    """trelix search <repo> <query> must exit 0 after indexing."""
    result = _run("search", str(indexed_repo), "function")
    assert result.returncode == 0, (
        f"trelix search failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_search_stdout_not_empty(indexed_repo: Path) -> None:
    """trelix search must produce non-empty stdout."""
    result = _run("search", str(indexed_repo), "function")
    assert result.stdout.strip(), f"trelix search produced empty stdout.\nstderr: {result.stderr}"


def test_search_json_flag(indexed_repo: Path) -> None:
    """trelix search --json must output valid JSON with status=ok."""
    result = _run("search", str(indexed_repo), "function", "--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "ok"
    assert "results" in data


def test_stats_exits_zero(indexed_repo: Path) -> None:
    """trelix stats <repo> must exit 0 after indexing."""
    result = _run("stats", str(indexed_repo))
    assert result.returncode == 0, (
        f"trelix stats failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_stats_output_contains_counts(indexed_repo: Path) -> None:
    """trelix stats output must mention files and symbols."""
    result = _run("stats", str(indexed_repo))
    combined = result.stdout + result.stderr
    # The Rich table should include these labels
    assert "Files" in combined or "files" in combined, (
        f"'Files' not in output.\nstdout: {result.stdout}"
    )


def test_update_index_exits_zero(indexed_repo: Path) -> None:
    """trelix update-index must exit 0 on a file that was already indexed."""
    calc_py = indexed_repo / "calc.py"
    result = _run(
        "update-index",
        str(indexed_repo),
        str(calc_py),
        "--provider",
        "local",
    )
    assert result.returncode == 0, (
        f"trelix update-index failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_update_index_returns_json(indexed_repo: Path) -> None:
    """trelix update-index must print valid JSON with status=ok."""
    calc_py = indexed_repo / "calc.py"
    result = _run(
        "update-index",
        str(indexed_repo),
        str(calc_py),
        "--provider",
        "local",
    )
    data = json.loads(result.stdout)
    assert data["status"] == "ok", f"Expected status=ok, got {data}"


def test_query_exits_zero(indexed_repo: Path) -> None:
    """trelix query <repo> <query> must exit 0 after indexing."""
    result = _run("query", str(indexed_repo), "add function")
    assert result.returncode == 0, (
        f"trelix query failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_ask_exits_zero_local(indexed_repo: Path) -> None:
    """trelix ask with provider=local (no API key) must exit 0 and print context."""
    result = _run("ask", str(indexed_repo), "what does add do?", "--provider", "local")
    assert result.returncode == 0, (
        f"trelix ask failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_invalid_repo_path_exits_one() -> None:
    """trelix index on a non-existent path must exit with code 1."""
    result = _run("index", "/does/not/exist/at/all")
    assert result.returncode == 1, f"Expected exit code 1 for invalid path, got {result.returncode}"


def test_stats_nonexistent_repo_exits_one() -> None:
    """trelix stats on a non-existent repo must exit with code 1."""
    result = _run("stats", "/no/such/path")
    assert result.returncode == 1, f"Expected exit code 1 for invalid path, got {result.returncode}"


def test_search_nonexistent_repo_exits_one() -> None:
    """trelix search on a non-existent repo must exit with code 1."""
    result = _run("search", "/no/such/path", "query")
    assert result.returncode == 1, f"Expected exit code 1 for invalid path, got {result.returncode}"


def test_search_json_finds_the_right_symbol(indexed_repo: Path) -> None:
    """Content correctness, not just "non-empty": a query must surface the
    specific symbol it targets, not merely any result.

    Every other search assertion in this file (test_search_exits_zero,
    test_search_stdout_not_empty, test_search_json_flag) only checks that
    SOMETHING came back — none of them can tell a correct ranking from a
    degenerate one that always returns the same top hit regardless of query.
    """
    result = _run("search", str(indexed_repo), "add two numbers together", "--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "ok"
    symbols = {r["symbol"] for r in data["results"]}
    assert "add" in symbols, f"expected the add() function to surface; got {sorted(symbols)}"


def test_cli_version_matches_the_installed_package() -> None:
    """The installed console script's --version must agree with the real
    __version__ this test process's own trelix import reports.

    Catches the exact class of drift a stale editable install (pip install -e
    . run once, then source edited without a reinstall) or a half-applied
    release stamp can hide — the console script and the importable package
    could silently disagree about which version is running.
    """
    from trelix import __version__

    result = _run("--version")
    assert result.returncode == 0
    assert __version__ in result.stdout
