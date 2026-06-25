"""
Integration tests for the trelix CLI (Phase 14).

Each test runs the `trelix` binary via subprocess against a small temporary
Python repo to verify the full end-to-end CLI wiring.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TRELIX_BIN = str(Path(__file__).parent.parent.parent / ".venv" / "bin" / "trelix")


def _env() -> dict[str, str]:
    """Build a minimal subprocess environment using the project venv."""
    venv = Path(__file__).parent.parent.parent / ".venv"
    env = os.environ.copy()
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv)
    # Silence sentence-transformers / torch progress output in tests
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _run(*args: str, repo: Path) -> subprocess.CompletedProcess:
    """Run `trelix <args>` inside the venv, capturing output."""
    return subprocess.run(
        [TRELIX_BIN, *args],
        capture_output=True,
        text=True,
        env=_env(),
    )


# ---------------------------------------------------------------------------
# Fixture: minimal Python repo
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_repo(tmp_path: Path) -> Path:
    """Create a minimal Python repo with a couple of functions."""
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
    (tmp_path / "calc.py").write_text(py_src, encoding="utf-8")

    ts_src = textwrap.dedent("""\
        function greet(name: string): string {
            return `Hello, ${name}!`;
        }
    """)
    (tmp_path / "greet.ts").write_text(ts_src, encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_help_exits_zero() -> None:
    """trelix --help must exit 0."""
    result = subprocess.run(
        [TRELIX_BIN, "--help"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "trelix" in result.stdout.lower()


def test_index_exits_zero(mini_repo: Path) -> None:
    """trelix index <repo> --provider local must exit 0."""
    result = _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    assert result.returncode == 0, (
        f"trelix index failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_index_creates_db(mini_repo: Path) -> None:
    """After indexing, .trelix/index.db must exist."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    db_path = mini_repo / ".trelix" / "index.db"
    assert db_path.exists(), f"DB not found at {db_path}"


def test_search_exits_zero(mini_repo: Path) -> None:
    """trelix search <repo> <query> must exit 0 after indexing."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    result = _run("search", str(mini_repo), "function", repo=mini_repo)
    assert result.returncode == 0, (
        f"trelix search failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_search_stdout_not_empty(mini_repo: Path) -> None:
    """trelix search must produce non-empty stdout."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    result = _run("search", str(mini_repo), "function", repo=mini_repo)
    assert result.stdout.strip(), f"trelix search produced empty stdout.\nstderr: {result.stderr}"


def test_search_json_flag(mini_repo: Path) -> None:
    """trelix search --json must output valid JSON with status=ok."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    result = _run("search", str(mini_repo), "function", "--json", repo=mini_repo)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "ok"
    assert "results" in data


def test_stats_exits_zero(mini_repo: Path) -> None:
    """trelix stats <repo> must exit 0 after indexing."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    result = _run("stats", str(mini_repo), repo=mini_repo)
    assert result.returncode == 0, (
        f"trelix stats failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_stats_output_contains_counts(mini_repo: Path) -> None:
    """trelix stats output must mention files and symbols."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    result = _run("stats", str(mini_repo), repo=mini_repo)
    combined = result.stdout + result.stderr
    # The Rich table should include these labels
    assert "Files" in combined or "files" in combined, (
        f"'Files' not in output.\nstdout: {result.stdout}"
    )


def test_update_index_exits_zero(mini_repo: Path) -> None:
    """trelix update-index must exit 0 on a file that was already indexed."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    calc_py = mini_repo / "calc.py"
    result = _run(
        "update-index",
        str(mini_repo),
        str(calc_py),
        "--provider",
        "local",
        repo=mini_repo,
    )
    assert result.returncode == 0, (
        f"trelix update-index failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_update_index_returns_json(mini_repo: Path) -> None:
    """trelix update-index must print valid JSON with status=ok."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    calc_py = mini_repo / "calc.py"
    result = _run(
        "update-index",
        str(mini_repo),
        str(calc_py),
        "--provider",
        "local",
        repo=mini_repo,
    )
    data = json.loads(result.stdout)
    assert data["status"] == "ok", f"Expected status=ok, got {data}"


def test_query_exits_zero(mini_repo: Path) -> None:
    """trelix query <repo> <query> must exit 0 after indexing."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    result = _run("query", str(mini_repo), "add function", repo=mini_repo)
    assert result.returncode == 0, (
        f"trelix query failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_ask_exits_zero_local(mini_repo: Path) -> None:
    """trelix ask with provider=local (no API key) must exit 0 and print context."""
    _run("index", str(mini_repo), "--provider", "local", repo=mini_repo)
    result = _run("ask", str(mini_repo), "what does add do?", "--provider", "local", repo=mini_repo)
    assert result.returncode == 0, (
        f"trelix ask failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_invalid_repo_path_exits_one() -> None:
    """trelix index on a non-existent path must exit with code 1."""
    result = subprocess.run(
        [TRELIX_BIN, "index", "/does/not/exist/at/all"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert result.returncode == 1, f"Expected exit code 1 for invalid path, got {result.returncode}"


def test_stats_nonexistent_repo_exits_one() -> None:
    """trelix stats on a non-existent repo must exit with code 1."""
    result = subprocess.run(
        [TRELIX_BIN, "stats", "/no/such/path"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert result.returncode == 1, f"Expected exit code 1 for invalid path, got {result.returncode}"


def test_search_nonexistent_repo_exits_one() -> None:
    """trelix search on a non-existent repo must exit with code 1."""
    result = subprocess.run(
        [TRELIX_BIN, "search", "/no/such/path", "query"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert result.returncode == 1, f"Expected exit code 1 for invalid path, got {result.returncode}"
