"""Pins the smoke commands scripts/verify_release.py runs in each freshly installed PyPI package.

The script is run by hand once per release, after the artifacts are public, so nothing else in the
tree notices when its smoke commands stop proving anything. This file runs `check_pypi_installs`
with every subprocess, venv and temp directory faked (no network, no pip) and pins what it runs.

WHY. `trelix-mcp` was smoked with `import trelix_mcp; print(trelix_mcp.__version__)`, which reads a
4-line `__init__`. A published `trelix-mcp` that exits 1 on start (its server module imports core
names that an older core, kept by a stale `trelix>=` floor, does not have) therefore passed. It is
now run as `trelix-mcp --version`, like `trelix`, and its server module is imported in-process as
well; the two libraries (trelix-langchain, trelix-llama-index) have no console script and keep the
import.

Every expected value is a literal; the script is imported by path (scripts/ is not a package) and
nothing is imported from trelix. Mutations that fail it: drop "trelix-mcp" from `_CONSOLE_SCRIPTS`;
empty `_IMPORT_SMOKE_MODULES` (both are the import-only smoke again); `"--version"` -> `"--help"`;
drop the `dist_name != "trelix"` condition; `printed == expected` -> `expected in printed`;
`expected in printed` -> `printed == expected`; remove the exit-status check; return a failure
list that does not stop at the first failing command.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "verify_release.py"

_MCP_VERSION_CMD = ("<venv>/bin/trelix-mcp", "--version")
_MCP_IMPORT_CMD = (
    "<venv>/bin/python",
    "-c",
    "import trelix_mcp.server; print(trelix_mcp.__version__)",
)
_LANGCHAIN_CMD = (
    "<venv>/bin/python",
    "-c",
    "import trelix_langchain; print(trelix_langchain.__version__)",
)

# The commands, in order, with the venv directory written as <venv>, and what each one prints
# when the package is healthy. The version asked about is 9.9.9.
_HEALTHY: dict[tuple[str, ...], tuple[int, str, str]] = {
    ("<venv>/bin/trelix", "--version"): (0, "trelix 9.9.9\n", ""),
    _MCP_VERSION_CMD: (0, "trelix-mcp 9.9.9\n", ""),
    _MCP_IMPORT_CMD: (0, "9.9.9\n", ""),
    _LANGCHAIN_CMD: (0, "9.9.9\n", ""),
    (
        "<venv>/bin/python",
        "-c",
        "import trelix_llama_index; print(trelix_llama_index.__version__)",
    ): (0, "9.9.9\n", ""),
}


def _load_script() -> ModuleType:
    """Import scripts/verify_release.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_trelix_verify_release", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _run_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[tuple[str, ...], tuple[int, str, str]],
) -> tuple[list[str], list[tuple[str, ...]]]:
    """`check_pypi_installs("9.9.9")` with faked subprocesses: (failures, smoke commands run)."""
    responses = {**_HEALTHY, **overrides}
    venv_dir = re.compile(re.escape(str(tmp_path)) + r"/trelix-verify-pypi-[^/]+/venv")
    smoke_commands: list[tuple[str, ...]] = []

    def fake_mkdtemp(prefix: str) -> str:
        path = tmp_path / prefix
        path.mkdir()
        return str(path)

    def fake_create_venv(target: Path, *, with_pip: bool, symlinks: bool) -> None:
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "python").touch()

    def fake_run(
        cmd: list[str], *, cwd: Path | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        if cmd[1:3] == ["-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        key = tuple(venv_dir.sub("<venv>", part) for part in cmd)
        smoke_commands.append(key)
        returncode, stdout, stderr = responses[key]
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(script, "_run", fake_run)
    monkeypatch.setattr(script, "tempfile", SimpleNamespace(mkdtemp=fake_mkdtemp))
    monkeypatch.setattr(script, "venv", SimpleNamespace(create=fake_create_venv))
    return script.check_pypi_installs("9.9.9"), smoke_commands


def test_each_distribution_is_smoked_with_its_real_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failures, smoke_commands = _run_check(monkeypatch, tmp_path, {})

    assert failures == []
    assert smoke_commands == [
        ("<venv>/bin/trelix", "--version"),
        ("<venv>/bin/trelix-mcp", "--version"),
        (
            "<venv>/bin/python",
            "-c",
            "import trelix_mcp.server; print(trelix_mcp.__version__)",
        ),
        (
            "<venv>/bin/python",
            "-c",
            "import trelix_langchain; print(trelix_langchain.__version__)",
        ),
        (
            "<venv>/bin/python",
            "-c",
            "import trelix_llama_index; print(trelix_llama_index.__version__)",
        ),
    ]


_STALE_FLOOR_STDERR = "ModuleNotFoundError: No module named 'trelix.core.index_check'"

_OUTCOMES: dict[str, tuple[dict[tuple[str, ...], tuple[int, str, str]], str]] = {
    # The audited defect: the package imports fine, the command exits 1.
    "trelix-mcp-console-script-exits-1": (
        {_MCP_VERSION_CMD: (1, "", _STALE_FLOOR_STDERR)},
        "trelix-mcp: smoke command `trelix-mcp --version` failed (exit 1): " + _STALE_FLOOR_STDERR,
    ),
    # A stale floor makes both fail; only the first failing command of a distribution is reported.
    "trelix-mcp-command-and-server-import-both-fail": (
        {
            _MCP_VERSION_CMD: (1, "", _STALE_FLOOR_STDERR),
            _MCP_IMPORT_CMD: (1, "", _STALE_FLOOR_STDERR),
        },
        "trelix-mcp: smoke command `trelix-mcp --version` failed (exit 1): " + _STALE_FLOOR_STDERR,
    ),
    "trelix-mcp-console-script-prints-another-version": (
        {_MCP_VERSION_CMD: (0, "trelix-mcp 0.0.1\n", "")},
        "trelix-mcp: smoke command `trelix-mcp --version` printed 'trelix-mcp 0.0.1', "
        "expected 'trelix-mcp 9.9.9'",
    ),
    "trelix-console-script-prints-another-version": (
        {("<venv>/bin/trelix", "--version"): (0, "trelix 0.0.1\n", "")},
        "trelix: smoke command `trelix --version` printed 'trelix 0.0.1', expected 'trelix 9.9.9'",
    ),
    "trelix-mcp-server-import-fails": (
        {_MCP_IMPORT_CMD: (1, "", "ImportError: cannot import name 'x'")},
        "trelix-mcp: smoke command `python -c import trelix_mcp.server; "
        "print(trelix_mcp.__version__)` failed (exit 1): ImportError: cannot import name 'x'",
    ),
    "library-prints-another-version": (
        {_LANGCHAIN_CMD: (0, "0.0.1\n", "")},
        "trelix-langchain: smoke command `python -c import trelix_langchain; "
        "print(trelix_langchain.__version__)` printed '0.0.1', expected '9.9.9'",
    ),
    "library-prints-a-longer-version": (
        {_LANGCHAIN_CMD: (0, "19.9.9\n", "")},
        "trelix-langchain: smoke command `python -c import trelix_langchain; "
        "print(trelix_langchain.__version__)` printed '19.9.9', expected '9.9.9'",
    ),
}


@pytest.mark.parametrize(("overrides", "failure"), _OUTCOMES.values(), ids=_OUTCOMES.keys())
def test_a_broken_distribution_is_reported_once_by_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[tuple[str, ...], tuple[int, str, str]],
    failure: str,
) -> None:
    failures, _ = _run_check(monkeypatch, tmp_path, overrides)

    assert failures == [failure]


def test_a_console_script_may_print_more_than_its_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The console scripts are matched by containment, as they were before trelix-mcp joined."""
    overrides = {_MCP_VERSION_CMD: (0, "notice\ntrelix-mcp 9.9.9\n", "")}

    failures, _ = _run_check(monkeypatch, tmp_path, overrides)

    assert failures == []
