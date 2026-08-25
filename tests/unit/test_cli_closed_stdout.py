"""OPS-04 — `trelix stats | head` (or `| grep -q`) must not fail or crash.

The Windows binary smoke test in build-binaries.yml asserts
`dist/trelix.exe stats <repo> | grep -q "Symbols"`. `grep -q` exits the instant
it matches, which is *before* stats has finished writing (PR #163's fe125ce added a
provenance block after the table, widening the window). Every subsequent write then
hits a pipe with no reader. Two distinct failures follow, and both were repeatedly
filed as CI flakiness:

  * POSIX (errno EPIPE 32) — rich's Console.on_broken_pipe and click's own EPIPE
    handler each unilaterally `sys.exit(1)`. GitHub Actions runs `shell: bash` as
    `bash -e -o pipefail`, so a 1 from the *left* side of the pipeline fails the
    step even though `grep -q` matched. Silent: no traceback, nothing to read.
  * Windows (errno EINVAL 22) — WriteFile on a reader-less pipe does not map to
    BrokenPipeError, so rich (`except BrokenPipeError`) does not catch it and
    click (`if e.errno == errno.EPIPE`) re-raises it. The OSError escapes, and in
    the frozen binary that prints a rich traceback plus PyInstaller's
    "Failed to execute script main".

A closed consumer is not an error for a CLI — it means the reader is done. These
tests pin the well-behaved response: exit 0, write no traceback.

The subprocess shape is not incidental. The failure lives in process teardown
(interpreter-shutdown flush) and in the entry point's exception handling, neither
of which typer's in-process CliRunner exercises.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

import pytest

# The console script installed by `[project.scripts] trelix = "trelix.cli.main:app"`
# does exactly this, so the test drives the real entry point rather than `-m`.
_ENTRY = "import sys; from trelix.cli.main import app; sys.exit(app())"


def _child_env() -> dict[str, str]:
    """Env that makes the child import the SAME trelix as this test process.

    Not `dict(os.environ)` alone: the venv may carry an editable install pointing at
    a different checkout, and then the child would exercise someone else's source
    and pass or fail for reasons unrelated to this worktree.
    """
    import trelix

    src_root = str(Path(trelix.__file__).resolve().parents[1])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing}" if existing else src_root
    return env


def _run_with_closed_consumer(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI with stdout on a pipe whose read end is already gone.

    The read end is closed *before* the child is spawned, so the pipe never has a
    reader and every write fails. `| head -1` is the same condition with a race in
    it — the CI failure is non-deterministic precisely because a small payload can
    fit the 64 KiB pipe buffer and finish before the consumer exits. Removing the
    race is what makes this a test rather than a coin flip.

    `_ENTRY` is written to a real `.py` file rather than passed via `python -c`.
    `-c` gives the child's top-level frame a `co_filename` of the literal string
    `"<string>"`, which has no meaning as a filesystem path. That is invisible under
    a normal `pytest` run, but discovered as a real trap while widening this
    project's mutation-coverage scope to `graph`: `_child_env()` below deliberately
    points the child's `PYTHONPATH` at whatever `trelix` package this test process
    itself has loaded, and under `scripts/mutation.py`'s mutmut driver that is the
    trampoline-instrumented copy inside `mutants/`. Calling into any trampoline-
    wrapped function (e.g. this test's `"graph"` parametrization constructing a
    `GraphBuilder`) makes mutmut's `record_trampoline_hit` walk the call stack via
    `Path(frame.f_code.co_filename).resolve(strict=True)`, and `Path("<string>")`
    has no filesystem entry to resolve — `FileNotFoundError`, uncaught, crashing the
    child with a traceback instead of the closed-consumer exit 0 this test asserts.
    A real file has a resolvable path regardless of what loads it, so this is a
    general robustness fix, not a mutmut-only workaround: identical behaviour for
    the CLI under test either way, verified by running this file, unmutated,
    before and after the change.
    """
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as entry_file:
        entry_file.write(_ENTRY)
        entry_path = entry_file.name
    try:
        proc = subprocess.Popen(
            [sys.executable, entry_path, *args],
            stdout=write_fd,
            stderr=subprocess.PIPE,
            env=_child_env(),
            text=True,
        )
        os.close(write_fd)
        _, stderr = proc.communicate(timeout=180)
        return subprocess.CompletedProcess(proc.args, proc.returncode, "", stderr)
    finally:
        os.unlink(entry_path)


@pytest.fixture
def indexed_repo(tmp_path: Path) -> Path:
    """A repo with an index present, so `stats` reaches its stdout writes.

    Only the schema is needed: `stats` counts rows and renders a table, and an
    empty index still emits the "Symbols" row the smoke test greps for. Built with
    Database() directly because `trelix index` needs an embedder.
    """
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "util.py").write_text("def add(a, b):\n    return a + b\n")

    db_path = IndexConfig(repo_path=str(repo)).db_path_absolute
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with Database(db_path):
        pass
    return repo


def test_stats_exits_zero_when_the_consumer_closed_the_pipe(indexed_repo: Path) -> None:
    """`trelix stats <repo> | grep -q Symbols` must not fail the pipeline.

    Under `bash -o pipefail` a non-zero status here fails the CI step even though
    grep matched — which is the entire OPS-04 red, mistaken for flakiness twice
    across two days.
    """
    result = _run_with_closed_consumer("stats", str(indexed_repo))

    assert result.returncode == 0, (
        f"exit {result.returncode} on a closed consumer; stderr={result.stderr!r}"
    )
    assert "Traceback" not in result.stderr
    assert result.stderr == "", f"unexpected diagnostics: {result.stderr!r}"


def test_stats_exits_zero_and_stays_quiet_on_a_windows_einval_stdout(
    indexed_repo: Path,
) -> None:
    """The Windows half: errno 22 from a reader-less pipe must not escape.

    Reproduced rather than skipped on POSIX, because the errno is the whole defect
    and a `skipif(sys.platform != "win32")` test would never run in this project's
    CI matrix (ci.yml is ubuntu-only for unit tests) — i.e. it would assert nothing
    and read as green. The shim replaces sys.stdout with a stream that fails the
    way Windows fails: OSError(EINVAL) instead of BrokenPipeError.
    """
    shim = (
        "import errno, io, sys\n"
        "class WindowsClosedPipe(io.TextIOBase):\n"
        # Windows' WriteFile on a pipe with no reader yields ERROR_NO_DATA /
        # ERROR_INVALID_HANDLE, which CPython surfaces as OSError(EINVAL) — not
        # BrokenPipeError, which is why rich and click both miss it.
        "    encoding = 'utf-8'\n"
        "    errors = 'replace'\n"
        "    def writable(self): return True\n"
        "    def isatty(self): return False\n"
        "    def fileno(self): return 1\n"
        "    def write(self, s): raise OSError(errno.EINVAL, 'Invalid argument')\n"
        "    def flush(self): raise OSError(errno.EINVAL, 'Invalid argument')\n"
        "sys.stdout = WindowsClosedPipe()\n"
        f"sys.argv = ['trelix', 'stats', {str(indexed_repo)!r}]\n"
        "from trelix.cli.main import app\n"
        "sys.exit(app())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", shim],
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"exit {result.returncode} on an EINVAL stdout; stderr={result.stderr!r}"
    )
    assert "Traceback" not in result.stderr, result.stderr
    assert "Invalid argument" not in result.stderr, result.stderr
    # 120 is CPython's "flushing sys.stdout at shutdown failed" code. It appears
    # whenever the handler exits without neutralising the stream, and it is a
    # distinct red from the one the handler was written to remove.
    assert result.returncode != 120


@pytest.mark.parametrize("command", ["telemetry", "graph"])
def test_sibling_output_commands_exit_zero_when_the_consumer_closed_the_pipe(
    indexed_repo: Path, command: str
) -> None:
    """Not a stats-only defect: every command writing through the shared consoles.

    `stats`, `telemetry` and `graph` all render through the module-level
    `console`/`err_console`, so a fix inside `stats` would leave these exposed. All
    three exited 1 before the fix. This is the guard against re-fixing the site.
    """
    result = _run_with_closed_consumer(command, str(indexed_repo))

    assert result.returncode == 0, (
        f"`{command}` exited {result.returncode} on a closed consumer; stderr={result.stderr!r}"
    )
    assert "Traceback" not in result.stderr, result.stderr


def test_only_closed_consumer_errnos_are_classified_as_one() -> None:
    """The guard must not become a blanket `except OSError: exit 0`.

    ENOSPC while writing output is a genuine failure a script must be able to
    detect. Only the closed-consumer errnos may be translated into exit 0.
    """
    from trelix.cli.main import _is_closed_consumer_error

    assert _is_closed_consumer_error(OSError(errno.EPIPE, "Broken pipe")) is True
    assert _is_closed_consumer_error(OSError(errno.EINVAL, "Invalid argument")) is True
    assert _is_closed_consumer_error(OSError(errno.ENOSPC, "No space left")) is False
    assert _is_closed_consumer_error(OSError(errno.EACCES, "Permission denied")) is False
    # errno=None happens for OSError raised without one; it is not a closed consumer.
    assert _is_closed_consumer_error(OSError("no errno at all")) is False


def _throwaway_app(exc: OSError) -> object:
    """A `_PipeSafeTyper` whose `boom` command raises `exc`.

    Two commands, not one: typer collapses a single-command app into a bare command
    and then rejects "boom" as an unexpected argument. Throwaway rather than the
    real `app` so a test never mutates the shipped command set.
    """
    from trelix.cli.main import _PipeSafeTyper

    throwaway = _PipeSafeTyper()

    @throwaway.command()
    def boom() -> None:
        raise exc

    @throwaway.command()
    def quiet() -> None:
        """Second command; exists only to keep the app a group."""

    return throwaway


def test_typer_entry_routes_a_closed_consumer_error_to_the_quiet_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entry wrapper, in isolation: an EINVAL write reaches the quiet exit.

    This is the layer covering writers the CLI does not own — typer's help renderer,
    Indexer's private Console — for the errno that produces the crash. The silencer
    is stubbed because the real one dup2()s /dev/null onto this pytest process's own
    fd 1, which breaks output capture for every test that runs after it.

    Only EINVAL is driven through the app: click intercepts EPIPE inside
    Command.main and exits 1 before the wrapper can see it, which is exactly why
    _PipeSafeConsole below is required and this wrapper alone is not enough.
    """
    from trelix.cli import main as cli_main

    class _Reached(Exception):
        pass

    def _stub() -> NoReturn:
        raise _Reached

    monkeypatch.setattr(cli_main, "_exit_quietly_after_closed_consumer", _stub)

    with pytest.raises(_Reached):
        _throwaway_app(OSError(errno.EINVAL, "Invalid argument"))(["boom"])  # type: ignore[operator]


def test_typer_entry_lets_a_real_oserror_reach_the_user() -> None:
    """Same wrapper, negative case: a non-pipe OSError must not be silenced.

    No stub needed — reaching the silencer at all would be the failure.
    """
    with pytest.raises(OSError) as excinfo:
        _throwaway_app(OSError(errno.ENOSPC, "No space left on device"))(["boom"])  # type: ignore[operator]
    assert excinfo.value.errno == errno.ENOSPC


def test_the_cli_entry_and_consoles_are_pipe_safe() -> None:
    """Pinned because the regression is a one-word revert.

    Putting `Console()` back for either console restores the silent POSIX exit 1
    while the EINVAL tests above still pass, so nothing else in this file would
    notice.
    """
    from rich.console import Console

    from trelix.cli.main import _PipeSafeConsole, _PipeSafeTyper, app, console, err_console

    assert isinstance(app, _PipeSafeTyper)
    assert isinstance(console, _PipeSafeConsole)
    assert isinstance(err_console, _PipeSafeConsole)
    assert _PipeSafeConsole.on_broken_pipe is not Console.on_broken_pipe


def test_broken_pipe_console_policy_is_exit_zero() -> None:
    """rich's on_broken_pipe raises SystemExit(1); the override must be 0.

    Run in a subprocess: the override redirects the process's own stdout fd to the
    null device, which is the correct behaviour and unusable in-process. The exit
    code is the whole result, so nothing needs to be read back from stdout.
    """
    probe = (
        "import sys\n"
        "from trelix.cli.main import _PipeSafeConsole\n"
        "try:\n"
        "    _PipeSafeConsole().on_broken_pipe()\n"
        "except SystemExit as exc:\n"
        # 3 distinguishes "wrong code" from "did not exit at all" (4).
        "    raise SystemExit(0 if exc.code == 0 else 3)\n"
        "raise SystemExit(4)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"rc={result.returncode} stderr={result.stderr!r}"
