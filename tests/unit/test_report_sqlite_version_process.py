"""Runs scripts/report_sqlite_version.py in a real child process and pins what the CI step sees.

"Informational" means the step can never turn a job red. That is a promise about the
PROCESS exit status, and `main()` returning 0 does not settle it (see
tests/unit/test_report_sqlite_version_exit_status.py for the in-process branches):

  * after `main` returns, the interpreter flushes stdout and stderr once more at shutdown,
    and a failure there replaces the status with 120 ("Exception ignored in ...");
  * a closed stdout (`>&-`) makes `sys.stdout` None, which used to be an AttributeError
    traceback and status 1;
  * `SystemExit` raised while importing is not an `Exception`.

The pipe tests have the shape of tests/unit/test_cli_closed_stdout.py (OPS-04): the read end
is closed BEFORE the child starts, so there is no race. The interpreter is bare (`-S`: no
site-packages, so no trelix), the way the Docker images and a fresh runner run it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from trelix.store.db import wal_reset_risk

from .test_report_sqlite_version import _SCRIPT

_BARE_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}


def _run_script(
    tmp_path: Path, fake_sqlite3: str | None, *, via_stdin: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run the script in a bare interpreter (`-S`: no site-packages, so no trelix).

    `fake_sqlite3` is written to `<tmp>/sqlite3.py` and put first on PYTHONPATH, so the
    script sees whatever SQLite this test chooses. None leaves the real one.
    """
    env = dict(_BARE_ENV)
    if fake_sqlite3 is not None:
        fake_dir = tmp_path / "fake"
        fake_dir.mkdir()
        (fake_dir / "sqlite3.py").write_text(fake_sqlite3, encoding="utf-8")
        env["PYTHONPATH"] = str(fake_dir)
    argv = [sys.executable, "-S", "-" if via_stdin else str(_SCRIPT)]
    stdin = _SCRIPT.read_text(encoding="utf-8") if via_stdin else None
    return subprocess.run(  # noqa: S603
        argv, input=stdin, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60
    )


def _fake_sqlite(dotted: str) -> str:
    parts = ", ".join(dotted.split("."))
    return f'sqlite_version = "{dotted}"\nsqlite_version_info = ({parts})\n'


@pytest.mark.parametrize("via_stdin", [False, True], ids=["file", "stdin"])
def test_affected_sqlite_is_a_warning_and_exit_zero(tmp_path: Path, via_stdin: bool) -> None:
    done = _run_script(tmp_path, _fake_sqlite("3.46.1"), via_stdin=via_stdin)
    assert done.returncode == 0
    lines = done.stdout.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("SQLite 3.46.1 linked by ")
    assert lines[0].endswith(
        "range table: embedded copy (trelix not importable: ModuleNotFoundError)"
    )
    assert lines[1].startswith("::warning::SQLite 3.46.1 (") and lines[1].endswith(
        " Informational only; see docs/TROUBLESHOOTING.md."
    )
    assert done.stderr == ""


def test_fixed_sqlite_is_a_notice_and_exit_zero(tmp_path: Path) -> None:
    done = _run_script(tmp_path, _fake_sqlite("3.51.3"))
    assert done.returncode == 0
    assert done.stdout.splitlines()[1].startswith("::notice::SQLite 3.51.3 (")


def test_python_without_sqlite3_warns_and_exits_zero(tmp_path: Path) -> None:
    done = _run_script(tmp_path, 'raise ImportError("no sqlite3 in this build")\n')
    assert done.returncode == 0
    assert done.stdout.splitlines() == [
        "::warning::The SQLite version report could not run (ImportError: no sqlite3 in this "
        "build). Informational only; this does not fail the job."
    ]


def test_a_sqlite3_without_version_attributes_warns_and_exits_zero(tmp_path: Path) -> None:
    done = _run_script(tmp_path, "")
    assert done.returncode == 0
    assert done.stdout.startswith("::warning::The SQLite version report could not run (Attr")


def test_a_sqlite3_that_exits_while_being_imported_still_exits_zero(tmp_path: Path) -> None:
    done = _run_script(tmp_path, "import sys\nsys.exit(3)\n")
    assert done.returncode == 0
    assert done.stdout.splitlines() == [
        "::warning::The SQLite version report could not run (SystemExit: 3). "
        "Informational only; this does not fail the job."
    ]


def test_real_interpreter_run_agrees_with_the_shipped_check() -> None:
    import sqlite3

    done = subprocess.run(  # noqa: S603
        [sys.executable, str(_SCRIPT)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0
    summary, annotation = done.stdout.splitlines()
    assert summary.startswith(f"SQLite {sqlite3.sqlite_version} linked by ")
    kind = "::warning::" if wal_reset_risk(sqlite3.sqlite_version_info) else "::notice::"
    assert annotation.startswith(f"{kind}SQLite {sqlite3.sqlite_version} (")


@contextlib.contextmanager
def _pipe_nobody_reads() -> Iterator[int]:
    """The write end of a pipe whose read end is already closed."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        yield write_fd
    finally:
        os.close(write_fd)


def _run_bare(
    *,
    stdout: int,
    stderr: int,
    prefix: tuple[str, ...] = (),
    via_stdin: bool = False,
) -> subprocess.CompletedProcess[str]:
    """The script in a bare interpreter, with the given stdout and stderr (fds or PIPE)."""
    argv = [*prefix, sys.executable, "-S", "-" if via_stdin else str(_SCRIPT)]
    source = _SCRIPT.read_text(encoding="utf-8") if via_stdin else None
    return subprocess.run(  # noqa: S603
        argv, input=source, stdout=stdout, stderr=stderr, env=_BARE_ENV, text=True, timeout=60
    )


@pytest.mark.parametrize("via_stdin", [False, True], ids=["file", "stdin"])
def test_exit_status_is_zero_when_stdout_is_a_pipe_nobody_reads(via_stdin: bool) -> None:
    """Before: status 120 and "Exception ignored in <stdout>" at interpreter shutdown."""
    with _pipe_nobody_reads() as write_fd:
        done = _run_bare(stdout=write_fd, stderr=subprocess.PIPE, via_stdin=via_stdin)

    assert done.returncode == 0, done.stderr
    assert "could not write the report" in done.stderr  # the failure path really ran
    assert "Traceback" not in done.stderr
    assert "Exception ignored" not in done.stderr


def test_exit_status_is_zero_when_stdout_and_stderr_are_both_pipes_nobody_reads() -> None:
    """The stderr note fails too; the interpreter flushes stderr at shutdown as well."""
    with _pipe_nobody_reads() as out_fd, _pipe_nobody_reads() as err_fd:
        done = _run_bare(stdout=out_fd, stderr=err_fd)
    assert done.returncode == 0


_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="closing a descriptor before exec needs a POSIX shell; the unit matrix is ubuntu-only",
)


@_POSIX_ONLY
@pytest.mark.parametrize(
    ("redirect", "stdout_is_unread_pipe"),
    [(">&-", False), (">&- 2>&-", False), ("2>&-", True)],
    ids=["stdout-closed", "stdout-and-stderr-closed", "stdout-unread-stderr-closed"],
)
def test_exit_status_is_zero_when_a_standard_stream_is_closed(
    redirect: str, stdout_is_unread_pipe: bool
) -> None:
    """`python report_sqlite_version.py >&-` leaves sys.stdout None (was: status 1)."""
    prefix = ("/bin/sh", "-c", f'exec "$0" "$@" {redirect}')
    if stdout_is_unread_pipe:
        with _pipe_nobody_reads() as write_fd:
            done = _run_bare(stdout=write_fd, stderr=subprocess.PIPE, prefix=prefix)
    else:
        done = _run_bare(stdout=subprocess.PIPE, stderr=subprocess.PIPE, prefix=prefix)

    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr
