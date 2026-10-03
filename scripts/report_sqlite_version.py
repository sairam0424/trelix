#!/usr/bin/env python3
"""Report which SQLite library the running Python links, as a CI annotation.

WHY. SQLite 3.7.0 through 3.51.2 can lose committed WAL writes when two or more
connections write or checkpoint the same database at the same instant. The fix is in
3.51.3 and was backported to 3.50.7 and 3.44.6. trelix warns at runtime
(`trelix.store.db.wal_reset_risk`), but nobody knew which SQLite the interpreters CI
tests on, the Docker images and the interpreters PyInstaller freezes into the binaries
actually link. The CI runs that build or test them print it, so the answer comes from data.

WHAT. One plain line (version, interpreter, which range table decided) and one workflow
command: `::warning::` when the version is in the affected range, `::notice::` when it
is not. The range table is `trelix.store.db.wal_reset_risk` when trelix is importable,
and `embedded_wal_reset_risk` below otherwise. The embedded copy is deliberately a copy,
so this file runs on a bare interpreter; tests/unit/test_report_sqlite_version.py keeps it
in step with the real function. The number printed is the UPSTREAM one
(`sqlite3.sqlite_version`): a distro build can carry the fix under a lower number, so a
line saying "in the range" cannot settle whether a Debian image is really affected.

INFORMATIONAL ONLY. The PROCESS exits 0 on every path a step can meet: a Python built
without sqlite3, a module that calls `sys.exit` while being imported, a stdout that is a
pipe nobody reads, a stdout that is closed. A failure to report becomes a `::warning::`,
never a red job. The version line survives a failing range check or platform query: the
embedded copy answers instead, or the interpreter is named "unknown", and the label says
so. The one thing deliberately not contained is KeyboardInterrupt, so Ctrl-C still stops
the script. The workflows call this plainly, with no `continue-on-error` and no `|| true`,
because the exit code is this file's promise and tests pin it, including the exit status
of a real child process with a broken stdout (`main()` returning 0 is not enough: the
interpreter flushes stdout again at shutdown, and a failure there replaces the status with
120; see `silence_stream`). Only the script is held to that: a failing `docker run` (a
daemon error, an image without `python`) still fails its step.

RUNNING IT. `python scripts/report_sqlite_version.py`. For an image that does not carry
scripts/: `docker run --rm -i --entrypoint python <image> - < scripts/report_sqlite_version.py`.
`python -` reads the program from stdin, where `__file__` is `<stdin>` and names nothing on
disk, so nothing here may use it to find another file.

Everything that reaches a workflow command is clipped, ASCII-escaped and `%`/newline
escaped first (`sanitize_field`), so a strange version or platform string cannot start a
second command of its own.
"""

from __future__ import annotations

import contextlib
import os
import platform
import sys
from collections.abc import Callable
from typing import TextIO

RiskCheck = Callable[[tuple[int, ...]], bool]

# Longest a dynamic field (version, interpreter, exception text) may be. A real SQLite
# version is under 20 characters; anything longer is clipped rather than echoed.
_MAX_FIELD_CHARS = 80

# Printed with every annotation, whichever range table decided. It is text, so it is NOT
# derived from `trelix.store.db.wal_reset_risk`: when that table changes (a new fixed
# line, a moved boundary) this sentence and `embedded_wal_reset_risk` must be edited
# together with it. tests/unit/test_report_sqlite_version.py derives the boundaries from
# the real function over a sweep and fails when this sentence stops matching them.
_RANGE_SENTENCE = "(3.7.0 through 3.51.2; fixed in 3.51.3, 3.50.7 and 3.44.6)"


def embedded_wal_reset_risk(version_info: tuple[int, ...]) -> bool:
    """A copy of `trelix.store.db.wal_reset_risk`, for an interpreter without trelix.

    Affected: 3.7.0 (the first WAL release) and later, unless fixed. Fixed: 3.51.3 and
    later, 3.50.7 through 3.50.x, 3.44.6 through 3.44.x. Upstream numbers only; a distro
    build can carry the fix under a lower number.
    """
    if version_info < (3, 7, 0):
        return False
    is_fixed = (
        version_info >= (3, 51, 3)
        or (3, 50, 7) <= version_info < (3, 51, 0)
        or (3, 44, 6) <= version_info < (3, 45, 0)
    )
    return not is_fixed


def resolve_wal_reset_risk() -> tuple[RiskCheck, str]:
    """The range check to use, and a label saying which one it is.

    trelix's own function wins when it can be imported, so CI reports with the table the
    shipped warning uses. Any failure to import falls back to the embedded copy and says
    so in the label, rather than failing the report.
    """
    try:
        from trelix.store.db import wal_reset_risk
    except (Exception, SystemExit) as exc:
        # SystemExit too: a module that calls `sys.exit` while trelix is imported must
        # not end this script with that status.
        reason = type(exc).__name__
        return embedded_wal_reset_risk, f"embedded copy (trelix not importable: {reason})"
    return wal_reset_risk, "trelix.store.db.wal_reset_risk"


def escape_workflow_data(text: str) -> str:
    """Escape `text` for use as the data of a GitHub workflow command.

    `%` first, so the escapes added for CR and LF are not escaped again.
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def sanitize_field(text: str) -> str:
    """Make one dynamic value safe to print on a workflow-command line.

    Clipped to `_MAX_FIELD_CHARS`, non-ASCII turned into backslash escapes (a Windows
    console in a legacy code page cannot encode it), then command-escaped. Linear in the
    length of the clipped text.
    """
    clipped = text[:_MAX_FIELD_CHARS] + ("..." if len(text) > _MAX_FIELD_CHARS else "")
    ascii_only = clipped.encode("ascii", "backslashreplace").decode("ascii")
    return escape_workflow_data(ascii_only)


def describe_interpreter() -> str:
    """For example `CPython 3.12.3 on Linux x86_64`.

    Never raises. The SQLite version is what this script exists to collect, so a platform
    query that fails costs the description (named by the exception type), not the line.
    """
    try:
        return (
            f"{platform.python_implementation()} {platform.python_version()} "
            f"on {platform.system()} {platform.machine()}"
        )
    except Exception as exc:
        return f"unknown interpreter ({type(exc).__name__})"


def evaluate_risk(
    risk_check: RiskCheck, table_source: str, version_info: tuple[int, ...]
) -> tuple[bool, str]:
    """Whether `version_info` is in the range, and the label of the table that said so.

    A range check that raises (or calls `sys.exit`) falls back to the embedded copy and
    says so in the label, so the version line is still printed.
    """
    try:
        return risk_check(version_info), table_source
    except (Exception, SystemExit) as exc:
        label = f"embedded copy (range check failed: {type(exc).__name__})"
        return embedded_wal_reset_risk(version_info), label


def format_annotation(*, at_risk: bool, version: str, interpreter: str) -> str:
    """The workflow command: `::warning::` when `at_risk`, `::notice::` when not."""
    subject = f"SQLite {sanitize_field(version)} ({sanitize_field(interpreter)})"
    if at_risk:
        return (
            f"::warning::{subject} is in the range where SQLite can lose committed WAL "
            f"writes {_RANGE_SENTENCE}. A distro build may carry the fix under a lower "
            "number. Informational only; see docs/TROUBLESHOOTING.md."
        )
    return (
        f"::notice::{subject} is outside the range where SQLite can lose committed WAL "
        f"writes {_RANGE_SENTENCE}."
    )


def format_summary_line(*, version: str, interpreter: str, table_source: str) -> str:
    """The plain log line printed before the annotation."""
    return (
        f"SQLite {sanitize_field(version)} linked by {sanitize_field(interpreter)}; "
        f"range table: {sanitize_field(table_source)}"
    )


def format_failure_annotation(exc: BaseException) -> str:
    """The warning printed when the report itself could not be produced."""
    what = sanitize_field(f"{type(exc).__name__}: {exc}")
    return (
        f"::warning::The SQLite version report could not run ({what}). "
        "Informational only; this does not fail the job."
    )


def collect_lines() -> list[str]:
    """The lines to print. May raise; `main` turns a raise into a warning."""
    # Imported here, not at module level: a Python built without sqlite3 must still get
    # as far as `main`, which reports it and exits 0.
    import sqlite3

    risk_check, table_source = resolve_wal_reset_risk()
    version = str(sqlite3.sqlite_version)
    interpreter = describe_interpreter()
    at_risk, table_source = evaluate_risk(
        risk_check, table_source, tuple(sqlite3.sqlite_version_info)
    )
    return [
        format_summary_line(version=version, interpreter=interpreter, table_source=table_source),
        format_annotation(at_risk=at_risk, version=version, interpreter=interpreter),
    ]


def emit(lines: list[str], out: TextIO) -> None:
    for line in lines:
        out.write(line + "\n")
    out.flush()


def silence_stream(stream: TextIO) -> None:
    """Aim `stream`'s file descriptor at the null device, so a later flush cannot fail.

    The interpreter flushes sys.stdout once more at shutdown. When the bytes that could
    not be written are still buffered, that flush fails again, Python prints "Exception
    ignored in ..." and the process exits 120 whatever `main` returned. Pointing the
    descriptor at the null device lets that last flush succeed.

    Best effort, and a no-op for a stream with no usable descriptor (a test double, a
    closed file). Changes the descriptor for the whole process, so `main` only applies it
    to the process's own stdout.
    """
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        # If this fails nothing more can be done; the stderr note already says why.
        with contextlib.suppress(OSError):
            os.dup2(null_fd, fd)
    finally:
        os.close(null_fd)


def note_write_failure(exc: Exception) -> None:
    """Say on stderr that the report could not be written, if stderr still exists.

    Last channel: when stderr is closed (`None`) there is nothing to report through. When
    stderr is broken too (both ends of a pipeline gone), it gets the same treatment as
    stdout: the interpreter also flushes stderr at shutdown and also answers a failure
    there with exit status 120.
    """
    stderr = sys.stderr
    if stderr is None:
        return
    try:
        stderr.write(
            f"report_sqlite_version: could not write the report: {sanitize_field(str(exc))}\n"
        )
        stderr.flush()
    except (OSError, ValueError):
        silence_stream(stderr)


def main(out: TextIO | None = None) -> int:
    """Print the report and return 0, whatever happened (KeyboardInterrupt excepted)."""
    stream: TextIO | None = sys.stdout if out is None else out
    if stream is None:
        # stdout is closed (`>&-`) or there is no console: nowhere to print, nothing to do.
        return 0
    try:
        lines = collect_lines()
    except (Exception, SystemExit) as exc:
        lines = [format_failure_annotation(exc)]
    try:
        emit(lines, stream)
    except (OSError, ValueError) as exc:
        # stdout is broken (a pipe nobody reads) or closed, so there is nothing to print
        # to. Neutralise the real stdout so the shutdown flush cannot turn this into
        # exit 120, then try stderr once.
        if stream is sys.stdout:
            silence_stream(stream)
        note_write_failure(exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
