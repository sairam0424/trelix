"""Pins the EXIT STATUS of scripts/report_sqlite_version.py.

"Informational" means the step that runs the script can never turn a job red. That is a
promise about the PROCESS exit status, and `main()` returning 0 does not settle it:

  * after `main` returns, the interpreter flushes stdout and stderr once more at shutdown,
    and a failure there replaces the status with 120 ("Exception ignored in ...");
  * a closed stdout (`>&-`) makes `sys.stdout` None, which used to be an AttributeError
    traceback and status 1;
  * `SystemExit` raised while importing is not an `Exception`.

This file drives each branch of `main` and of `silence_stream` in process. The tests that
see the real status, in a child process, are in
tests/unit/test_report_sqlite_version_process.py.

In-process tests never let `silence_stream` touch this pytest process's own descriptors
(it would redirect pytest's output capture): the real function is only pointed at a
private pipe, everything else records the call.
"""

from __future__ import annotations

import contextlib
import errno
import io
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from .test_report_sqlite_version import (
    _HOSTILE_INPUTS,
    _LARGE_HOSTILE_INPUTS,
    _TIME_LIMIT_SECONDS,
    _seconds,
    script,
)


class _BrokenStream:
    """A stdout whose reader went away: write and flush raise BrokenPipeError."""

    def write(self, text: str) -> int:
        raise BrokenPipeError("closed")

    def flush(self) -> None:
        raise BrokenPipeError("closed")


class _FailingStream:
    """A stream whose write raises `error`, the way a dead stdout raises it."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def write(self, text: str) -> int:
        raise self._error

    def flush(self) -> None:
        raise self._error


# ---------------------------------------------------------------------------
# main(): the in-process branches
# ---------------------------------------------------------------------------


def test_main_prints_two_lines_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert script.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("SQLite ")
    assert lines[1].startswith(("::warning::", "::notice::"))


def test_main_turns_a_crash_into_a_warning_and_still_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom() -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(script, "collect_lines", boom)
    assert script.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "::warning::The SQLite version report could not run (RuntimeError: boom). "
        "Informational only; this does not fail the job."
    ]


def test_main_turns_a_sys_exit_into_a_warning_and_still_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def leave() -> list[str]:
        raise SystemExit(3)

    monkeypatch.setattr(script, "collect_lines", leave)
    assert script.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "::warning::The SQLite version report could not run (SystemExit: 3). "
        "Informational only; this does not fail the job."
    ]


def test_main_does_not_swallow_a_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C has to stop the script: it is the one thing deliberately not contained."""

    def interrupted() -> list[str]:
        raise KeyboardInterrupt

    monkeypatch.setattr(script, "collect_lines", interrupted)
    with pytest.raises(KeyboardInterrupt):
        script.main()


@pytest.mark.parametrize(
    ("error", "note"),
    [
        (BrokenPipeError("closed"), "closed"),
        (ValueError("I/O operation on closed file"), "I/O operation on closed file"),
        (OSError(errno.EINVAL, "Invalid argument"), "[Errno 22] Invalid argument"),  # Windows
    ],
    ids=["broken-pipe", "closed-file", "windows-einval"],
)
def test_main_returns_zero_and_says_so_on_stderr_when_stdout_cannot_be_written(
    error: Exception, note: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert script.main(_FailingStream(error)) == 0
    assert capsys.readouterr().err == f"report_sqlite_version: could not write the report: {note}\n"


def test_the_note_on_stderr_cannot_start_a_workflow_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hostile = ValueError("boom\n::error::injected")
    assert script.main(_FailingStream(hostile)) == 0
    err = capsys.readouterr().err
    assert err == "report_sqlite_version: could not write the report: boom%0A::error::injected\n"


# The stderr note is the seventh place a dynamic value reaches a line (the other six are in
# test_report_sqlite_version.py): the message of the exception that broke stdout.


def test_the_note_on_stderr_is_ascii_so_any_console_can_print_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert script.main(_FailingStream(OSError("broken é pipe"))) == 0
    err = capsys.readouterr().err
    assert err.isascii()
    assert err == "report_sqlite_version: could not write the report: broken \\xe9 pipe\n"


def test_an_overlong_message_is_clipped_in_the_note_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert script.main(_FailingStream(OSError("Q" * 5_000))) == 0
    err = capsys.readouterr().err
    assert err == "report_sqlite_version: could not write the report: " + "Q" * 80 + "...\n"


# Both sizes of hostile text from test_report_sqlite_version.py (50,000 and 200,000
# characters; see the comment there for why a call site needs the larger one).
_NOTE_HOSTILE_INPUTS = {
    f"{name}-{len(text)}": text
    for source in (_HOSTILE_INPUTS, _LARGE_HOSTILE_INPUTS)
    for name, text in source.items()
}


@pytest.mark.parametrize("name", list(_NOTE_HOSTILE_INPUTS))
def test_the_note_on_stderr_is_linear_on_hostile_text(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hostile = _NOTE_HOSTILE_INPUTS[name]
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert _seconds(script.note_write_failure, OSError(hostile)) < _TIME_LIMIT_SECONDS


@pytest.mark.parametrize(
    "error",
    [BrokenPipeError("closed"), ValueError("I/O operation on closed file")],
    ids=["broken-pipe", "closed-file"],
)
def test_main_returns_zero_when_stderr_is_broken_too(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(sys, "stderr", _FailingStream(error))
    assert script.main(_BrokenStream()) == 0


def test_main_returns_zero_when_stderr_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sys.stderr` is None when fd 2 was closed before the interpreter started."""
    monkeypatch.setattr(sys, "stderr", None)
    assert script.main(_BrokenStream()) == 0


def test_main_returns_zero_when_stdout_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sys.stdout` is None when fd 1 was closed before the interpreter started (`>&-`)."""
    monkeypatch.setattr(sys, "stdout", None)
    assert script.main() == 0


# ---------------------------------------------------------------------------
# silence_stream(): the shutdown-flush trap
# ---------------------------------------------------------------------------


def _record_silenced(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace `silence_stream` with a recorder; returns the list it appends to."""
    silenced: list[Any] = []
    monkeypatch.setattr(script, "silence_stream", silenced.append)
    return silenced


def test_main_silences_the_process_stdout_when_it_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _BrokenStream()
    monkeypatch.setattr(sys, "stdout", broken)
    silenced = _record_silenced(monkeypatch)
    assert script.main() == 0
    assert silenced == [broken]


def test_main_leaves_a_stream_that_is_not_the_process_stdout_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silencing redirects a descriptor for the whole process: only stdout's, never a caller's."""
    silenced = _record_silenced(monkeypatch)
    assert script.main(_BrokenStream()) == 0
    assert silenced == []


def test_a_broken_stderr_is_silenced_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The interpreter flushes stderr at shutdown as well, and a failure there is also 120."""
    broken_stderr = _BrokenStream()
    monkeypatch.setattr(sys, "stderr", broken_stderr)
    silenced = _record_silenced(monkeypatch)
    assert script.main(_BrokenStream()) == 0
    assert silenced == [broken_stderr]


class _FlushFailsStream:
    """Accepts the write into a buffer and fails later, when the buffer is flushed."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        raise BrokenPipeError("closed")


def test_a_stderr_that_fails_only_when_flushed_is_silenced_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr = _FlushFailsStream()
    monkeypatch.setattr(sys, "stderr", stderr)
    silenced = _record_silenced(monkeypatch)
    assert script.main(_BrokenStream()) == 0
    assert silenced == [stderr]


def test_a_working_stderr_is_not_silenced(monkeypatch: pytest.MonkeyPatch) -> None:
    silenced = _record_silenced(monkeypatch)
    assert script.main(_BrokenStream()) == 0
    assert silenced == []


@pytest.fixture
def unread_pipe_writer() -> Iterator[io.TextIOWrapper]:
    """A text stream on a private pipe whose read end is already closed."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    stream = os.fdopen(write_fd, "w", encoding="utf-8")
    try:
        yield stream
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def test_silence_stream_makes_a_broken_pipe_flushable(
    unread_pipe_writer: io.TextIOWrapper,
) -> None:
    """The mechanism behind exit 120: unflushed bytes are retried at shutdown and fail again.

    After `silence_stream` the retry lands in the null device. Private pipe only: the
    descriptor that is redirected belongs to this test.
    """
    unread_pipe_writer.write("report\n")
    with pytest.raises(OSError):  # BrokenPipeError on POSIX, EINVAL on Windows
        unread_pipe_writer.flush()
    with pytest.raises(OSError):  # still failing: the bytes stayed buffered
        unread_pipe_writer.flush()

    script.silence_stream(unread_pipe_writer)

    unread_pipe_writer.flush()  # must not raise
    unread_pipe_writer.close()  # nor must the close that follows


class _FilenoRaisesOSError:
    def fileno(self) -> int:
        raise OSError(errno.EBADF, "Bad file descriptor")


def _closed_file(tmp_path: Path) -> io.TextIOWrapper:
    stream = (tmp_path / "closed.txt").open("w", encoding="utf-8")
    stream.close()
    return stream


@pytest.mark.parametrize(
    "make_stream",
    [
        lambda tmp_path: io.StringIO(),  # fileno() raises io.UnsupportedOperation
        _closed_file,  # fileno() raises ValueError
        lambda tmp_path: _BrokenStream(),  # no fileno at all: AttributeError
        lambda tmp_path: _FilenoRaisesOSError(),  # a plain OSError
    ],
    ids=["no-descriptor", "closed-file", "no-fileno-method", "fileno-raises-oserror"],
)
def test_silence_stream_ignores_a_stream_without_a_usable_descriptor(
    make_stream: Any, tmp_path: Path
) -> None:
    assert script.silence_stream(make_stream(tmp_path)) is None


def _script_os(**replaced: Any) -> SimpleNamespace:
    """What the script sees as `os`: the real functions, with `replaced` swapped out.

    Only the script's own reference is replaced. Patching the real `os.dup2` is not safe
    in-process: pytest's output capture calls it between test phases.
    """
    members: dict[str, Any] = {
        "open": os.open,
        "dup2": os.dup2,
        "close": os.close,
        "devnull": os.devnull,
        "O_WRONLY": os.O_WRONLY,
    }
    members.update(replaced)
    return SimpleNamespace(**members)


def test_silence_stream_survives_a_failing_dup2_and_leaves_no_descriptor_open(
    monkeypatch: pytest.MonkeyPatch, unread_pipe_writer: io.TextIOWrapper
) -> None:
    opened: list[int] = []

    def spy_open(*args: Any, **kwargs: Any) -> int:
        fd = os.open(*args, **kwargs)
        opened.append(fd)
        return fd

    def failing_dup2(src: int, dst: int) -> None:
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(script, "os", _script_os(open=spy_open, dup2=failing_dup2))

    script.silence_stream(unread_pipe_writer)

    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])  # the null-device descriptor was closed again


def test_silence_stream_survives_a_null_device_that_cannot_be_opened(
    monkeypatch: pytest.MonkeyPatch, unread_pipe_writer: io.TextIOWrapper
) -> None:
    redirected: list[tuple[int, int]] = []

    def failing_open(*args: Any, **kwargs: Any) -> int:
        raise OSError(errno.EACCES, "Permission denied")

    def recording_dup2(src: int, dst: int) -> None:
        redirected.append((src, dst))

    monkeypatch.setattr(script, "os", _script_os(open=failing_open, dup2=recording_dup2))

    script.silence_stream(unread_pipe_writer)

    assert redirected == []
