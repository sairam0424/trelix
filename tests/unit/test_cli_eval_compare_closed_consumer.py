"""`trelix eval-compare` when nobody reads a stream: the exit status stays the verdict.

The shared consoles turn a reader that has gone away into exit 0 (see test_cli_closed_stdout.py).
That suits `stats | grep -q` and is wrong here, where only PASS may exit 0: under `pipefail`,
`trelix eval-compare ... | head -n 20` must not report a FAIL, an INCONCLUSIVE or a REFUSED as
success once `head` has closed its end.

The real thing is run in a child process on a pipe whose read end is already closed (the race of
a real `| head` removed), because the fix lives in how the process ends. Windows reports a closed
pipe as OSError(EINVAL), not BrokenPipeError; a stdout that fails that way stands in for it.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `except SystemExit` in `_print_verdict_line` re-raising or removed  (TestARealClosedPipe)
2. the `OSError` branch removed, or made to swallow every errno         (the ENOSPC test)
3. `_silence_closed_stream` not called for an EINVAL stream            (TestAWindowsStyleClosedPipe)
4. the stderr lines printed with the stdout console                    (the stderr test)
"""

from __future__ import annotations

import errno
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.unit.eval_compare_fixtures import SUITE, results_doc, write_json, write_prereg
from trelix.cli import main as cli_main

OTHER_SUITE = {**SUITE, "repo_sha": "d" * 40}

# (verdict, exit code, candidate nDCG, candidate suite, what stderr carries)
OUTCOMES = [
    pytest.param("FAIL", 1, 0.4375, SUITE, "", id="fail"),
    pytest.param("INCONCLUSIVE", 2, 0.5, SUITE, "", id="inconclusive"),
    pytest.param(
        "REFUSED",
        3,
        0.5625,
        OTHER_SUITE,
        "refused: suite identity differs in: repo_sha\n",
        id="refused",
    ),
]

# A stdout that fails the way a Windows pipe without a reader does.
WINDOWS_CLOSED_PIPE = (
    "import errno, io, sys\n"
    "class WindowsClosedPipe(io.TextIOBase):\n"
    "    encoding = 'utf-8'\n"
    "    errors = 'replace'\n"
    "    def writable(self): return True\n"
    "    def isatty(self): return False\n"
    "    def fileno(self): return 1\n"
    "    def write(self, s): raise OSError(errno.EINVAL, 'Invalid argument')\n"
    "    def flush(self): raise OSError(errno.EINVAL, 'Invalid argument')\n"
    "sys.stdout = WindowsClosedPipe()\n"
    "sys.argv = ['trelix', 'eval-compare', *{args!r}]\n"
    "from trelix.cli.main import app\n"
    "sys.exit(app())\n"
)


def _child_env() -> dict[str, str]:
    """Env that makes the child import the SAME trelix as this test process."""
    import trelix

    src_root = str(Path(trelix.__file__).resolve().parents[1])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing}" if existing else src_root
    return env


def _cli_args(tmp_path: Path, cand_ndcg: float, cand_suite: dict[str, str]) -> list[str]:
    base = write_json(tmp_path, "base.json", results_doc("baseline", 0.5))
    cand = write_json(tmp_path, "cand.json", results_doc("flag-on", cand_ndcg, suite=cand_suite))
    return [base, cand, "--prereg", write_prereg(tmp_path)]


def _run_with_closed(stream: str, cli_args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `eval-compare` with `stream` on a pipe whose read end is gone; the other is captured."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    targets = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, stream: write_fd}
    try:
        return subprocess.run(
            [sys.executable, "-m", "trelix.cli.main", "eval-compare", *cli_args],
            stdout=targets["stdout"],
            stderr=targets["stderr"],
            env=_child_env(),
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        os.close(write_fd)


class TestARealClosedPipe:
    @pytest.mark.parametrize(("verdict", "exit_code", "cand_ndcg", "cand_suite", "err"), OUTCOMES)
    def test_the_exit_status_is_the_verdict_when_stdout_has_no_reader(
        self,
        tmp_path: Path,
        verdict: str,
        exit_code: int,
        cand_ndcg: float,
        cand_suite: dict[str, str],
        err: str,
    ) -> None:
        result = _run_with_closed("stdout", _cli_args(tmp_path, cand_ndcg, cand_suite))

        assert result.returncode == exit_code, f"{verdict}: stderr={result.stderr!r}"
        assert result.stderr == err

    def test_the_verdict_line_is_still_printed_when_stderr_has_no_reader(
        self, tmp_path: Path
    ) -> None:
        result = _run_with_closed("stderr", _cli_args(tmp_path, 0.5625, OTHER_SUITE))

        assert result.returncode == 3
        assert result.stdout == "verdict: REFUSED\n"


class TestAWindowsStyleClosedPipe:
    @pytest.mark.parametrize(("verdict", "exit_code", "cand_ndcg", "cand_suite", "err"), OUTCOMES)
    def test_the_exit_status_is_the_verdict_when_stdout_fails_with_einval(
        self,
        tmp_path: Path,
        verdict: str,
        exit_code: int,
        cand_ndcg: float,
        cand_suite: dict[str, str],
        err: str,
    ) -> None:
        entry = tmp_path / "entry.py"
        entry.write_text(
            WINDOWS_CLOSED_PIPE.format(args=_cli_args(tmp_path, cand_ndcg, cand_suite)),
            encoding="utf-8",
        )

        result = subprocess.run(
            [sys.executable, str(entry)],
            capture_output=True,
            env=_child_env(),
            text=True,
            timeout=180,
            check=False,
        )

        # 120 is CPython's "flushing sys.stdout at shutdown failed": the stream was not silenced.
        assert result.returncode == exit_code, f"{verdict}: stderr={result.stderr!r}"
        # startswith, not ==: CPython 3.14 reports "Exception ignored while flushing sys.stdout"
        # at shutdown for this emulated stream (a Python object whose write raises EINVAL, which
        # the null-device redirect cannot reach). That trailing text is not part of the contract.
        assert result.stderr.startswith(err)


class _RaisingConsole(Console):
    """A console whose `print` raises `exc`; its `file` is a buffer nobody reads."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__(file=io.StringIO())
        self._exc = exc

    def print(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - rich's signature
        raise self._exc


@pytest.fixture
def silenced(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record the streams `_silence_closed_stream` is asked to silence, without doing it.

    The real one points this pytest process's own fd at the null device, which breaks output
    capture for every test that follows.
    """
    streams: list[object] = []
    monkeypatch.setattr(cli_main, "_silence_closed_stream", streams.append)
    return streams


class TestTheLineHelper:
    def test_an_einval_write_silences_that_stream_and_returns(self, silenced: list[object]) -> None:
        console = _RaisingConsole(OSError(errno.EINVAL, "Invalid argument"))

        cli_main._print_verdict_line(console, "verdict: FAIL")

        assert silenced == [console.file]

    def test_an_exit_from_the_console_is_absorbed_without_silencing_again(
        self, silenced: list[object]
    ) -> None:
        console = _RaisingConsole(SystemExit(0))

        cli_main._print_verdict_line(console, "verdict: FAIL")

        assert silenced == []

    def test_an_oserror_that_is_not_a_closed_pipe_reaches_the_caller(
        self, silenced: list[object]
    ) -> None:
        console = _RaisingConsole(OSError(errno.ENOSPC, "No space left on device"))

        with pytest.raises(OSError, match="No space left on device"):
            cli_main._print_verdict_line(console, "verdict: FAIL")

        assert silenced == []
