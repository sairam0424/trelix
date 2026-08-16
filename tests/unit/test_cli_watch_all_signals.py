"""Regression tests for `trelix watch-all` signal handling.

watch-all installed its SIGINT/SIGTERM handlers on the loop returned by
asyncio.get_event_loop() *before* calling asyncio.run(). asyncio.run builds
its own loop, so the handlers landed on a loop that never ran a single
iteration: measured, the two loop objects have different id()s and
`_signal_handlers[SIGTERM]` is None inside the coroutine.

Worse than "the handler is late": loop.add_signal_handler() also repoints the
process-level SIGINT disposition at asyncio's no-op handler, so the
`except KeyboardInterrupt` fallback below it stopped firing too. A `docker
stop` / `kubectl delete` SIGTERM was swallowed entirely — measured 4s+ with
no shutdown, i.e. the container's grace period elapsed and the process was
hard-killed before printing the re-indexed/skipped stats block.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

import trelix.indexing.multi_watcher as multi_watcher_mod
from trelix.cli.main import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _restore_process_signal_dispositions():
    """Keep the test session's own SIGINT/SIGTERM handling intact.

    asyncio's loop.close() resets the dispositions it touched to
    default_int_handler / SIG_DFL rather than to whatever pytest had, so a
    test that drives add_signal_handler leaks that reset into later tests.
    """
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            if handler is not None:
                signal.signal(sig, handler)


def _write_registry(tmp_path: Path) -> Path:
    config = tmp_path / "repos.json"
    config.write_text(json.dumps({"repos": [{"alias": "demo", "path": str(tmp_path)}]}))
    return config


def test_watch_all_installs_signal_handlers_on_the_loop_that_runs(tmp_path, monkeypatch):
    """The handlers must exist on the loop asyncio.run actually drives."""
    config = _write_registry(tmp_path)
    observed: dict[str, object] = {}

    class _ProbeWatcher:
        def __init__(self, registry: object) -> None:
            self._registry = registry

        async def run(self, stop_event: asyncio.Event) -> None:
            loop = asyncio.get_running_loop()
            handlers = getattr(loop, "_signal_handlers", {})
            observed["sigint"] = signal.SIGINT in handlers
            observed["sigterm"] = signal.SIGTERM in handlers

        def stats(self) -> dict[str, int]:
            return {"files_reindexed": 0, "files_skipped_unchanged": 0}

    monkeypatch.setattr(multi_watcher_mod, "MultiRepoWatcher", _ProbeWatcher)

    result = runner.invoke(app, ["watch-all", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert observed, "watcher.run() was never awaited"
    assert observed["sigterm"] is True, "SIGTERM handler missing from the running loop"
    assert observed["sigint"] is True, "SIGINT handler missing from the running loop"


def test_watch_all_shuts_down_and_prints_stats_on_real_sigterm(tmp_path):
    """A real SIGTERM must reach the stats block, the way `docker stop` does.

    Run in a subprocess: the signal is delivered to the process, so the only
    honest way to assert the container-stop path is to own the process.
    """
    config = _write_registry(tmp_path)
    script = textwrap.dedent(
        f"""
        import os, signal, threading, time
        import trelix.indexing.multi_watcher as mw

        class _SleepWatcher:
            def __init__(self, registry):
                pass

            async def run(self, stop_event):
                await stop_event.wait()

            def stats(self):
                return {{"files_reindexed": 7, "files_skipped_unchanged": 3}}

        mw.MultiRepoWatcher = _SleepWatcher

        from trelix.cli.main import app

        def _send():
            time.sleep(1.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_send, daemon=True).start()
        app(["watch-all", "--config", {str(config)!r}], standalone_mode=False)
        print("REACHED-END")
        """
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "watch-all ignored SIGTERM and hung — the handler is not on the running loop. "
            f"stdout so far: {exc.stdout!r}"
        )

    assert result.returncode == 0, result.stderr
    assert "Watch stopped" in result.stdout, result.stdout
    assert "Re-indexed: 7" in result.stdout, result.stdout
    assert "REACHED-END" in result.stdout, result.stdout
