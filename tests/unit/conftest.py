"""
Unit test configuration.

The env-isolation table lives in ``tests/_env_isolation.py`` and is shared with
the integration and eval suites — see that module for why each variable is
pinned the way it is, and why three copies of the table was a bug rather than
duplication.
"""

import logging
import os
from collections.abc import Iterator

# ── Hub offline, set BEFORE any test imports transformers or huggingface_hub ──────────
#
# Set at conftest MODULE level, not in a fixture: `TRANSFORMERS_OFFLINE` is read at
# transformers import time, and a fixture runs far too late — the first test to import a
# model library would already have decided it may reach the network.
#
# WHY, measured. 48.3% of this suite's wall clock (238.75s of 494.70s) was live HTTP to
# huggingface.co. With `--disable-socket` in addopts and these two variables UNSET, 51
# unit tests fail; with them set, the same three files go from 41 failed / 147 passed to
# **188 passed in 11.79s**. So those tests were never testing the network — they construct
# a real embedder, and offline mode makes it load the already-cached snapshot instead of
# revalidating against the Hub.
#
# That distinction decides the fix. Marking those 51 `requires_network` and re-enabling
# their sockets would have kept the traffic and the cost; making them offline removes both.
#
# Scoped to tests/unit deliberately. tests/integration carries a comment that deliberately
# permits the download, and this file cannot affect it.
#
# `setdefault`, not assignment: a developer debugging a genuine Hub-fetch problem can
# export `HF_HUB_OFFLINE=0` and have it respected.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest  # noqa: E402 - must follow the env vars above

from tests._env_isolation import apply_env_isolation  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override beast-mode feature flags to false so unit tests see code defaults."""
    apply_env_isolation(monkeypatch)


@pytest.fixture(autouse=True)
def _no_leaked_log_handler() -> Iterator[None]:
    """Undo any root-logger handler a test leaves behind, for EVERY unit test.

    `_setup_logging()` — which every real CLI command calls — builds a bare
    `logging.StreamHandler()`. Under `CliRunner` that binds to a capture buffer which is
    closed when the invocation ends, while the handler itself survives on the root logger.
    The next test in the session that logs anything then writes to a closed file and pytest
    dumps a "--- Logging error ---" traceback into *that* test's captured output, breaking
    assertions which are substring checks over it.

    This guard already existed, but only inside `tests/unit/test_cli_markup_safety.py`,
    which protected that file and nothing else. Measured: `tests/unit` in REVERSE collection
    order failed three `test_cli_audit.py` tests for exactly this reason — and those tests
    are canaries, not casualties. They assert on a `_LOGGING_ERROR_MARKER` and fail with
    "an earlier test leaked a logging.StreamHandler bound to a now-closed CliRunner buffer",
    which is a correct diagnosis of someone else's mess.

    Promoted here because the leak is not specific to one module: any test that invokes a
    command through CliRunner can cause it, and the victim is whichever test logs next. A
    per-file guard makes the outcome depend on collection order, which is the property this
    is supposed to remove.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
