"""
Unit test configuration.

The env-isolation table lives in ``tests/_env_isolation.py`` and is shared with
the integration and eval suites — see that module for why each variable is
pinned the way it is, and why three copies of the table was a bug rather than
duplication.
"""

import logging
from collections.abc import Iterator

import pytest

from tests._env_isolation import apply_env_isolation


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
