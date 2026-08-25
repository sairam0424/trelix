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

from tests._env_isolation import (  # noqa: E402
    apply_env_isolation,
    disable_litellm_dotenv_autoload,
    scrub_operator_env,
)

# Shared fixtures (tests/fixtures/): re-exported here, per-fixture, per pytest's
# documented conftest-sharing mechanism, so every test under tests/unit/ can
# request `tmp_db` or `index_config` without a per-file import. The `as name`
# form is an explicit re-export (ruff recognizes `import x as x`; a plain
# `import x` here would be flagged unused since neither name is otherwise
# referenced in this file).
from tests.fixtures.config import index_config as index_config  # noqa: E402
from tests.fixtures.db import tmp_db as tmp_db  # noqa: E402

# ── litellm may not publish the operator's dotenv; set BEFORE any test imports it ──────
#
# Same class of problem as the two Hub variables above, and the same reason it cannot be a
# fixture: `litellm/__init__.py` calls `dotenv.load_dotenv()` at import time.
#
# The mechanism is wider than 'it finds the repo you are standing in': `find_dotenv()`
# walks up from the CALLER'S FRAME directory -- `site-packages/litellm/` -- so it climbs
# out of `.venv` and finds the `.env` of the repository that owns the venv, from ANY cwd.
# Measured: run from `~` with no `.env` in any ancestor, the repo's token still appears.
#
# Measured effect in one process: `EmbedderConfig().provider` is "local" before
# `import litellm` and "azure" after. tests/unit/test_retry.py imports litellm from inside
# a test body, so without this the injection lands MID-TEST, after that test's isolation
# fixture has already run.
#
# This is a CORRECTNESS problem, not a spend one: `--disable-socket` already stops the
# outbound call (measured: zero outbound connect attempts). What leaks is which provider
# the test actually exercises.
#
# Belt and braces on purpose. This line stops the pollution; `scrub_operator_env` below
# cleans up anything that gets in anyway -- including the operator's plain shell exports,
# which no import-time flag can prevent.
disable_litellm_dotenv_autoload()


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deny the operator's env, then pin the flags that survive it to code defaults.

    Order matters: `apply_env_isolation` SETS `TRELIX_*` names and the scrub DELETES every
    `TRELIX_*` name, so scrubbing second would undo the pins.

    Only tests/unit does this. tests/integration and tests/eval exist to reach live Azure
    and Bedrock and read those very credentials, so applying the scrub to the shared helper
    would "fix" the hermetic suite by breaking the two that are supposed to see operator
    config. tests/_env_isolation.py documents the asymmetry so it does not read as an
    omission.
    """
    scrub_operator_env(monkeypatch)
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
