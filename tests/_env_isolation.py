"""
Shared env isolation for the unit, integration and eval suites.

Beast-mode feature flags default to False in code, but a developer's ``.env``
may have them set to True for local testing. Every suite has to override them
so tests observe the shipped code defaults regardless of ``.env``.

WHY THIS IS ONE MODULE
----------------------
This table used to be copy-pasted into ``tests/{unit,integration,eval}/
conftest.py``. The copies drifted: only the unit copy grew ``_CODE_DEFAULTS``
and ``_ENV_PREFIXES_TO_SCRUB`` when scalar compression tuning landed, so a
developer with ``TRELIX_RETRIEVAL_COMPRESSION_RATIO=0.2`` in ``.env`` got that
value inside integration and eval runs while unit runs correctly saw 0.45.
That is the exact contamination the eval conftest calls "load-bearing": the
compression A/B scores faithfulness against the assembled context, so an "off"
baseline built with a dev's ratio is not comparable to CI's. One table means a
new setting cannot be pinned in one suite and forgotten in the other two.

The tables are ``MappingProxyType``/tuples because three conftests now share
them: an in-place edit from one suite's fixture would silently change what the
other two observe.

pydantic-settings reads env vars BEFORE the .env file, so setenv("X", "false")
overrides whatever .env contains for that key. RetrievalConfig uses
``extra="ignore"``, so neutralizing a compression flag that hasn't landed as a
field yet is harmless (ignored now, auto-neutralized once the field exists).

SCOPE LIMIT — read this before adding isolation here
----------------------------------------------------
``monkeypatch`` is IN-PROCESS ONLY. Nothing here affects a `trelix` binary
spawned via ``subprocess``: the child re-reads the process environment *and*
the ``.env`` file in its own cwd from scratch. Tests that shell out
(``tests/integration/test_cli.py``) must isolate their child explicitly — that
file scrubs ``TRELIX_*`` out of the child env and runs it from a directory with
no ``.env``. Adding a var here will NOT fix a subprocess test.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType

import pytest

# Boolean beast-mode flags. Union of what the three suites pinned separately;
# the compression aliases carry both plausible spellings so the off-baseline
# stays byte-identical no matter which the field ultimately binds to.
BEAST_MODE_DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        # Context compression (A/B isolation -- the load-bearing entry here).
        "TRELIX_RETRIEVAL_COMPRESSION": "false",
        "TRELIX_RETRIEVAL_COMPRESSION_ENABLED": "false",
        # Other beast-mode retrieval flags.
        "TRELIX_FILE_SUMMARIES_ENABLED": "false",
        "TRELIX_RETRIEVAL_FILE_SUMMARY_LEG": "false",
        "TRELIX_RETRIEVAL_HYDE_FALLBACK": "false",
        "TRELIX_RETRIEVAL_MULTI_QUERY": "false",
        "TRELIX_RETRIEVAL_FLARE": "false",
        "TRELIX_RETRIEVAL_PAGERANK_BOOST": "false",
        "TRELIX_TELEMETRY_ENABLED": "false",
        "TRELIX_OTEL_ENABLED": "false",
    }
)

# Non-flag settings pinned to their CODE defaults. Same rationale as
# BEAST_MODE_DEFAULTS, but these carry a value rather than a boolean, so they
# must be pinned to the default itself instead of "false" — a developer's .env
# tuning compression for a local experiment must not change what the
# "unconfigured" tests observe.
CODE_DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        "TRELIX_RETRIEVAL_COMPRESSION_PROVIDER": "extractive",
        "TRELIX_RETRIEVAL_COMPRESSION_RATIO": "0.45",
        "TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS": "120",
    }
)

# Per-intent override families read straight from os.environ (NOT via
# pydantic-settings), so pinning the scalar above does not neutralize them —
# every var under these prefixes has to be removed by name. The trailing
# underscore is deliberate: it matches the per-intent suffixes only, never the
# bare scalar that CODE_DEFAULTS pins.
ENV_PREFIXES_TO_SCRUB: tuple[str, ...] = ("TRELIX_RETRIEVAL_COMPRESSION_RATIO_",)

# Non-boolean settings that must stay unset (not "false") for tests to see the
# "unconfigured" code default — same rationale as BEAST_MODE_DEFAULTS above,
# but these aren't flags, so setenv("...", "false") would misconfigure them
# instead of disabling them. delenv() alone is NOT enough here: it only
# removes the var from the process environment, but pydantic-settings'
# .env-file source reads the file directly and independently — if a
# developer's .env has a real value for one of these (e.g. a live
# TRELIX_LINEAR_API_KEY for manual connector testing), removing the
# process-env copy doesn't stop the file fallback from supplying it
# anyway. Overriding to "" (falsy, same as unset for every `if not val`
# validate_config() check in this codebase) is what actually neutralizes
# the .env value for the test process.
UNSET_BY_DEFAULT: tuple[str, ...] = ("TRELIX_API_AUTH_TOKEN",)
EMPTY_STRING_BY_DEFAULT: tuple[str, ...] = (
    "TRELIX_LINEAR_API_KEY",
    "TRELIX_LINEAR_TEAM_KEY",
    "TRELIX_JIRA_BASE_URL",
    "TRELIX_JIRA_EMAIL",
    "TRELIX_JIRA_API_TOKEN",
    "TRELIX_JIRA_PROJECT_KEY",
    "TRELIX_TESTRAIL_BASE_URL",
    "TRELIX_TESTRAIL_USERNAME",
    "TRELIX_TESTRAIL_API_KEY",
)
# Int-typed connector fields (project_id: int | None) can't take the ""
# override above — pydantic would fail to parse "" as an int the same way
# it fails on a malformed real value. "0" parses fine and is falsy, so
# every `if not val` validate_config() check in this codebase still treats
# it as "missing" — same effective behavior as the string fields above.
ZERO_INT_BY_DEFAULT: tuple[str, ...] = ("TRELIX_TESTRAIL_PROJECT_ID",)


def apply_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin beast-mode flags and connector settings to their code defaults.

    Called from each suite's autouse ``_isolate_beast_mode_flags`` fixture, so
    every override is undone at teardown by ``monkeypatch``.
    """
    for var, val in BEAST_MODE_DEFAULTS.items():
        monkeypatch.setenv(var, val)
    for var, val in CODE_DEFAULTS.items():
        monkeypatch.setenv(var, val)
    for prefix in ENV_PREFIXES_TO_SCRUB:
        for var in [k for k in os.environ if k.startswith(prefix)]:
            monkeypatch.delenv(var, raising=False)
    for var in UNSET_BY_DEFAULT:
        monkeypatch.delenv(var, raising=False)
    for var in EMPTY_STRING_BY_DEFAULT:
        monkeypatch.setenv(var, "")
    for var in ZERO_INT_BY_DEFAULT:
        monkeypatch.setenv(var, "0")
