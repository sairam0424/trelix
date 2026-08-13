"""
Eval test configuration.

Mirrors ``tests/unit/conftest.py``: beast-mode feature flags default to False
in code, but a developer's ``.env`` may have them set to True for local
testing. Override them to "false" here so eval/golden-set runs see the shipped
code defaults regardless of ``.env``.

This is load-bearing for context-compression A/B evals: the harness scores
faithfulness/completeness/hallucination on the assembled context, so the "off"
baseline must be byte-identical to today. If a dev's ``.env`` flips
``TRELIX_RETRIEVAL_COMPRESSION`` on, the baseline would silently compress and
the A/B delta would be meaningless. Neutralizing the flag here removes that
contamination.

pydantic-settings reads env vars BEFORE the .env file, so setenv("X", "false")
overrides whatever .env contains for that key. RetrievalConfig uses
``extra="ignore"``, so neutralizing a compression flag that hasn't landed as a
field yet is harmless (ignored now, auto-neutralized once the field exists).
"""

import pytest

_BEAST_MODE_DEFAULTS: dict[str, str] = {
    # Context compression (A/B isolation -- the load-bearing entry here).
    # Both plausible alias spellings are covered so the off-baseline stays
    # byte-identical no matter which the field ultimately binds to.
    "TRELIX_RETRIEVAL_COMPRESSION": "false",
    "TRELIX_RETRIEVAL_COMPRESSION_ENABLED": "false",
    # Other beast-mode retrieval flags (same set as unit's conftest).
    "TRELIX_FILE_SUMMARIES_ENABLED": "false",
    "TRELIX_RETRIEVAL_FILE_SUMMARY_LEG": "false",
    "TRELIX_RETRIEVAL_HYDE_FALLBACK": "false",
    "TRELIX_RETRIEVAL_MULTI_QUERY": "false",
    "TRELIX_RETRIEVAL_FLARE": "false",
    "TRELIX_RETRIEVAL_PAGERANK_BOOST": "false",
    "TRELIX_TELEMETRY_ENABLED": "false",
    "TRELIX_OTEL_ENABLED": "false",
}

# Non-boolean settings that must stay unset (not "false") for tests to see the
# "unconfigured" code default. See unit/conftest.py for the full rationale:
# delenv() alone is not enough because pydantic-settings' .env-file source
# reads the file directly and independently of the process environment.
_UNSET_BY_DEFAULT: tuple[str, ...] = ("TRELIX_API_AUTH_TOKEN",)
_EMPTY_STRING_BY_DEFAULT: tuple[str, ...] = (
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
# Int-typed connector fields can't take the "" override (pydantic can't parse
# "" as int); "0" parses fine and is falsy, so every `if not val` check still
# treats it as "missing" -- same effective behavior as the string fields above.
_ZERO_INT_BY_DEFAULT: tuple[str, ...] = ("TRELIX_TESTRAIL_PROJECT_ID",)


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override beast-mode feature flags to false so eval runs see code defaults."""
    for var, val in _BEAST_MODE_DEFAULTS.items():
        monkeypatch.setenv(var, val)
    for var in _UNSET_BY_DEFAULT:
        monkeypatch.delenv(var, raising=False)
    for var in _EMPTY_STRING_BY_DEFAULT:
        monkeypatch.setenv(var, "")
    for var in _ZERO_INT_BY_DEFAULT:
        monkeypatch.setenv(var, "0")
