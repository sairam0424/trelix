"""
Unit test configuration.

Beast-mode feature flags default to False in code, but a developer's .env
may have them set to True for local testing. Override them to "false" here
so unit tests that assert "disabled by default" pass regardless of .env.

pydantic-settings reads env vars BEFORE the .env file, so setenv("X", "false")
overrides whatever .env contains for that key.
"""

import pytest

_BEAST_MODE_DEFAULTS: dict[str, str] = {
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
# "unconfigured" code default — same rationale as _BEAST_MODE_DEFAULTS above,
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
_UNSET_BY_DEFAULT: tuple[str, ...] = ("TRELIX_API_AUTH_TOKEN",)
_EMPTY_STRING_BY_DEFAULT: tuple[str, ...] = (
    "TRELIX_LINEAR_API_KEY",
    "TRELIX_LINEAR_TEAM_KEY",
    "TRELIX_JIRA_BASE_URL",
    "TRELIX_JIRA_EMAIL",
    "TRELIX_JIRA_API_TOKEN",
    "TRELIX_JIRA_PROJECT_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override beast-mode feature flags to false so unit tests see code defaults."""
    for var, val in _BEAST_MODE_DEFAULTS.items():
        monkeypatch.setenv(var, val)
    for var in _UNSET_BY_DEFAULT:
        monkeypatch.delenv(var, raising=False)
    for var in _EMPTY_STRING_BY_DEFAULT:
        monkeypatch.setenv(var, "")
