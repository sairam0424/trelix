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
}


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override beast-mode feature flags to false so unit tests see code defaults."""
    for var, val in _BEAST_MODE_DEFAULTS.items():
        monkeypatch.setenv(var, val)
