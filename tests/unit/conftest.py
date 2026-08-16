"""
Unit test configuration.

The env-isolation table lives in ``tests/_env_isolation.py`` and is shared with
the integration and eval suites — see that module for why each variable is
pinned the way it is, and why three copies of the table was a bug rather than
duplication.
"""

import pytest

from tests._env_isolation import apply_env_isolation


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override beast-mode feature flags to false so unit tests see code defaults."""
    apply_env_isolation(monkeypatch)
