"""
Integration test configuration.

The env-isolation table lives in ``tests/_env_isolation.py`` and is shared with
the unit and eval suites — see that module for why each variable is pinned the
way it is, and for the subprocess scope limit (nothing here reaches a `trelix`
binary spawned via ``subprocess``; ``test_cli.py`` isolates its own child).

This matters especially for context-compression A/B runs: if a dev's ``.env``
flips ``TRELIX_RETRIEVAL_COMPRESSION`` on — or merely retunes
``TRELIX_RETRIEVAL_COMPRESSION_RATIO``, which this file used to miss because it
carried its own partial copy of the table — the "off" baseline would silently
compress and the comparison would be contaminated.
"""

from pathlib import Path

import pytest

from tests._env_isolation import apply_env_isolation

_INTEGRATION_DIR = Path(__file__).parent


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override beast-mode feature flags to false so integration tests see code defaults."""
    apply_env_isolation(monkeypatch)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag everything under tests/integration/ with ``integration`` and ``enable_socket``.

    CONTRIBUTING.md documents ``pytest -m "not integration"`` as the
    credential-free run, but no test in this tree ever carried the marker, so
    that command deselected nothing — collection was identical to bare pytest
    (2,965 tests, 0 deselected when this was fixed) and it exercised the live
    Azure/Bedrock paths it claimed to skip. Applying the marker by directory
    here (rather than by hand on each of the 104 items in it) means a new file
    added to this directory is credential-gated on arrival instead of the day
    someone remembers to decorate it.

    ``enable_socket`` is applied by the same rule and for the same reason.
    ``--disable-socket`` lives in the root ``addopts``, so it applies to EVERY
    directory — including this one, whose entire purpose is to reach live
    services. Without this the ban produced 188 ``SocketBlockedError`` failures
    in the Integration Tests job ("A test tried to use socket.getaddrinfo"),
    which is the ban working exactly as configured and pointed at the wrong
    tree. Re-enabling per directory rather than weakening the global flag keeps
    the unit suite hermetic, which is where the 232s and the accidental spend
    came from.

    ``pytest_collection_modifyitems`` from a subdirectory conftest is still
    handed the WHOLE session's items, hence the explicit path filter.
    """
    for item in items:
        if _INTEGRATION_DIR in item.path.parents:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.enable_socket)
            item.add_marker(pytest.mark.requires_network)
