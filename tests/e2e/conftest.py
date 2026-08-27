"""
E2E test configuration — mirrors tests/integration/conftest.py's exact pattern.

These tests spawn real subprocesses (the installed `trelix` console script,
`trelix-mcp` over real stdio, fresh venvs built from freshly-built wheels), so
they need `enable_socket` to survive the repo-wide `--disable-socket` ban and
`requires_network` for the same reason tests/integration/ does.

Distinct from tests/integration/: nothing here needs live LLM credentials or
paid APIs — every test uses the `local` embedder provider. The two directories
are gated for different reasons and must stay separate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._env_isolation import apply_env_isolation

_E2E_DIR = Path(__file__).parent


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same isolation tests/integration/conftest.py applies — a real subprocess
    inherits the current os.environ, so a developer's .env can contaminate a
    real `trelix index`/`search` run exactly as it could an integration test."""
    apply_env_isolation(monkeypatch)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag everything under tests/e2e/ with ``e2e``, ``enable_socket``, and
    ``requires_network`` by directory, not by hand on each test — see
    tests/integration/conftest.py's docstring for why: a marker applied by
    hand is a marker someone eventually forgets to add to a new file.
    """
    for item in items:
        if _E2E_DIR in item.path.parents:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.enable_socket)
            item.add_marker(pytest.mark.requires_network)
