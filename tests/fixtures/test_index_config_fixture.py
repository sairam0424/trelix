"""Standalone proof that ``index_config`` (tests/fixtures/config.py) works in
isolation, BEFORE any call site migrates to it. Uses ONLY the
``index_config`` fixture.

FALSIFIED BY: ``index_config`` raising (e.g. ``repo_path`` pointing at a
directory that does not exist -- the exact ``ValueError`` the real
``repo_must_exist`` validator raises), or returning a config whose
``repo_path`` does not match this test's own ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.config import index_config as index_config  # re-exported (ruff F401)
from trelix.core.config import IndexConfig


def test_index_config_is_a_real_config_rooted_at_this_tests_tmp_path(
    index_config: IndexConfig, tmp_path: Path
) -> None:
    assert isinstance(index_config, IndexConfig)
    # repo_must_exist's field_validator calls Path(v).resolve(), so compare
    # resolved paths -- tmp_path itself may already be a symlink-resolved
    # path depending on platform, but resolving both sides makes the
    # comparison robust either way.
    assert Path(index_config.repo_path) == tmp_path.resolve()


def test_index_config_rejects_a_nonexistent_repo_path(tmp_path: Path) -> None:
    """Not testing the fixture itself here -- testing that the real
    validator the fixture relies on for its "repo_path always exists"
    guarantee actually enforces that, so a future change to
    repo_must_exist that weakens it is caught by the SAME test file that
    documents the guarantee `index_config` depends on."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="repo_path does not exist"):
        IndexConfig(repo_path=str(missing))
