"""``index_config``: a real ``IndexConfig`` (src/trelix/core/config.py) rooted
at a fresh ``tmp_path``, function-scoped.

``IndexConfig.repo_path`` is validated by a real field_validator
(``repo_must_exist``) that requires the path to exist on disk -- ``tmp_path``
satisfies that for free, which is exactly why this fixture is worth having:
every call site that previously wrote ``IndexConfig(repo_path=str(tmp_path))``
inline can share this one instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.config import IndexConfig


@pytest.fixture
def index_config(tmp_path: Path) -> IndexConfig:
    """A real ``IndexConfig`` whose ``repo_path`` is this test's own
    ``tmp_path`` -- satisfies ``repo_must_exist`` without a caller having to
    know that constraint exists."""
    return IndexConfig(repo_path=str(tmp_path))
