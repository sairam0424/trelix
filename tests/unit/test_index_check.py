"""The shared "does this repository have an index?" check.

Every read surface (CLI read commands, MCP tools, REST read routes, the LangChain and
LlamaIndex retrievers, `search-all`, `review`) asks this before it builds anything that
opens the index database. Opening a missing one creates it, so the check must answer from
the filesystem alone and leave no `.trelix/` behind.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fixtures.indexed import mark_indexed
from trelix.core.config import IndexConfig, StoreConfig
from trelix.core.index_check import IndexNotFoundError, require_index


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return root


def test_raises_for_a_repository_that_was_never_indexed(repo: Path) -> None:
    config = IndexConfig(repo_path=str(repo))

    with pytest.raises(IndexNotFoundError):
        require_index(config)


def test_the_message_is_the_cli_message(repo: Path) -> None:
    config = IndexConfig(repo_path=str(repo))

    with pytest.raises(IndexNotFoundError) as excinfo:
        require_index(config)

    resolved = repo.resolve()
    assert str(excinfo.value) == (
        f"No index found at {resolved}/.trelix/index.db. Run trelix index {resolved} first."
    )
    assert excinfo.value.db_path == resolved / ".trelix" / "index.db"
    assert excinfo.value.repo_path == str(resolved)


def test_checking_creates_neither_the_directory_nor_its_gitignore(repo: Path) -> None:
    config = IndexConfig(repo_path=str(repo))

    with pytest.raises(IndexNotFoundError):
        require_index(config)

    assert not (repo / ".trelix").exists()


def test_passes_once_the_index_exists(repo: Path) -> None:
    mark_indexed(repo)

    require_index(IndexConfig(repo_path=str(repo)))


def test_the_error_is_a_file_not_found_error(repo: Path) -> None:
    """A caller that already handles a missing file (or any OSError) handles this one."""
    config = IndexConfig(repo_path=str(repo))

    with pytest.raises(FileNotFoundError):
        require_index(config)
    assert issubclass(IndexNotFoundError, OSError)


@pytest.mark.parametrize(
    "clone",
    [
        copy.copy,
        copy.deepcopy,
        lambda error: pickle.loads(pickle.dumps(error)),  # noqa: S301 - our own exception
    ],
    ids=["copy", "deepcopy", "pickle"],
)
def test_the_error_survives_copying_and_pickling(
    repo: Path, clone: Callable[[IndexNotFoundError], IndexNotFoundError]
) -> None:
    """A process-pool worker ships this error to its parent by pickling it.

    The constructor takes two arguments but the exception's `args` holds one (the message), so
    the default reconstruction raised a `TypeError`, and the parent saw `BrokenProcessPool`
    instead of the message.
    """
    config = IndexConfig(repo_path=str(repo))
    with pytest.raises(IndexNotFoundError) as excinfo:
        require_index(config)

    cloned = clone(excinfo.value)

    resolved = repo.resolve()
    assert type(cloned) is IndexNotFoundError
    assert str(cloned) == (
        f"No index found at {resolved}/.trelix/index.db. Run trelix index {resolved} first."
    )
    assert cloned.db_path == resolved / ".trelix" / "index.db"
    assert cloned.repo_path == str(resolved)


def test_follows_a_relative_store_path(repo: Path) -> None:
    config = IndexConfig(repo_path=str(repo), store=StoreConfig(db_path="data/custom.db"))
    (repo / ".trelix").mkdir()
    (repo / ".trelix" / "index.db").write_bytes(b"")

    with pytest.raises(IndexNotFoundError) as excinfo:
        require_index(config)
    assert excinfo.value.db_path == repo.resolve() / "data" / "custom.db"

    (repo / "data").mkdir()
    (repo / "data" / "custom.db").write_bytes(b"")
    require_index(config)


def test_follows_an_absolute_store_path(repo: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / "index.db"
    config = IndexConfig(repo_path=str(repo), store=StoreConfig(db_path=str(elsewhere)))

    with pytest.raises(IndexNotFoundError) as excinfo:
        require_index(config)
    assert excinfo.value.db_path == elsewhere
    # The command to run names the repository, not the directory the custom store sits in.
    assert str(excinfo.value) == (
        f"No index found at {elsewhere}. Run trelix index {repo.resolve()} first."
    )

    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"")
    require_index(config)
