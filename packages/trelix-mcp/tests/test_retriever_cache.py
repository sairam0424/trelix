"""server.py's Retriever cache is an LRU bounded by TRELIX_MCP_RETRIEVER_CACHE_SIZE.

* The bound is read with the other limits (budget.py): default 8, at least 1, blank means the
  default, and an unusable value stops `trelix-mcp` at start-up with exit code 2.
* A hit counts as a use; past the bound the least recently used entry is dropped (not closed), and
  a later call for that repository constructs a Retriever again.

Every expected value is a literal written here. `Retriever` is patched to a counting factory, as
the cache tests in test_server.py do, and `_get_retriever` is driven directly. This package's tests
are governed by packages/trelix-mcp/pyproject.toml, which has no socket ban and no timeout, so
the markers below supply both.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import trelix_mcp.server as srv
from trelix_mcp.budget import limits_from_env

pytestmark = [pytest.mark.disable_socket, pytest.mark.timeout(60)]


def _resolved(repo_path: str) -> str:
    """The cache key `_get_retriever` uses for `repo_path`."""
    return str(Path(repo_path).resolve())


def test_retriever_cache_evicts_lru(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRELIX_MCP_RETRIEVER_CACHE_SIZE", "2")
    with (
        patch("trelix_mcp.server.check_repo_dir"),
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Retriever") as factory,
    ):
        factory.side_effect = lambda config: MagicMock(name="retriever")

        first_a = srv._get_retriever("/r/a")
        srv._get_retriever("/r/b")
        # A hit: nothing is constructed, and `a` becomes the most recently used entry.
        assert srv._get_retriever("/r/a") is first_a
        assert factory.call_count == 2

        srv._get_retriever("/r/c")
        # `b` was the least recently used, so it went; `a` stayed because it was touched.
        assert list(srv._retriever_cache) == [_resolved("/r/a"), _resolved("/r/c")]
        assert factory.call_count == 3

        # `b` is gone from the cache, so asking for it constructs again (and `a` goes).
        srv._get_retriever("/r/b")
        assert factory.call_count == 4
        assert list(srv._retriever_cache) == [_resolved("/r/c"), _resolved("/r/b")]


def test_default_size_is_eight() -> None:
    assert limits_from_env({}).retriever_cache_size == 8


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", "TRELIX_MCP_RETRIEVER_CACHE_SIZE must be an integer of at least 1, got '0'"),
        ("abc", "TRELIX_MCP_RETRIEVER_CACHE_SIZE must be an integer of at least 1, got 'abc'"),
    ],
    ids=["zero", "not-an-integer"],
)
def test_bad_cache_size_is_a_startup_error(
    value: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Like a bad TRELIX_MCP_MAX_K: a usage error before the server starts, not a tool error."""
    monkeypatch.setattr("sys.argv", ["trelix-mcp"])
    monkeypatch.setenv("TRELIX_MCP_RETRIEVER_CACHE_SIZE", value)
    mock_run = MagicMock()
    monkeypatch.setattr(srv.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        srv.main()

    assert exc_info.value.code == 2
    mock_run.assert_not_called()
    assert expected in capsys.readouterr().err
