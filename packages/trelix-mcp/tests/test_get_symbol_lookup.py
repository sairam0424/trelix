"""`get_symbol` finds a symbol by qualified name first, then by bare name, on a real index.

`seed(repo, 2)` indexes `target.run` plus two callers that share the bare name `uses`
(`callers.m0.uses` in caller_module_0000.py, `callers.m1.uses` in caller_module_0001.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trelix_mcp.server as srv
from budget_support import seed


@pytest.fixture(scope="module")
def shared_name_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return seed(tmp_path_factory.mktemp("shared-name"), 2)


@pytest.mark.parametrize(
    ("qualified_name", "found", "file"),
    [
        ("callers.m1.uses", "callers.m1.uses", "src/callers/caller_module_0001.py"),
        ("callers.m0.uses", "callers.m0.uses", "src/callers/caller_module_0000.py"),
        ("target.run", "target.run", "src/target.py"),
    ],
)
def test_a_qualified_name_finds_that_symbol_and_not_one_sharing_its_bare_name(
    shared_name_repo: Path, qualified_name: str, found: str, file: str
) -> None:
    symbol = srv.get_symbol(qualified_name, str(shared_name_repo))

    assert (symbol["qualified_name"], symbol["file"]) == (found, file)


@pytest.mark.parametrize(
    ("asked", "found"),
    [
        ("run", "target.run"),
        ("Elsewhere.run", "target.run"),
        ("pkg.sub.target.run", "target.run"),
    ],
    ids=["bare-name", "other-prefix", "longer-prefix"],
)
def test_a_name_with_no_exact_match_falls_back_to_its_last_segment(
    shared_name_repo: Path, asked: str, found: str
) -> None:
    assert srv.get_symbol(asked, str(shared_name_repo))["qualified_name"] == found


def test_a_bare_name_shared_by_two_symbols_returns_the_one_indexed_first(
    shared_name_repo: Path,
) -> None:
    assert srv.get_symbol("uses", str(shared_name_repo))["qualified_name"] == "callers.m0.uses"
    assert srv.get_symbol("Elsewhere.uses", str(shared_name_repo))["qualified_name"] == (
        "callers.m0.uses"
    )


@pytest.mark.parametrize("asked", ["nothing", "no.such.symbol", "target.missing", ""])
def test_a_name_the_index_does_not_hold_returns_none(shared_name_repo: Path, asked: str) -> None:
    assert srv.get_symbol(asked, str(shared_name_repo)) is None
