"""
Integration tests for Retriever.get_callers / get_callees / get_importers.

Indexes a small purpose-built fixture repo using provider=local (no API keys).
The fixture is module-scoped so indexing runs only once for all tests.

Runtime: ~10-25 s for the single index build (dominated by loading the local
sentence-transformers model), then microseconds per test.

History / why a fixture repo instead of the trelix worktree
-----------------------------------------------------------
This module used to point ``IndexConfig(repo_path=...)`` at the trelix repo
root and index the entire worktree. That exceeded 390 s on this repo, so the
run was killed by the harness timeout and all 16 tests produced *no result*
whatsoever — the file had to be excluded from CI. Nothing in these tests needs
a large repo: every assertion is about the *shape* of the graph API's return
value (source tag, score, 1-indexed contiguous ranks, sort order, empty-list
behavior) plus "at least one edge resolved". A four-file fixture repo exercises
exactly those shapes, and does it more strictly than the worktree did: the
sort-order tests below no longer have to ``skip`` for lack of a second result,
and ``test_callers_sorted_by_file_then_line`` now covers *both* halves of the
sort key (two callers in one file at different lines, plus a third in another
file).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig
from trelix.indexing.indexer import Indexer
from trelix.retrieval.retriever import Retriever

# ---------------------------------------------------------------------------
# Fixture repo source
# ---------------------------------------------------------------------------
#
# Deliberately NO module-level docstrings in these files. The Python extractor
# inserts its synthetic ``<module>`` symbol at index 0 of the symbol list
# *after* the walk has already recorded each call site's caller as an index into
# that same list, so a module docstring shifts every caller index by one and
# misattributes call edges. That is a real defect in
# ``src/trelix/indexing/parser/extractors/python.py`` (see
# ``test_module_docstring_does_not_shift_caller_attribution`` at the bottom of
# this file, which documents it as a strict xfail). Keeping the fixture
# docstring-free lets the tests below assert the *intended* graph semantics
# instead of the off-by-one ones.

_RETRIEVER_SRC = textwrap.dedent("""\
    class Retriever:
        def __init__(self, index=None):
            self.index = index or {}

        def retrieve(self, query):
            candidates = self._fetch(query)
            return self._rank(candidates)

        def _fetch(self, query):
            return [v for k, v in self.index.items() if query in k]

        def _rank(self, candidates):
            return sorted(candidates)
    """)

# Two callers in the same file at different lines -> exercises the "then line"
# half of get_callers' (rel_path, line_start) sort key.
_CLI_SRC = textwrap.dedent("""\
    from pkg.retriever import Retriever


    def main(query):
        return Retriever({"alpha": "A"}).retrieve(query)


    def run_batch(queries):
        engine = Retriever({"beta": "B"})
        return [engine.retrieve(q) for q in queries]
    """)

# A third caller in a *different* file -> exercises the "by file" half.
_SERVICE_SRC = textwrap.dedent("""\
    from pkg.retriever import Retriever


    def serve(query):
        return Retriever({"gamma": "C"}).retrieve(query)
    """)


def _write_fixture_repo(root: Path) -> Path:
    """Materialize the fixture repo under ``root`` and return the repo path."""
    pkg = root / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "retriever.py").write_text(_RETRIEVER_SRC, encoding="utf-8")
    (pkg / "cli.py").write_text(_CLI_SRC, encoding="utf-8")
    (pkg / "service.py").write_text(_SERVICE_SRC, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Module-scoped fixture: index the small fixture repo once
# ---------------------------------------------------------------------------


def _build_retriever(repo: Path, db_dir: Path) -> Retriever:
    """Index ``repo`` into a fresh db under ``db_dir`` and return a Retriever."""
    config = IndexConfig(
        repo_path=str(repo),
        incremental=False,
        # Explicit provider: an init kwarg outranks the .env file in
        # pydantic-settings, so a developer .env with an API-backed provider
        # cannot leak in here.
        embedder=EmbedderConfig(provider="local"),
        store=StoreConfig(db_path=str(db_dir / "trelix.db")),
    )
    Indexer(config).index()
    return Retriever(config)


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> Retriever:
    """Index the small fixture repo into a fresh temp dir and return a Retriever."""
    repo = _write_fixture_repo(tmp_path_factory.mktemp("trelix_graph_fixture_repo"))
    db_dir = tmp_path_factory.mktemp("trelix_graph_fixture_index")
    return _build_retriever(repo, db_dir)


# ---------------------------------------------------------------------------
# get_callers
# ---------------------------------------------------------------------------


class TestGetCallersIntegration:
    def test_callers_of_retrieve_are_non_empty(self, retriever: Retriever) -> None:
        """
        Retriever.retrieve is called by pkg/cli.py::main, pkg/cli.py::run_batch
        and pkg/service.py::serve — there must be at least one resolved
        internal caller.
        """
        results = retriever.get_callers("Retriever.retrieve")
        assert len(results) > 0, (
            "Expected at least one internal caller of Retriever.retrieve. "
            "Check that call edges were indexed (db.get_callers returned empty)."
        )

    def test_callers_source_is_graph_callers(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        assert all(r.source == "graph_callers" for r in results)

    def test_callers_score_is_one(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        assert all(r.score == 1.0 for r in results)

    def test_callers_rank_is_one_indexed_contiguous(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        assert results  # guard
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_callers_nonexistent_symbol_returns_empty(self, retriever: Retriever) -> None:
        results = retriever.get_callers("__this_symbol_does_not_exist_anywhere__")
        assert results == []

    def test_callers_sorted_by_file_then_line(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        if len(results) < 2:
            pytest.skip("Need at least 2 callers to check sort order")
        keys = [(r.file.rel_path, r.symbol.line_start) for r in results]
        assert keys == sorted(keys)
        # The fixture repo guarantees a same-file pair, so the secondary
        # (line_start) half of the sort key is genuinely exercised — not just
        # the rel_path half.
        assert len({path for path, _ in keys}) < len(keys), (
            "Fixture repo should produce >1 caller in the same file so that the "
            "line_start half of the sort key is exercised."
        )


# ---------------------------------------------------------------------------
# get_callees
# ---------------------------------------------------------------------------


class TestGetCalleesIntegration:
    def test_callees_of_retrieve_are_non_empty(self, retriever: Retriever) -> None:
        """
        Retriever.retrieve calls internal helpers (_fetch, _rank) — at least one
        should resolve internally.
        """
        results = retriever.get_callees("Retriever.retrieve")
        assert len(results) > 0, (
            "Expected at least one resolved internal callee of Retriever.retrieve."
        )

    def test_callees_source_is_graph_callees(self, retriever: Retriever) -> None:
        results = retriever.get_callees("Retriever.retrieve")
        assert all(r.source == "graph_callees" for r in results)

    def test_callees_score_is_one(self, retriever: Retriever) -> None:
        results = retriever.get_callees("Retriever.retrieve")
        assert all(r.score == 1.0 for r in results)

    def test_callees_nonexistent_symbol_returns_empty(self, retriever: Retriever) -> None:
        results = retriever.get_callees("__this_symbol_does_not_exist_anywhere__")
        assert results == []


# ---------------------------------------------------------------------------
# get_importers
# ---------------------------------------------------------------------------


class TestGetImportersIntegration:
    def test_importers_of_retriever_module_non_empty(self, retriever: Retriever) -> None:
        """
        pkg/retriever.py is imported by pkg/cli.py and pkg/service.py — there
        must be at least one importer.
        """
        results = retriever.get_importers("pkg/retriever.py")
        assert len(results) > 0, (
            "Expected at least one file importing pkg/retriever.py. "
            "Check that import edges were resolved during indexing."
        )

    def test_importers_source_is_graph_importers(self, retriever: Retriever) -> None:
        results = retriever.get_importers("pkg/retriever.py")
        assert all(r.source == "graph_importers" for r in results)

    def test_importers_score_is_one(self, retriever: Retriever) -> None:
        results = retriever.get_importers("pkg/retriever.py")
        assert all(r.score == 1.0 for r in results)

    def test_importers_unknown_module_returns_empty(self, retriever: Retriever) -> None:
        results = retriever.get_importers("__no_such_module_anywhere__")
        assert results == []

    def test_importers_sorted_by_rel_path(self, retriever: Retriever) -> None:
        results = retriever.get_importers("pkg/retriever.py")
        if len(results) < 2:
            pytest.skip("Need at least 2 importers to check sort order")
        paths = [r.file.rel_path for r in results]
        assert paths == sorted(paths)

    def test_importers_rank_is_one_indexed_contiguous(self, retriever: Retriever) -> None:
        results = retriever.get_importers("pkg/retriever.py")
        assert results  # guard
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))


# ---------------------------------------------------------------------------
# Known defect: module docstring shifts caller attribution
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect in src/trelix/indexing/parser/extractors/python.py: the "
        "synthetic '<module>' symbol is inserted at symbols[0] AFTER the walk "
        "recorded each call site's caller as an index into that list, so every "
        "caller index is off by one in any file that has a module docstring. "
        "Real effect on the shipped repo: get_callees('Retriever.retrieve') "
        "returns [] for src/trelix/retrieval/retriever.py (its callees are "
        "misattributed to Retriever.__init__), and get_callers reports "
        "'<module>' instead of the calling function. Remove this xfail when the "
        "extractor is fixed."
    ),
)
def test_module_docstring_does_not_shift_caller_attribution(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """
    Same fixture repo, but every file carries a module docstring.

    The graph API must report identical call edges either way: a docstring is
    documentation, not structure. This test exists so the docstring-free
    fixture used by the rest of this module does not silently hide the
    off-by-one described above.
    """
    repo = _write_fixture_repo(tmp_path_factory.mktemp("trelix_graph_docstring_repo"))
    for name in ("retriever.py", "cli.py", "service.py"):
        path = repo / "pkg" / name
        original = path.read_text(encoding="utf-8")
        path.write_text(f'"""Module docstring."""\n\n{original}', encoding="utf-8")
    db_dir = tmp_path_factory.mktemp("trelix_graph_docstring_index")
    documented = _build_retriever(repo, db_dir)

    callees = documented.get_callees("Retriever.retrieve")
    assert [r.symbol.qualified_name for r in callees] == [
        "Retriever._fetch",
        "Retriever._rank",
    ]
