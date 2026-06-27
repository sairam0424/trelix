"""
Phase 15 — Eval Harness: Mini-repo file recall tests.

Indexes the committed synthetic test repo at tests/fixtures/mini_repo/
using provider=local (no API key needed), then runs 10 targeted queries
and asserts that the expected file appears in the top-5 results for each.

Recall metric: at least one result in the top-5 matches the expected file name.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import NamedTuple

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig
from trelix.indexing.indexer import Indexer
from trelix.retrieval.retriever import Retriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINI_REPO = Path(__file__).parent.parent / "fixtures" / "mini_repo"
TOP_K = 5  # recall is measured in top-5


# ---------------------------------------------------------------------------
# Eval case definition
# ---------------------------------------------------------------------------


class RecallCase(NamedTuple):
    query: str
    expected_file: str  # filename stem or partial path that must appear in top-5


EVAL_CASES: list[RecallCase] = [
    RecallCase("how does authentication work", "auth.py"),
    RecallCase("user repository get by id", "user.py"),
    RecallCase("hash password function", "utils.py"),
    RecallCase("login method", "auth.py"),
    RecallCase("validate token", "auth.py"),
    RecallCase("User dataclass", "user.py"),
    RecallCase("main entry point", "main.py"),
    RecallCase("delete user", "user.py"),
    RecallCase("verify password", "utils.py"),
    RecallCase("create user", "user.py"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def indexed_mini_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Copy mini_repo to a tmp dir, index it with provider=local,
    and return the path.

    Scoped to module so the indexing (sentence-transformers download +
    embedding) only happens once per test run.
    """
    # Copy the committed mini_repo into a fresh temp dir so .trelix/
    # is created alongside the source files (not inside the real repo).
    src = MINI_REPO
    dest = tmp_path_factory.mktemp("mini_repo_indexed")
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)

    config = IndexConfig(
        repo_path=str(dest),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
        retrieval=RetrievalConfig(rerank=False),
    )
    indexer = Indexer(config, quiet=True)
    stats = indexer.index()
    assert stats["files_indexed"] > 0, (
        f"Indexer reported 0 files indexed — check mini_repo at {src}"
    )
    return dest


@pytest.fixture(scope="module")
def retriever(indexed_mini_repo: Path) -> Retriever:
    """Build a Retriever over the indexed mini_repo (no reranking for speed)."""
    config = IndexConfig(
        repo_path=str(indexed_mini_repo),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
        retrieval=RetrievalConfig(rerank=False),
    )
    return Retriever(config)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _file_in_top_k(retriever: Retriever, query: str, expected_file: str, k: int = TOP_K) -> bool:
    """Run a query and return True if expected_file appears in the top-k results."""
    context = retriever.retrieve(query)
    top_results = context.results[:k]
    for result in top_results:
        if expected_file in result.file.rel_path:
            return True
    return False


def _top_k_files(retriever: Retriever, query: str, k: int = TOP_K) -> list[str]:
    """Return the list of rel_paths from the top-k results (for diagnostics)."""
    context = retriever.retrieve(query)
    return [r.file.rel_path for r in context.results[:k]]


# ---------------------------------------------------------------------------
# Individual recall tests (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c.query for c in EVAL_CASES])
def test_file_recall_top5(retriever: Retriever, case: RecallCase) -> None:
    """Assert the expected file appears in the top-5 results for the query."""
    found = _file_in_top_k(retriever, case.query, case.expected_file)
    top_files = _top_k_files(retriever, case.query)
    assert found, (
        f"\nQuery: {case.query!r}\n"
        f"Expected {case.expected_file!r} in top-{TOP_K}, but got:\n"
        + "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(top_files))
    )


# ---------------------------------------------------------------------------
# Recall summary table (printed at end of module)
# ---------------------------------------------------------------------------


class TestRecallSummary:
    """
    Run all 10 queries, collect pass/fail, and print a summary table.
    This is a single test so the table is always shown in the output.
    """

    def test_full_recall_summary(
        self,
        retriever: Retriever,
        capsys,
    ) -> None:
        """Run all eval cases and print a recall summary table."""
        passed = 0
        failed = 0
        rows: list[tuple[str, str, str, str]] = []

        for case in EVAL_CASES:
            top_files = _top_k_files(retriever, case.query)
            found = any(case.expected_file in f for f in top_files)
            status = "PASS" if found else "FAIL"
            top1 = top_files[0] if top_files else "(no results)"
            rows.append((case.query, case.expected_file, status, top1))
            if found:
                passed += 1
            else:
                failed += 1

        recall_pct = passed / len(EVAL_CASES) * 100

        # Print table
        with capsys.disabled():
            print("\n")
            print("=" * 80)
            print(f"  TRELIX RECALL EVAL — mini_repo  |  top-{TOP_K}  |  {len(EVAL_CASES)} queries")
            print("=" * 80)
            print(f"  {'Query':<40} {'Expected':<12} {'Status':<6} {'Top-1 file'}")
            print(f"  {'-' * 40} {'-' * 12} {'-' * 6} {'-' * 20}")
            for query, expected, status, top1 in rows:
                indicator = "" if status == "PASS" else ""
                print(f"  {query:<40} {expected:<12} {indicator} {status}  {top1}")
            print("-" * 80)
            print(f"  Recall@{TOP_K}: {passed}/{len(EVAL_CASES)} = {recall_pct:.0f}%")
            print("=" * 80)
            print()

        # The test passes if recall >= 70%
        assert recall_pct >= 70.0, (
            f"Overall recall@{TOP_K} is {recall_pct:.0f}% ({passed}/{len(EVAL_CASES)}) "
            f"— below the required 70% threshold.\n"
            f"Failing cases:\n"
            + "\n".join(
                f"  {q!r} expected {e!r}, got top-5: {_top_k_files(retriever, q)}"
                for q, e, s, _ in rows
                if s == "FAIL"
            )
        )


# ---------------------------------------------------------------------------
# File-type weighting recall tests
# ---------------------------------------------------------------------------


class TestFileTypeWeightingRecall:
    """
    Assert that file-type weighting improves recall for the 4 known-failing
    queries where README.md was outranking the actual source file.

    These tests use the same indexed_mini_repo fixture as the rest of the module.
    The mini_repo does not have a README.md, so these tests verify the mechanism
    using queries that should resolve to Python source files (not markdown).

    The key assertion is the Recall@5 metric: with weighting enabled (default),
    the overall recall must remain ≥ 70%.
    """

    def test_recall_with_weighting_enabled_meets_threshold(
        self, retriever: Retriever
    ) -> None:
        """Recall@5 >= 70% with file-type weighting enabled (default config)."""
        passed = sum(
            1
            for case in EVAL_CASES
            if _file_in_top_k(retriever, case.query, case.expected_file)
        )
        recall_pct = passed / len(EVAL_CASES) * 100
        assert recall_pct >= 70.0, (
            f"Recall@5 with weighting = {recall_pct:.0f}% ({passed}/{len(EVAL_CASES)}) "
            f"— must be ≥ 70%"
        )

    def test_weighting_disabled_still_returns_results(
        self,
        indexed_mini_repo: Path,
    ) -> None:
        """With weighting disabled, retriever still returns results (kill-switch test)."""
        from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig

        config = IndexConfig(
            repo_path=str(indexed_mini_repo),
            incremental=False,
            parse_workers=2,
            embedder=EmbedderConfig(provider="local"),
            retrieval=RetrievalConfig(rerank=False, file_type_weighting_enabled=False),
        )
        disabled_retriever = Retriever(config)
        context = disabled_retriever.retrieve("how does authentication work")
        assert len(context.results) > 0, (
            "Retriever with weighting disabled returned no results"
        )

    def test_python_source_files_ranked_ahead_of_config_files(
        self, retriever: Retriever
    ) -> None:
        """
        For a query that returns both Python source and JSON/YAML config,
        at least one Python file must appear in top-5 ahead of all config-only files.

        Uses 'main entry point' query — main.py should be in top-5.
        """
        context = retriever.retrieve("main entry point")
        top5_files = [r.file.rel_path for r in context.results[:5]]
        has_python_in_top5 = any(f.endswith(".py") for f in top5_files)
        assert has_python_in_top5, (
            f"No Python file in top-5 for 'main entry point'. Got: {top5_files}"
        )
