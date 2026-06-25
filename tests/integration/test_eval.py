"""
U10 — Production Eval Harness integration test.

CI-friendly version: uses the committed mini_repo fixture (7 files, ~300 lines).
10 queries, asserts Recall@5 >= 0.80.

Full 50-query trelix self-eval runs separately via:
    make eval-full
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig

from tests.eval.harness import EvalHarness

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINI_REPO_PATH = Path(__file__).parent.parent / "fixtures" / "mini_repo"

# 10 representative queries mapped to mini_repo files (substring match).
# Kept consistent with the existing test_recall.py query set so CI tracks
# the same test surface.
MINI_REPO_CASES: list[tuple[str, str]] = [
    ("how does authentication work",   "auth.py"),
    ("user repository get by id",       "user.py"),
    ("hash password function",          "utils.py"),
    ("login method",                    "auth.py"),
    ("validate token",                  "auth.py"),
    ("User dataclass",                  "user.py"),
    ("main entry point",                "main.py"),
    ("delete user",                     "user.py"),
    ("verify password",                 "utils.py"),
    ("create user",                     "user.py"),
]

# ---------------------------------------------------------------------------
# Fixture — index mini_repo once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_repo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy mini_repo to a fresh tmp dir and return its path."""
    src = MINI_REPO_PATH
    dest = tmp_path_factory.mktemp("mini_repo_eval")
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
    return dest


@pytest.fixture(scope="module")
def mini_repo_config(mini_repo_dir: Path) -> IndexConfig:
    """IndexConfig pointing at the copied mini_repo, local embedder, no reranking."""
    return IndexConfig(
        repo_path=str(mini_repo_dir),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
        retrieval=RetrievalConfig(rerank=False),
    )


# ---------------------------------------------------------------------------
# Test: harness eval on mini_repo
# ---------------------------------------------------------------------------


def test_trelix_self_eval(
    mini_repo_dir: Path,
    mini_repo_config: IndexConfig,
) -> None:
    """
    Run multi-metric eval on mini_repo.

    Asserts:
      - Recall@5  >= 0.80
      - MRR       >= 0.55  (relaxed for mini_repo; full eval uses 0.75)
      - NDCG@10   >= 0.60  (relaxed for mini_repo; full eval uses 0.80)
    """
    harness = EvalHarness(
        repo_path=str(mini_repo_dir),
        config=mini_repo_config,
    )
    report = harness.run(MINI_REPO_CASES)

    # Assert minimum thresholds (relaxed for the small mini_repo)
    harness.assert_min_metrics(
        report,
        min_recall5=0.80,
        min_mrr=0.55,
        min_ndcg=0.60,
    )

    # Print a summary for easy CI log inspection
    print(
        f"\n[eval] mini_repo: n={report.n} "
        f"R@1={report.mean_recall_at_1:.3f} "
        f"R@5={report.mean_recall_at_5:.3f} "
        f"R@10={report.mean_recall_at_10:.3f} "
        f"MRR={report.mrr:.3f} "
        f"NDCG@10={report.mean_ndcg_at_10:.3f}"
    )
