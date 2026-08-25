"""
U10 — Production Eval Harness integration test.

CI-friendly version: uses the committed mini_repo fixture (7 files, ~300 lines).
10 queries, gated by a PER-QUERY RANK LEDGER (see ``RANK_BUDGET`` below), not a
mean threshold.

Full 50-query trelix self-eval runs separately via:
    make eval-full

WHY A LEDGER, NOT A MEAN. The old gate (``EvalHarness.assert_min_metrics``, still
available for scripts that want it) checked ``mean(nDCG@10) >= 0.60`` and
``mean(recall@5) >= 0.80`` over these 10 queries. Flipping ``graph_search_enabled``
to ``True`` — documented MEASURED HARMFUL and shipped default-off
(``RetrievalConfig.graph_search_enabled = False``; see
``docs/discoverability/ECOSYSTEM-ROADMAP.md`` and
``tests/unit/test_retriever_leg_gates_and_dedup.py``) — passes that mean gate on
THIS fixture with the config below (``parse_workers=2, incremental=False``):
mean nDCG@10 falls from 0.9631 to 0.8492 and mean recall@5 does not move at all
(1.000 -> 1.000), because "hash password function" and "verify password" both
drop from rank 1 to rank 4 — still inside a recall@5 window. A MEAN hides this
inside headroom; even a PER-QUERY recall@5 ledger would MISS it, because rank 4
still scores recall@5 = 1.0 for both queries individually. Only a rank budget
tighter than 5 catches it, which is what ``RANK_BUDGET`` below uses (3, not 5).
See ``test_graph_search_enabled_regresses_past_the_ledger_budget`` for the
adversarial self-test that pins this measurement and the numbers below.

A NOTE ON THE MAGNITUDE. A different indexing configuration (defaults:
``parse_workers=4, incremental=True`` — what ``EvalHarness()`` builds when no
``config`` is passed) reproduces a LARGER, differently-shaped regression on the
same two queries: rank 1 -> 9 and 1 -> 8, mean nDCG@10 -0.1384, mean recall@5
-0.20 dropping from a baseline that clears the old floor by +0.3631. Both
configurations demonstrate the SAME defect (a real per-query collapse that a
mean gate absorbs); the exact magnitude is sensitive to the indexing
configuration (thread count / incremental flag), not a universal constant, and
this file measures against the configuration `mini_repo_config` actually uses.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.eval.harness import EvalHarness
from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINI_REPO_PATH = Path(__file__).parent.parent / "fixtures" / "mini_repo"

# 10 representative queries mapped to mini_repo files (exact match — see
# tests/eval/harness.py:score_ranking, which matches ground truth by exact string
# equality against a SearchResult's rel_path, never by substring).
MINI_REPO_CASES: list[tuple[str, str]] = [
    ("how does authentication work", "auth.py"),
    ("user repository get by id", "user.py"),
    ("hash password function", "utils.py"),
    ("login method", "auth.py"),
    ("validate token", "auth.py"),
    ("User dataclass", "user.py"),
    ("main entry point", "main.py"),
    ("delete user", "user.py"),
    ("verify password", "utils.py"),
    ("create user", "user.py"),
]

# PER-QUERY RANK LEDGER. Every query in MINI_REPO_CASES must retrieve its expected
# file at rank <= 3. Measured with the shipped default RetrievalConfig()
# (graph_search_enabled=False, rerank=False, and this file's parse_workers=2,
# incremental=False): every query ranks its expected file at 1, except "validate
# token" at 2 — so a budget of 3 carries real slack over today's baseline while
# still sitting BELOW the rank 4 that graph_search_enabled=True measurably
# produces for two queries (a budget of 5 would not: see module docstring).
RANK_BUDGET: dict[str, int] = {query: 3 for query, _expected in MINI_REPO_CASES}


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


def _ledger_failures(report) -> list[str]:
    """Every query whose retrieved rank exceeds its RANK_BUDGET, as report lines.

    A missing budget is a bug in this file (a case added to MINI_REPO_CASES
    without a matching ledger entry), not a retrieval failure, so it raises
    (KeyError) rather than silently passing that query.
    """
    failures: list[str] = []
    for er in report.results:
        budget = RANK_BUDGET[er.query]
        if er.rank <= 0 or er.rank > budget:
            failures.append(
                f"{er.query!r} -> {er.expected_file}: rank={er.rank} exceeds budget "
                f"{budget} (recall@5={er.recall_at_5:.1f}, ndcg@10={er.ndcg_at_10:.3f})"
            )
    return failures


def _assert_ledger_passes(report) -> None:
    """THE GATE ITSELF. Raises AssertionError if any query's rank exceeds its budget.

    Called two ways in this file: unwrapped from ``test_trelix_self_eval`` (must
    pass today, on the shipped default config), and wrapped in
    ``pytest.raises(AssertionError)`` from
    ``test_graph_search_enabled_regresses_past_the_ledger_budget`` (must raise
    there, on the same config with ``graph_search_enabled=True``). Using the exact
    same function both places means "the gate that must pass" and "the gate that
    must fail" are provably the same code path, not two similar-looking checks
    that could drift apart.
    """
    failures = _ledger_failures(report)
    assert failures == [], "retrieval quality ledger failed:\n" + "\n".join(
        f"  - {f}" for f in failures
    )


# ---------------------------------------------------------------------------
# Test: harness eval on mini_repo, gated by the per-query rank ledger
# ---------------------------------------------------------------------------


def test_trelix_self_eval(
    mini_repo_dir: Path,
    mini_repo_config: IndexConfig,
) -> None:
    """Every query in MINI_REPO_CASES must retrieve its expected file within budget.

    MUTATION THAT MUST FAIL THIS TEST: set ``graph_search_enabled=True`` on
    ``mini_repo_config``'s retrieval config — see
    ``test_graph_search_enabled_regresses_past_the_ledger_budget`` below, which
    pins that exact mutation on an independent harness instance so this test
    itself stays a pure "does the shipped default pass" check.
    """
    harness = EvalHarness(repo_path=str(mini_repo_dir), config=mini_repo_config)
    report = harness.run(MINI_REPO_CASES)

    _assert_ledger_passes(report)

    # Print a summary for easy CI log inspection
    print(
        f"\n[eval] mini_repo: n={report.n} "
        f"R@1={report.mean_recall_at_1:.3f} "
        f"R@5={report.mean_recall_at_5:.3f} "
        f"R@10={report.mean_recall_at_10:.3f} "
        f"MRR={report.mrr:.3f} "
        f"NDCG@10={report.mean_ndcg_at_10:.3f}"
    )


# ---------------------------------------------------------------------------
# Adversarial self-test — the ledger must be ABLE to fail
# ---------------------------------------------------------------------------


def test_graph_search_enabled_regresses_past_the_ledger_budget(
    mini_repo_dir: Path,
    mini_repo_config: IndexConfig,
) -> None:
    """A gate that has never been shown to fail is decoration.

    Flipping ``graph_search_enabled`` to ``True`` here reuses the SAME index the
    sibling test built (``EvalHarness._ensure_indexed`` is idempotent — it skips
    indexing when ``.trelix/index.db`` already exists under ``mini_repo_dir``),
    so this test spends no extra indexing time and measures only the
    retrieval-config change. Whichever of the two tests in this module pytest
    runs first is the one that actually builds the index; that ordering is not
    relied upon, only the shared, idempotent side effect is.

    MEASURED on this fixture, with this exact config (``parse_workers=2,
    incremental=False`` — see ``mini_repo_config``), holding across repeated runs
    in both fresh processes and a shared in-process index:
      * "hash password function" and "verify password": rank 1 -> 4
      * "login method": rank 1 -> 2; "validate token": rank 2 -> 1
      * mean nDCG@10: 0.9631 -> 0.8492 (checked below with a wide tolerance)
      * mean recall@5: 1.000 -> 1.000 (UNCHANGED — rank 4 is still inside top-5)
      * mean MRR: 0.950 -> 0.800

    Both the OLD mean gate (min_recall5=0.80, min_mrr=0.55, min_ndcg=0.60) and a
    per-query recall@5 ledger would PASS on these numbers — recall@5 does not even
    move. Only a per-query rank budget tighter than 5 (this file uses 3) sees
    "hash password function" and "verify password" individually blow through it.
    """
    mutated_retrieval = mini_repo_config.retrieval.model_copy(update={"graph_search_enabled": True})
    mutated_config = mini_repo_config.model_copy(update={"retrieval": mutated_retrieval})

    harness = EvalHarness(repo_path=str(mini_repo_dir), config=mutated_config)
    report = harness.run(MINI_REPO_CASES)

    # THE ADVERSARIAL SELF-TEST: the same gate function that must PASS in
    # test_trelix_self_eval must RAISE here. `raises=` pinned per the suite's own
    # rule that a bare `pytest.raises(AssertionError)` would absorb ANY assertion
    # failure, including one from an unrelated bug in this test file itself.
    with pytest.raises(AssertionError, match=r"retrieval quality ledger failed"):
        _assert_ledger_passes(report)

    failures = _ledger_failures(report)
    assert failures != [], "unreachable: _assert_ledger_passes just raised on these"
    culprits = {f.split(" -> ", 1)[0].strip("'") for f in failures}
    assert culprits == {"hash password function", "verify password"}, (
        f"expected exactly the two documented queries to fail the ledger, got: "
        f"{sorted(culprits)} — re-measure and update this test's docstring"
    )

    # The mean gate this replaced (and a per-query recall@5 ledger) would have
    # PASSED on these same numbers — pin that explicitly, at a tolerance wide
    # enough to survive embedder/library point releases, so a future reader can
    # see the exact defect the tighter rank ledger fixes rather than trusting
    # the docstring's prose.
    assert report.mean_ndcg_at_10 == pytest.approx(0.8492, abs=0.02), (
        f"mean nDCG@10 {report.mean_ndcg_at_10:.4f} has drifted from the measured "
        "0.8492 by more than the tolerance — re-measure before relying on this as "
        "the 'old mean gate would have passed' control"
    )
    assert report.mean_recall_at_5 == pytest.approx(1.0, abs=1e-9), (
        f"mean recall@5 {report.mean_recall_at_5:.4f} moved from the measured 1.0 — "
        "re-measure before relying on this as the 'per-query recall@5 ledger would "
        "also have missed it' control"
    )
    assert report.mean_ndcg_at_10 >= 0.60
    assert report.mean_recall_at_5 >= 0.80
