"""
Cross-source edges vs. plain PageRank — a rigor eval for Decision #7 of the
original 5-phase cross-source-edges upgrade plan.

The team shipped plain nx.pagerank() (uniform teleport) in
rank_by_pagerank() (src/trelix/retrieval/graph.py) deliberately, deferring
Personalized PageRank (teleport mass concentrated on ticket/test nodes)
until there was real evidence it was needed. This script produces that
evidence by measuring trelix's OWN self-eval before and after real
generic_edges (from GitLinker walking trelix's own git history) are
present, on two axes:

  1. End-to-end retrieval quality (TRELIX_SELF_CASES via EvalHarness) —
     does cross-source-edge-influenced PageRank move the needle on the
     metrics a user actually experiences?
  2. Direct PageRank rank-position deltas for symbols that gained a
     generic_edge — does the signal even reach the ranking, before it can
     reach end-to-end metrics?

trelix's own commit history has ZERO "[A-Z]+-\\d+" (Jira-style) matches but
95+ "#\\d+" (GitHub-issue-style) matches, so this self-eval overrides
GitLinkerConfig.ticket_pattern to r"#\\d+" rather than using the Jira-style
default.

Usage (run as a module — `python tests/eval/...py` directly fails to import
the `tests.eval.*` package since the repo root isn't on sys.path that way)::

    source .venv/bin/activate
    python -m tests.eval.cross_source_pagerank_eval
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from tests.eval.datasets.trelix_self import TRELIX_SELF_CASES
from tests.eval.harness import EvalHarness, EvalReport
from trelix.core.config import EmbedderConfig, GitLinkerConfig, IndexConfig, RetrievalConfig
from trelix.indexing.git_linker import GitLinker
from trelix.retrieval.graph import rank_by_pagerank
from trelix.store.db import Database

REPO_PATH = str(Path(__file__).resolve().parents[2])
SAMPLE_SIZE = 25


def _select_candidate_symbol_ids(db: Database, limit: int = SAMPLE_SIZE) -> list[int]:
    """Symbol ids that appear as either a caller or a callee in `calls` —
    i.e. have real call-graph edges, per the task's sampling criteria."""
    rows = db._conn.execute(
        """
        SELECT sid FROM (
            SELECT DISTINCT caller_id AS sid FROM calls WHERE caller_id IS NOT NULL
            UNION
            SELECT DISTINCT callee_id AS sid FROM calls WHERE callee_id IS NOT NULL
        )
        ORDER BY sid
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def _rank_positions(ranked: list[tuple[int, float]]) -> dict[int, int]:
    """1-based rank position of each symbol within a rank_by_pagerank() result."""
    return {sid: i + 1 for i, (sid, _score) in enumerate(ranked)}


def _print_metrics_row(label: str, report: EvalReport) -> None:
    print(
        f"  {label:<8}  recall@5={report.mean_recall_at_5:.3f}   "
        f"mrr={report.mrr:.3f}   ndcg@10={report.mean_ndcg_at_10:.3f}   (n={report.n})"
    )


def main() -> None:
    print(f"Repo under test: {REPO_PATH}\n")

    config = IndexConfig(
        repo_path=REPO_PATH,
        embedder=EmbedderConfig(provider="local"),
        retrieval=RetrievalConfig(rerank=False),
    )
    harness = EvalHarness(REPO_PATH, config)

    # --- Step (a)/(b): index once (idempotent), run BEFORE eval -----------
    print("=" * 78)
    print("STEP (b): BEFORE — end-to-end eval, NO generic_edges present")
    print("=" * 78)
    t0 = time.time()
    before_report = harness.run(TRELIX_SELF_CASES)
    print(f"(indexing+eval wall time: {time.time() - t0:.1f}s)")

    db = Database(config.db_path_absolute)
    candidate_ids = _select_candidate_symbol_ids(db)
    print(f"\nSampled {len(candidate_ids)} candidate symbol_ids with real call-graph edges.")

    before_edge_count = db._conn.execute("SELECT COUNT(*) FROM generic_edges").fetchone()[0]
    print(f"generic_edges rows before linking: {before_edge_count}")

    before_ranked = rank_by_pagerank(candidate_ids, db)
    before_positions = _rank_positions(before_ranked)
    print(
        f"rank_by_pagerank() expanded the 25 seed candidates to a "
        f"{len(before_ranked)}-node call subgraph (seeds + their 1-hop callers/callees)."
    )

    # --- Step (c): run GitLinker against trelix's own real git history ----
    print("\n" + "=" * 78)
    print("STEP (c): running GitLinker on trelix's own commit history (real #NNN refs)")
    print("=" * 78)
    linker = GitLinker(db, GitLinkerConfig(ticket_pattern=r"#\d+"))
    inserted = linker.link(REPO_PATH)
    after_edge_count = db._conn.execute("SELECT COUNT(*) FROM generic_edges").fetchone()[0]
    distinct_symbols = db._conn.execute(
        "SELECT COUNT(DISTINCT from_symbol_id) FROM generic_edges"
    ).fetchone()[0]
    distinct_tickets = db._conn.execute(
        "SELECT COUNT(DISTINCT source_ref) FROM generic_edges"
    ).fetchone()[0]
    print(f"GitLinker.link() reported {inserted} edges inserted")
    print(
        f"generic_edges table now has {after_edge_count} rows "
        f"across {distinct_symbols} distinct symbols and {distinct_tickets} distinct tickets"
    )

    # --- Step (d): AFTER — re-run direct PageRank + end-to-end eval -------
    print("\n" + "=" * 78)
    print("STEP (d): AFTER — direct rank_by_pagerank() on the SAME candidate set")
    print("=" * 78)
    after_ranked = rank_by_pagerank(candidate_ids, db)
    after_positions = _rank_positions(after_ranked)

    # rank_by_pagerank() only calls get_generic_edge_targets() for the
    # symbol_ids it's directly given (candidate_ids below) — nodes pulled
    # into the ranked subgraph via call-graph traversal never have their
    # own generic_edges looked up, so a generic_edge on one of THOSE nodes
    # never actually enters the ranked graph. "Affected" must therefore be
    # restricted to candidate_ids, not all nodes in the expanded subgraph —
    # checking the DB across the full subgraph would count edges that were
    # never inserted into this particular ranking, diluting the real effect
    # size with unrelated zero-deltas.
    affected = [sid for sid in candidate_ids if db.get_generic_edge_targets(sid)]
    deltas = [
        before_positions[sid] - after_positions.get(sid, before_positions[sid]) for sid in affected
    ]
    changed = [d for d in deltas if d != 0]

    print(
        f"Of {len(candidate_ids)} seed candidates (which expand to a "
        f"{len(before_positions)}-node ranked call-subgraph), "
        f"{len(affected)} gained a generic_edge."
    )
    if affected:
        frac_changed = len(changed) / len(affected)
        print(
            f"  Fraction with a measurable rank-position change: "
            f"{frac_changed:.2%} ({len(changed)}/{len(affected)})"
        )
        print(f"  Mean rank-position delta (abs):   {statistics.mean(abs(d) for d in deltas):.2f}")
        print(
            f"  Median rank-position delta (abs): {statistics.median(abs(d) for d in deltas):.2f}"
        )
        print(f"  Raw deltas (before_rank - after_rank): {deltas}")
    else:
        print("  No candidate symbols gained a generic_edge — cannot compute rank deltas.")

    print("\n" + "=" * 78)
    print("STEP (d): AFTER — end-to-end eval, generic_edges now present")
    print("=" * 78)
    t0 = time.time()
    after_report = harness.run(TRELIX_SELF_CASES)
    print(f"(eval wall time: {time.time() - t0:.1f}s)")

    # --- Step (e): comparison table ----------------------------------------
    print("\n" + "=" * 78)
    print("STEP (e): BEFORE / AFTER COMPARISON")
    print("=" * 78)
    print("\nEnd-to-end retrieval metrics (TRELIX_SELF_CASES, 50 queries):")
    _print_metrics_row("BEFORE", before_report)
    _print_metrics_row("AFTER", after_report)
    recall5_delta = after_report.mean_recall_at_5 - before_report.mean_recall_at_5
    mrr_delta = after_report.mrr - before_report.mrr
    ndcg_delta = after_report.mean_ndcg_at_10 - before_report.mean_ndcg_at_10
    print(
        f"  {'DELTA':<8}  recall@5={recall5_delta:+.3f}   "
        f"mrr={mrr_delta:+.3f}   ndcg@10={ndcg_delta:+.3f}"
    )

    print(f"\nEdges inserted by GitLinker: {inserted}")
    print(f"  distinct symbols touched: {distinct_symbols}")
    print(f"  distinct tickets referenced: {distinct_tickets}")

    print(
        f"\nPageRank rank-position impact "
        f"(seeded from {len(candidate_ids)} candidates -> {len(before_positions)}-node subgraph, "
        f"{len(affected)} affected):"
    )
    if affected:
        print(f"  fraction changed: {len(changed) / len(affected):.2%}")
        print(f"  mean |delta|: {statistics.mean(abs(d) for d in deltas):.2f} positions")
        print(f"  median |delta|: {statistics.median(abs(d) for d in deltas):.2f} positions")
    else:
        print("  n/a — no affected candidates")

    db.close()


if __name__ == "__main__":
    main()
