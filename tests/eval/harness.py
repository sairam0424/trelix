"""
Eval harness for trelix retrieval quality.

Usage::

    from tests.eval.harness import EvalHarness, EvalReport
    from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig

    config = IndexConfig(
        repo_path="/path/to/repo",
        embedder=EmbedderConfig(provider="local"),
        retrieval=RetrievalConfig(rerank=False),
    )
    harness = EvalHarness(config)
    report  = harness.run(cases)
    harness.assert_min_metrics(report)

EvalHarness indexes the repo once (idempotent — skips if already indexed),
runs every (query, expected_file) pair, and returns an EvalReport with
per-query EvalResult objects and aggregate statistics.

Printing is done via Rich when available; falls back to plain text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.core.models import SearchResult
from trelix.indexing.indexer import Indexer
from trelix.retrieval.retriever import Retriever

from .metrics import (
    EvalResult,
    find_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger("trelix.eval")


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Aggregate statistics over all eval queries."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def mean_recall_at_1(self) -> float:
        return _mean(r.recall_at_1 for r in self.results)

    @property
    def mean_recall_at_5(self) -> float:
        return _mean(r.recall_at_5 for r in self.results)

    @property
    def mean_recall_at_10(self) -> float:
        return _mean(r.recall_at_10 for r in self.results)

    @property
    def mrr(self) -> float:
        return _mean(r.mrr for r in self.results)

    @property
    def mean_ndcg_at_10(self) -> float:
        return _mean(r.ndcg_at_10 for r in self.results)

    @property
    def mean_judge_score(self) -> float | None:
        scores = [r.judge_score for r in self.results if r.judge_score is not None]
        return sum(scores) / len(scores) if scores else None


def _mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Run multi-metric retrieval evaluation against an indexed repository."""

    def __init__(
        self,
        repo_path: str,
        config: IndexConfig | None = None,
    ) -> None:
        if config is None:
            from trelix.core.config import EmbedderConfig, RetrievalConfig

            config = IndexConfig(
                repo_path=repo_path,
                embedder=EmbedderConfig(provider="local"),
                retrieval=RetrievalConfig(rerank=False),
            )
        else:
            # Allow callers to pass a config; honour repo_path override.
            object.__setattr__(config, "repo_path", repo_path) if False else None
            config = config.model_copy(update={"repo_path": repo_path})

        self._config = config
        self._repo_path = repo_path
        self._indexed = False

    # ------------------------------------------------------------------
    # Index (idempotent)
    # ------------------------------------------------------------------

    def _ensure_indexed(self) -> None:
        if self._indexed:
            return
        db_path = Path(self._repo_path) / ".trelix" / "index.db"
        if db_path.exists():
            logger.info("Skipping indexing — index already exists at %s", db_path)
        else:
            logger.info("Indexing repo at %s", self._repo_path)
            indexer = Indexer(self._config, quiet=True)
            stats = indexer.index()
            logger.info("Indexing complete: %s", stats)
        self._indexed = True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, cases: list[tuple[str, str]]) -> EvalReport:
        """
        Run eval for every (query, expected_file) pair.

        Args:
            cases: list of (query_string, expected_rel_path_fragment)

        Returns:
            EvalReport with per-query EvalResult and aggregate stats.
        """
        self._ensure_indexed()
        retriever = Retriever(self._config)

        eval_results: list[EvalResult] = []
        for query, expected_file in cases:
            context = retriever.retrieve(query)
            results: list[SearchResult] = context.results

            r1 = recall_at_k(results, expected_file, k=1)
            r5 = recall_at_k(results, expected_file, k=5)
            r10 = recall_at_k(results, expected_file, k=10)
            rr = reciprocal_rank(results, expected_file)
            ndcg = ndcg_at_k(results, expected_file, k=10)
            rank = find_rank(results, expected_file)

            eval_results.append(
                EvalResult(
                    query=query,
                    expected_file=expected_file,
                    recall_at_1=r1,
                    recall_at_5=r5,
                    recall_at_10=r10,
                    mrr=rr,
                    ndcg_at_10=ndcg,
                    rank=rank,
                )
            )

        report = EvalReport(results=eval_results)
        self._print_report(report)
        return report

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    def assert_min_metrics(
        self,
        report: EvalReport,
        min_recall5: float = 0.90,
        min_mrr: float = 0.75,
        min_ndcg: float = 0.80,
    ) -> None:
        """Raise AssertionError if any metric falls below the minimum threshold."""
        failures: list[str] = []

        if report.mean_recall_at_5 < min_recall5:
            failures.append(
                f"Recall@5 = {report.mean_recall_at_5:.3f} < required {min_recall5:.3f}"
            )
        if report.mrr < min_mrr:
            failures.append(f"MRR = {report.mrr:.3f} < required {min_mrr:.3f}")
        if report.mean_ndcg_at_10 < min_ndcg:
            failures.append(f"NDCG@10 = {report.mean_ndcg_at_10:.3f} < required {min_ndcg:.3f}")

        if failures:
            detail_lines = []
            for er in report.results:
                if er.recall_at_5 < 1.0 or er.mrr < 0.5:
                    detail_lines.append(
                        f"  [{er.rank:>4}] {er.query!r} -> {er.expected_file}"
                        f"  (R@5={er.recall_at_5:.1f}, MRR={er.mrr:.3f}, NDCG={er.ndcg_at_10:.3f})"
                    )
            msg = (
                "Retrieval quality below threshold:\n"
                + "\n".join(f"  - {f}" for f in failures)
                + "\n\nFailing / weak cases:\n"
                + "\n".join(detail_lines)
            )
            raise AssertionError(msg)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_report(self, report: EvalReport) -> None:
        try:
            self._print_rich(report)
        except ImportError:
            self._print_plain(report)

    def _print_rich(self, report: EvalReport) -> None:
        from rich import box as rich_box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"Trelix Eval — {report.n} queries",
            box=rich_box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Query", style="dim", max_width=45)
        table.add_column("Expected file", max_width=35)
        table.add_column("Rank", justify="right")
        table.add_column("R@1", justify="right")
        table.add_column("R@5", justify="right")
        table.add_column("R@10", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("NDCG@10", justify="right")

        for er in report.results:
            rank_str = str(er.rank) if er.rank > 0 else "—"
            row_style = "green" if er.recall_at_5 >= 1.0 else "red"
            table.add_row(
                er.query[:45],
                er.expected_file[-35:],
                rank_str,
                f"{er.recall_at_1:.0f}",
                f"{er.recall_at_5:.0f}",
                f"{er.recall_at_10:.0f}",
                f"{er.mrr:.3f}",
                f"{er.ndcg_at_10:.3f}",
                style=row_style,
            )

        table.add_section()
        table.add_row(
            "[bold]AGGREGATE[/bold]",
            "",
            "",
            f"{report.mean_recall_at_1:.3f}",
            f"{report.mean_recall_at_5:.3f}",
            f"{report.mean_recall_at_10:.3f}",
            f"{report.mrr:.3f}",
            f"{report.mean_ndcg_at_10:.3f}",
            style="bold white",
        )
        console.print(table)

    def _print_plain(self, report: EvalReport) -> None:
        col = 46
        print("\n" + "=" * 100)
        print(f"  TRELIX EVAL REPORT — {report.n} queries")
        print("=" * 100)
        hdr_cols = f"{'Rank':>5} {'R@1':>5} {'R@5':>5} {'R@10':>5} {'MRR':>7} {'NDCG@10':>8}"
        header = f"  {'Query':<{col}} {'Expected':<32} {hdr_cols}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for er in report.results:
            rank_str = f"{er.rank:>5}" if er.rank > 0 else "    —"
            mark = "" if er.recall_at_5 >= 1.0 else ""
            print(
                f"  {er.query[:col]:<{col}} {er.expected_file[-32:]:<32} "
                f"{rank_str} {er.recall_at_1:>5.0f} {er.recall_at_5:>5.0f} "
                f"{er.recall_at_10:>5.0f} {er.mrr:>7.3f} {er.ndcg_at_10:>8.3f} {mark}"
            )
        print("-" * 100)
        print(
            f"  {'AGGREGATE':<{col}} {'':32} "
            f"{'':>5} "
            f"{report.mean_recall_at_1:>5.3f} {report.mean_recall_at_5:>5.3f} "
            f"{report.mean_recall_at_10:>5.3f} {report.mrr:>7.3f} {report.mean_ndcg_at_10:>8.3f}"
        )
        print("=" * 100 + "\n")
