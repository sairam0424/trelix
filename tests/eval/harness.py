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

Metric arithmetic lives in exactly one place: ``trelix.eval.ndcg``, which
``trelix eval`` ships from and which agrees with an independent (sklearn)
implementation to 1e-12 — see ``tests/unit/test_eval_metric_single_implementation.py``.
This module used to carry a *second* implementation (``tests/eval/metrics.py``,
now deleted) that diverged from the shipped one by up to 0.6309 nDCG@10 on
identical rankings, because it scored raw chunk positions instead of distinct
files and matched ground truth by substring instead of exact string equality.
``score_ranking`` below is the only boundary between this harness and the
shipped metrics; it does no arithmetic of its own beyond building the
integer-ID universe the shipped functions require.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from trelix.core.config import IndexConfig
from trelix.eval.ndcg import mrr, ndcg_at_k, recall_at_k
from trelix.indexing.indexer import Indexer
from trelix.retrieval.retriever import Retriever

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trelix.core.models import SearchResult

logger = logging.getLogger("trelix.eval")


# ---------------------------------------------------------------------------
# Scoring — the only place this module talks to the shipped metric module
# ---------------------------------------------------------------------------


def _rank_from_mrr(reciprocal: float) -> int:
    """1-based rank of the first relevant hit, recovered from its reciprocal.

    ``round()``, not ``int()``: measured over ranks 1..100000, ``int(1.0 / (1.0 /
    r)) != r`` for 5,850 of them (smallest: rank 93, where
    ``1.0 / (1.0 / 93) == 92.99999999999999`` and truncation reports 92).
    ``round()`` recovers every one of them exactly.
    """
    if reciprocal <= 0.0:
        return -1
    return round(1.0 / reciprocal)


def score_ranking(
    ranked_files: list[str], relevant_files: Sequence[str]
) -> tuple[float, float, float, float, float, int]:
    """Score one query's ranked file list against its ground truth.

    ``ranked_files`` are exact repo-relative ``rel_path`` strings in rank order
    (chunk order — the same file may repeat; the shipped metrics collapse
    repeats to each file's best rank, so ``@k`` counts distinct files). Matching
    against ``relevant_files`` is exact string equality, never substring: a bare
    stem like ``"user.py"`` will not match ``"src/superuser.py"``.

    Returns ``(recall@1, recall@5, recall@10, mrr, ndcg@10, rank)``. ``rank`` is
    the 1-based position of the first relevant file, or ``-1`` if it was never
    retrieved.
    """
    relevant = set(relevant_files)
    universe = {f: i for i, f in enumerate(set(ranked_files) | relevant)}
    ranked_ids = [universe[f] for f in ranked_files]
    relevant_ids = {universe[f] for f in relevant}

    r1 = recall_at_k(ranked_ids, relevant_ids, k=1)
    r5 = recall_at_k(ranked_ids, relevant_ids, k=5)
    r10 = recall_at_k(ranked_ids, relevant_ids, k=10)
    rr = mrr(ranked_ids, relevant_ids)
    ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=10)
    rank = _rank_from_mrr(rr)
    return r1, r5, r10, rr, ndcg, rank


# ---------------------------------------------------------------------------
# Per-query result + aggregate report
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Per-query eval metrics."""

    query: str
    expected_file: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float  # reciprocal rank for this query
    ndcg_at_10: float
    rank: int  # 1-based rank of first match, -1 if not found
    judge_score: float | None = None  # LLM-as-judge score (0.0–1.0); None if judge not run
    retrieved_files: list[str] = field(
        default_factory=list
    )  # rel_paths of retrieved files (optional)


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
            config = config.model_copy(update={"repo_path": repo_path})

        self._config = config
        self._repo_path = repo_path
        self._indexed = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_cases(self, cases: list[tuple[str, str]]) -> None:
        """Refuse any case whose ground truth cannot exist under ``repo_path``.

        Matching is exact-string (see ``score_ranking``), so a stale fixture —
        a bare stem, a renamed file, a path from a different repo — scores 0.0
        on every query and reads exactly like a genuine retrieval failure.
        Called before indexing, so an unusable golden set fails fast without
        spending an embedder or a query planner call.
        """
        root = Path(self._repo_path)
        for query, expected in cases:
            if not (root / expected).exists():
                raise ValueError(
                    f"{expected!r} does not exist under {self._repo_path!r} (query: {query!r})"
                )

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
            cases: list of (query_string, expected_rel_path). ``expected_rel_path``
                must be the exact repo-relative path — see ``_validate_cases`` and
                ``score_ranking``, both of which match exactly, never by substring.

        Returns:
            EvalReport with per-query EvalResult and aggregate stats.
        """
        self._validate_cases(cases)
        self._ensure_indexed()
        retriever = Retriever(self._config)

        eval_results: list[EvalResult] = []
        for query, expected_file in cases:
            context = retriever.retrieve(query)
            results: list[SearchResult] = context.results
            ranked_files = [r.file.rel_path for r in results]

            r1, r5, r10, rr, ndcg, rank = score_ranking(ranked_files, [expected_file])

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
        """Raise AssertionError if any MEAN metric falls below the minimum threshold.

        Kept for callers that want a coarse mean-over-the-set check (e.g. a
        standalone script comparing two configurations). The CI retrieval-quality
        gate itself (``tests/integration/test_eval.py::test_trelix_self_eval``) no
        longer calls this: a mean over a golden set can hide a large regression on
        a few queries behind headroom on the rest — measured concretely, flipping
        ``graph_search_enabled`` to ``True`` costs -0.1384 mean nDCG@10 while the
        gate's floor sits 0.3631 below the baseline mean, so the mean gate still
        returns PASS. The gate uses a per-query rank ledger instead.
        """
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
