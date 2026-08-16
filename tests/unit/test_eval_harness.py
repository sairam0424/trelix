"""Tests for CoIR-style evaluation harness and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trelix.eval.ndcg import mrr, ndcg_at_k, recall_at_k


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        score = ndcg_at_k(ranked, relevant, k=5)
        assert score == pytest.approx(1.0)

    def test_no_relevant_in_top_k(self) -> None:
        ranked = [10, 11, 12]
        relevant = {99}
        assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        ranked = [1, 10, 2, 11, 12]
        relevant = {1, 2}
        score = ndcg_at_k(ranked, relevant, k=5)
        assert 0.0 < score < 1.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_relevant_found(self) -> None:
        assert recall_at_k([1, 2, 3], {1, 2}, k=3) == pytest.approx(1.0)

    def test_none_found(self) -> None:
        assert recall_at_k([10, 11], {1}, k=2) == pytest.approx(0.0)

    def test_partial(self) -> None:
        assert recall_at_k([1, 10, 11], {1, 2}, k=3) == pytest.approx(0.5)


class TestMRR:
    def test_first_hit_at_rank_1(self) -> None:
        assert mrr([1, 2, 3], {1}) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self) -> None:
        assert mrr([10, 1, 2], {1}) == pytest.approx(0.5)

    def test_no_hit(self) -> None:
        assert mrr([10, 11, 12], {1}) == pytest.approx(0.0)


class TestEvalHarness:
    def test_run_returns_metrics_dict(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.eval.harness import EvalHarness

        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            json.dumps({"query": "how does auth work", "relevant_files": ["auth.py"]}) + "\n"
        )

        mock_ctx = MagicMock()
        mock_result = MagicMock()
        mock_result.file.rel_path = "auth.py"
        mock_ctx.results = [mock_result]

        config = IndexConfig(repo_path=str(tmp_path))
        harness = EvalHarness(config)

        with patch.object(harness, "_retriever") as mock_r:
            mock_r.retrieve.return_value = mock_ctx
            metrics = harness.run(str(golden))

        assert "ndcg@10" in metrics
        assert "recall@10" in metrics
        assert "mrr" in metrics
        assert 0.0 <= metrics["ndcg@10"] <= 1.0


class TestMetricsStayWithinTheirDocumentedRange:
    """nDCG and recall must stay in [0, 1] even when an ID repeats in the ranking.

    Both functions document a `[0, 1]` result, and `docs/CLI_REFERENCE.md` shows
    example output in that range. Neither deduplicated, so a repeated ID scored
    once per occurrence.

    That is not a corner case for trelix: `EvalHarness` maps chunk-level results
    onto file-level IDs (`r.file.rel_path`), and this index averages ~77 chunks per
    file, so a single relevant file routinely supplies several of the top 10 hits.
    Measured before the fix: one relevant file appearing 5 times in the top 10
    produced recall@10 = 5.0 and nDCG@10 = 2.52.
    """

    def test_recall_cannot_exceed_one(self) -> None:
        ranked = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]  # the one relevant file, 5 times
        assert recall_at_k(ranked, {0}, k=10) == 1.0

    def test_recall_cannot_exceed_one_when_every_hit_repeats(self) -> None:
        assert recall_at_k([0] * 10, {0}, k=10) == 1.0

    def test_ndcg_cannot_exceed_one(self) -> None:
        ranked = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]
        assert 0.0 <= ndcg_at_k(ranked, {0}, k=10) <= 1.0

    def test_ndcg_is_one_for_a_perfect_repeated_ranking(self) -> None:
        """A relevant file first, repeated below, is still a perfect ranking."""
        assert ndcg_at_k([0] * 10, {0}, k=10) == pytest.approx(1.0)

    def test_repeats_do_not_beat_a_cleaner_ranking(self) -> None:
        """Ranking A at 1 must not score below ranking B just because A repeats.

        This is the ordering property the inflation destroyed: padding a result
        list with duplicates used to raise the score.
        """
        relevant = {0, 1}
        padded = ndcg_at_k([0, 0, 0, 0, 1], relevant, k=10)
        clean = ndcg_at_k([0, 1], relevant, k=10)
        assert clean >= padded

    def test_partial_relevance_is_still_partial(self) -> None:
        """Deduplicating must not turn a partial hit into a perfect one."""
        # Two relevant files (0, 1); only 0 is retrieved, three times.
        assert recall_at_k([0, 0, 0, 9], {0, 1}, k=10) == pytest.approx(0.5)


class TestHarnessRanksFilesNotChunks:
    """Ground truth is file-level, so the ranking the metrics see must be too."""

    @staticmethod
    def _harness_metrics(tmp_path: Path, ranked_rel_paths: list[str], relevant: list[str]):
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.eval.harness import EvalHarness

        golden = tmp_path / "golden.jsonl"
        golden.write_text(json.dumps({"query": "q", "relevant_files": relevant}) + "\n")

        mock_ctx = MagicMock()
        mock_ctx.results = []
        for path in ranked_rel_paths:
            hit = MagicMock()
            hit.file.rel_path = path
            mock_ctx.results.append(hit)

        harness = EvalHarness(IndexConfig(repo_path=str(tmp_path)))
        with patch.object(harness, "_retriever") as mock_r:
            mock_r.retrieve.return_value = mock_ctx
            return harness.run(str(golden))

    def test_many_chunks_from_one_file_score_once(self, tmp_path: Path) -> None:
        """The realistic shape: every top hit is a different chunk of one file."""
        metrics = self._harness_metrics(
            tmp_path, ["src/walker.py"] * 8 + ["src/other.py"] * 2, ["src/walker.py"]
        )
        assert metrics["recall@10"] == 1.0
        assert 0.0 <= metrics["ndcg@10"] <= 1.0

    def test_k_counts_distinct_files(self, tmp_path: Path) -> None:
        """A relevant file at chunk-rank 11 is within the top 10 *files*.

        Ten chunks of one file are one result as far as file-level ground truth is
        concerned; collapsing them is what makes `@10` mean ten files.
        """
        ranked = ["src/noise.py"] * 10 + ["src/target.py"]
        metrics = self._harness_metrics(tmp_path, ranked, ["src/target.py"])
        assert metrics["recall@10"] == 1.0, (
            "src/target.py is the 2nd distinct file retrieved and must count as a hit"
        )

    def test_a_genuine_miss_is_still_a_miss(self, tmp_path: Path) -> None:
        """Deduplication must not manufacture hits that were never retrieved."""
        ranked = [f"src/noise{i}.py" for i in range(12)]
        metrics = self._harness_metrics(tmp_path, ranked, ["src/target.py"])
        assert metrics["recall@10"] == 0.0
        assert metrics["ndcg@10"] == 0.0
        assert metrics["mrr"] == 0.0


class TestMrrDeduplicates:
    """MRR must agree with nDCG/recall about what a rank is."""

    def test_repeats_before_the_hit_do_not_depress_the_score(self) -> None:
        # A occupies ranks 1-3 as three chunks of one file; B is the 2nd distinct file.
        assert mrr([0, 0, 0, 1], {1}) == pytest.approx(0.5)

    def test_first_rank_still_scores_one(self) -> None:
        assert mrr([1, 1, 0], {1}) == pytest.approx(1.0)

    def test_no_hit_is_still_zero(self) -> None:
        assert mrr([0, 0, 2], {9}) == 0.0
