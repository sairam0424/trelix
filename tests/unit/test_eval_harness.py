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
