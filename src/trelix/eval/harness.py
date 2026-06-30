"""
EvalHarness — run a golden JSONL file through trelix retrieval and report metrics.

Golden file format (one JSON object per line):
    {"query": "how does JWT auth work", "relevant_files": ["src/auth.py"]}

Usage:
    harness = EvalHarness(config)
    metrics = harness.run("golden.jsonl")
    # -> {"ndcg@10": 0.74, "recall@10": 0.81, "mrr": 0.66, "n_queries": 12}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.eval.ndcg import mrr, ndcg_at_k, recall_at_k

logger = logging.getLogger("trelix.eval")


class EvalHarness:
    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        from trelix.retrieval.retriever import Retriever

        self._retriever = Retriever(config)

    def run(self, golden_path: str) -> dict[str, float]:
        """
        Run all queries in the golden file and return aggregate metrics.

        Returns dict with keys: ndcg@10, recall@10, mrr, n_queries.
        """
        path = Path(golden_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden file not found: {golden_path}")

        queries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not queries:
            return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr": 0.0, "n_queries": 0.0}

        ndcg_scores: list[float] = []
        recall_scores: list[float] = []
        mrr_scores: list[float] = []

        for item in queries:
            query = item["query"]
            relevant_files: set[str] = set(item.get("relevant_files", []))
            if not relevant_files:
                continue

            try:
                ctx = self._retriever.retrieve(query)
            except Exception as exc:
                logger.warning("Query %r failed: %s", query[:60], exc)
                ndcg_scores.append(0.0)
                recall_scores.append(0.0)
                mrr_scores.append(0.0)
                continue

            # Use file rel_path as the ID for matching
            ranked_files = [r.file.rel_path for r in ctx.results]
            # Convert to integer IDs for metric functions (hash-based)
            file_to_id = {f: i for i, f in enumerate(set(ranked_files) | relevant_files)}
            ranked_ids = [file_to_id[f] for f in ranked_files]
            relevant_ids = {file_to_id[f] for f in relevant_files if f in file_to_id}

            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant_ids, k=10))
            recall_scores.append(recall_at_k(ranked_ids, relevant_ids, k=10))
            mrr_scores.append(mrr(ranked_ids, relevant_ids))

        n = len(ndcg_scores)
        if n == 0:
            return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr": 0.0, "n_queries": 0.0}

        return {
            "ndcg@10": sum(ndcg_scores) / n,
            "recall@10": sum(recall_scores) / n,
            "mrr": sum(mrr_scores) / n,
            "n_queries": float(n),
        }
