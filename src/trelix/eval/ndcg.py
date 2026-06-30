"""
Pure metric functions for retrieval evaluation.

All functions are stateless, dependency-free, and O(k log k).
Compatible with CoIR benchmark format (arXiv:2407.02883).
"""

from __future__ import annotations

import math


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """
    Compute nDCG@k.

    Args:
        ranked_ids: list of retrieved IDs in rank order (best first)
        relevant_ids: set of relevant (ground-truth) IDs
        k: cutoff

    Returns:
        nDCG@k score in [0, 1]
    """
    if not relevant_ids:
        return 0.0

    def dcg(ids: list[int], rel: set[int], k: int) -> float:
        return sum(
            1.0 / math.log2(rank + 2) for rank, doc_id in enumerate(ids[:k]) if doc_id in rel
        )

    actual = dcg(ranked_ids, relevant_ids, k)
    # Ideal: all relevant docs at top positions
    ideal_ranked = list(relevant_ids)[:k]
    ideal = dcg(ideal_ranked, relevant_ids, k)
    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean Reciprocal Rank — reciprocal of the first relevant rank."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
