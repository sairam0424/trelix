"""
Retrieval quality metrics for the trelix eval harness.

Provides: Recall@K, Reciprocal Rank (MRR component), and NDCG@K.

All functions take a list[SearchResult] and an expected_file str (the
rel_path that should appear in the results), returning a float in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trelix.core.models import SearchResult


def recall_at_k(results: list[SearchResult], expected_file: str, k: int) -> float:
    """Return 1.0 if expected_file appears in the top-k results, else 0.0.

    Matching is substring-based so callers can pass a short stem like
    "config.py" and it will match "src/trelix/core/config.py".
    """
    for r in results[:k]:
        if expected_file in r.file.rel_path:
            return 1.0
    return 0.0


def reciprocal_rank(results: list[SearchResult], expected_file: str) -> float:
    """Return 1/rank for the first result whose file matches expected_file.

    Returns 0.0 if expected_file is not found anywhere in results.
    Used to compute Mean Reciprocal Rank (MRR) across a query set.
    """
    for i, r in enumerate(results, start=1):
        if expected_file in r.file.rel_path:
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: list[SearchResult], expected_file: str, k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Relevance model: binary, one relevant document (the first result whose
    file matches expected_file). We award gain only on the FIRST hit to avoid
    inflating DCG when multiple symbols from the same file appear in results.

    DCG@K = 1 / log2(rank + 1)  where rank is the 1-based position of the
            first matching result, or 0.0 if no match in top-K.
    IDCG  = 1 / log2(2) = 1.0  (optimal: match at rank 1).
    NDCG@K = DCG@K / IDCG = DCG@K.

    Returns a value in [0.0, 1.0].
    """
    for i, r in enumerate(results[:k], start=1):
        if expected_file in r.file.rel_path:
            dcg = 1.0 / math.log2(i + 1)
            # IDCG = 1.0 (perfect ranking places the relevant doc at rank 1)
            return dcg / 1.0
    return 0.0


def find_rank(results: list[SearchResult], expected_file: str) -> int:
    """Return the 1-based rank of the first matching result, or -1 if not found."""
    for i, r in enumerate(results, start=1):
        if expected_file in r.file.rel_path:
            return i
    return -1


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
