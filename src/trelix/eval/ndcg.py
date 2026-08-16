"""
Pure metric functions for retrieval evaluation.

All functions are stateless, dependency-free, and O(k log k).
Compatible with CoIR benchmark format (arXiv:2407.02883).
"""

from __future__ import annotations

import math


def _dedupe(ranked_ids: list[int]) -> list[int]:
    """Keep each ID's best (earliest) rank and drop later repeats.

    A ranking may legitimately name the same document more than once — trelix's own
    harness ranks chunks and then maps them onto file IDs, so one file can occupy
    several positions. A document is still *one* document, though: scoring it once per
    appearance let both metrics run past the `[0, 1]` they promise (a single relevant
    file appearing five times in the top ten measured recall@10 = 5.0, nDCG@10 = 2.52),
    and it meant padding a result list with duplicates raised the score.
    """
    seen: set[int] = set()
    unique: list[int] = []
    for doc_id in ranked_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """
    Compute nDCG@k.

    Repeated IDs are collapsed to their best rank first (see `_dedupe`), so `k` counts
    distinct documents.

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

    actual = dcg(_dedupe(ranked_ids), relevant_ids, k)
    # Ideal: all relevant docs at top positions
    ideal_ranked = list(relevant_ids)[:k]
    ideal = dcg(ideal_ranked, relevant_ids, k)
    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """Fraction of relevant documents found in the top-k distinct results."""
    if not relevant_ids:
        return 0.0
    hits = len(set(_dedupe(ranked_ids)[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def mrr(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean Reciprocal Rank — reciprocal of the first relevant rank.

    Deduplicated for the same reason as the metrics above, but note the effect runs the
    other way: repeats *before* the first relevant hit inflate its rank number and so
    depress the score. `[A, A, A, B]` scores 1/4 undeduplicated and 1/2 deduplicated.
    Leaving this function alone would have made the three metrics disagree about what a
    rank is.
    """
    for rank, doc_id in enumerate(_dedupe(ranked_ids), start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
