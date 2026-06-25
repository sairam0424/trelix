"""
Reciprocal Rank Fusion (RRF) — combines multiple ranked lists into one.

Formula:  score(doc) = Σ  1 / (k + rank_i)
where k=60 is the standard constant (Cormack et al. 2009).

Why RRF instead of score normalization:
- Scores from different systems (BM25 vs cosine) are not comparable
- RRF only uses rank position, making it robust across any mix of retrievers
- Simple, fast, no training needed
"""

from __future__ import annotations

from collections import defaultdict

from trelix.core.models import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """
    Fuse multiple ranked result lists using RRF.

    Args:
        ranked_lists: list of result lists, each sorted by relevance (best first)
        k: RRF constant (default 60 from the original paper)

    Returns:
        Single merged list sorted by fused RRF score, best first.
    """
    # Map chunk_id → accumulated RRF score
    rrf_scores: dict[int, float] = defaultdict(float)
    # Keep the best SearchResult object per chunk (highest contributing list)
    best_result: dict[int, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result.chunk.symbol_id   # use symbol_id as dedup key
            rrf_scores[chunk_id] += 1.0 / (k + rank)
            # Keep first-seen result: source reflects which leg first found it.
            # Do NOT replace based on raw score — scores across legs (cosine vs
            # BM25) are not comparable, so score comparison would always favor
            # vector (0.7–0.95 range) over BM25 (0.05–0.5 range).
            if chunk_id not in best_result:
                best_result[chunk_id] = result

    # Sort by fused score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused: list[SearchResult] = []
    for new_rank, chunk_id in enumerate(sorted_ids, start=1):
        result = best_result[chunk_id]
        # Overwrite score with the RRF score for downstream reranking
        result.score = rrf_scores[chunk_id]
        result.rank = new_rank
        fused.append(result)

    return fused
