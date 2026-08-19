"""
Reciprocal Rank Fusion (RRF) — combines multiple ranked lists into one.

Formula:  score(doc) = Σ  1 / (k + rank_i)
where k=60 is the standard constant (Cormack et al. 2009).

Why RRF instead of score normalization:
- Scores from different systems (BM25 vs cosine) are not comparable
- RRF only uses rank position, making it robust across any mix of retrievers
- Simple, fast, no training needed

This module owns the ONE dedupe key in the retrieval path, and it must be
globally unique — the input lists are sometimes legs of one repo and sometimes
whole separate repos (federated fan-out). See _fusion_identity().
"""

from __future__ import annotations

from collections import defaultdict

from trelix.core.models import SearchResult


def _fusion_identity(result: SearchResult) -> tuple[str, int]:
    """The key two SearchResults must agree on to be the same row.

    `chunk.symbol_id` alone is NOT that key, and using it alone was EXE-02.
    It is `symbols.id`, an `INTEGER PRIMARY KEY AUTOINCREMENT` in ONE repo's
    `.trelix/index.db`, so it is unique only within the database that issued it
    — every repo numbers its first symbol 1, so cross-repo collisions are
    guaranteed, not unlucky. Federated fan-out hands this module one ranked list
    per repo, so a bare symbol_id key made repo B's symbol 1 a duplicate of repo
    A's symbol 1 and the first-seen-wins branch below deleted it. "First" was
    whichever `ThreadPoolExecutor` future completed first, so `search_all` erased
    an entire repo's results nondeterministically — five runs of one query gave
    per_repo={sample-a: 17} once and {sample-b: 5, sample-a: 12} another time —
    while the envelope reported repos_searched=2, repos_skipped=0 every run.

    `IndexedFile.path` is the ABSOLUTE path on disk (`walker.py` builds it from
    the resolved repo root), so it identifies the issuing database and not merely
    the file: two distinct repos cannot share one. `rel_path` cannot do this job
    — it is repo-relative, so two repos with the same layout both report
    `src/app.py`. FederatedRetriever._query_repos() used to run a SECOND dedupe
    keyed on exactly that, `{rel_path}:{symbol_id}`, and it erased the second repo
    all over again after this function was fixed; it has been deleted, because
    this function is now the single place cross-repo identity is decided.

    Pairing path with symbol_id keeps intra-repo dedupe intact, which is the half
    that must not regress: the vector, BM25, grep and summary legs of one repo
    all hydrate from the same database, so the same chunk yields the same
    (path, symbol_id) and still collapses onto a single fused row.

    Not `make_scip_symbol_id()` (`trelix.federation.retriever`): that module
    imports this one, so importing it back here is a circular import — and this
    function also runs for ordinary single-repo retrieval, where there is no
    package/version pair to hash.
    """
    chunk, indexed_file = result.chunk, result.file
    return (indexed_file.path, chunk.symbol_id)


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    list_weights: list[float] | None = None,
) -> list[SearchResult]:
    """
    Fuse multiple ranked result lists using RRF, then optionally apply
    per-language file-type weight multipliers.

    Args:
        ranked_lists: list of result lists, each sorted by relevance (best first)
        k:            RRF constant (default 60, Cormack et al. 2009)
        weights:      optional dict mapping Language enum value (str) to a
                      multiplicative weight applied after RRF accumulation.
                      None or empty dict → no weighting (backward compatible).
        list_weights: optional per-list multiplier (same length/order as
                      ranked_lists) applied to each list's RRF rank
                      contribution before summing — e.g. federated search
                      weighting one repo's results above another's. Orthogonal
                      to `weights` (which scales by result language, not by
                      source list). None → no weighting (backward compatible).

    Returns:
        Single merged list sorted by fused (weighted) RRF score, best first,
        deduplicated on _fusion_identity() — one row per (absolute file path,
        symbol_id). Callers must NOT add a second dedupe pass on this output:
        every distinct row here is already distinct, so any further pass can only
        delete correct rows, which is exactly how a whole repo went missing.
    """
    # Map globally-unique row identity → accumulated RRF score. The key is
    # (absolute file path, symbol_id), NOT symbol_id alone — see
    # _fusion_identity() for the cross-repo erasure a bare symbol_id caused.
    rrf_scores: dict[tuple[str, int], float] = defaultdict(float)
    # Keep the best SearchResult object per identity (highest contributing list)
    best_result: dict[tuple[str, int], SearchResult] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        list_weight = list_weights[list_idx] if list_weights else 1.0
        for rank, result in enumerate(ranked_list, start=1):
            identity = _fusion_identity(result)
            rrf_scores[identity] += list_weight / (k + rank)
            # Keep first-seen result: source reflects which leg first found it.
            # Do NOT replace based on raw score — scores across legs (cosine vs
            # BM25) are not comparable, so score comparison would always favor
            # vector (0.7–0.95 range) over BM25 (0.05–0.5 range).
            # First-seen is only safe because `identity` is now globally unique:
            # while it was a per-database rowid, "first" silently meant "whichever
            # repo's thread finished first" and the rest were discarded.
            if identity not in best_result:
                best_result[identity] = result

    # Apply file-type weight multiplier (new step — skipped when weights is None/empty)
    if weights:
        for identity, result in best_result.items():
            lang = result.file.language  # Language enum (StrEnum → str)
            multiplier = weights.get(str(lang), 1.0)
            rrf_scores[identity] *= multiplier

    # Sort by fused score descending
    sorted_ids = sorted(rrf_scores, key=lambda ident: rrf_scores[ident], reverse=True)

    fused: list[SearchResult] = []
    for new_rank, identity in enumerate(sorted_ids, start=1):
        result = best_result[identity]
        # Overwrite score with the RRF score for downstream reranking
        result.score = rrf_scores[identity]
        result.rank = new_rank
        fused.append(result)

    return fused
