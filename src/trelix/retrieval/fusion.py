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


def _fusion_identity(result: SearchResult) -> tuple[str, int, int | None]:
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

    (path, symbol_id) alone is ALSO not enough, as of chunker.py's oversized-
    symbol split (CHANGELOG [3.3.4], docs/reports/chunking-strategy-spike-
    2026-09-18.md addendum): a symbol over `max_tokens_per_chunk` now yields
    MULTIPLE `chunks` rows sharing one `symbol_id` — sequential pieces, not a
    truncated single row. If two pieces of the same split symbol both rank in
    one query's fused input (reachable on the always-on vector leg, whose ANN
    search can legitimately return more than one piece of the same oversized
    symbol), (path, symbol_id) collapsed them into one row and the first-seen
    branch below silently dropped the second piece's distinct text, even
    though it was a real, separate hit — that was the bug this key now fixes.
    `chunk.id` (`chunks.id`, the per-piece `INTEGER PRIMARY KEY AUTOINCREMENT`)
    is the piece-level discriminator: two pieces of one split symbol get two
    distinct `chunks.id` values, so `(path, symbol_id, chunk.id)` keeps them
    apart. Bare `chunk.id` would repeat EXE-02 one level down — it is exactly
    as per-database an autoincrement rowid as `symbols.id` — so it is never
    used unpaired from `path`; it only ever narrows an already-globally-unique
    `(path, symbol_id)` prefix.

    Pairing path with symbol_id AND chunk.id keeps intra-repo dedupe intact,
    which is the half that must not regress: the vector, BM25, grep and
    summary legs of one repo all hydrate the SAME chunk row through
    `Database.get_chunk_with_context`/`get_first_chunk_for_symbol`/
    `get_chunk_by_id` — all three set `Chunk.id` from the real `chunks.id`
    column — so a chunk found by more than one leg still yields the same
    (path, symbol_id, chunk.id) and still collapses onto a single fused row
    with every leg's RRF contribution summed.

    `chunk.id` is `None` on results hydrated through the graph-expansion
    family (`hydrate_symbol`, `_hydrate_symbol_id` in retriever.py; every
    `expand_with_*`/`seed_from_import_paths` helper in graph.py; the
    `_hydrate`/`bm25_search` fallbacks in grep_search.py/bm25.py;
    `graph_search` in graph/search.py) exactly when
    `Database.get_first_chunk_for_symbol` returns None and the caller falls
    back to a synthetic, unsaved `Chunk(symbol_id=..., ...)` with no `id=`
    argument — reachable whenever a symbol has no row in `chunks` at all.
    This is not a collision risk: two such synthetic chunks for the SAME
    (path, symbol_id) both carry chunk.id=None, so their identity tuples are
    still equal and they still collapse — the fallback degrades to exactly
    the pre-fix (path, symbol_id) behavior for that entry, never regressing
    it, which is why no extra `is None` branch is needed here.

    `Database.get_first_chunk_for_symbol` always anchors to the FIRST chunk
    row for a symbol ("the first (and usually only) chunk", per its own
    docstring), so BM25/grep/graph-expansion legs can only ever surface piece
    1 of a split symbol, never piece 2+, regardless of which piece is
    actually the best match — only the vector leg's ANN search can reach a
    later piece directly, since each piece is embedded and indexed
    separately. A vector hit on piece 2 and a BM25 hit on piece 1 of the SAME
    split symbol therefore now correctly do NOT collapse (their chunk.id
    differs) — but they also do not get to sum RRF contributions the way two
    legs finding the exact same chunk do. That is a pre-existing per-leg
    granularity limitation (which piece a given leg can even see), not
    something this identity-key fix is responsible for solving, and it is
    strictly better than the silent content-drop it replaces.

    `chunk.id` must NOT be used for `source == "sub_chunk"` results, and this is
    a second, distinct reason from the ones above — not a collision-safety
    concern, a deliberate design one, confirmed by a pinned mutation test on
    `Retriever._dedup()` (`test_two_sub_chunks_of_one_symbol_collapse_to_a_
    single_row`, `tests/unit/test_retriever_row_identity_and_leg_weights.py`)
    that this module's own sibling reduction must also honor: MGS3's sub-chunk
    leg (`_sub_chunk_search`) emits one `SearchResult` per `sub_chunks` rowid,
    a finer-grained, overlapping/redundant VIEW into a symbol whose full body
    the primary chunk already covers -- unlike a split PRIMARY chunk (piece 1
    vs. piece 2 of one oversized symbol), which is genuinely disjoint content.
    Keying sub-chunk hits on `chunk.id` would let N sub-chunks of one symbol
    all survive as N separate fused rows, each spending assembler budget on
    text the caller already has via the primary chunk. Sub-chunk hits
    therefore collapse on `(path, symbol_id)` alone, exactly as before this
    fix; every other source uses the 3-tuple.

    Not `make_scip_symbol_id()` (`trelix.federation.retriever`): that module
    imports this one, so importing it back here is a circular import — and this
    function also runs for ordinary single-repo retrieval, where there is no
    package/version pair to hash.
    """
    chunk, indexed_file = result.chunk, result.file
    if result.source == "sub_chunk":
        return (indexed_file.path, chunk.symbol_id, None)
    return (indexed_file.path, chunk.symbol_id, chunk.id)


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
        symbol_id, chunk id), except sub_chunk-sourced results, which collapse
        on (path, symbol_id) alone (see _fusion_identity()'s docstring).
        Callers must NOT add a second dedupe pass on this output: every
        distinct row here is already distinct, so any further pass can only
        delete correct rows, which is exactly how a whole repo went missing.
    """
    # Map globally-unique row identity → accumulated RRF score. The key is
    # (absolute file path, symbol_id, chunk id), NOT symbol_id alone and NOT
    # (path, symbol_id) alone — see _fusion_identity() for the cross-repo
    # erasure a bare symbol_id caused (EXE-02) and the split-symbol-piece
    # collapse a bare (path, symbol_id) caused.
    rrf_scores: dict[tuple[str, int, int | None], float] = defaultdict(float)
    # Keep the best SearchResult object per identity (highest contributing list)
    best_result: dict[tuple[str, int, int | None], SearchResult] = {}

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
