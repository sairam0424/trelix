"""
Call graph + import graph expansion — inspired by Aider's PageRank approach.

Two expansion strategies run after RRF fusion:

  1. Call graph expansion  — follow caller/callee edges (1–N hops).
     Candidates are ranked by PageRank centrality so the most structurally
     important symbols fill the context budget first.

  2. Import graph expansion — follow resolved import edges to related files.
     If we retrieved symbol X from file A, we surface top symbols from files
     that A imports and from files that import A.

Both strategies discount scores by hop distance so they don't crowd out
the directly retrieved results during reranking.
"""

from __future__ import annotations

from trelix.core.models import Chunk, SearchResult
from trelix.store.db import Database


def expand_with_call_graph(
    db: Database,
    results: list[SearchResult],
    depth: int = 1,
    max_extra: int = 10,
) -> list[SearchResult]:
    """
    Expand result set by following call graph edges (callers + callees).

    Collects ALL reachable candidates within `depth` hops, ranks them by
    PageRank centrality (so central hub symbols surface first), then
    hydrates and returns the top `max_extra`.
    """
    if not results:
        return []

    seen_ids: set[int] = {r.chunk.symbol_id for r in results}
    # candidates: (symbol_id, hop_distance)
    candidates: list[tuple[int, int]] = []
    frontier = [r.chunk.symbol_id for r in results]

    for hop in range(1, depth + 1):
        next_frontier: list[int] = []
        for symbol_id in frontier:
            for neighbor_id in db.get_callees(symbol_id) + db.get_callers(symbol_id):
                if neighbor_id in seen_ids:
                    continue
                seen_ids.add(neighbor_id)
                next_frontier.append(neighbor_id)
                candidates.append((neighbor_id, hop))
        frontier = next_frontier

    if not candidates:
        return []

    # Only apply PageRank re-sorting when the call graph is rich enough to matter.
    # On sparse graphs (few resolved callee_ids) PageRank scores are near-uniform
    # and the sort just shuffles BFS order, which hurts more than it helps.
    total_resolved = sum(
        1 for sid, _ in candidates if db.get_callees(sid) or db.get_callers(sid)
    )
    if total_resolved >= 3:
        all_ids = [r.chunk.symbol_id for r in results] + [sid for sid, _ in candidates]
        pr_scores = dict(rank_by_pagerank(all_ids, db))
        # Closer hops win; PageRank breaks ties within the same hop
        candidates.sort(key=lambda x: (x[1], -pr_scores.get(x[0], 0.0)))

    base_score = results[0].score if results else 0.5
    extra: list[SearchResult] = []

    for symbol_id, hop in candidates[:max_extra]:
        sym_file = db.get_symbol_with_file(symbol_id)
        if sym_file is None:
            continue
        symbol, file = sym_file

        chunk = db.get_first_chunk_for_symbol(symbol_id)
        if chunk is None:
            chunk = Chunk(
                symbol_id=symbol_id,
                chunk_text=symbol.body[:2000],
                token_count=0,
            )

        extra.append(SearchResult(
            chunk=chunk,
            symbol=symbol,
            file=file,
            score=base_score * (0.5 ** hop),
            rank=len(extra) + 1,
            source="graph_expansion",
        ))

    return extra


def expand_with_imports(
    db: Database,
    results: list[SearchResult],
    max_extra: int = 5,
    depth: int = 1,
    direction: str = "both",
) -> list[SearchResult]:
    """
    Expand result set by following resolved import edges.

    Parameters
    ----------
    max_extra   : max symbols to add from import expansion
    depth       : hops through the import graph (1=direct, 2=transitive)
    direction   : "both"    — outgoing (what this file imports) + incoming (what imports this file)
                  "forward" — outgoing only  → enumerate dependencies of X
                  "reverse" — incoming only  → enumerate dependents on X (blast radius)

    Works across all languages — relies on imports.imported_file_id being
    populated by Indexer's second-pass resolve_import_file_ids().

    Score is discounted below real RRF results so import-expanded symbols
    don't crowd out directly retrieved ones during reranking.
    """
    if not results or max_extra <= 0:
        return []

    base_score = results[0].score if results else 0.5
    score_discount = 0.15

    # ── Multi-hop BFS over the file import graph ──────────────────────────────
    # seed_file_ids: files from direct retrieval results (not expanded)
    seed_file_ids: set[int] = {r.symbol.file_id for r in results}
    visited_file_ids: set[int] = set(seed_file_ids)

    # Safety cap: don't traverse more than 60 unique files (handles widely-imported
    # utility modules that could fan out to hundreds of dependents).
    MAX_EXPAND_FILES = 60

    frontier: set[int] = set(seed_file_ids)

    for _hop in range(depth):
        if not frontier:
            break
        next_frontier: set[int] = set()
        for file_id in frontier:
            if direction in ("both", "forward"):
                for fid in db.get_file_imports_resolved(file_id):
                    if fid not in visited_file_ids:
                        visited_file_ids.add(fid)
                        next_frontier.add(fid)
                        if len(visited_file_ids) >= MAX_EXPAND_FILES:
                            break
            if direction in ("both", "reverse"):
                for fid in db.get_files_importing(file_id):
                    if fid not in visited_file_ids:
                        visited_file_ids.add(fid)
                        next_frontier.add(fid)
                        if len(visited_file_ids) >= MAX_EXPAND_FILES:
                            break
            if len(visited_file_ids) >= MAX_EXPAND_FILES:
                break
        frontier = next_frontier

    # ── Collect top symbols from all discovered (non-seed) files ─────────────
    seen_symbol_ids: set[int] = {r.chunk.symbol_id for r in results}
    extra: list[SearchResult] = []

    for file_id in visited_file_ids - seed_file_ids:
        for symbol_id in db.get_top_symbols_for_file(file_id, limit=3):
            if symbol_id in seen_symbol_ids:
                continue
            seen_symbol_ids.add(symbol_id)

            sym_file = db.get_symbol_with_file(symbol_id)
            if sym_file is None:
                continue
            symbol, file = sym_file

            chunk = db.get_first_chunk_for_symbol(symbol_id)
            if chunk is None:
                chunk = Chunk(
                    symbol_id=symbol_id,
                    chunk_text=symbol.body[:2000],
                    token_count=0,
                )

            extra.append(SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=base_score * score_discount,
                rank=len(extra) + 1,
                source="import_expansion",
            ))

            if len(extra) >= max_extra:
                return extra

    return extra


def expand_with_type_edges(
    db: Database,
    results: list[SearchResult],
    max_extra: int = 5,
) -> list[SearchResult]:
    """
    Expand result set by following type hierarchy edges (extends/implements/trait_impl).

    For each retrieved class/struct:
      - Pull parent types (what it extends/implements) — useful for understanding the contract
      - Pull sibling implementors (other classes that extend the same type) — less often, limit 2

    Score is discounted more than call-graph since type relationships are structural
    context rather than direct code paths.
    """
    if not results:
        return []

    seen_ids: set[int] = {r.chunk.symbol_id for r in results}
    extra: list[SearchResult] = []
    base_score = results[0].score if results else 0.5
    score_discount = 0.2

    for r in results:
        symbol_id = r.chunk.symbol_id

        # Parents: types this symbol inherits/implements
        parent_ids = db.get_type_parents(symbol_id)
        # Children: types that implement/extend this symbol (cap at 15 to cover
        # "what extends X?" queries that expect many subclasses)
        child_ids = db.get_type_children(symbol_id)[:15]

        for neighbor_id in parent_ids + child_ids:
            if neighbor_id in seen_ids:
                continue
            seen_ids.add(neighbor_id)

            sym_file = db.get_symbol_with_file(neighbor_id)
            if sym_file is None:
                continue
            symbol, file = sym_file

            chunk = db.get_first_chunk_for_symbol(neighbor_id)
            if chunk is None:
                chunk = Chunk(
                    symbol_id=neighbor_id,
                    chunk_text=symbol.body[:2000],
                    token_count=0,
                )

            extra.append(SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=base_score * score_discount,
                rank=len(extra) + 1,
                source="type_expansion",
            ))

            if len(extra) >= max_extra:
                return extra

    return extra


def seed_from_import_paths(
    db: Database,
    patterns: list[str],
    max_extra: int = 30,
) -> list[SearchResult]:
    """
    Seed retrieval from the imports table by matching raw import path strings.

    Handles path aliases (@shared, @core, etc.) that never appear in symbol
    bodies but DO appear in imports.imported_from. Used as a first pass for
    blast_radius queries so that "what breaks if @shared changes" finds the
    actual importing files rather than Markdown docs that mention the alias.

    For each matched file_id, surfaces the top symbols from that file.
    """
    if not patterns or max_extra <= 0:
        return []

    # Collect all file_ids that import any of the patterns
    matched_file_ids: set[int] = set()
    for pattern in patterns:
        matched_file_ids.update(db.get_files_by_import_path(pattern))

    if not matched_file_ids:
        return []

    seen_symbol_ids: set[int] = set()
    extra: list[SearchResult] = []

    for file_id in sorted(matched_file_ids):
        for symbol_id in db.get_top_symbols_for_file(file_id, limit=3):
            if symbol_id in seen_symbol_ids:
                continue
            seen_symbol_ids.add(symbol_id)

            sym_file = db.get_symbol_with_file(symbol_id)
            if sym_file is None:
                continue
            symbol, file = sym_file

            chunk = db.get_first_chunk_for_symbol(symbol_id)
            if chunk is None:
                chunk = Chunk(
                    symbol_id=symbol_id,
                    chunk_text=symbol.body[:2000],
                    token_count=0,
                )

            extra.append(SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=0.3,   # below real retrieval scores; reranker will re-sort
                rank=len(extra) + 1,
                source="import_path_seed",
            ))

            if len(extra) >= max_extra:
                return extra

    return extra


def rank_by_pagerank(
    symbol_ids: list[int],
    db: Database,
) -> list[tuple[int, float]]:
    """
    Run PageRank on the call subgraph of the given symbols.
    Returns (symbol_id, pagerank_score) sorted descending.

    Falls back to uniform scores if networkx is not installed.
    Stolen from Aider's approach of using graph centrality to prioritize
    which symbols are most important for a limited context window.
    """
    try:
        import networkx as nx
    except ImportError:
        return [(sid, 1.0) for sid in symbol_ids]

    G: nx.DiGraph = nx.DiGraph()
    for symbol_id in symbol_ids:
        for callee in db.get_callees(symbol_id):
            G.add_edge(symbol_id, callee)
        for caller in db.get_callers(symbol_id):
            G.add_edge(caller, symbol_id)

    if not G.nodes:
        return [(sid, 1.0) for sid in symbol_ids]

    scores = nx.pagerank(G, alpha=0.85)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
