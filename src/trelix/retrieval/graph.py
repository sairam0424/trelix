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
    personalization_enabled: bool = False,
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
    # candidates: (symbol_id, hop_distance, via_symbol_id, direction)
    # via_symbol_id is the frontier symbol this neighbor was discovered through;
    # direction is "callee" (neighbor is called BY via_symbol_id, i.e. found via
    # db.get_callees(via_symbol_id)) or "caller" (neighbor CALLS via_symbol_id,
    # found via db.get_callers(via_symbol_id)). Iterating callees and callers as
    # two separate loops (rather than concatenating them into one list before
    # iterating) preserves the exact same discovery order as the prior
    # `db.get_callees(symbol_id) + db.get_callers(symbol_id)` combined list —
    # `for x in a + b` and `for x in a: ...; for x in b: ...` visit elements in
    # the same order — while letting each loop tag its own direction.
    candidates: list[tuple[int, int, int, str]] = []
    frontier = [r.chunk.symbol_id for r in results]

    for hop in range(1, depth + 1):
        next_frontier: list[int] = []
        for symbol_id in frontier:
            for neighbor_id in db.get_callees(symbol_id):
                if neighbor_id in seen_ids:
                    continue
                seen_ids.add(neighbor_id)
                next_frontier.append(neighbor_id)
                candidates.append((neighbor_id, hop, symbol_id, "callee"))
            for neighbor_id in db.get_callers(symbol_id):
                if neighbor_id in seen_ids:
                    continue
                seen_ids.add(neighbor_id)
                next_frontier.append(neighbor_id)
                candidates.append((neighbor_id, hop, symbol_id, "caller"))
        frontier = next_frontier

    if not candidates:
        return []

    # Only apply PageRank re-sorting when the call graph is rich enough to matter.
    # On sparse graphs (few resolved callee_ids) PageRank scores are near-uniform
    # and the sort just shuffles BFS order, which hurts more than it helps.
    total_resolved = sum(
        1 for sid, _hop, _via, _dir in candidates if db.get_callees(sid) or db.get_callers(sid)
    )
    if total_resolved >= 3:
        all_ids = [r.chunk.symbol_id for r in results] + [c[0] for c in candidates]
        pr_scores = dict(rank_by_pagerank(all_ids, db, personalization_enabled))
        # Closer hops win; PageRank breaks ties within the same hop
        candidates.sort(key=lambda x: (x[1], -pr_scores.get(x[0], 0.0)))

    base_score = results[0].score if results else 0.5
    extra: list[SearchResult] = []

    for symbol_id, hop, via_symbol_id, direction in candidates[:max_extra]:
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

        extra.append(
            SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=base_score * (0.5**hop),
                rank=len(extra) + 1,
                source="graph_expansion",
                graph_context=_render_graph_context(db, via_symbol_id, direction, hop),
            )
        )

    return extra


def _render_graph_context(db: Database, via_symbol_id: int, direction: str, hop: int) -> str:
    """
    Human-readable, self-contained description of how a call-graph-expanded
    result relates to the frontier symbol it was discovered through.

    direction="callee" — this result is CALLED BY via_symbol_id (discovered
        via db.get_callees(via_symbol_id)).
    direction="caller" — this result CALLS via_symbol_id (discovered via
        db.get_callers(via_symbol_id)).

    Names the actual via-parent (the immediate frontier symbol one hop closer
    to the seed), not the original seed — so a 2+ hop result's context stays
    truthful about the path it was actually reached through.
    """
    via_sym_file = db.get_symbol_with_file(via_symbol_id)
    via_name = via_sym_file[0].qualified_name if via_sym_file else f"symbol#{via_symbol_id}"
    verb = "called by" if direction == "callee" else "calls"
    hop_word = "hop" if hop == 1 else "hops"
    return f"{verb} {via_name} ({hop} {hop_word})"


def expand_with_dataflow(
    db: Database,
    results: list[SearchResult],
    max_extra: int = 10,
) -> list[SearchResult]:
    """
    Expand result set with callees that a seed symbol's OWN local data flows into.

    CodeRAG-style dataflow leg. For each seed symbol, pulls its intra-procedural
    def-use spans (DataFlowExtractor / def_use_edges, via db.get_data_flows) and
    its resolved call sites WITH line numbers (db.get_call_edges) and keeps only
    the callees whose call site line falls inside a def-use span
    (min(def_line, use_line) <= call.line <= max(def_line, use_line)).

    That correlation is the new signal here: expand_with_call_graph already
    returns "every function this symbol calls" unconditionally; this returns
    the narrower "functions this symbol calls that a specific tracked variable
    is live across" — distinguishing call topology from actual data flow.

    Degrades to [] (never raises) when the seed symbol has no def_use_edges —
    either ParserConfig.dataflow_enabled was False at index time (the default),
    or the symbol genuinely has no local variables — or when none of its
    resolved callees' call sites land inside a span.
    """
    if not results:
        return []

    seen_ids: set[int] = {r.chunk.symbol_id for r in results}
    base_score = results[0].score if results else 0.5
    score_discount = 0.4

    # Collect matching callee_ids per seed, then interleave round-robin across
    # seeds before truncating -- an earlier version truncated (and returned)
    # as soon as a single seed's matches filled max_extra, which silently
    # starved every later seed's candidates whenever one early, higher-ranked
    # seed alone had more correlated calls than the budget. Round-robin gives
    # every seed a turn before any seed gets a second slot, instead of
    # exhausting the first seed's matches before later seeds are examined.
    per_seed_candidates: list[list[int]] = []

    for r in results:
        symbol_id = r.chunk.symbol_id

        flows = db.get_data_flows(symbol_id)
        if not flows:
            continue

        spans = [(min(e.def_line, e.use_line), max(e.def_line, e.use_line)) for e in flows]

        seed_candidates: list[int] = []
        for call in db.get_call_edges(symbol_id):
            if call.callee_id is None or call.callee_id in seen_ids:
                continue
            if not any(start <= call.line <= end for start, end in spans):
                continue
            seen_ids.add(call.callee_id)
            seed_candidates.append(call.callee_id)

        if seed_candidates:
            per_seed_candidates.append(seed_candidates)

    if not per_seed_candidates:
        return []

    candidates: list[int] = []
    max_per_seed = max(len(c) for c in per_seed_candidates)
    for i in range(max_per_seed):
        for seed_candidates in per_seed_candidates:
            if i < len(seed_candidates):
                candidates.append(seed_candidates[i])

    extra: list[SearchResult] = []

    for callee_id in candidates[:max_extra]:
        sym_file = db.get_symbol_with_file(callee_id)
        if sym_file is None:
            continue
        symbol, file = sym_file

        chunk = db.get_first_chunk_for_symbol(callee_id)
        if chunk is None:
            chunk = Chunk(
                symbol_id=callee_id,
                chunk_text=symbol.body[:2000],
                token_count=0,
            )

        extra.append(
            SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=base_score * score_discount,
                rank=len(extra) + 1,
                source="dataflow_expansion",
            )
        )

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

            extra.append(
                SearchResult(
                    chunk=chunk,
                    symbol=symbol,
                    file=file,
                    score=base_score * score_discount,
                    rank=len(extra) + 1,
                    source="import_expansion",
                )
            )

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

            extra.append(
                SearchResult(
                    chunk=chunk,
                    symbol=symbol,
                    file=file,
                    score=base_score * score_discount,
                    rank=len(extra) + 1,
                    source="type_expansion",
                )
            )

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

            extra.append(
                SearchResult(
                    chunk=chunk,
                    symbol=symbol,
                    file=file,
                    score=0.3,  # below real retrieval scores; reranker will re-sort
                    rank=len(extra) + 1,
                    source="import_path_seed",
                )
            )

            if len(extra) >= max_extra:
                return extra

    return extra


def rank_by_pagerank(
    symbol_ids: list[int],
    db: Database,
    personalization_enabled: bool = False,
) -> list[tuple[int, float]]:
    """
    Run PageRank on the call subgraph of the given symbols.
    Returns (symbol_id, pagerank_score) sorted descending.

    Falls back to uniform scores if networkx is not installed.
    Stolen from Aider's approach of using graph centrality to prioritize
    which symbols are most important for a limited context window.

    `personalization_enabled` (default False — zero behavior change unless
    a caller opts in via RetrievalConfig.pagerank_personalization_enabled):
    when True, replaces the uniform 1/n teleport vector with a Personalized
    PageRank teleport vector concentrated on nodes with a cross-source
    generic_edge in THIS subgraph — the classic topic-sensitive PageRank
    construction (uniform mass 1/|T| over seed set T; see Haveliwala 2003).
    A prior eval proved the graph-edge change alone shifts real rankings
    (95% of ticket-linked symbols changed PageRank position) — this makes
    the algorithm itself aware of which nodes that signal actually touched,
    instead of treating every node as equally likely to be "home base".
    """
    try:
        import networkx as nx
    except ImportError:
        return [(sid, 1.0) for sid in symbol_ids]

    G: nx.DiGraph = nx.DiGraph()
    cross_source_nodes: set[int] = set()
    for symbol_id in symbol_ids:
        for callee in db.get_callees(symbol_id):
            G.add_edge(symbol_id, callee)
        for caller in db.get_callers(symbol_id):
            G.add_edge(caller, symbol_id)
        # Cross-source edges (e.g. ticket/artifact references from the
        # git-log linker or ArtifactLinker) — a symbol referenced by many
        # tickets AND called by many other symbols gets a higher rank than
        # call-graph centrality alone would give it.
        #
        # Bidirectional, same reasoning as code_graph.py's GENERIC edge loop:
        # PageRank propagates via INCOMING edges, so a single symbol->ticket
        # edge would only raise the ticket's rank, not the symbol's.
        generic_targets = db.get_generic_edge_targets(symbol_id)
        for source_ref in generic_targets:
            G.add_edge(symbol_id, source_ref)
            G.add_edge(source_ref, symbol_id)
        if generic_targets:
            cross_source_nodes.add(symbol_id)

    if not G.nodes:
        return [(sid, 1.0) for sid in symbol_ids]

    personalization = None
    if personalization_enabled and cross_source_nodes:
        mass = 1.0 / len(cross_source_nodes)
        personalization = {node: mass for node in cross_source_nodes}

    scores = nx.pagerank(G, alpha=0.85, personalization=personalization)
    # Drop synthetic artifact nodes (source_ref strings) from the returned
    # ranking — callers expect (symbol_id: int, score) pairs only.
    symbol_scores = {sid: score for sid, score in scores.items() if isinstance(sid, int)}
    return sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
