"""Graph-aware search: BFS over CodeGraph to surface structurally related symbols."""
from __future__ import annotations

import logging
from collections import deque

from trelix.core.models import Chunk, SearchResult
from trelix.graph.code_graph import CodeGraph
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.search")


def graph_search(
    db: Database,
    cg: CodeGraph,
    query_symbol_ids: list[int],
    depth: int = 2,
    max_results: int = 15,
) -> list[SearchResult]:
    """
    BFS over CodeGraph starting from query_symbol_ids.

    Returns hydrated SearchResult objects for all reachable neighbors
    within `depth` hops, scored by hop distance (closer = higher score).
    Source label: "graph_search".

    Returns [] when query_symbol_ids is empty.
    """
    if not query_symbol_ids:
        return []

    seen: set[int] = set(query_symbol_ids)
    # Queue: (symbol_id, hop_distance)
    queue: deque[tuple[int, int]] = deque()
    for sid in query_symbol_ids:
        for neighbor in cg.neighbors(sid):
            if neighbor not in seen:
                queue.append((neighbor, 1))
                seen.add(neighbor)

    candidates: list[tuple[int, int]] = []  # (symbol_id, hop)

    while queue and len(candidates) < max_results * 3:
        symbol_id, hop = queue.popleft()
        candidates.append((symbol_id, hop))
        if hop < depth:
            for neighbor in cg.neighbors(symbol_id):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, hop + 1))

    # Score: 0.5^hop (closer hops score higher)
    results: list[SearchResult] = []
    for symbol_id, hop in candidates[:max_results]:
        sym_file = db.get_symbol_with_file(symbol_id)
        if sym_file is None:
            continue
        symbol, file = sym_file
        chunk = db.get_first_chunk_for_symbol(symbol_id)
        if chunk is None:
            chunk = Chunk(
                symbol_id=symbol_id,
                chunk_text=symbol.body[:512],
                token_count=len(symbol.body.split()),
                id=None,
            )
        score = 0.5 ** hop
        results.append(
            SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=score,
                rank=len(results) + 1,
                source="graph_search",
            )
        )

    return results


def get_community_context(cg: CodeGraph, symbol_id: int) -> list[int]:
    """Return all symbol IDs in the same community as symbol_id."""
    if symbol_id not in cg.nx:
        return [symbol_id]
    target_community = cg.nx.nodes[symbol_id].get("community")
    if target_community is None:
        return [symbol_id]
    return [
        nid
        for nid, attrs in cg.nx.nodes(data=True)
        if attrs.get("community") == target_community
    ]
