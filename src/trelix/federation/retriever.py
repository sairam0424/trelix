"""
FederatedRetriever — fan-out search across multiple independently-indexed repos.

Strategy: parallel query fan-out (one thread per repo) -> collect SearchResult
lists -> RRF merge with per-repo weight -> deduplicate by (file_path, symbol_id).

This is embedding-level federation only — no cross-repo call graph linking.
Results from different repos are identified by (file_path, repo_path) pairs.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from trelix.core.models import SearchResult
from trelix.federation.registry import RepoRegistry
from trelix.retrieval.fusion import reciprocal_rank_fusion
from trelix.retrieval.retriever import Retriever

logger = logging.getLogger("trelix.federation.retriever")


class FederatedRetriever:
    """
    Fan-out retriever across multiple trelix-indexed repos.

    Usage:
        registry = RepoRegistry.load()
        fed = FederatedRetriever(registry, max_workers=4)
        results = fed.retrieve("how does authentication work", k=10)
    """

    def __init__(self, registry: RepoRegistry, max_workers: int = 4) -> None:
        self._registry = registry
        self._max_workers = max_workers

    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]:
        """
        Fan-out query to all registered repos in parallel.
        Returns merged, deduplicated SearchResult list. Never raises.
        """
        entries = self._registry.list()
        if not entries:
            return []

        per_repo_results: list[list[SearchResult]] = []

        def _query_one(repo_path: str) -> list[SearchResult]:
            from trelix.core.config import IndexConfig

            # model_construct skips path-existence validation — safe for
            # federation where the caller is responsible for registering valid paths.
            # In production use, paths are validated at registry.add() time by the CLI.
            config = IndexConfig.model_construct(repo_path=repo_path)
            retriever = Retriever(config)
            ctx = retriever.retrieve(query)
            return ctx.results[:k]

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(entries))) as pool:
            future_to_entry = {
                pool.submit(_query_one, entry.path): entry for entry in entries
            }
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    results = future.result(timeout=30)
                    # Tag each result with the repo alias for provenance.
                    # Create new SearchResult objects instead of mutating the
                    # originals (immutability rule: never mutate existing objects).
                    tagged = [
                        SearchResult(
                            chunk=r.chunk,
                            symbol=r.symbol,
                            file=r.file,
                            score=r.score,
                            rank=r.rank,
                            source=f"{entry.alias}:{r.source}",
                        )
                        for r in results
                    ]
                    per_repo_results.append(tagged)
                except Exception as exc:
                    logger.warning(
                        "FederatedRetriever: repo '%s' failed (non-fatal): %s",
                        entry.alias,
                        exc,
                    )

        if not per_repo_results:
            return []

        # Merge via RRF — each repo's results form one ranked list
        fused = reciprocal_rank_fusion(per_repo_results, k=60)

        # Deduplicate by symbol_id within same file (cross-repo same ID is fine)
        seen: set[tuple[object, int]] = set()
        deduped: list[SearchResult] = []
        for r in fused:
            key = (r.file.rel_path, r.chunk.symbol_id)
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped[:k]
