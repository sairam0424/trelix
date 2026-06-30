"""
Retriever: orchestrates hybrid search at query time.

Flow:
  1. QueryPlan is either supplied externally or produced by the internal
     QueryPlanner (which calls the LLM when API keys are present, and falls
     back to default_plan() when they are not).
  2. Intent-based routing dispatches to the right retrieval path.
  3. Standard path: three retrieval legs -> RRF fusion -> graph expansion
     -> rerank -> assemble.
     File/project paths: DB-direct lookup -> assemble (no fusion/rerank overhead).

Debug tracing: every query writes a structured JSON file to .trelix/debug/
relative to the project root configured in IndexConfig.repo_path.
Each file captures all pipeline stages: plan -> legs -> fusion -> expansion
-> rerank -> assembly.
To disable: comment out the self._trace(...) calls in this file.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from trelix.core.config import IndexConfig
from trelix.core.models import Chunk, RetrievedContext, SearchResult
from trelix.embedder.base import BaseEmbedder, make_embedder
from trelix.store.db import Database
from trelix.store.vector import BaseVectorStore, make_vector_store

from .bm25 import bm25_search
from .fusion import reciprocal_rank_fusion
from .graph import (
    expand_with_call_graph,
    expand_with_imports,
    expand_with_type_edges,
    seed_from_import_paths,
)
from .grep_search import grep_search
from .planner.agent import QueryPlanner
from .planner.models import (
    IntentType,
    QueryPlan,
    RetrievalStrategy,
    RoutingTier,
    SubQuery,
    default_plan,
)
from .reranker import rerank

# Thread-local storage so parallel eval workers don't mix each other's traces
_trace_local = threading.local()

logger = logging.getLogger("trelix.retrieval")


class Retriever:
    """
    Top-level retriever. Call ``retrieve(query)`` to get context for a query.

    Usage::

        retriever = Retriever(config)
        context = retriever.retrieve("how does authentication work?")
        print(context.context_text)

    The QueryPlanner is instantiated internally and makes one LLM call when
    API keys are present. When provider=local (no keys), it falls back to
    default_plan() automatically — zero LLM calls.
    """

    def __init__(self, config: IndexConfig) -> None:
        self.config = config
        self.db = Database(config.db_path_absolute)
        raw_embedder: BaseEmbedder = make_embedder(config.embedder)
        # Wrap with LRU query cache when enabled (default: 256 entries).
        # embed_query() hits are returned in <1ms; embed() passthrough unchanged.
        if config.retrieval.query_cache_size > 0:
            from trelix.embedder.cache import CachingEmbedder

            self.embedder: BaseEmbedder = CachingEmbedder(
                raw_embedder, max_size=config.retrieval.query_cache_size
            )
        else:
            self.embedder = raw_embedder
        self.vector_store: BaseVectorStore = make_vector_store(
            config=config,
            dimension=self.embedder.dimension,
        )
        # Instantiate the LLM query planner. Falls back gracefully to
        # default_plan() when no API key is set (provider=local).
        self._planner = QueryPlanner(config.embedder)
        # Wrap with LRU plan cache when enabled (default: 128 entries).
        # plan() hits are returned in <1ms; cold misses delegate to the LLM unchanged.
        if config.retrieval.plan_cache_size > 0:
            from trelix.retrieval.plan_cache import CachingPlanner

            self._planner = CachingPlanner(  # type: ignore[assignment]
                self._planner, max_size=config.retrieval.plan_cache_size
            )

        # Debug output dir: <repo_root>/.trelix/debug/
        self._debug_dir = Path(config.repo_path) / ".trelix" / "debug"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, plan: QueryPlan | None = None) -> RetrievedContext:
        """
        Retrieve context for a query.

        Args:
            query: The raw user query string.
            plan:  Pre-formed QueryPlan (e.g. from an external planner or test).
                   When None, the internal QueryPlanner is invoked, which in turn
                   calls the LLM (if available) or falls back to default_plan().
        """
        t_start = time.perf_counter()
        plan_source = "external" if plan is not None else "planner"
        logger.info("Retrieval start: query=%r plan_source=%s", query, plan_source)

        # Initialise a fresh per-query trace in thread-local storage.
        _trace_local.data = {
            "query": query,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "plan_source": plan_source,
        }

        if plan is None:
            plan = self._planner.plan(query)

        # -- Trace: planner decision --
        self._trace(
            "planner",
            {
                "intent": plan.intent.value,
                "execution_mode": plan.execution_mode,
                "sub_queries": [
                    {
                        "semantic_query": sq.semantic_query,
                        "hyde_snippet": sq.hyde_snippet[:120] if sq.hyde_snippet else "",
                        "bm25_tokens": sq.bm25_tokens,
                        "grep_hints": sq.grep_hints,
                        "file_hints": sq.file_hints,
                        "depends_on": sq.depends_on,
                    }
                    for sq in plan.sub_queries
                ],
            },
        )

        context = self._execute_plan(plan)
        context.elapsed_seconds = round(time.perf_counter() - t_start, 3)

        # -- Trace: final assembly output --
        self._trace(
            "assembly",
            {
                "intent": plan.intent.value,
                "results_count": len(context.results),
                "tokens_used": context.total_tokens,
                "token_budget": self.config.retrieval.context_token_budget,
                "budget_pct": round(
                    context.total_tokens / max(1, self.config.retrieval.context_token_budget) * 100,
                    1,
                ),
                "sources": context.retrieval_sources,
                "top5_symbols": [
                    {
                        "name": r.symbol.name,
                        "kind": r.symbol.kind,
                        "file": r.file.rel_path,
                        "score": round(r.score, 4),
                    }
                    for r in context.results[:5]
                ],
                "elapsed_s": context.elapsed_seconds,
            },
        )
        self._flush_trace()

        logger.info(
            "Retrieval complete: intent=%s results=%d tokens=%d (%.0f%%) sources=%s elapsed=%.3fs",
            plan.intent.value,
            len(context.results),
            context.total_tokens,
            context.total_tokens / max(1, self.config.retrieval.context_token_budget) * 100,
            context.retrieval_sources,
            context.elapsed_seconds,
        )
        return context

    # ------------------------------------------------------------------
    # Intent router
    # ------------------------------------------------------------------

    def _execute_plan(self, plan: QueryPlan) -> RetrievedContext:
        # Tier 1 DIRECT: skip all retrieval legs — answer from project overview only.
        # The router has already classified this as a trivial factual query.
        if getattr(plan, "routing_tier", None) == RoutingTier.TIER_1_DIRECT:
            logger.info("Tier 1 DIRECT path: skipping retrieval legs for query=%r", plan.raw_query)
            return self._retrieve_project_overview(plan)

        if plan.intent == IntentType.FILE_OVERVIEW:
            return self._retrieve_file_overview(plan)
        if plan.intent == IntentType.PROJECT_OVERVIEW:
            return self._retrieve_project_overview(plan)
        if plan.intent == IntentType.CONFIG_LOOKUP:
            return self._retrieve_config(plan)
        # SYMBOL_LOOKUP, FEATURE_FLOW, COMPARISON, DEPENDENCY_MAP, BLAST_RADIUS
        # Tier 3 MULTI also lands here — execution_mode="parallel" runs all sub-queries.
        return self._retrieve_standard(plan)

    # ------------------------------------------------------------------
    # Standard pipeline (symbol_lookup / feature_flow / comparison /
    #                    dependency_map / blast_radius)
    # ------------------------------------------------------------------

    def _retrieve_standard(self, plan: QueryPlan) -> RetrievedContext:
        cfg = self.config.retrieval
        strategy = plan.strategy

        # Run sub-queries in parallel when the planner says they're independent.
        if plan.execution_mode == "parallel" and len(plan.sub_queries) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as pool:
                futures = [
                    pool.submit(self._run_subquery_legs, sq, strategy) for sq in plan.sub_queries
                ]
                leg_results_list = [f.result() for f in futures]
        else:
            leg_results_list = [self._run_subquery_legs(sq, strategy) for sq in plan.sub_queries]

        # Merge per-leg results across all sub-queries for RRF
        vector_results: list[SearchResult] = [r for lr in leg_results_list for r in lr["vector"]]
        bm25_results: list[SearchResult] = [r for lr in leg_results_list for r in lr["bm25"]]
        grep_results: list[SearchResult] = [r for lr in leg_results_list for r in lr["grep"]]

        # 5th leg: file-summary search (RAPTOR-style, off by default)
        summary_results: list[SearchResult] = []
        if cfg.file_summary_leg_enabled and plan.sub_queries:
            embed_text = (
                plan.sub_queries[0].hyde_snippet
                if plan.sub_queries[0].hyde_snippet.strip()
                else plan.sub_queries[0].semantic_query
            )
            query_embedding: list[float] = self.embedder.embed_query(embed_text)
            summary_results = self._summary_search(query_embedding, k=cfg.top_k_file_summary)

        logger.info(
            "Pre-fusion leg sizes: vector=%d bm25=%d grep=%d summary=%d",
            len(vector_results),
            len(bm25_results),
            len(grep_results),
            len(summary_results),
        )

        # -- Trace: per-leg results --
        self._trace(
            "retrieval_legs",
            {
                "intent": plan.intent.value,
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "grep_count": len(grep_results),
                "summary_count": len(summary_results),
                "top_vector": [
                    {"name": r.symbol.name, "file": r.file.rel_path, "score": round(r.score, 4)}
                    for r in vector_results[:5]
                ],
                "top_bm25": [
                    {"name": r.symbol.name, "file": r.file.rel_path, "score": round(r.score, 4)}
                    for r in bm25_results[:5]
                ],
                "top_grep": [
                    {"name": r.symbol.name, "file": r.file.rel_path, "score": round(r.score, 4)}
                    for r in grep_results[:5]
                ],
                "top_summary": [
                    {"name": r.symbol.name, "file": r.file.rel_path, "score": round(r.score, 4)}
                    for r in summary_results[:5]
                ],
            },
        )

        _weights = cfg.file_type_weights if cfg.file_type_weighting_enabled else None
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results, grep_results, summary_results],
            k=cfg.rrf_k,
            weights=_weights,
        )

        # -- Trace: post-fusion ranking --
        self._trace(
            "post_fusion",
            {
                "total": len(fused),
                "top5": [
                    {
                        "name": r.symbol.name,
                        "file": r.file.rel_path,
                        "rrf_score": round(r.score, 6),
                        "source": r.source,
                    }
                    for r in fused[:5]
                ],
            },
        )

        # Graph expansion — all parameters driven by intent strategy
        top = fused[: cfg.graph_expansion_max_symbols]
        call_expanded = expand_with_call_graph(
            self.db, top, depth=strategy.expand_depth, max_extra=cfg.graph_expansion_max_symbols
        )
        import_expanded = expand_with_imports(
            self.db,
            top,
            max_extra=strategy.import_max_extra,
            depth=strategy.import_depth,
            direction=strategy.import_direction,
        )
        type_expanded = expand_with_type_edges(self.db, top, max_extra=15)

        # For blast_radius: also seed from raw import path strings (@aliases)
        import_path_seeded: list[SearchResult] = []
        if plan.intent == IntentType.BLAST_RADIUS:
            patterns = [h for sq in plan.sub_queries for h in sq.grep_hints if h.startswith("@")]
            if patterns:
                import_path_seeded = seed_from_import_paths(
                    self.db, patterns, max_extra=strategy.import_max_extra
                )

        # Graph search leg (optional 4th retrieval leg — CodeGraph BFS)
        graph_search_results: list[SearchResult] = []
        if cfg.graph_search_enabled:
            try:
                from trelix.graph.code_graph import CodeGraph
                from trelix.graph.search import graph_search

                cg = CodeGraph(self.db)
                seed_ids = [r.chunk.symbol_id for r in fused[:10] if r.chunk.symbol_id]
                graph_search_results = graph_search(
                    db=self.db,
                    cg=cg,
                    query_symbol_ids=seed_ids,
                    depth=cfg.graph_search_depth,
                    max_results=cfg.graph_search_max_results,
                )
            except Exception as exc:
                logger.warning("Graph search leg failed (non-fatal): %s", exc)

        candidates = self._dedup(
            fused
            + call_expanded
            + import_expanded
            + type_expanded
            + import_path_seeded
            + graph_search_results
        )

        logger.info(
            "Post-expansion candidates: fused=%d call_exp=%d import_exp=%d "
            "type_exp=%d path_seed=%d graph_search=%d total=%d",
            len(fused),
            len(call_expanded),
            len(import_expanded),
            len(type_expanded),
            len(import_path_seeded),
            len(graph_search_results),
            len(candidates),
        )

        # -- Trace: graph expansion --
        self._trace(
            "expansion",
            {
                "call_expanded": len(call_expanded),
                "import_expanded": len(import_expanded),
                "type_expanded": len(type_expanded),
                "import_path_seeded": len(import_path_seeded),
                "total_candidates": len(candidates),
                "import_strategy": {
                    "depth": strategy.import_depth,
                    "max_extra": strategy.import_max_extra,
                    "direction": strategy.import_direction,
                },
                "top_import_files": list({r.file.rel_path for r in import_expanded}),
                "import_path_seed_files": list({r.file.rel_path for r in import_path_seeded})[:10],
            },
        )

        # Rerank — skipped when strategy says exact ordering is already correct.
        if cfg.rerank and candidates and not strategy.skip_reranker:
            candidates = rerank(
                query=plan.raw_query,
                results=candidates,
                config=cfg,
                top_n=strategy.rerank_top_n,
            )

            # -- Trace: post-rerank ordering --
            self._trace(
                "post_rerank",
                {
                    "total": len(candidates),
                    "top5": [
                        {"name": r.symbol.name, "file": r.file.rel_path, "score": round(r.score, 4)}
                        for r in candidates[:5]
                    ],
                },
            )

        return self._assemble(
            plan.raw_query,
            candidates,
            intent=plan.intent.value,
            assembly_mode=plan.strategy.assembly_mode,
        )

    # ------------------------------------------------------------------
    # File overview (file_overview intent)
    # ------------------------------------------------------------------

    def _retrieve_file_overview(self, plan: QueryPlan) -> RetrievedContext:
        """
        Bypass retrieval legs entirely. Find the file by name, fetch all its
        symbols in structural order, and let the context assembler apply the
        token budget.
        """
        file_hints = [h for sq in plan.sub_queries for h in sq.file_hints]
        # Also treat grep_hints that look like filenames (contain a dot) as file hints
        for sq in plan.sub_queries:
            for hint in sq.grep_hints:
                if "." in hint and hint not in file_hints:
                    file_hints.append(hint)

        results: list[SearchResult] = []
        visited_file_ids: set[int] = set()

        for hint in file_hints:
            for file_id in self.db.find_file_by_path_fragment(hint)[:2]:
                if file_id in visited_file_ids:
                    continue
                visited_file_ids.add(file_id)
                for rank, sid in enumerate(self.db.get_all_symbols_for_file(file_id), start=1):
                    r = self.hydrate_symbol(
                        sid, score=1.0 - rank * 0.001, rank=rank, source="file_direct"
                    )
                    if r:
                        results.append(r)

        self._trace(
            "file_overview",
            {
                "file_hints": file_hints,
                "files_matched": len(visited_file_ids),
                "symbols_fetched": len(results),
            },
        )

        if not results:
            logger.info(
                "file_overview: no file matched hints %r — falling back to standard", file_hints
            )
            return self._retrieve_standard(default_plan(plan.raw_query))

        return self._assemble(plan.raw_query, self._dedup(results), intent=plan.intent.value)

    # ------------------------------------------------------------------
    # Project overview (project_overview intent)
    # ------------------------------------------------------------------

    def _retrieve_project_overview(self, plan: QueryPlan) -> RetrievedContext:
        """
        Fetch README, project manifests, and module-level summary symbols.
        No retrieval legs needed — these files answer "what does this project do?" directly.
        """
        symbol_ids = self.db.get_module_and_readme_symbols(limit=40)
        results: list[SearchResult] = []
        for rank, sid in enumerate(symbol_ids, start=1):
            r = self.hydrate_symbol(sid, score=1.0 - rank * 0.001, rank=rank, source="file_direct")
            if r:
                results.append(r)

        self._trace(
            "project_overview",
            {
                "symbol_ids_from_db": len(symbol_ids),
                "hydrated": len(results),
                "files": list({r.file.rel_path for r in results}),
            },
        )

        if not results:
            logger.info("project_overview: no overview symbols found — falling back to standard")
            return self._retrieve_standard(default_plan(plan.raw_query))

        return self._assemble(plan.raw_query, results, intent=plan.intent.value)

    # ------------------------------------------------------------------
    # Config lookup (config_lookup intent)
    # ------------------------------------------------------------------

    def _retrieve_config(self, plan: QueryPlan) -> RetrievedContext:
        """
        Try file_direct for known config filenames; fall back to standard retrieval.
        """
        file_hints = [h for sq in plan.sub_queries for h in sq.file_hints + sq.grep_hints]
        config_extensions = {".json", ".yaml", ".yml", ".toml", ".js", ".ts"}

        results: list[SearchResult] = []
        visited: set[int] = set()

        for hint in file_hints:
            if any(hint.endswith(ext) for ext in config_extensions) or "config" in hint.lower():
                for file_id in self.db.find_file_by_path_fragment(hint)[:2]:
                    if file_id in visited:
                        continue
                    visited.add(file_id)
                    for rank, sid in enumerate(self.db.get_all_symbols_for_file(file_id), start=1):
                        r = self.hydrate_symbol(sid, score=1.0, rank=rank, source="file_direct")
                        if r:
                            results.append(r)

        self._trace(
            "config_lookup",
            {
                "file_hints": file_hints,
                "files_matched": len(visited),
                "symbols_fetched": len(results),
            },
        )

        if not results:
            logger.info("config_lookup: no config file matched — falling back to standard")
            return self._retrieve_standard(default_plan(plan.raw_query))

        return self._assemble(plan.raw_query, self._dedup(results), intent=plan.intent.value)

    # ------------------------------------------------------------------
    # Sub-query execution — one unit of retrieval per sub-query
    # ------------------------------------------------------------------

    def _run_subquery_legs(
        self,
        sq: SubQuery,
        strategy: RetrievalStrategy,
    ) -> dict[str, list[SearchResult]]:
        """
        Run all retrieval legs for a single sub-query.
        Returns {"vector": [...], "bm25": [...], "grep": [...]} so callers
        can merge per-leg before RRF fusion.

        Safe to call from a ThreadPoolExecutor — reads only, no writes.
        """
        cfg = self.config.retrieval
        out: dict[str, list[SearchResult]] = {"vector": [], "bm25": [], "grep": []}

        if "vector" in strategy.legs:
            # HyDE: embed the hypothetical code snippet if the planner provided one.
            embed_text = sq.hyde_snippet if sq.hyde_snippet.strip() else sq.semantic_query
            embedding = self.embedder.embed_query(embed_text)
            out["vector"] = self._vector_search(embedding, k=cfg.top_k_vector)

        if "bm25" in strategy.legs:
            bm25_query = " ".join(sq.bm25_tokens) if sq.bm25_tokens else sq.semantic_query
            out["bm25"] = bm25_search(self.db, bm25_query, k=cfg.top_k_bm25)

        if "grep" in strategy.legs:
            hints = sq.grep_hints if sq.grep_hints else [sq.semantic_query]
            for hint in hints:
                out["grep"].extend(grep_search(self.db, hint, k=cfg.top_k_grep))

        return out

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _summary_search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        """Search file-summary embeddings (5th retrieval leg).

        Returns SearchResult objects where the symbol is the first symbol in the file
        (used as a representative for the file-level summary context).
        Returns empty list when no summaries are indexed or file_summary_leg_enabled=False.
        """
        results: list[SearchResult] = []
        try:
            pairs = self.vector_store.search_file_summaries(query_embedding, k=k)
            for file_id, score in pairs:
                file_obj = self.db.get_file_by_id(file_id)
                if file_obj is None:
                    continue
                summary_text = self.db.get_file_summary(file_id)
                if not summary_text:
                    continue
                # Build a synthetic Chunk representing the file-level summary
                synthetic_chunk = Chunk(
                    id=-(file_id),  # negative = summary sentinel
                    symbol_id=0,
                    chunk_text=summary_text,
                    token_count=len(summary_text.split()),
                )
                # Pick the first symbol in the file as the representative symbol
                symbols = self.db.get_symbols_for_file(file_id)
                if not symbols:
                    continue
                rep_symbol = min(symbols, key=lambda s: s.line_start)
                results.append(
                    SearchResult(
                        chunk=synthetic_chunk,
                        symbol=rep_symbol,
                        file=file_obj,
                        score=score,
                        rank=0,
                        source="file_summary",
                    )
                )
        except Exception as exc:
            logger.warning("File summary leg failed (non-fatal): %s", exc)
        return results

    def _vector_search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        raw = self.vector_store.search(query_embedding, k=k)
        results: list[SearchResult] = []
        for rank, (chunk_id, distance) in enumerate(raw, start=1):
            score = max(0.0, 1.0 - distance)
            result = self._hydrate_chunk(chunk_id, score=score, rank=rank, source="vector")
            if result:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Hydration
    # ------------------------------------------------------------------

    def _hydrate_chunk(
        self,
        chunk_id: int,
        score: float,
        rank: int,
        source: str,
    ) -> SearchResult | None:
        row = self.db.get_chunk_with_context(chunk_id)
        if row is None:
            return None
        chunk, symbol, file = row
        return SearchResult(
            chunk=chunk, symbol=symbol, file=file, score=score, rank=rank, source=source
        )

    def hydrate_symbol(
        self,
        symbol_id: int,
        score: float,
        rank: int,
        source: str,
    ) -> SearchResult | None:
        sym_file = self.db.get_symbol_with_file(symbol_id)
        if sym_file is None:
            return None
        symbol, file = sym_file

        chunk = self.db.get_first_chunk_for_symbol(symbol_id)
        if chunk is None:
            chunk = Chunk(
                symbol_id=symbol_id,
                chunk_text=symbol.body[:2000],
                token_count=0,
            )

        return SearchResult(
            chunk=chunk, symbol=symbol, file=file, score=score, rank=rank, source=source
        )

    # ------------------------------------------------------------------
    # Public graph API
    # ------------------------------------------------------------------

    def _hydrate_symbol_id(self, symbol_id: int, source: str) -> SearchResult | None:
        """
        Hydrate a raw symbol_id into a SearchResult.
        Returns None when the symbol is no longer in the db (stale index).
        Score is fixed at 1.0 — graph queries are exact, not ranked.
        """
        sym_file = self.db.get_symbol_with_file(symbol_id)
        if sym_file is None:
            return None
        symbol, file = sym_file
        chunk = self.db.get_first_chunk_for_symbol(symbol_id)
        if chunk is None:
            chunk = Chunk(
                symbol_id=symbol_id,
                chunk_text=symbol.body[:2000],
                token_count=0,
            )
        return SearchResult(
            chunk=chunk,
            symbol=symbol,
            file=file,
            score=1.0,
            rank=0,
            source=source,
        )

    def get_callers(self, symbol_name: str) -> list[SearchResult]:
        """
        Return the symbols that call ``symbol_name`` (1-hop incoming call edges).

        ``symbol_name`` may be a bare name (``"retrieve"``) or a qualified name
        (``"Retriever.retrieve"``).  All matching symbols are tried; results are
        deduplicated by symbol id and sorted by file path + line for determinism.

        Returns an empty list when the symbol is not found or has no callers.
        """
        symbols = self.db.get_symbol_by_name(symbol_name)
        if not symbols:
            return []
        caller_ids: set[int] = set()
        for sym in symbols:
            if sym.id is not None:
                caller_ids.update(self.db.get_callers(sym.id))
        results: list[SearchResult] = []
        for cid in caller_ids:
            r = self._hydrate_symbol_id(cid, "graph_callers")
            if r is not None:
                results.append(r)
        results.sort(key=lambda r: (r.file.rel_path, r.symbol.line_start))
        for i, r in enumerate(results, start=1):
            r.rank = i
        return results

    def get_callees(self, symbol_name: str) -> list[SearchResult]:
        """
        Return the symbols that ``symbol_name`` calls (1-hop outgoing call edges,
        resolved internal calls only — external/stdlib calls are excluded).

        Same name resolution and deduplication rules as ``get_callers``.
        """
        symbols = self.db.get_symbol_by_name(symbol_name)
        if not symbols:
            return []
        callee_ids: set[int] = set()
        for sym in symbols:
            if sym.id is not None:
                callee_ids.update(self.db.get_callees(sym.id))
        results: list[SearchResult] = []
        for cid in callee_ids:
            r = self._hydrate_symbol_id(cid, "graph_callees")
            if r is not None:
                results.append(r)
        results.sort(key=lambda r: (r.file.rel_path, r.symbol.line_start))
        for i, r in enumerate(results, start=1):
            r.rank = i
        return results

    def get_importers(self, module_path: str) -> list[SearchResult]:
        """
        Return the top symbol from each file that imports ``module_path``.

        ``module_path`` is matched against ``files.rel_path`` by suffix.
        For each importing file, only the first symbol (lowest line_start) is
        returned.

        Returns an empty list when the module is not indexed or has no importers.
        """
        file_id = self.db.get_file_by_rel_path_suffix(module_path)
        if file_id is None:
            return []
        importer_file_ids = self.db.get_files_importing(file_id)
        results: list[SearchResult] = []
        for fid in importer_file_ids:
            syms = self.db.get_symbols_for_file(fid)
            if not syms:
                continue
            first_sym = min(syms, key=lambda s: s.line_start)
            if first_sym.id is None:
                continue
            r = self._hydrate_symbol_id(first_sym.id, "graph_importers")
            if r is not None:
                results.append(r)
        results.sort(key=lambda r: r.file.rel_path)
        for i, r in enumerate(results, start=1):
            r.rank = i
        return results

    # ------------------------------------------------------------------
    # Dedup + assemble
    # ------------------------------------------------------------------

    def _dedup(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate symbols, keeping highest score."""
        seen: dict[int, SearchResult] = {}
        for r in results:
            sid = r.chunk.symbol_id
            if sid not in seen or r.score > seen[sid].score:
                seen[sid] = r
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    def _assemble(
        self,
        query: str,
        results: list[SearchResult],
        intent: str | None = None,
        assembly_mode: str = "greedy",
    ) -> RetrievedContext:
        from trelix.retrieval.assembler import ContextAssembler

        assembler = ContextAssembler(token_budget=self.config.retrieval.context_token_budget)
        return assembler.assemble(
            query=query,
            results=results,
            intent=intent,
            assembly_mode=assembly_mode,
        )

    # ------------------------------------------------------------------
    # Structured per-query trace
    # ------------------------------------------------------------------

    def _trace(self, section: str, data: dict[str, Any]) -> None:
        """Write a named section into the current query's in-memory trace."""
        try:
            _trace_local.data[section] = data
        except AttributeError:
            pass  # trace not initialised (called outside retrieve())

    def _flush_trace(self) -> None:
        """Write the accumulated trace to .trelix/debug/<timestamp>_<slug>.json."""
        try:
            trace = _trace_local.data
            ts = trace.get("ts", datetime.datetime.now().isoformat(timespec="seconds"))
            query = trace.get("query", "unknown")
            slug = "".join(c if c.isalnum() or c == " " else "_" for c in query[:60])
            slug = "_".join(slug.split())[:60]
            filename = f"{ts.replace(':', '-').replace('T', '_')}_{slug}.json"
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            (self._debug_dir / filename).write_text(
                json.dumps(trace, indent=2, ensure_ascii=False, default=str)
            )
        except Exception:
            pass  # never let debug tracing break retrieval
