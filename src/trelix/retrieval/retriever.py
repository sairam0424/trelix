"""
Retriever: orchestrates hybrid search at query time.

Flow:
  1. QueryPlan is either supplied externally or produced by the internal
     QueryPlanner (which calls the LLM when API keys are present, and falls
     back to default_plan() when they are not).
  2. Intent-based routing dispatches to the right retrieval path.
  3. Standard path: three retrieval legs -> RRF fusion -> graph expansion
     -> rerank -> assemble.
     File/project/config paths: DB-direct lookup -> assemble (no fusion/rerank
     overhead) — but see the breadth floor below: a direct lookup that resolves too
     few files AND too few symbols now ALSO runs the standard path and merges, with
     direct hits ordered first, instead of returning a thin result as complete.
     The `breadth_floor` trace section records the decision on every such query.

Debug tracing: every query writes a structured JSON file to .trelix/debug/
relative to the project root configured in IndexConfig.repo_path.
Each file captures all pipeline stages: plan -> legs -> fusion -> expansion
-> rerank -> assembly.
To disable: comment out the self._trace(...) calls in this file.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from .otel_tracing import pipeline_stage_span, retrieval_leg_span, with_current_context
from .planner.agent import QueryPlanner
from .planner.models import (
    IntentType,
    QueryPlan,
    RetrievalStrategy,
    RoutingTier,
    SubQuery,
    compression_ratio_for_intent,
    default_plan,
)
from .reranker import rerank_with_outcome

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trelix.compression.base import Compressor

# Thread-local storage so parallel eval workers don't mix each other's traces
_trace_local = threading.local()

logger = logging.getLogger("trelix.retrieval")

# Suffixes that mark a file as configuration or infrastructure.
_CONFIG_SUFFIXES = frozenset(
    {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".js",
        ".ts",
        ".ini",
        ".cfg",
        ".cnf",
        ".conf",
        ".properties",
        ".tf",
        ".hcl",
        ".gradle",
        ".mk",
    }
)

# Config files that carry NO extension at all, which is the entire bug: a suffix test can
# never match one, because `Path("Dockerfile").suffix` is `""`. Every ecosystem's most
# important config file is in this shape by convention.
_CONFIG_FILENAMES = frozenset(
    {
        "dockerfile",
        "containerfile",
        "makefile",
        "gnumakefile",
        "procfile",
        "jenkinsfile",
        "vagrantfile",
        "rakefile",
        "gemfile",
        "brewfile",
        "justfile",
        "caddyfile",
        "cmakelists.txt",
        ".gitignore",
        ".dockerignore",
        ".editorconfig",
        ".gitattributes",
        ".nvmrc",
    }
)

# Checked BEFORE anything else, and deliberately not just "absent from the sets above".
#
# These are the conventional homes for credentials, and `_retrieve_config` resolves a hint
# straight to a file's symbols, which then go into an LLM prompt verbatim — there is no
# redaction seam anywhere in retrieval (`grep -rn 'redact|scrub_secret|mask_secret'
# src/trelix/` finds nothing). Reaching them requires the file to be in `files`, which
# today it is not because `walker.detect_language` returns UNKNOWN for all of these. That
# is the only thing standing in the way, it is a coupling rather than a decision, and
# adding ops artifacts to `EXTENSION_MAP` is exactly what this release just did.
#
# A deny-list rather than omission because the `"config" in name` fallback at the end of
# `_looks_like_config` admits things no allow-list mentions — `kubeconfig` passed on that
# fallback alone.
_SECRET_FILENAMES = frozenset({".env", ".envrc", ".npmrc", ".netrc", ".pgpass", "kubeconfig"})
_SECRET_SUFFIXES = frozenset({".env", ".tfvars", ".pem", ".key", ".p12", ".pfx", ".keystore"})


def _looks_like_config(hint: str) -> bool:
    """Does this planner hint name a configuration or infrastructure file?

    The gate cannot simply be removed. `file_hints` is concatenated with `grep_hints`,
    which `planner/models.py:63` documents as "exact symbol names", so unfiltered hints
    like `EXPOSE` or `ports:` would be fed to `find_file_by_path_fragment` and match
    arbitrary files.

    What it got wrong was testing only six suffixes. `Path("Dockerfile").suffix` is `""`,
    so the check silently dropped Dockerfile, Makefile, Procfile, setup.cfg, nginx.conf
    and every `*.tf` — for every repository, not just this one. `eval/golden.jsonl`
    records the consequence: "what port does the container expose and what is its
    entrypoint" resolved exactly one file (`docker-compose.yml`) and answered
    confidently without ever consulting the Dockerfile, whose symbols were indexed and
    retrievable the whole time.

    Note the trap this function itself fell into on the first pass: `.env` was listed in
    `_CONFIG_SUFFIXES`, where it could never match, because `Path(".env").suffix` is `""`
    — a leading dot makes the whole thing a stem. That is the same "a suffix test cannot
    match this" defect described above, reproduced while fixing it. `.env` now belongs to
    the deny-list instead, which is where it wanted to be regardless.
    """
    name = Path(hint).name.lower()

    # Deny first, so neither the allow-lists nor the `"config" in name` fallback can
    # route a credential file into a prompt.
    if name in _SECRET_FILENAMES or Path(name).suffix in _SECRET_SUFFIXES:
        return False
    # "prod.env", "staging.env" — a variant name on a secret suffix.
    if name.rsplit(".", 1)[-1] in {s.lstrip(".") for s in _SECRET_SUFFIXES}:
        return False

    if name in _CONFIG_FILENAMES:
        return True
    if Path(name).suffix in _CONFIG_SUFFIXES:
        return True
    # "Dockerfile.prod", "Makefile.local" — a base config name with a variant suffix.
    stem = name.split(".", 1)[0]
    if stem in _CONFIG_FILENAMES:
        return True
    return "config" in name


# --------------------------------------------------------------------------
# Breadth floor for the three direct-lookup paths
# --------------------------------------------------------------------------
#
# `_retrieve_file_overview`, `_retrieve_project_overview` and `_retrieve_config` each
# widened to standard retrieval only on `if not results:` — a total miss. One matched
# file yielding a handful of symbols therefore returned exclusively those symbols and
# suppressed the vector, BM25 and grep legs completely. Measured on this tree with a
# mocked DB (1 file, 2 symbols): `_retrieve_standard` was never called and assembly saw
# 1 result. Widening the config filename gate made that reachable rather than rarer —
# a query whose only matching hint was `Dockerfile` went from `files_matched=0` plus a
# full-breadth fallback to short-circuiting on 2 section chunks.
#
# MIN_SYMBOLS is `graph_expansion_max_symbols`' default (10): the standard path already
# calls 10 the number of candidates worth expanding from, so fewer than that from a
# direct lookup is below the pipeline's own idea of enough material.
#
# MIN_FILES is the assembler's `_breadth_first_candidates(max_per_file=2)` premise read
# the other way round — breadth means covering more than one file, and the measured
# failure was exactly `files_matched=1`.
#
# The two are ANDed, not ORed. A legitimate single-file overview ("what does
# retriever.py do?") resolves 1 file and ~40 symbols; OR would fire the floor on it and
# charge the cheapest path in the system three retrieval legs plus a rerank round-trip
# on every such query. Requiring both counts to be short is what separates it from the
# Dockerfile case (1 file, 2 symbols). Known gap this leaves: 3 files at 1 symbol each
# passes the file test and does not fire. The trace records both counts on every query
# so that gap is measurable rather than assumed.
#
# The thresholds live on RetrievalConfig (see core/config.py) rather than being read from
# os.environ here, which is where they started. Raw os.environ does not see `.env`, so the
# kill switch the release notes document was silently inert in the one place a user would
# set it.


def _breadth_floor_thresholds(retrieval: Any) -> tuple[bool, int, int]:
    """Resolve ``(enabled, min_files, min_symbols)`` from RetrievalConfig.

    `TRELIX_RETRIEVAL_BREADTH_FLOOR=false` restores the pre-3.1.2 all-or-nothing
    short-circuit exactly, which is what makes the floor's effect measurable as a diff;
    `..._MIN_FILES` / `..._MIN_SYMBOLS` tune it. All three now go through
    pydantic-settings, so `.env` and the process environment both work and a malformed
    value is a startup validation error naming the field rather than a silent fallback.
    """
    return (
        bool(retrieval.breadth_floor_enabled),
        int(retrieval.breadth_floor_min_files),
        int(retrieval.breadth_floor_min_symbols),
    )


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
        if config.store.bm25_read_pool_size > 0:
            self.db.enable_bm25_read_pool(config.store.bm25_read_pool_size)
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
        self._planner = QueryPlanner(config.embedder, retrieval_config=config.retrieval)
        # Wrap with LRU plan cache when enabled (default: 128 entries).
        # plan() hits are returned in <1ms; cold misses delegate to the LLM unchanged.
        if config.retrieval.plan_cache_size > 0:
            from trelix.retrieval.plan_cache import CachingPlanner

            self._planner = CachingPlanner(  # type: ignore[assignment]
                self._planner, max_size=config.retrieval.plan_cache_size
            )

        # Dimension guard: detect provider switch mismatches at startup
        try:
            from trelix.store.dimension_guard import DimensionGuard, DimensionMismatchError

            DimensionGuard.check(
                self.db,
                current_dimension=self.embedder.dimension,
                provider=config.embedder.provider,
            )
        except DimensionMismatchError:
            raise  # Re-raise with the clear user-facing message
        except Exception as exc:
            logger.debug("DimensionGuard.check failed (non-fatal): %s", exc)

        # Debug output dir: <repo_root>/.trelix/debug/
        self._debug_dir = Path(config.repo_path) / ".trelix" / "debug"

        # Memoized SparseEmbedder — instantiated at most once per Retriever.
        # _run_subquery_legs() is called once per sub-query; without this slot the
        # SparseEmbedder lazy-loads the SPLADE model (several seconds via
        # from_pretrained) on EVERY sub-query call when sparse_enabled=True.
        # Initialised lazily on first use so the import remains optional.
        self._sparse_embedder: object | None = None
        self._sparse_embedder_lock = threading.Lock()

        # Latch for the "graph_metadata is empty" warning in _apply_pagerank_boost:
        # the condition is a property of the index, not of the query, so it is worth
        # exactly one line per Retriever instead of one per query.
        self._pagerank_empty_warned = False

        # Resolve effective context budget at startup (memoized for the session)
        self._effective_budget = self._resolve_effective_budget()
        # Scale retrieval ceilings if requested
        self._effective_top_k_vector = config.retrieval.top_k_vector
        self._effective_rerank_top_n = config.retrieval.rerank_top_n
        if config.retrieval.scale_top_k_to_budget and config.retrieval.context_token_budget is None:
            scale_factor = self._effective_budget / 12_000
            self._effective_top_k_vector = max(1, int(config.retrieval.top_k_vector * scale_factor))
            self._effective_rerank_top_n = max(1, int(config.retrieval.rerank_top_n * scale_factor))
            logger.info(
                "Scaled retrieval ceilings: top_k_vector=%d→%d, rerank_top_n=%d→%d (scale=%.2fx)",
                config.retrieval.top_k_vector,
                self._effective_top_k_vector,
                config.retrieval.rerank_top_n,
                self._effective_rerank_top_n,
                scale_factor,
            )

    # ------------------------------------------------------------------
    # Budget resolution
    # ------------------------------------------------------------------

    def _resolve_effective_budget(self) -> int:
        """
        Resolve the effective context token budget.

        Returns the explicit budget when context_token_budget is an int.
        When context_token_budget is None, auto-derives from model window:
          effective_budget = window_size * context_window_fraction

        Falls back to 12,000 when:
        - Model name is not recognized by resolve_window()
        - LLM config is invalid/missing

        Logged at INFO level so operators can see the resolved budget in logs.
        """
        cfg = self.config.retrieval

        # Explicit budget (default 12000) — preserves v2.12.0 behavior
        if cfg.context_token_budget is not None:
            logger.info(
                "Using explicit context budget: %d tokens (no auto-scaling)",
                cfg.context_token_budget,
            )
            return cfg.context_token_budget

        # Auto-derive from model window
        try:
            from trelix.llm.context_windows import resolve_window

            model = self.config.llm.model
            window = resolve_window(model)
            if window is None:
                logger.warning(
                    "Model %r not recognized by context_windows — falling back to 12,000 tokens",
                    model,
                )
                return 12_000

            effective = int(window * cfg.context_window_fraction)
            logger.info(
                "Auto-derived context budget from model %r: window=%d × fraction=%.2f = %d tokens",
                model,
                window,
                cfg.context_window_fraction,
                effective,
            )
            return effective

        except Exception as exc:
            logger.warning(
                "Failed to resolve model context window (falling back to 12,000): %s", exc
            )
            return 12_000

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
        # Reset expansion result for this query (set later by _retrieve_standard if used)
        _trace_local.expand_result = None
        # Reset the rerank outcome. None means the rerank stage was never entered; the
        # stage overwrites it with a RerankOutcome when it runs. Must be reset per query
        # or a later query inherits an earlier one's verdict off this thread.
        _trace_local.rerank_outcome = None

        cfg = self.config.retrieval
        with pipeline_stage_span(cfg, "retrieve", {"query_len": len(query)}):
            if plan is None:
                with pipeline_stage_span(cfg, "planner"):
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
                    "token_budget": self._effective_budget,
                    "budget_pct": round(
                        context.total_tokens / max(1, self._effective_budget) * 100,
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
                "Retrieval complete: intent=%s results=%d tokens=%d (%.0f%%) "
                "sources=%s elapsed=%.3fs",
                plan.intent.value,
                len(context.results),
                context.total_tokens,
                context.total_tokens / max(1, self._effective_budget) * 100,
                context.retrieval_sources,
                context.elapsed_seconds,
            )

            # Telemetry — record timing, result count, expansion metadata (no-op when disabled)
            if self.config.telemetry_enabled:
                from trelix.retrieval.telemetry import TelemetryWriter

                elapsed_ms = (time.perf_counter() - t_start) * 1000
                expand_result = getattr(_trace_local, "expand_result", None)
                TelemetryWriter(self.db, enabled=True).record(
                    context, elapsed_ms=elapsed_ms, expansion_result=expand_result
                )

            # Attach what the rerank stage actually did, so a caller reporting a number
            # can say which pipeline produced it. A new object rather than an in-place
            # write: the assembler's return value is not ours to mutate, and callers
            # holding it must not see it change under them.
            return dataclasses.replace(
                context, rerank=getattr(_trace_local, "rerank_outcome", None)
            )

    # ------------------------------------------------------------------
    # Intent router
    # ------------------------------------------------------------------

    def _execute_plan(self, plan: QueryPlan) -> RetrievedContext:
        # Tier 1 DIRECT: skip all retrieval legs — answer from project overview only.
        # The router has already classified this as a trivial factual query.
        #
        # This check reads the tier and IGNORES plan.strategy, so a TIER_1_DIRECT stamp
        # on any other intent silently discards the legs that intent resolved. That is
        # not hypothetical: plan_from_intent_hint() stamped it on all eight intents and
        # every intent_hint on both the REST route and the MCP tool returned the same
        # project-overview chunks with 0 code files. The invariant the two plan builders
        # now hold — AdaptiveRouter._tier1_plan() and plan_from_intent_hint() — is that
        # TIER_1_DIRECT is only ever paired with PROJECT_OVERVIEW. A mismatch here means
        # a third builder broke that pairing, so it is logged at WARNING rather than
        # being absorbed the way the last one was.
        if getattr(plan, "routing_tier", None) == RoutingTier.TIER_1_DIRECT:
            if plan.intent is not IntentType.PROJECT_OVERVIEW:
                logger.warning(
                    "Tier 1 DIRECT plan carries intent=%s, not project_overview: its "
                    "strategy legs %s will NOT run. Whoever built this plan broke the "
                    "TIER_1_DIRECT/PROJECT_OVERVIEW pairing.",
                    plan.intent.value,
                    plan.strategy.legs,
                )
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
        candidates = self._standard_candidates(plan)
        with pipeline_stage_span(
            self.config.retrieval, "assembly", {"candidate_count": len(candidates)}
        ):
            return self._assemble(
                plan.raw_query,
                candidates,
                intent=plan.intent.value,
                assembly_mode=plan.strategy.assembly_mode,
            )

    def _standard_candidates(self, plan: QueryPlan) -> list[SearchResult]:
        """The standard pipeline up to (but not including) assembly.

        Split out of `_retrieve_standard` so the breadth floor can merge these
        candidates with high-precision direct hits and pack the union ONCE. Merging
        `_retrieve_standard`'s RetrievedContext instead would mean assembling twice:
        the inner pack has already truncated to the token budget without knowing the
        direct hits exist, so the outer pack would only ever see a pool one budget
        wide that was chosen against the wrong candidate set — and with compression
        enabled it would run the compressor over bodies the inner pass had already
        compressed. Returning the candidate list avoids both.
        """
        cfg = self.config.retrieval
        strategy = plan.strategy

        # Run sub-queries in parallel when the planner says they're independent.
        if plan.execution_mode == "parallel" and len(plan.sub_queries) > 1:
            from concurrent.futures import ThreadPoolExecutor

            # with_current_context: OTel's context is contextvars-based and does
            # NOT cross a ThreadPoolExecutor boundary on its own — this makes
            # each worker's leg spans nest under the caller's active span
            # instead of starting as new, unparented traces.
            traced_run = with_current_context(self._run_subquery_legs)
            with ThreadPoolExecutor() as pool:
                futures = [pool.submit(traced_run, sq, strategy) for sq in plan.sub_queries]
                leg_results_list = [f.result() for f in futures]
        else:
            leg_results_list = [self._run_subquery_legs(sq, strategy) for sq in plan.sub_queries]

        # --- Multi-query expansion (optional 8th leg boost) ---
        # When enabled, the primary sub-query is expanded into N variants via LLM,
        # each variant runs through all retrieval legs in parallel, and results are
        # merged into the existing leg buckets before RRF fusion.
        # Deduplication by symbol_id happens later in _dedup().
        # expand_result is stored in thread-local so retrieve() can pass it to telemetry.
        expand_result = None
        if cfg.multi_query_enabled and plan.sub_queries:
            try:
                from trelix.retrieval.query_expansion import MultiQueryExpander

                primary_query = plan.sub_queries[0].semantic_query
                expander = MultiQueryExpander(
                    llm_config=self.config.llm,
                    n=cfg.multi_query_count,
                )
                expand_result = expander.expand(primary_query)
                # Store in thread-local so retrieve() can forward it to telemetry
                try:
                    _trace_local.expand_result = expand_result
                except Exception:
                    pass
                variants = expand_result.queries
                # variants[0] is always the original — skip it (already run above)
                extra_variants = variants[1:]

                if extra_variants:
                    from concurrent.futures import ThreadPoolExecutor as _MQExecutor

                    # Build minimal SubQuery objects for each variant
                    # (no grep hints — pure semantic with simple token split for BM25)
                    variant_sqs = [
                        SubQuery(
                            semantic_query=v,
                            bm25_tokens=v.split()[:5],
                            grep_hints=[],
                            file_hints=[],
                            hyde_snippet="",
                            depends_on=[],
                        )
                        for v in extra_variants
                    ]

                    # Run variant legs in parallel using existing ThreadPoolExecutor
                    traced_variant_run = with_current_context(self._run_subquery_legs)
                    with _MQExecutor() as pool:
                        variant_futures = [
                            pool.submit(traced_variant_run, vsq, plan.strategy)
                            for vsq in variant_sqs
                        ]
                        variant_leg_results = [f.result() for f in variant_futures]

                    # Merge variant results into the main leg buckets
                    leg_results_list = [*leg_results_list, *variant_leg_results]

            except Exception as exc:
                logger.warning("Multi-query expansion failed (non-fatal): %s", exc)

        # Merge per-leg results across all sub-queries for RRF
        vector_results: list[SearchResult] = [r for lr in leg_results_list for r in lr["vector"]]
        bm25_results: list[SearchResult] = [r for lr in leg_results_list for r in lr["bm25"]]
        grep_results: list[SearchResult] = [r for lr in leg_results_list for r in lr["grep"]]
        sparse_results: list[SearchResult] = [
            r for lr in leg_results_list for r in lr.get("sparse", [])
        ]

        # 5th leg: file-summary search (RAPTOR-style, off by default)
        summary_results: list[SearchResult] = []
        if cfg.file_summary_leg_enabled and plan.sub_queries:
            embed_text = (
                plan.sub_queries[0].hyde_snippet
                if plan.sub_queries[0].hyde_snippet.strip()
                else plan.sub_queries[0].semantic_query
            )
            with retrieval_leg_span(
                cfg, "file_summary", query_text=embed_text, top_k=cfg.top_k_file_summary
            ) as span:
                query_embedding: list[float] = self.embedder.embed_query(embed_text)
                summary_results = self._summary_search(query_embedding, k=cfg.top_k_file_summary)
                span.set_result_count(len(summary_results))

        # 6th leg: sub-chunk search (MGS3 block/statement granularity, off by default)
        sub_chunk_results: list[SearchResult] = []
        if cfg.sub_chunk_search_enabled and plan.sub_queries:
            sc_embed_text = (
                plan.sub_queries[0].hyde_snippet
                if plan.sub_queries[0].hyde_snippet.strip()
                else plan.sub_queries[0].semantic_query
            )
            with retrieval_leg_span(
                cfg, "sub_chunk", query_text=sc_embed_text, top_k=cfg.top_k_sub_chunk
            ) as span:
                sc_query_embedding: list[float] = self.embedder.embed_query(sc_embed_text)
                sub_chunk_results = self._sub_chunk_search(
                    sc_query_embedding, k=cfg.top_k_sub_chunk
                )
                span.set_result_count(len(sub_chunk_results))

        logger.info(
            "Pre-fusion leg sizes: vector=%d bm25=%d grep=%d summary=%d sub_chunk=%d sparse=%d",
            len(vector_results),
            len(bm25_results),
            len(grep_results),
            len(summary_results),
            len(sub_chunk_results),
            len(sparse_results),
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
                "sub_chunk_count": len(sub_chunk_results),
                "sparse_count": len(sparse_results),
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
                "top_sparse": [
                    {"name": r.symbol.name, "file": r.file.rel_path, "score": round(r.score, 4)}
                    for r in sparse_results[:5]
                ],
            },
        )

        _weights = cfg.file_type_weights if cfg.file_type_weighting_enabled else None
        # Order must match the ranked_lists order below exactly.
        _list_weights = [
            cfg.leg_weights.get("vector", 1.0),
            cfg.leg_weights.get("bm25", 1.0),
            cfg.leg_weights.get("grep", 1.0),
            cfg.leg_weights.get("summary", 1.0),
            cfg.leg_weights.get("sub_chunk", 1.0),
            cfg.leg_weights.get("sparse", 1.0),
        ]
        with pipeline_stage_span(cfg, "fusion", {"rrf_k": cfg.rrf_k}):
            fused = reciprocal_rank_fusion(
                [
                    vector_results,
                    bm25_results,
                    grep_results,
                    summary_results,
                    sub_chunk_results,
                    sparse_results,
                ],
                k=cfg.rrf_k,
                weights=_weights,
                list_weights=_list_weights,
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
        with pipeline_stage_span(cfg, "expansion"):
            top = fused[: cfg.graph_expansion_max_symbols]
            call_expanded = expand_with_call_graph(
                self.db,
                top,
                depth=strategy.expand_depth,
                max_extra=cfg.graph_expansion_max_symbols,
                personalization_enabled=cfg.pagerank_personalization_enabled,
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
                patterns = [
                    h for sq in plan.sub_queries for h in sq.grep_hints if h.startswith("@")
                ]
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
                    "import_path_seed_files": list({r.file.rel_path for r in import_path_seeded})[
                        :10
                    ],
                },
            )

        # Rerank — skipped when strategy says exact ordering is already correct.
        # Use scaled rerank_top_n when budget scaling is enabled.
        effective_rerank_top_n = (
            self._effective_rerank_top_n
            if cfg.scale_top_k_to_budget and cfg.context_token_budget is None
            else strategy.rerank_top_n
        )
        if cfg.rerank and candidates and not strategy.skip_reranker:
            with pipeline_stage_span(cfg, "rerank", {"top_n": effective_rerank_top_n}):
                candidates, _rerank_outcome = rerank_with_outcome(
                    query=plan.raw_query,
                    results=candidates,
                    config=cfg,
                    top_n=effective_rerank_top_n,
                )
                # Recorded so retrieve() can attach it to the returned context. Every
                # provider degrades to "unranked head" on a missing library, credential
                # or internal error, logging a warning and nothing else — which leaves a
                # caller reporting a measurement unable to say which pipeline produced it.
                _trace_local.rerank_outcome = _rerank_outcome

                # -- Trace: post-rerank ordering --
                self._trace(
                    "post_rerank",
                    {
                        "total": len(candidates),
                        "top5": [
                            {
                                "name": r.symbol.name,
                                "file": r.file.rel_path,
                                "score": round(r.score, 4),
                            }
                            for r in candidates[:5]
                        ],
                    },
                )

        # PageRank boost — applied post-rerank, pre-assemble (no-op internally
        # when cfg.pagerank_boost_enabled is False, so always safe to wrap).
        with pipeline_stage_span(cfg, "pagerank_boost"):
            return self._apply_pagerank_boost(candidates)

    # ------------------------------------------------------------------
    # Breadth floor — shared by the three direct-lookup paths
    # ------------------------------------------------------------------

    def _assemble_direct(
        self,
        plan: QueryPlan,
        direct: list[SearchResult],
        *,
        path: str,
    ) -> RetrievedContext:
        """Assemble direct-lookup results, widening to standard retrieval when thin.

        `direct` must arrive in its intended presentation order, deduped as its own
        path requires (project_overview's ids come from one query and are already
        unique; the other two run `_dedup` first).
        Direct hits keep that order in the merged list and are NOT rescored: the
        greedy pack every direct path uses consumes the list in order, so ordering
        alone gives them precedence, and inventing synthetic scores to express the
        same thing would push fabricated numbers into the trace and telemetry that
        record real fusion/rerank scores. Caveat worth naming: under
        `context_budget_per_source=True` (off by default) the assembler slices the
        budget per `source`, so `file_direct` gets a slice proportional to its count
        instead of first refusal — the merge cannot express direct-first there.
        """
        enabled, min_files, min_symbols = _breadth_floor_thresholds(self.config.retrieval)
        files = {r.file.rel_path for r in direct}
        fired = enabled and len(files) < min_files and len(direct) < min_symbols

        decision: dict[str, Any] = {
            "path": path,
            "enabled": enabled,
            "direct_files": len(files),
            "direct_symbols": len(direct),
            "min_files": min_files,
            "min_symbols": min_symbols,
            "fired": fired,
        }

        if not fired:
            self._trace("breadth_floor", decision)
            return self._assemble(plan.raw_query, direct, intent=plan.intent.value)

        # Same default_plan() the total-miss fallback below already uses, so the widened
        # leg is the exact pipeline the pre-floor fallback was measured on.
        logger.info(
            "%s: direct lookup thin (files=%d<%d symbols=%d<%d) — also running standard "
            "retrieval and merging, direct hits first",
            path,
            len(files),
            min_files,
            len(direct),
            min_symbols,
        )
        try:
            standard = self._standard_candidates(default_plan(plan.raw_query))
        except Exception as exc:
            # Direct-only would have answered this query before the floor existed;
            # failing it now would be a regression caused by the widening itself. Loud
            # and recorded, because a floor that silently stops widening is the same
            # invisible thinness it was added to remove.
            logger.warning(
                "%s: breadth-floor standard leg failed — assembling direct only: %s", path, exc
            )
            decision["standard_error"] = repr(exc)
            self._trace("breadth_floor", decision)
            return self._assemble(plan.raw_query, direct, intent=plan.intent.value)

        seen = {r.chunk.symbol_id for r in direct}
        merged = [*direct, *(r for r in standard if r.chunk.symbol_id not in seen)]
        decision["standard_candidates"] = len(standard)
        decision["merged_symbols"] = len(merged)
        decision["merged_files"] = len({r.file.rel_path for r in merged})
        self._trace("breadth_floor", decision)

        return self._assemble(plan.raw_query, merged, intent=plan.intent.value)

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
        promoted_grep_hints: list[str] = []
        for sq in plan.sub_queries:
            for hint in sq.grep_hints:
                if "." in hint and hint not in file_hints:
                    file_hints.append(hint)
                    promoted_grep_hints.append(hint)

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
                # Recorded separately because the "contains a dot" test above is far
                # looser than _looks_like_config: `self.db` and `os.path` are promoted
                # to file hints as readily as `retriever.py`. Naming which hints came in
                # that way makes a thin file_overview traceable to the promotion rule
                # instead of looking like a genuinely small file.
                "promoted_grep_hints": promoted_grep_hints,
                "files_matched": len(visited_file_ids),
                "symbols_fetched": len(results),
            },
        )

        if not results:
            logger.info(
                "file_overview: no file matched hints %r — falling back to standard", file_hints
            )
            return self._retrieve_standard(default_plan(plan.raw_query))

        return self._assemble_direct(plan, self._dedup(results), path="file_overview")

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

        return self._assemble_direct(plan, results, path="project_overview")

    # ------------------------------------------------------------------
    # Config lookup (config_lookup intent)
    # ------------------------------------------------------------------
    # See _looks_like_config below for why this gate exists and what it used to miss.

    def _retrieve_config(self, plan: QueryPlan) -> RetrievedContext:
        """
        Try file_direct for known config filenames; fall back to standard retrieval.
        """
        file_hints = [h for sq in plan.sub_queries for h in sq.file_hints + sq.grep_hints]

        results: list[SearchResult] = []
        visited: set[int] = set()

        for hint in file_hints:
            if _looks_like_config(hint):
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

        return self._assemble_direct(plan, self._dedup(results), path="config_lookup")

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

        Safe to call from a ThreadPoolExecutor. The lazy SparseEmbedder
        memoization is guarded by self._sparse_embedder_lock (double-checked
        locking) so concurrent sub-query calls cannot race on first-use
        construction or the underlying model load.
        """
        cfg = self.config.retrieval
        out: dict[str, list[SearchResult]] = {"vector": [], "bm25": [], "grep": []}

        # Skip vector ANN for short keyword queries marked lexical_only=True.
        # Research: CoREB (arXiv:2605.04615) shows all embedding models score
        # near-zero nDCG@10 on short keyword queries. BM25+grep wins at this
        # query length. Activated by AdaptiveRouter when
        # TRELIX_RETRIEVAL_SHORT_QUERY_LEXICAL=true.
        skip_vector = getattr(sq, "lexical_only", False)

        if "vector" in strategy.legs and not skip_vector:
            # HyDE: embed the hypothetical code snippet if the planner provided one.
            embed_text = sq.hyde_snippet if sq.hyde_snippet.strip() else sq.semantic_query
            # HyDE fallback: if planner left hyde_snippet empty and fallback is enabled,
            # generate a synthetic snippet now (single LLM call, result replaces
            # semantic_query embed). Skipped when planner already set a snippet.
            if cfg.hyde_fallback_enabled and not sq.hyde_snippet.strip():
                from trelix.retrieval.query_expansion import HyDEExpander

                snippet = HyDEExpander(self.config.llm).expand(sq.semantic_query)
                if snippet:
                    embed_text = snippet
            # Use scaled top_k_vector when budget scaling is enabled
            effective_k = (
                self._effective_top_k_vector
                if cfg.scale_top_k_to_budget and cfg.context_token_budget is None
                else cfg.top_k_vector
            )
            with retrieval_leg_span(
                cfg, "vector", query_text=embed_text, top_k=effective_k
            ) as span:
                embedding = self.embedder.embed_query(embed_text)
                out["vector"] = self._vector_search(
                    embedding, k=effective_k, path_filter=sq.path_filter
                )
                span.set_result_count(len(out["vector"]))

        if "bm25" in strategy.legs:
            bm25_query = " ".join(sq.bm25_tokens) if sq.bm25_tokens else sq.semantic_query
            with retrieval_leg_span(
                cfg, "bm25", query_text=bm25_query, top_k=cfg.top_k_bm25
            ) as span:
                out["bm25"] = bm25_search(
                    self.db,
                    bm25_query,
                    k=cfg.top_k_bm25,
                    path_filter=sq.path_filter,
                    declaration_boost_weight=(
                        cfg.declaration_boost_weight if cfg.declaration_boost_enabled else 1.0
                    ),
                )
                span.set_result_count(len(out["bm25"]))

        if "grep" in strategy.legs:
            hints = sq.grep_hints if sq.grep_hints else [sq.semantic_query]
            with retrieval_leg_span(
                cfg, "grep", query_text=", ".join(hints), top_k=cfg.top_k_grep
            ) as span:
                for hint in hints:
                    out["grep"].extend(
                        grep_search(self.db, hint, k=cfg.top_k_grep, path_filter=sq.path_filter)
                    )
                span.set_result_count(len(out["grep"]))

        # Sparse leg (SPLADE-Code, 7th leg — off by default)
        out["sparse"] = []
        if cfg.sparse_enabled:
            try:
                from trelix.embedder.sparse import SparseEmbedder
                from trelix.retrieval.sparse_search import sparse_search
                from trelix.store.sparse_store import SparseStore

                # Reuse the memoized SparseEmbedder so the SPLADE model is only
                # loaded once per Retriever instance, not once per sub-query call.
                # Double-checked locking: the outer check avoids lock contention
                # on the common already-memoized path; the inner re-check (held
                # under self._sparse_embedder_lock) closes the TOCTOU race where
                # concurrent ThreadPoolExecutor workers could otherwise both
                # observe self._sparse_embedder is None and both construct it.
                if self._sparse_embedder is None:
                    with self._sparse_embedder_lock:
                        if self._sparse_embedder is None:  # re-check inside the lock
                            self._sparse_embedder = SparseEmbedder(
                                model_name=self.config.sparse.model,
                                top_k=self.config.sparse.top_k_tokens,
                            )
                sparse_emb: SparseEmbedder = self._sparse_embedder  # type: ignore[assignment]
                query_sparse = sparse_emb.embed_query(sq.semantic_query)
                if query_sparse:
                    with retrieval_leg_span(
                        cfg, "sparse", query_text=sq.semantic_query, top_k=cfg.top_k_sparse
                    ) as span:
                        sparse_store = SparseStore(self.config.db_path_absolute)
                        out["sparse"] = sparse_search(
                            sparse_store, self.db, query_sparse, k=cfg.top_k_sparse
                        )
                        span.set_result_count(len(out["sparse"]))
            except Exception as exc:
                logger.warning("Sparse search leg failed (non-fatal): %s", exc)

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
                    symbol_id=-(file_id),  # unique per file so dedup keeps all summaries
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

    def _sub_chunk_search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        """Search sub-chunk embeddings (6th retrieval leg, MGS3).

        Returns SearchResult objects using the parent symbol's metadata.
        Returns empty list when no sub-chunks are indexed or sub_chunk_search_enabled=False.
        """
        results: list[SearchResult] = []
        try:
            pairs = self.vector_store.search_sub_chunks(query_embedding, k=k)
            for sub_chunk_id, score in pairs:
                sc = self.db.get_sub_chunk_by_id(sub_chunk_id)
                if sc is None:
                    continue
                sym_file = self.db.get_symbol_with_file(sc.parent_symbol_id)
                if sym_file is None:
                    continue
                symbol, file_obj = sym_file
                results.append(
                    SearchResult(
                        chunk=Chunk(
                            id=sub_chunk_id,
                            symbol_id=sc.parent_symbol_id,
                            chunk_text=sc.chunk_text,
                            token_count=sc.token_count,
                        ),
                        symbol=symbol,
                        file=file_obj,
                        score=score,
                        rank=0,
                        source="sub_chunk",
                    )
                )
        except Exception as exc:
            logger.warning("Sub-chunk search leg failed (non-fatal): %s", exc)
        return results

    def _vector_search(
        self, query_embedding: list[float], k: int, path_filter: str | None = None
    ) -> list[SearchResult]:
        """
        `path_filter`, when set, restricts results to files whose rel_path
        starts with that prefix. The ANN index has no metadata predicate to
        push this into (unlike BM25/grep's SQL joins), so this over-fetches
        by `path_filter_oversample`x and post-filters by prefix after
        hydration, then truncates back to `k` — protects recall against the
        filter discarding some of the raw ANN results.

        Without a `path_filter` the fetch is exactly `k`, so every hydration
        miss is a permanently lost result slot: nothing is left to backfill
        from. A miss means the ANN index still points at a `chunk_id` whose row
        is gone from the DB — the two stores have drifted — and the honest
        response is to name it, not to quietly re-fetch more vectors until the
        count looks right. Over-fetching would return `k` results from a stale
        index and leave the drift undiagnosable; the WARNING below is what
        turns "retrieval feels thin" into "run `trelix index <repo>`".
        """
        fetch_k = k
        if path_filter:
            fetch_k = k * self.config.retrieval.path_filter_oversample
        raw = self.vector_store.search(query_embedding, k=fetch_k)
        results: list[SearchResult] = []
        dead_chunk_ids: list[int] = []
        examined = 0
        for rank, (chunk_id, distance) in enumerate(raw, start=1):
            examined += 1
            score = max(0.0, 1.0 - distance)
            result = self._hydrate_chunk(chunk_id, score=score, rank=rank, source="vector")
            if result is None:
                dead_chunk_ids.append(chunk_id)
                continue
            if path_filter and not result.file.rel_path.startswith(path_filter):
                continue
            results.append(result)
            if len(results) >= k:
                break

        # Only hydration misses are reported. A prefix reject is expected (it is the
        # reason path_filter oversamples) and an ANN index smaller than k is not a
        # defect — neither says anything about store consistency, so neither warns.
        if dead_chunk_ids:
            logger.warning(
                "Vector leg returned %d result(s) for k=%d: %d of %d ANN hit(s) had no "
                "chunk row in the index DB (chunk_ids %s%s) — the vector store and the "
                "index DB have drifted; re-run `trelix index <repo>`",
                len(results),
                k,
                len(dead_chunk_ids),
                examined,
                ", ".join(str(cid) for cid in dead_chunk_ids[:5]),
                ", …" if len(dead_chunk_ids) > 5 else "",
            )
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
    # PageRank boost
    # ------------------------------------------------------------------

    def _apply_pagerank_boost(self, results: list[SearchResult]) -> list[SearchResult]:
        """Boost RRF scores for high-centrality symbols (post-rerank, pre-assemble)."""
        cfg = self.config.retrieval
        if not cfg.pagerank_boost_enabled:
            return results
        try:
            from trelix.graph.persistence import get_top_central_symbols

            top_ids = set(get_top_central_symbols(self.db, top_n=200))
            if not top_ids:
                # The boost is enabled but has nothing to boost with. Saying so once at
                # WARNING is the difference between "this feature is on" and "this
                # feature is on and working" — the previous DEBUG-only failure path made
                # an inert setting indistinguishable from an active one.
                #
                # "Once" has to be enforced by a flag: centrality does not change
                # mid-session, so repeating the line on every query turns the one
                # actionable warning in the log into the kind of per-query noise
                # operators filter out — which is the original silence again, wearing
                # a different hat. Per-instance rather than module-global, so a fresh
                # Retriever (new process, new index) still reports the state.
                if not self._pagerank_empty_warned:
                    self._pagerank_empty_warned = True
                    logger.warning(
                        "PageRank boost is enabled but graph_metadata is empty — "
                        "run `trelix graph <repo>` to populate centrality, "
                        "or set TRELIX_RETRIEVAL_PAGERANK_BOOST=false"
                    )
                return results
            boosted: list[SearchResult] = []
            for r in results:
                if r.symbol.id is not None and r.symbol.id in top_ids:
                    boosted.append(
                        SearchResult(
                            chunk=r.chunk,
                            symbol=r.symbol,
                            file=r.file,
                            score=r.score * cfg.pagerank_boost_factor,
                            rank=r.rank,
                            source=r.source,
                        )
                    )
                else:
                    boosted.append(r)
            return sorted(boosted, key=lambda x: x.score, reverse=True)
        except Exception as exc:
            logger.debug("PageRank boost failed (non-fatal): %s", exc)
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

        cfg = self.config.retrieval
        compressor, ratio = self._make_compressor(intent)
        assembler = ContextAssembler(
            token_budget=self._effective_budget,
            per_source_budget=cfg.context_budget_per_source,
            compressor=compressor,
            compression_ratio=ratio,
            compression_min_tokens=cfg.compression_min_tokens,
        )
        context = assembler.assemble(
            query=query,
            results=results,
            intent=intent,
            assembly_mode=assembly_mode,
            query_embedding=self._cached_query_embedding(query) if compressor else None,
        )
        # Traced whenever the feature is ON — including the intent-opted-out case
        # (active=False), so "why did nothing compress?" is answerable from the
        # trace alone. Nothing is written when the feature is OFF, keeping today's
        # trace files byte-identical too.
        if cfg.compression_enabled:
            self._trace(
                "compression",
                {
                    "provider": cfg.compression_provider,
                    "ratio": ratio,
                    "active": compressor is not None,
                    "path": getattr(compressor, "last_path", None),
                    **(assembler.last_compression_stats or {}),
                },
            )
        return context

    def _make_compressor(self, intent: str | None) -> tuple[Compressor | None, float]:
        """
        Build the compressor and resolve the per-intent target ratio.

        Returns ``(None, 1.0)`` — i.e. today's exact uncompressed assembly — when
        compression is disabled, when the intent opts out (ratio 1.0, e.g.
        symbol_lookup / config_lookup, where the body IS the answer), or when
        constructing the provider fails (graceful degradation: log + carry on,
        same contract as the reranker).
        """
        cfg = self.config.retrieval
        if not cfg.compression_enabled:
            return None, 1.0
        ratio = compression_ratio_for_intent(intent)
        if ratio is None:  # unknown/absent intent — fall back to the global target
            ratio = cfg.compression_target_ratio
        if ratio >= 1.0:
            return None, 1.0
        try:
            from trelix.compression import make_compressor

            return make_compressor(self.config, self.db, self.embedder), ratio
        except Exception as exc:
            logger.warning("Compressor init failed (%s); assembling uncompressed", exc)
            return None, 1.0

    def _cached_query_embedding(self, query: str) -> list[float] | None:
        """
        Peek the embed_query LRU for an embedding we ALREADY paid for.

        Deliberately a peek, never a call: compression must not add inference or
        network cost to assembly. A miss simply returns None and the extractive
        compressor falls back to its zero-inference lexical path.
        """
        cache = getattr(self.embedder, "_cache", None)
        if not isinstance(cache, dict):
            return None
        return cache.get(query.strip().lower())

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
