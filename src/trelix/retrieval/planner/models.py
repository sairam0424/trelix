"""
Data models for the query planner.

QueryPlan is the structured output from the LLM planner agent.
It drives all retrieval decisions downstream: which legs run, how deep
graph expansion goes, and what each leg searches for.

RetrievalStrategy is the single source of truth for every retrieval
parameter for a given intent. Adding a new intent = one new entry in
INTENT_STRATEGIES. No changes needed anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RoutingTier(int, Enum):
    """
    Adaptive routing tier assigned by AdaptiveRouter.

    TIER_1_DIRECT: trivial factual queries — skip retrieval entirely and
                   answer directly from project overview symbols.
    TIER_2_SINGLE: default single-step plan (current behaviour for most queries).
    TIER_3_MULTI:  complex multi-part queries — LLM decomposes into 2–3 focused
                   sub-queries executed in parallel.
    """
    TIER_1_DIRECT = 1
    TIER_2_SINGLE = 2
    TIER_3_MULTI  = 3


class IntentType(str, Enum):
    SYMBOL_LOOKUP    = "symbol_lookup"    # "what does text_to_code() do"
    FILE_OVERVIEW    = "file_overview"    # "tell me about auth.py"
    FEATURE_FLOW     = "feature_flow"     # "how does the indexing pipeline work end-to-end"
    PROJECT_OVERVIEW = "project_overview" # "what does this project do"
    COMPARISON       = "comparison"       # "compare Go and Python parsers"
    CONFIG_LOOKUP    = "config_lookup"    # "what's in jest.config"
    DEPENDENCY_MAP   = "dependency_map"   # "what are the key dependencies of X" — forward import graph
    BLAST_RADIUS     = "blast_radius"     # "what breaks if X changes" — reverse import graph


@dataclass
class SubQuery:
    """
    One retrieval unit within a plan.

    Multiple sub-queries handle compound questions. Each carries
    leg-specific search hints so every retrieval leg gets a query
    optimised for its strengths rather than the raw NL string.
    """
    semantic_query: str        # rephrased as a technical description (NOT a question)
    hyde_snippet: str          # hypothetical code snippet — embedded instead of the NL query (HyDE)
    bm25_tokens: list[str]     # clean keyword tokens — no stop words, no questions
    grep_hints: list[str]      # exact symbol names / filename fragments for grep
    file_hints: list[str]      # filename fragments to bias retrieval toward
    depends_on: list[int] = field(default_factory=list)  # 0-based indices of required prior sub-queries


@dataclass
class RetrievalStrategy:
    """
    Per-intent retrieval configuration.

    The retriever reads every field from this object and executes accordingly —
    there is no intent-switching logic inside the retriever itself.
    All intelligence lives here.
    """
    # ── Call-graph expansion ──────────────────────────────────────────
    expand_depth: int       # call-graph hops: 0=none, 1=callers+callees, 2=deep
    # ── Retrieval legs ───────────────────────────────────────────────
    legs: list[str]         # "vector" | "bm25" | "grep" | "file_direct"
    skip_reranker: bool     # True when structural order is already correct (file_direct)
    # ── Import-graph expansion ───────────────────────────────────────
    import_depth: int       # hops through the import graph: 1=direct, 2=transitive
    import_max_extra: int   # max symbols to surface from import expansion
    import_direction: str   # "both" | "forward" (what X imports) | "reverse" (what imports X)
    # ── Context assembly ─────────────────────────────────────────────
    assembly_mode: str      # "greedy" (depth-first by score) | "breadth_first" (1-2 per file, many files)
    # ── Reranker budget ──────────────────────────────────────────
    rerank_top_n: int       # candidates passed to the reranker; higher = more recall at cost of latency


# Pre-baked strategies — the ONLY place that controls retrieval behaviour per intent.
# Adding a new intent = add one line here. Zero changes elsewhere.
INTENT_STRATEGIES: dict[IntentType, RetrievalStrategy] = {
    # ── Exact symbol questions ("what does X do?") ───────────────────────────
    IntentType.SYMBOL_LOOKUP: RetrievalStrategy(
        expand_depth=1, legs=["grep", "bm25", "vector"], skip_reranker=False,
        import_depth=1, import_max_extra=3,  import_direction="both",    assembly_mode="greedy",
        rerank_top_n=20,
    ),
    # ── File-level overview ("tell me about auth.py") ────────────────────────
    IntentType.FILE_OVERVIEW: RetrievalStrategy(
        expand_depth=0, legs=["file_direct"],            skip_reranker=True,
        import_depth=0, import_max_extra=0,  import_direction="both",    assembly_mode="greedy",
        rerank_top_n=20,
    ),
    # ── End-to-end feature flows ("how does indexing work?") ─────────────────
    IntentType.FEATURE_FLOW: RetrievalStrategy(
        expand_depth=2, legs=["vector", "bm25"],         skip_reranker=False,
        import_depth=2, import_max_extra=15, import_direction="both",    assembly_mode="greedy",
        rerank_top_n=30,
    ),
    # ── Overall project architecture ─────────────────────────────────────────
    IntentType.PROJECT_OVERVIEW: RetrievalStrategy(
        expand_depth=0, legs=["file_direct"],            skip_reranker=True,
        import_depth=0, import_max_extra=0,  import_direction="both",    assembly_mode="greedy",
        rerank_top_n=20,
    ),
    # ── Comparisons ("compare X and Y") ──────────────────────────────────────
    IntentType.COMPARISON: RetrievalStrategy(
        expand_depth=1, legs=["vector", "bm25", "grep"], skip_reranker=False,
        import_depth=1, import_max_extra=8,  import_direction="both",    assembly_mode="greedy",
        rerank_top_n=35,
    ),
    # ── Config file lookups ───────────────────────────────────────────────────
    IntentType.CONFIG_LOOKUP: RetrievalStrategy(
        expand_depth=0, legs=["file_direct", "grep"],    skip_reranker=True,
        import_depth=0, import_max_extra=0,  import_direction="both",    assembly_mode="greedy",
        rerank_top_n=20,
    ),
    # ── "What does X depend on / what services does Y use?" ──────────────────
    # Forward import walk (2 hops) to enumerate all transitive dependencies.
    # breadth_first assembly ensures every dependency file gets representation.
    IntentType.DEPENDENCY_MAP: RetrievalStrategy(
        expand_depth=1, legs=["vector", "bm25"],         skip_reranker=False,
        import_depth=2, import_max_extra=20, import_direction="forward", assembly_mode="breadth_first",
        rerank_top_n=30,
    ),
    # ── "What breaks if X changes / what imports Y?" ─────────────────────────
    # grep-first to seed from exact matches, then reverse import walk to find
    # every file that depends on the found symbols/files.
    # breadth_first assembly shows 1-2 symbols from many affected files.
    IntentType.BLAST_RADIUS: RetrievalStrategy(
        expand_depth=0, legs=["grep", "vector", "bm25"], skip_reranker=False,
        import_depth=1, import_max_extra=30, import_direction="reverse", assembly_mode="breadth_first",
        rerank_top_n=40,
    ),
}


@dataclass
class QueryPlan:
    """
    The complete retrieval plan produced by the planner agent.

    Retriever reads `intent`, `strategy`, `routing_tier`, and `sub_queries` —
    no conditional intent-switch logic inside the retriever itself.
    """
    intent: IntentType
    execution_mode: str       # "parallel" | "sequential"
    strategy: RetrievalStrategy
    sub_queries: list[SubQuery]
    raw_query: str            # original user query, used as final fallback
    routing_tier: RoutingTier = field(default=RoutingTier.TIER_2_SINGLE)


def default_plan(raw_query: str) -> QueryPlan:
    """Fallback plan when the LLM planner is unavailable or fails."""
    intent = IntentType.FEATURE_FLOW
    return QueryPlan(
        intent=intent,
        execution_mode="parallel",
        strategy=INTENT_STRATEGIES[intent],
        sub_queries=[SubQuery(
            semantic_query=raw_query,
            hyde_snippet="",
            bm25_tokens=raw_query.split(),
            grep_hints=[],
            file_hints=[],
        )],
        raw_query=raw_query,
    )
