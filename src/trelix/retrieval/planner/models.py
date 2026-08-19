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

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, StrEnum

logger = logging.getLogger(__name__)


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
    TIER_3_MULTI = 3


class IntentType(StrEnum):
    SYMBOL_LOOKUP = "symbol_lookup"  # "what does text_to_code() do"
    FILE_OVERVIEW = "file_overview"  # "tell me about auth.py"
    FEATURE_FLOW = "feature_flow"  # "how does the indexing pipeline work end-to-end"
    PROJECT_OVERVIEW = "project_overview"  # "what does this project do"
    COMPARISON = "comparison"  # "compare Go and Python parsers"
    CONFIG_LOOKUP = "config_lookup"  # "what's in jest.config"
    DEPENDENCY_MAP = "dependency_map"  # "what are the key dependencies of X" — forward import graph
    BLAST_RADIUS = "blast_radius"  # "what breaks if X changes" — reverse import graph


@dataclass
class SubQuery:
    """
    One retrieval unit within a plan.

    Multiple sub-queries handle compound questions. Each carries
    leg-specific search hints so every retrieval leg gets a query
    optimised for its strengths rather than the raw NL string.
    """

    semantic_query: str  # rephrased as a technical description (NOT a question)
    hyde_snippet: str  # hypothetical code snippet — embedded instead of the NL query (HyDE)
    bm25_tokens: list[str]  # clean keyword tokens — no stop words, no questions
    grep_hints: list[str]  # exact symbol names / filename fragments for grep
    file_hints: list[str]  # filename fragments to bias retrieval toward
    depends_on: list[int] = field(
        default_factory=list
    )  # 0-based indices of required prior sub-queries
    lexical_only: bool = False  # v2.6.0: True for short keyword queries — skips vector ANN
    path_filter: str | None = None  # restrict every leg to files under this rel_path prefix


@dataclass
class RetrievalStrategy:
    """
    Per-intent retrieval configuration.

    The retriever reads every field from this object and executes accordingly —
    there is no intent-switching logic inside the retriever itself.
    All intelligence lives here.
    """

    # ── Call-graph expansion ──────────────────────────────────────────
    expand_depth: int  # call-graph hops: 0=none, 1=callers+callees, 2=deep
    # ── Retrieval legs ───────────────────────────────────────────────
    legs: list[str]  # "vector" | "bm25" | "grep" | "file_direct"
    skip_reranker: bool  # True when structural order is already correct (file_direct)
    # ── Import-graph expansion ───────────────────────────────────────
    import_depth: int  # hops through the import graph: 1=direct, 2=transitive
    import_max_extra: int  # max symbols to surface from import expansion
    import_direction: str  # "both" | "forward" (what X imports) | "reverse" (what imports X)
    # ── Context assembly ─────────────────────────────────────────────
    assembly_mode: (
        str  # "greedy" (depth-first by score) | "breadth_first" (1-2 per file, many files)
    )
    # ── Reranker budget ──────────────────────────────────────────
    rerank_top_n: int  # candidates passed to the reranker; higher = more recall at cost of latency
    # ── Context compression ──────────────────────────────────────
    # Target fraction of an oversized body to keep when it would otherwise be
    # dropped for not fitting the budget. 1.0 = never compress this intent
    # (the answer depends on reading the body verbatim). Only consulted when
    # RetrievalConfig.compression_enabled is True; defaults to 1.0 so any
    # strategy built without this field is a guaranteed no-op.
    compression_ratio: float = 1.0


# Pre-baked strategies — the ONLY place that controls retrieval behaviour per intent.
# Adding a new intent = add one line here. Zero changes elsewhere.
INTENT_STRATEGIES: dict[IntentType, RetrievalStrategy] = {
    # ── Exact symbol questions ("what does X do?") ───────────────────────────
    IntentType.SYMBOL_LOOKUP: RetrievalStrategy(
        expand_depth=1,
        legs=["grep", "bm25", "vector"],
        skip_reranker=False,
        import_depth=1,
        import_max_extra=3,
        import_direction="both",
        assembly_mode="greedy",
        rerank_top_n=20,
        compression_ratio=1.0,  # the body IS the answer — never elide it
    ),
    # ── File-level overview ("tell me about auth.py") ────────────────────────
    IntentType.FILE_OVERVIEW: RetrievalStrategy(
        expand_depth=0,
        legs=["file_direct"],
        skip_reranker=True,
        import_depth=0,
        import_max_extra=0,
        import_direction="both",
        assembly_mode="greedy",
        rerank_top_n=20,
        compression_ratio=1.0,  # structural walk of one file — verbatim
    ),
    # ── End-to-end feature flows ("how does indexing work?") ─────────────────
    IntentType.FEATURE_FLOW: RetrievalStrategy(
        expand_depth=2,
        legs=["vector", "bm25"],
        skip_reranker=False,
        import_depth=2,
        import_max_extra=15,
        import_direction="both",
        assembly_mode="greedy",
        rerank_top_n=30,
        compression_ratio=0.45,  # many hops matter more than any one full body
    ),
    # ── Overall project architecture ─────────────────────────────────────────
    IntentType.PROJECT_OVERVIEW: RetrievalStrategy(
        expand_depth=0,
        legs=["file_direct"],
        skip_reranker=True,
        import_depth=0,
        import_max_extra=0,
        import_direction="both",
        assembly_mode="greedy",
        rerank_top_n=20,
        compression_ratio=1.0,  # already summary-level (module/README symbols)
    ),
    # ── Comparisons ("compare X and Y") ──────────────────────────────────────
    IntentType.COMPARISON: RetrievalStrategy(
        expand_depth=1,
        legs=["vector", "bm25", "grep"],
        skip_reranker=False,
        import_depth=1,
        import_max_extra=8,
        import_direction="both",
        assembly_mode="greedy",
        rerank_top_n=35,
        compression_ratio=0.65,  # both sides must fit, but detail still matters
    ),
    # ── Config file lookups ───────────────────────────────────────────────────
    IntentType.CONFIG_LOOKUP: RetrievalStrategy(
        expand_depth=0,
        legs=["file_direct", "grep"],
        skip_reranker=True,
        import_depth=0,
        import_max_extra=0,
        import_direction="both",
        assembly_mode="greedy",
        rerank_top_n=20,
        compression_ratio=1.0,  # config values are the answer — never elide
    ),
    # ── "What does X depend on / what services does Y use?" ──────────────────
    # Forward import walk (2 hops) to enumerate all transitive dependencies.
    # breadth_first assembly ensures every dependency file gets representation.
    IntentType.DEPENDENCY_MAP: RetrievalStrategy(
        expand_depth=1,
        legs=["vector", "bm25"],
        skip_reranker=False,
        import_depth=2,
        import_max_extra=20,
        import_direction="forward",
        assembly_mode="breadth_first",
        rerank_top_n=30,
        compression_ratio=0.30,  # breadth over depth — coverage is the answer
    ),
    # ── "What breaks if X changes / what imports Y?" ─────────────────────────
    # grep-first to seed from exact matches, then reverse import walk to find
    # every file that depends on the found symbols/files.
    # breadth_first assembly shows 1-2 symbols from many affected files.
    IntentType.BLAST_RADIUS: RetrievalStrategy(
        expand_depth=0,
        legs=["grep", "vector", "bm25"],
        skip_reranker=False,
        import_depth=1,
        import_max_extra=30,
        import_direction="reverse",
        assembly_mode="breadth_first",
        rerank_top_n=40,
        compression_ratio=0.30,  # "what breaks" = how many callers, not their guts
    ),
}

# Per-intent compression-ratio env override — one var per intent, mirroring the
# TRELIX_RETRIEVAL_LEG_WEIGHT_<LEG> / TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_<LANG>
# precedent in RetrievalConfig.model_post_init(). Example:
#     TRELIX_RETRIEVAL_COMPRESSION_RATIO_BLAST_RADIUS=0.5
_COMPRESSION_RATIO_ENV_PREFIX = "TRELIX_RETRIEVAL_COMPRESSION_RATIO_"

# Same bounds as RetrievalConfig.compression_target_ratio — an out-of-range or
# unparseable override is logged and ignored rather than raised (the planner
# never raises into the retrieval path).
_COMPRESSION_RATIO_MIN = 0.1
_COMPRESSION_RATIO_MAX = 1.0


def compression_ratio_for_intent(intent: IntentType | str | None) -> float | None:
    """
    Resolve the compression ratio for ``intent``.

    Precedence: ``TRELIX_RETRIEVAL_COMPRESSION_RATIO_<INTENT>`` env override >
    the intent's baked ``RetrievalStrategy.compression_ratio``.

    Returns ``None`` for an unknown/absent intent so the caller can fall back to
    ``RetrievalConfig.compression_target_ratio``. A returned ``1.0`` means "do
    not compress this intent" and callers MUST skip compression entirely.
    """
    if intent is None:
        return None
    try:
        resolved = IntentType(intent)
    except ValueError:
        return None
    strategy = INTENT_STRATEGIES.get(resolved)
    if strategy is None:
        return None

    baked = strategy.compression_ratio
    raw = os.environ.get(f"{_COMPRESSION_RATIO_ENV_PREFIX}{resolved.name}")
    if raw is None:
        return baked
    try:
        override = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring %s%s=%r — not a float; using baked ratio %s",
            _COMPRESSION_RATIO_ENV_PREFIX,
            resolved.name,
            raw,
            baked,
        )
        return baked
    if not _COMPRESSION_RATIO_MIN <= override <= _COMPRESSION_RATIO_MAX:
        logger.warning(
            "Ignoring %s%s=%s — outside [%s, %s]; using baked ratio %s",
            _COMPRESSION_RATIO_ENV_PREFIX,
            resolved.name,
            override,
            _COMPRESSION_RATIO_MIN,
            _COMPRESSION_RATIO_MAX,
            baked,
        )
        return baked
    return override


@dataclass
class QueryPlan:
    """
    The complete retrieval plan produced by the planner agent.

    Retriever reads `intent`, `strategy`, `routing_tier`, and `sub_queries` —
    no conditional intent-switch logic inside the retriever itself.
    """

    intent: IntentType
    execution_mode: str  # "parallel" | "sequential"
    strategy: RetrievalStrategy
    sub_queries: list[SubQuery]
    raw_query: str  # original user query, used as final fallback
    routing_tier: RoutingTier = field(default=RoutingTier.TIER_2_SINGLE)


def plan_from_intent_hint(
    raw_query: str, intent_hint: str, hyde_snippet_hint: str | None = None
) -> QueryPlan | None:
    """
    Build a QueryPlan directly from a caller-supplied intent hint — skips the
    internal LLM intent-classification call when the caller already knows the
    intent (e.g. an agent that already classified the query itself).

    Skipping the CLASSIFIER is not skipping RETRIEVAL. This function used to
    stamp routing_tier=TIER_1_DIRECT on every intent while also resolving
    strategy=INTENT_STRATEGIES[intent], and Retriever._execute_plan() tests the
    tier BEFORE the intent — so it returned _retrieve_project_overview(plan) and
    the strategy was never read. Measured on this repo: all eight IntentType
    values returned byte-identical output (40 README sections, 0 code files, 0
    overlap with the correct result set, 0.02s because no leg ran), on both the
    REST route and the MCP tool, which share this one function. The tier is
    therefore paired the way AdaptiveRouter._tier1_plan() pairs it — TIER_1_DIRECT
    ONLY with PROJECT_OVERVIEW, whose strategy really is a single file_direct
    lookup. Every other hint carries its legs through to execution as
    TIER_2_SINGLE, the tier _single_step_plan() stamps for the same pipeline.

    Returns None on an invalid/unrecognized intent_hint value — callers
    MUST fall through to normal server-side classification in that case,
    never hard-reject, matching QueryPlanner's own "never raise, always
    fall back" posture (see module docstring).

    A hint is trusted, not re-validated against the query text — the caller
    is assumed to have done real classification work; treat it as a
    narrowing instruction, not a suggestion to second-guess.
    """
    try:
        intent = IntentType(intent_hint)
    except ValueError:
        return None
    if intent not in INTENT_STRATEGIES:
        return None

    return QueryPlan(
        intent=intent,
        routing_tier=(
            RoutingTier.TIER_1_DIRECT
            if intent is IntentType.PROJECT_OVERVIEW
            else RoutingTier.TIER_2_SINGLE
        ),
        execution_mode="parallel",
        strategy=INTENT_STRATEGIES[intent],
        sub_queries=[
            SubQuery(
                semantic_query=raw_query,
                hyde_snippet=hyde_snippet_hint or "",
                bm25_tokens=raw_query.split(),
                grep_hints=[],
                file_hints=[],
            )
        ],
        raw_query=raw_query,
    )


def default_plan(raw_query: str) -> QueryPlan:
    """Fallback plan when the LLM planner is unavailable or fails."""
    intent = IntentType.FEATURE_FLOW
    return QueryPlan(
        intent=intent,
        execution_mode="parallel",
        strategy=INTENT_STRATEGIES[intent],
        sub_queries=[
            SubQuery(
                semantic_query=raw_query,
                hyde_snippet="",
                bm25_tokens=raw_query.split(),
                grep_hints=[],
                file_hints=[],
            )
        ],
        raw_query=raw_query,
    )
