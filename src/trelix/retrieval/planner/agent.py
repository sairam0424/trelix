"""
LLM Query Planner agent.

QueryPlanner makes a single tool-call to an LLM (OpenAI or Azure OpenAI)
to classify the query intent and decompose it into focused sub-queries with
per-retrieval-leg hints.

On ANY failure (missing API key, network error, parse error, invalid tool
call) it silently falls back to default_plan() — the retriever always gets
a valid QueryPlan.

AdaptiveRouter wraps QueryPlanner with 3-tier routing:
  Tier 1 (DIRECT)  — trivial factual queries, skip retrieval
  Tier 2 (SINGLE)  — default single-step plan (existing LLM call)
  Tier 3 (MULTI)   — complex multi-part queries, LLM decomposes into 2-3 sub-queries
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    RoutingTier,
    SubQuery,
    default_plan,
)
from trelix.retrieval.planner.prompts import (
    DECOMPOSITION_PROMPT,
    PLANNER_TOOL_SCHEMA,
    SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

logger = logging.getLogger(__name__)

# Chat model to use for the planner (cheap + fast — we only need structured output)
_PLANNER_MODEL_OPENAI = "gpt-4o-mini"
_PLANNER_MODEL_AZURE = "gpt-4o"  # deployment name; caller can override via config


class AdaptiveRouter:
    """
    3-tier adaptive query router.

    Tier 1 (DIRECT): trivial factual queries matched by regex — skip retrieval
                     entirely and return a PROJECT_OVERVIEW plan backed by
                     file_direct lookup (very cheap).
    Tier 2 (SINGLE): default single-step plan — delegates to the LLM planner
                     (existing behaviour, handles ~90 % of queries).
    Tier 3 (MULTI):  complex multi-part queries — LLM decomposes the question
                     into 2–3 focused sub-queries run in parallel.

    Usage::

        router = AdaptiveRouter(config)
        plan = router.route("what is trelix?")        # → Tier 1
        plan = router.route("how does auth work?")    # → Tier 2
        plan = router.route("walk me through how …")  # → Tier 3
    """

    # ------------------------------------------------------------------
    # Tier 1: trivial factual queries — no retrieval needed
    # ------------------------------------------------------------------
    _TIER_1_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"^what (is|are) \w+\??$", re.IGNORECASE),
        re.compile(r"^(list|show) all ", re.IGNORECASE),
        re.compile(r"^define ", re.IGNORECASE),
    ]

    # ------------------------------------------------------------------
    # Tier 3 signals — any match escalates to multi-step decomposition
    # ------------------------------------------------------------------
    _TIER_3_PHRASES: tuple[str, ...] = (
        "from ... to ...",
        "end-to-end",
        "step by step",
        "walk me through",
        "full flow",
    )

    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        # Lazy — only built when an LLM call is actually needed.
        self._planner: QueryPlanner | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str, project_context: dict | None = None) -> QueryPlan:
        """
        Route *query* to the appropriate tier and return a QueryPlan.

        Never raises — any failure falls back to default_plan().
        """
        try:
            if self._is_tier1(query):
                logger.debug("AdaptiveRouter: Tier 1 (direct) for query=%r", query)
                return self._tier1_plan(query)

            if self._is_tier3(query):
                logger.debug("AdaptiveRouter: Tier 3 (multi-step) for query=%r", query)
                return self._multi_step_plan(query, project_context)

            logger.debug("AdaptiveRouter: Tier 2 (single-step) for query=%r", query)
            return self._single_step_plan(query, project_context)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AdaptiveRouter: routing failed (%s), falling back to default plan.", exc
            )
            return default_plan(query)

    # ------------------------------------------------------------------
    # Tier detection helpers
    # ------------------------------------------------------------------

    def _is_tier1(self, query: str) -> bool:
        """Return True when *query* matches any Tier 1 trivial-factual pattern."""
        q = query.strip()
        return any(pattern.match(q) for pattern in self._TIER_1_PATTERNS)

    def _is_tier3(self, query: str) -> bool:
        """Return True when *query* signals a complex multi-step question."""
        q_lower = query.lower()
        # Explicit phrase signals
        if any(phrase in q_lower for phrase in self._TIER_3_PHRASES):
            return True
        # Long query with multiple conjunctions
        if len(query) > 80 and q_lower.count(" and ") >= 2:
            return True
        return False

    # ------------------------------------------------------------------
    # Tier 1: direct answer from project overview (no retrieval legs)
    # ------------------------------------------------------------------

    def _tier1_plan(self, query: str) -> QueryPlan:
        intent = IntentType.PROJECT_OVERVIEW
        return QueryPlan(
            intent=intent,
            routing_tier=RoutingTier.TIER_1_DIRECT,
            execution_mode="parallel",
            strategy=INTENT_STRATEGIES[intent],
            sub_queries=[
                SubQuery(
                    semantic_query=query,
                    hyde_snippet="",
                    bm25_tokens=query.split(),
                    grep_hints=[],
                    file_hints=[],
                )
            ],
            raw_query=query,
        )

    # ------------------------------------------------------------------
    # Tier 2: delegate to the LLM single-step planner (existing behaviour)
    # ------------------------------------------------------------------

    def _single_step_plan(self, query: str, project_context: dict | None) -> QueryPlan:
        # Call _plan_direct() (not plan()) to avoid re-entering the router loop.
        plan = self._get_planner()._plan_direct(query, project_context)
        # Stamp the tier (planner doesn't know about tiers)
        plan.routing_tier = RoutingTier.TIER_2_SINGLE
        return plan

    # ------------------------------------------------------------------
    # Tier 3: LLM decomposes query → 2-3 parallel sub-queries
    # ------------------------------------------------------------------

    def _multi_step_plan(self, query: str, project_context: dict | None) -> QueryPlan:
        """
        Ask the LLM to decompose *query* into 2–3 focused sub-questions and
        build a parallel QueryPlan from the result.

        Falls back to single-step on any parse error.
        """
        planner = self._get_planner()
        if planner._client is None:
            # No LLM available — single-step fallback with Tier 3 stamp
            plan = default_plan(query)
            plan.routing_tier = RoutingTier.TIER_3_MULTI
            return plan

        try:
            sub_questions = self._decompose_via_llm(planner, query)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AdaptiveRouter: decomposition failed (%s), falling back to single-step.", exc
            )
            plan = planner._plan_direct(query, project_context)
            plan.routing_tier = RoutingTier.TIER_3_MULTI
            return plan

        # Build one SubQuery per decomposed sub-question.
        sub_queries = [
            SubQuery(
                semantic_query=sq_text,
                hyde_snippet="",
                bm25_tokens=sq_text.split(),
                grep_hints=[],
                file_hints=[],
            )
            for sq_text in sub_questions
        ]

        if not sub_queries:
            plan = planner._plan_direct(query, project_context)
            plan.routing_tier = RoutingTier.TIER_3_MULTI
            return plan

        intent = IntentType.FEATURE_FLOW
        return QueryPlan(
            intent=intent,
            routing_tier=RoutingTier.TIER_3_MULTI,
            execution_mode="parallel",
            strategy=INTENT_STRATEGIES[intent],
            sub_queries=sub_queries,
            raw_query=query,
        )

    def _decompose_via_llm(self, planner: QueryPlanner, query: str) -> list[str]:
        """
        Call the LLM with DECOMPOSITION_PROMPT and parse the returned JSON array.

        Returns a list of 2–3 sub-question strings.
        Raises ValueError if parsing fails.
        """
        prompt = DECOMPOSITION_PROMPT.format(query=query)

        response = planner._client.chat.completions.create(  # type: ignore[union-attr]
            model=planner._model_name(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            timeout=15.0,
        )

        raw = response.choices[0].message.content or ""
        # Strip markdown fences if the model wraps the JSON
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        sub_questions: list[str] = json.loads(raw)
        if not isinstance(sub_questions, list) or not sub_questions:
            raise ValueError(f"Unexpected decomposition response: {raw!r}")

        # Clamp to 2–3 sub-questions
        sub_questions = [str(sq).strip() for sq in sub_questions[:3]]
        if len(sub_questions) < 2:
            raise ValueError(f"Too few sub-questions decomposed: {sub_questions!r}")

        return sub_questions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_planner(self) -> QueryPlanner:
        """Lazily create the QueryPlanner (builds the LLM client once)."""
        if self._planner is None:
            self._planner = QueryPlanner(self._config)
        return self._planner


class QueryPlanner:
    """
    LLM-backed query planner — thin wrapper around AdaptiveRouter.

    Usage::

        config = EmbedderConfig()
        planner = QueryPlanner(config)
        plan = planner.plan("how does the indexing pipeline work?")

    Internally delegates to AdaptiveRouter which applies 3-tier routing:
      Tier 1 — trivial factual queries (direct, no retrieval)
      Tier 2 — single-step LLM plan (default, existing behaviour)
      Tier 3 — multi-step decomposition for complex queries

    On ANY failure falls back to default_plan() — the retriever always gets
    a valid QueryPlan.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._client = self._build_client(config)
        # AdaptiveRouter is initialised lazily on first plan() call to avoid
        # circular reference issues during __init__ of the router itself.
        self._router: AdaptiveRouter | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, query: str, project_context: dict | None = None) -> QueryPlan:
        """
        Produce a QueryPlan for *query* via adaptive 3-tier routing.

        Args:
            query:           The raw natural-language question from the user.
            project_context: Optional dict of project-level hints passed to the
                             LLM (e.g. {"language": "Python", "framework": "FastAPI"}).
                             Currently appended to the user message as JSON.

        Returns:
            A fully populated QueryPlan with routing_tier set.
            Never raises — falls back to default_plan() on any error.
        """
        if self._router is None:
            self._router = AdaptiveRouter(self._config)
        return self._router.route(query, project_context)

    # ------------------------------------------------------------------
    # Direct LLM call (used internally by AdaptiveRouter for Tier 2)
    # ------------------------------------------------------------------

    def _plan_direct(self, query: str, project_context: dict | None = None) -> QueryPlan:
        """
        Produce a single-step QueryPlan via one LLM tool-call.

        This is the original plan() body, preserved for AdaptiveRouter._single_step_plan()
        to call directly without triggering the router loop.
        Falls back to default_plan() on any failure.
        """
        if self._client is None:
            logger.debug("QueryPlanner: no LLM client available, using default plan.")
            return default_plan(query)

        try:
            return self._call_llm(query, project_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryPlanner: LLM call failed (%s), falling back to default plan.", exc)
            return default_plan(query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self, config: EmbedderConfig) -> object | None:
        """
        Instantiate the appropriate OpenAI client.

        Returns None when no usable API key is present (e.g. provider=local),
        which causes plan() to fall back immediately.
        """
        if config.provider == "azure":
            if not config.azure_api_key or not config.azure_endpoint:
                logger.debug("QueryPlanner: Azure credentials not set, planner disabled.")
                return None
            try:
                from openai import AzureOpenAI

                return AzureOpenAI(
                    api_key=config.azure_api_key,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.azure_api_version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("QueryPlanner: could not create AzureOpenAI client: %s", exc)
                return None

        if config.provider == "openai":
            if not config.openai_api_key:
                logger.debug("QueryPlanner: OPENAI_API_KEY not set, planner disabled.")
                return None
            try:
                from openai import OpenAI

                return OpenAI(api_key=config.openai_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("QueryPlanner: could not create OpenAI client: %s", exc)
                return None

        # provider == "local" — no chat API available
        return None

    def _model_name(self) -> str:
        """Return the chat model name to use based on the configured provider."""
        if self._config.provider == "azure":
            return self._config.azure_chat_deployment
        return self._config.openai_chat_model

    def _build_user_message(self, query: str, project_context: dict | None) -> str:
        """Construct the user message, optionally including project context."""
        if project_context:
            context_str = json.dumps(project_context, indent=2)
            return f"Project context:\n{context_str}\n\nQuery: {query}"
        return f"Query: {query}"

    def _call_llm(self, query: str, project_context: dict | None) -> QueryPlan:
        """
        Make ONE tool-call to the LLM and parse the result into a QueryPlan.

        Raises on any failure so the caller can fall back cleanly.
        """
        from openai import AzureOpenAI, OpenAI  # noqa: F401 — needed for type narrowing

        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_message(query, project_context)},
            ],
            tools=[PLANNER_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "produce_query_plan"}},
            temperature=0.0,
            timeout=15.0,
        )

        return self._parse_response(response, query)

    def _parse_response(self, response: object, raw_query: str) -> QueryPlan:
        """
        Parse the LLM tool-call response into a QueryPlan.

        Raises ValueError / KeyError on malformed output so the caller falls back.
        """
        # Navigate to the tool call arguments
        choice = response.choices[0]  # type: ignore[union-attr]
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            raise ValueError("LLM did not return a tool call.")

        args_raw = tool_calls[0].function.arguments
        args: dict = json.loads(args_raw)

        # Validate & coerce intent
        intent_str: str = args["intent"]
        intent = IntentType(intent_str)

        if intent not in INTENT_STRATEGIES:
            raise ValueError(f"Intent {intent!r} not in INTENT_STRATEGIES.")

        strategy = INTENT_STRATEGIES[intent]
        execution_mode: str = args.get("execution_mode", "parallel")

        # Build SubQuery list
        sub_queries: list[SubQuery] = []
        for sq_raw in args["sub_queries"]:
            sub_queries.append(
                SubQuery(
                    semantic_query=sq_raw["semantic_query"],
                    hyde_snippet=sq_raw.get("hyde_snippet", ""),
                    bm25_tokens=sq_raw.get("bm25_tokens", []),
                    grep_hints=sq_raw.get("grep_hints", []),
                    file_hints=sq_raw.get("file_hints", []),
                    depends_on=sq_raw.get("depends_on", []),
                )
            )

        if not sub_queries:
            raise ValueError("LLM returned an empty sub_queries list.")

        return QueryPlan(
            intent=intent,
            execution_mode=execution_mode,
            strategy=strategy,
            sub_queries=sub_queries,
            raw_query=raw_query,
        )
