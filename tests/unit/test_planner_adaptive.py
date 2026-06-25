"""
Unit tests for the AdaptiveRouter (U7 — 3-tier adaptive query router).

All tests run without any API key.  The router falls back to
default_plan() / single-step when no LLM client is available.
Tier 1 and Tier 3 classification is purely local (regex / heuristics),
so those tests are deterministic even without a live model.
"""

from __future__ import annotations

import pytest

from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    RoutingTier,
    SubQuery,
    default_plan,
)


# ---------------------------------------------------------------------------
# Helper: build a local-provider AdaptiveRouter (no API key, no LLM calls)
# ---------------------------------------------------------------------------

def _make_router():
    from trelix.core.config import EmbedderConfig
    from trelix.retrieval.planner.agent import AdaptiveRouter
    config = EmbedderConfig(provider="local")
    return AdaptiveRouter(config)


# ---------------------------------------------------------------------------
# RoutingTier enum
# ---------------------------------------------------------------------------

class TestRoutingTierEnum:
    def test_has_three_tiers(self) -> None:
        assert len(RoutingTier) == 3

    def test_tier_values(self) -> None:
        assert RoutingTier.TIER_1_DIRECT.value == 1
        assert RoutingTier.TIER_2_SINGLE.value == 2
        assert RoutingTier.TIER_3_MULTI.value == 3

    def test_is_int_enum(self) -> None:
        assert isinstance(RoutingTier.TIER_1_DIRECT, int)


# ---------------------------------------------------------------------------
# QueryPlan.routing_tier field
# ---------------------------------------------------------------------------

class TestQueryPlanRoutingTierField:
    def test_default_plan_has_tier2(self) -> None:
        plan = default_plan("some query")
        assert plan.routing_tier == RoutingTier.TIER_2_SINGLE

    def test_can_override_routing_tier(self) -> None:
        plan = default_plan("q")
        plan.routing_tier = RoutingTier.TIER_1_DIRECT
        assert plan.routing_tier == RoutingTier.TIER_1_DIRECT

    def test_explicit_tier3_construction(self) -> None:
        intent = IntentType.FEATURE_FLOW
        plan = QueryPlan(
            intent=intent,
            execution_mode="parallel",
            strategy=INTENT_STRATEGIES[intent],
            sub_queries=[SubQuery(
                semantic_query="q",
                hyde_snippet="",
                bm25_tokens=["q"],
                grep_hints=[],
                file_hints=[],
            )],
            raw_query="q",
            routing_tier=RoutingTier.TIER_3_MULTI,
        )
        assert plan.routing_tier == RoutingTier.TIER_3_MULTI


# ---------------------------------------------------------------------------
# Tier 1 detection (_is_tier1)
# ---------------------------------------------------------------------------

class TestTier1Detection:
    """Tier 1 is driven by regex — deterministic, no LLM needed."""

    def _router(self):
        return _make_router()

    def test_what_is_pattern(self) -> None:
        assert self._router()._is_tier1("what is trelix?")

    def test_what_are_pattern(self) -> None:
        assert self._router()._is_tier1("what are chunks?")

    def test_define_pattern(self) -> None:
        assert self._router()._is_tier1("define embedding")

    def test_list_all_pattern(self) -> None:
        assert self._router()._is_tier1("list all parsers")

    def test_show_all_pattern(self) -> None:
        assert self._router()._is_tier1("show all symbols")

    def test_case_insensitive(self) -> None:
        assert self._router()._is_tier1("WHAT IS trelix?")
        assert self._router()._is_tier1("Define symbol")

    def test_complex_query_not_tier1(self) -> None:
        assert not self._router()._is_tier1("how does the indexing pipeline work?")

    def test_walk_through_not_tier1(self) -> None:
        assert not self._router()._is_tier1("walk me through the query flow")

    def test_auth_query_not_tier1(self) -> None:
        assert not self._router()._is_tier1("how does authentication work")


# ---------------------------------------------------------------------------
# Tier 3 detection (_is_tier3)
# ---------------------------------------------------------------------------

class TestTier3Detection:
    def _router(self):
        return _make_router()

    def test_walk_me_through(self) -> None:
        assert self._router()._is_tier3(
            "walk me through how a query goes from CLI to LLM answer"
        )

    def test_step_by_step(self) -> None:
        assert self._router()._is_tier3(
            "explain step by step how the indexer works"
        )

    def test_end_to_end(self) -> None:
        assert self._router()._is_tier3(
            "show the end-to-end flow for code search"
        )

    def test_full_flow(self) -> None:
        assert self._router()._is_tier3(
            "describe the full flow from indexing to retrieval"
        )

    def test_long_and_multiple_conjunctions(self) -> None:
        # >80 chars + "and" appearing 2+ times
        q = (
            "how does trelix parse files and embed code chunks "
            "and search them using vectors and bm25"
        )
        assert len(q) > 80
        assert q.lower().count(" and ") >= 2
        assert self._router()._is_tier3(q)

    def test_short_and_query_not_tier3(self) -> None:
        # Short query with "and" is NOT tier 3
        assert not self._router()._is_tier3("embedding and indexing")

    def test_simple_how_query_not_tier3(self) -> None:
        assert not self._router()._is_tier3("how does authentication work")


# ---------------------------------------------------------------------------
# Tier 2 default (no special signals)
# ---------------------------------------------------------------------------

class TestTier2Default:
    def _router(self):
        return _make_router()

    def test_auth_query_routes_tier2(self) -> None:
        """Standard query with no API key → Tier 2 via default_plan fallback."""
        plan = self._router().route("how does authentication work")
        # With local provider the planner falls back to default_plan which is
        # FEATURE_FLOW; routing_tier should be TIER_2_SINGLE.
        assert plan.routing_tier == RoutingTier.TIER_2_SINGLE

    def test_tier2_returns_query_plan(self) -> None:
        plan = self._router().route("explain the chunker module")
        assert isinstance(plan, QueryPlan)

    def test_tier2_execution_mode(self) -> None:
        plan = self._router().route("what does embed_batch do")
        # Single-step plans always parallel (one sub-query)
        assert plan.execution_mode in {"parallel", "sequential"}


# ---------------------------------------------------------------------------
# Tier 1 routing via route() — end-to-end
# ---------------------------------------------------------------------------

class TestTier1Routing:
    def _router(self):
        return _make_router()

    def test_what_is_trelix_routes_tier1(self) -> None:
        plan = self._router().route("what is trelix?")
        assert plan.routing_tier == RoutingTier.TIER_1_DIRECT

    def test_tier1_intent_is_project_overview(self) -> None:
        plan = self._router().route("what is trelix?")
        assert plan.intent == IntentType.PROJECT_OVERVIEW

    def test_tier1_has_sub_query(self) -> None:
        plan = self._router().route("what is trelix?")
        assert len(plan.sub_queries) >= 1

    def test_tier1_sub_query_carries_original_query(self) -> None:
        raw = "what is trelix?"
        plan = self._router().route(raw)
        assert plan.sub_queries[0].semantic_query == raw

    def test_tier1_execution_mode_parallel(self) -> None:
        plan = self._router().route("what are chunks?")
        assert plan.execution_mode == "parallel"

    def test_tier1_never_raises(self) -> None:
        plan = self._router().route("define embedding")
        assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# Tier 3 routing via route() — end-to-end (no LLM, falls back gracefully)
# ---------------------------------------------------------------------------

class TestTier3Routing:
    """
    With provider=local the decomposition LLM call is unavailable, so
    _multi_step_plan falls back to default_plan().  We verify routing_tier
    is set to TIER_3_MULTI and the plan is valid regardless.
    """

    def _router(self):
        return _make_router()

    def test_walk_through_routes_tier3(self) -> None:
        plan = self._router().route(
            "walk me through how a query goes from CLI to LLM answer"
        )
        assert plan.routing_tier == RoutingTier.TIER_3_MULTI

    def test_tier3_returns_query_plan(self) -> None:
        plan = self._router().route(
            "walk me through how a query goes from CLI to LLM answer"
        )
        assert isinstance(plan, QueryPlan)

    def test_tier3_has_sub_queries(self) -> None:
        """Even after fallback the plan must have at least one sub-query."""
        plan = self._router().route(
            "walk me through how a query goes from CLI to LLM answer"
        )
        assert len(plan.sub_queries) >= 1

    def test_tier3_raw_query_preserved(self) -> None:
        raw = "walk me through how a query goes from CLI to LLM answer"
        plan = self._router().route(raw)
        assert plan.raw_query == raw

    def test_tier3_never_raises(self) -> None:
        plan = self._router().route("step by step how does indexing work?")
        assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# Multi-step plan via mock LLM (verifies 2-3 sub-queries produced)
# ---------------------------------------------------------------------------

class TestMultiStepPlanWithMockLLM:
    """
    Simulate a live LLM by monkey-patching the client on the planner so that
    _decompose_via_llm returns a known JSON array.
    """

    def _make_mock_client(self, decomposed: list[str]):
        """Return a minimal mock that mimics the OpenAI chat completions API."""
        import json
        import types

        mock_client = types.SimpleNamespace()

        def create(**kwargs):
            content = json.dumps(decomposed)
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )
            return types.SimpleNamespace(choices=[choice])

        mock_client.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )
        return mock_client

    def test_multi_step_produces_two_sub_queries(self) -> None:
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        config = EmbedderConfig(provider="local")
        router = AdaptiveRouter(config)

        # Inject a mock LLM client into the internal planner
        planner = router._get_planner()
        planner._client = self._make_mock_client([
            "how does the CLI parse and dispatch a user query",
            "how does the retrieval pipeline process a query into context",
        ])

        plan = router._multi_step_plan(
            "walk me through how a query goes from CLI to LLM answer",
            project_context=None,
        )
        assert len(plan.sub_queries) == 2
        assert plan.routing_tier == RoutingTier.TIER_3_MULTI

    def test_multi_step_produces_three_sub_queries(self) -> None:
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        config = EmbedderConfig(provider="local")
        router = AdaptiveRouter(config)

        planner = router._get_planner()
        planner._client = self._make_mock_client([
            "how does the CLI parse and dispatch a user query",
            "how does the retrieval pipeline process a query into context",
            "how does the synthesizer produce the final LLM answer",
        ])

        plan = router._multi_step_plan(
            "walk me through how a query goes from CLI to LLM answer",
            project_context=None,
        )
        assert len(plan.sub_queries) == 3
        assert plan.routing_tier == RoutingTier.TIER_3_MULTI

    def test_multi_step_sub_queries_are_sub_query_instances(self) -> None:
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        config = EmbedderConfig(provider="local")
        router = AdaptiveRouter(config)

        planner = router._get_planner()
        planner._client = self._make_mock_client([
            "CLI query parsing and dispatch",
            "retrieval pipeline context assembly",
        ])

        plan = router._multi_step_plan(
            "walk me through how a query goes from CLI to LLM answer",
            project_context=None,
        )
        for sq in plan.sub_queries:
            assert isinstance(sq, SubQuery)

    def test_multi_step_execution_mode_parallel(self) -> None:
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        config = EmbedderConfig(provider="local")
        router = AdaptiveRouter(config)

        planner = router._get_planner()
        planner._client = self._make_mock_client([
            "CLI query parsing",
            "retrieval pipeline processing",
        ])

        plan = router._multi_step_plan(
            "walk me through query handling end-to-end",
            project_context=None,
        )
        # Tier 3 sub-queries are independent → parallel
        assert plan.execution_mode == "parallel"

    def test_multi_step_clamps_to_three(self) -> None:
        """LLM returning 4 items must be clamped to 3."""
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        config = EmbedderConfig(provider="local")
        router = AdaptiveRouter(config)

        planner = router._get_planner()
        planner._client = self._make_mock_client([
            "q1", "q2", "q3", "q4",
        ])

        plan = router._multi_step_plan(
            "walk me through the full pipeline step by step",
            project_context=None,
        )
        assert len(plan.sub_queries) <= 3


# ---------------------------------------------------------------------------
# QueryPlanner.plan() delegates to AdaptiveRouter
# ---------------------------------------------------------------------------

class TestQueryPlannerDelegatesRouter:
    """
    Verify QueryPlanner.plan() is now a thin wrapper — it returns plans with
    routing_tier stamped, delegating classification to AdaptiveRouter.
    """

    def _make_local_config(self):
        from trelix.core.config import EmbedderConfig
        return EmbedderConfig(provider="local")

    def test_plan_returns_query_plan(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner
        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan("how does the retrieval pipeline work?")
        assert isinstance(plan, QueryPlan)

    def test_plan_tier1_query_returns_tier1(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner
        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan("what is trelix?")
        assert plan.routing_tier == RoutingTier.TIER_1_DIRECT

    def test_plan_tier3_query_returns_tier3(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner
        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan(
            "walk me through how a query goes from CLI to LLM answer"
        )
        assert plan.routing_tier == RoutingTier.TIER_3_MULTI

    def test_plan_standard_query_returns_tier2(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner
        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan("how does authentication work")
        assert plan.routing_tier == RoutingTier.TIER_2_SINGLE

    def test_plan_never_raises(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner
        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan("")
        assert isinstance(plan, QueryPlan)
