"""
Unit tests for the LLM query planner (Phase 12).

All tests run without any API key — the planner falls back to default_plan()
when no LLM client is available, which is the only behaviour we can test
deterministically without mocking the OpenAI wire protocol.
"""

from __future__ import annotations

from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    RetrievalStrategy,
    RoutingTier,
    SubQuery,
    default_plan,
    plan_from_intent_hint,
)

# ---------------------------------------------------------------------------
# Models / default_plan
# ---------------------------------------------------------------------------


class TestDefaultPlan:
    def test_returns_query_plan(self) -> None:
        plan = default_plan("how does the indexing pipeline work?")
        assert isinstance(plan, QueryPlan)

    def test_intent_is_feature_flow(self) -> None:
        plan = default_plan("some query")
        assert plan.intent == IntentType.FEATURE_FLOW

    def test_execution_mode_is_parallel(self) -> None:
        plan = default_plan("some query")
        assert plan.execution_mode == "parallel"

    def test_sub_queries_non_empty(self) -> None:
        plan = default_plan("find the auth module")
        assert len(plan.sub_queries) >= 1

    def test_raw_query_preserved(self) -> None:
        raw = "what does embed_batch() do?"
        plan = default_plan(raw)
        assert plan.raw_query == raw

    def test_strategy_matches_feature_flow(self) -> None:
        plan = default_plan("q")
        assert plan.strategy is INTENT_STRATEGIES[IntentType.FEATURE_FLOW]

    def test_sub_query_semantic_query_equals_raw(self) -> None:
        raw = "explain the chunker"
        plan = default_plan(raw)
        assert plan.sub_queries[0].semantic_query == raw

    def test_sub_query_bm25_tokens_non_empty(self) -> None:
        plan = default_plan("indexing pipeline batch processing")
        tokens = plan.sub_queries[0].bm25_tokens
        assert len(tokens) > 0

    def test_sub_query_grep_hints_empty(self) -> None:
        plan = default_plan("some query")
        assert plan.sub_queries[0].grep_hints == []

    def test_sub_query_file_hints_empty(self) -> None:
        plan = default_plan("some query")
        assert plan.sub_queries[0].file_hints == []

    def test_sub_query_hyde_snippet_empty_string(self) -> None:
        plan = default_plan("some query")
        assert plan.sub_queries[0].hyde_snippet == ""

    def test_sub_query_depends_on_empty(self) -> None:
        plan = default_plan("some query")
        assert plan.sub_queries[0].depends_on == []


# ---------------------------------------------------------------------------
# plan_from_intent_hint — caller-supplied intent/HyDE override
# ---------------------------------------------------------------------------


class TestPlanFromIntentHint:
    def test_valid_intent_returns_matching_plan(self) -> None:
        plan = plan_from_intent_hint("how does auth work?", "symbol_lookup")
        assert plan is not None
        assert plan.intent == IntentType.SYMBOL_LOOKUP
        assert plan.strategy is INTENT_STRATEGIES[IntentType.SYMBOL_LOOKUP]

    def test_invalid_intent_returns_none(self) -> None:
        assert plan_from_intent_hint("some query", "not_a_real_intent") is None

    def test_empty_string_intent_returns_none(self) -> None:
        assert plan_from_intent_hint("some query", "") is None

    def test_valid_intent_stamps_tier1_direct(self) -> None:
        """Skips the LLM entirely, same as AdaptiveRouter._tier1_plan()."""
        plan = plan_from_intent_hint("q", "file_overview")
        assert plan is not None
        assert plan.routing_tier == RoutingTier.TIER_1_DIRECT

    def test_hyde_snippet_hint_is_used_when_provided(self) -> None:
        plan = plan_from_intent_hint("q", "symbol_lookup", "def foo(): ...")
        assert plan is not None
        assert plan.sub_queries[0].hyde_snippet == "def foo(): ..."

    def test_hyde_snippet_hint_defaults_to_empty_string(self) -> None:
        plan = plan_from_intent_hint("q", "symbol_lookup")
        assert plan is not None
        assert plan.sub_queries[0].hyde_snippet == ""

    def test_raw_query_preserved(self) -> None:
        raw = "what does embed_batch() do?"
        plan = plan_from_intent_hint(raw, "symbol_lookup")
        assert plan is not None
        assert plan.raw_query == raw
        assert plan.sub_queries[0].semantic_query == raw

    def test_every_intent_type_value_is_accepted(self) -> None:
        """Every real IntentType member must round-trip through the hint."""
        for intent in IntentType:
            plan = plan_from_intent_hint("q", intent.value)
            assert plan is not None
            assert plan.intent == intent


# ---------------------------------------------------------------------------
# IntentType enum
# ---------------------------------------------------------------------------


class TestIntentType:
    EXPECTED_VALUES = {
        "symbol_lookup",
        "file_overview",
        "feature_flow",
        "project_overview",
        "comparison",
        "config_lookup",
        "dependency_map",
        "blast_radius",
    }

    def test_has_eight_values(self) -> None:
        assert len(IntentType) == 8

    def test_all_expected_values_present(self) -> None:
        actual = {it.value for it in IntentType}
        assert actual == self.EXPECTED_VALUES

    def test_string_coercion(self) -> None:
        assert IntentType("symbol_lookup") == IntentType.SYMBOL_LOOKUP

    def test_value_is_string(self) -> None:
        # IntentType extends str — .value IS the raw string identifier
        assert IntentType.BLAST_RADIUS.value == "blast_radius"
        assert isinstance(IntentType.BLAST_RADIUS.value, str)


# ---------------------------------------------------------------------------
# INTENT_STRATEGIES coverage
# ---------------------------------------------------------------------------


class TestIntentStrategies:
    def test_all_intent_types_covered(self) -> None:
        for intent in IntentType:
            assert intent in INTENT_STRATEGIES, f"Missing strategy for {intent!r}"

    def test_no_extra_keys(self) -> None:
        intent_set = set(IntentType)
        strategy_set = set(INTENT_STRATEGIES.keys())
        assert strategy_set == intent_set

    def test_each_strategy_is_retrieval_strategy(self) -> None:
        for intent, strategy in INTENT_STRATEGIES.items():
            assert isinstance(strategy, RetrievalStrategy), (
                f"Strategy for {intent!r} is not a RetrievalStrategy"
            )

    def test_each_strategy_has_legs(self) -> None:
        for intent, strategy in INTENT_STRATEGIES.items():
            assert len(strategy.legs) >= 1, f"No legs for {intent!r}"

    def test_each_strategy_has_valid_import_direction(self) -> None:
        valid_directions = {"both", "forward", "reverse"}
        for intent, strategy in INTENT_STRATEGIES.items():
            assert strategy.import_direction in valid_directions, (
                f"Invalid import_direction for {intent!r}: {strategy.import_direction!r}"
            )

    def test_each_strategy_has_valid_assembly_mode(self) -> None:
        valid_modes = {"greedy", "breadth_first"}
        for intent, strategy in INTENT_STRATEGIES.items():
            assert strategy.assembly_mode in valid_modes, (
                f"Invalid assembly_mode for {intent!r}: {strategy.assembly_mode!r}"
            )

    def test_each_strategy_rerank_top_n_positive(self) -> None:
        for intent, strategy in INTENT_STRATEGIES.items():
            assert strategy.rerank_top_n > 0, f"rerank_top_n must be positive for {intent!r}"

    def test_each_strategy_expand_depth_non_negative(self) -> None:
        for intent, strategy in INTENT_STRATEGIES.items():
            assert strategy.expand_depth >= 0, f"expand_depth must be >= 0 for {intent!r}"

    def test_each_strategy_import_depth_non_negative(self) -> None:
        for intent, strategy in INTENT_STRATEGIES.items():
            assert strategy.import_depth >= 0, f"import_depth must be >= 0 for {intent!r}"

    def test_blast_radius_uses_reverse_direction(self) -> None:
        strategy = INTENT_STRATEGIES[IntentType.BLAST_RADIUS]
        assert strategy.import_direction == "reverse"

    def test_dependency_map_uses_forward_direction(self) -> None:
        strategy = INTENT_STRATEGIES[IntentType.DEPENDENCY_MAP]
        assert strategy.import_direction == "forward"

    def test_file_overview_skip_reranker(self) -> None:
        strategy = INTENT_STRATEGIES[IntentType.FILE_OVERVIEW]
        assert strategy.skip_reranker is True

    def test_feature_flow_has_max_expand_depth(self) -> None:
        strategy = INTENT_STRATEGIES[IntentType.FEATURE_FLOW]
        assert strategy.expand_depth == 2


# ---------------------------------------------------------------------------
# QueryPlanner fallback (no API key)
# ---------------------------------------------------------------------------


class TestQueryPlannerFallback:
    def _make_local_config(self):
        """Return an EmbedderConfig with provider=local (no API keys)."""
        from trelix.core.config import EmbedderConfig

        return EmbedderConfig(provider="local")

    def test_local_provider_returns_query_plan(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan("how does the retrieval pipeline work?")
        assert isinstance(plan, QueryPlan)

    def test_local_provider_fallback_intent_is_feature_flow(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan("what does the indexer do?")
        assert plan.intent == IntentType.FEATURE_FLOW

    def test_local_provider_sub_queries_non_empty(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan("explain the chunker module")
        assert len(plan.sub_queries) >= 1

    def test_local_provider_raw_query_preserved(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner

        raw = "what breaks if I change the embedder?"
        planner = QueryPlanner(self._make_local_config())
        plan = planner.plan(raw)
        assert plan.raw_query == raw

    def test_openai_no_key_falls_back(self) -> None:
        """openai provider with no API key must fall back silently."""
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import QueryPlanner

        config = EmbedderConfig(provider="openai", openai_api_key=None)
        planner = QueryPlanner(config)
        plan = planner.plan("what is the project structure?")
        assert isinstance(plan, QueryPlan)
        assert plan.intent == IntentType.FEATURE_FLOW

    def test_plan_never_raises(self) -> None:
        """plan() must NEVER raise — any exception becomes a fallback plan."""
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(self._make_local_config())
        # Should not raise regardless of input
        plan = planner.plan("")
        assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# SubQuery dataclass
# ---------------------------------------------------------------------------


class TestSubQuery:
    def test_defaults_depends_on_empty(self) -> None:
        sq = SubQuery(
            semantic_query="batch embedding logic",
            hyde_snippet="def embed_batch(texts): ...",
            bm25_tokens=["embed", "batch"],
            grep_hints=["embed_batch"],
            file_hints=["embedder"],
        )
        assert sq.depends_on == []

    def test_explicit_depends_on(self) -> None:
        sq = SubQuery(
            semantic_query="callers of embed_batch",
            hyde_snippet="",
            bm25_tokens=["embed_batch", "callers"],
            grep_hints=[],
            file_hints=[],
            depends_on=[0],
        )
        assert sq.depends_on == [0]


class TestQueryPlannerCarriesCredentialsFromEmbedderConfig:
    """Credentials on the EmbedderConfig must reach the planner's LLM client.

    `QueryPlanner.__init__` builds its own `LLMConfig` with `_env_file=None`, which
    disables dotenv loading. Its two siblings do the same and then call
    `model_copy(update={...})` to carry the credentials across from the
    EmbedderConfig — `Synthesizer` for the same reason, and `graph_rag` under an
    explicit `# Carry over credentials` comment. `QueryPlanner` omitted that step.

    The consequence was silent and total: with credentials supplied via `.env` — the
    documented route, and what `.env.example` is for — the planner's client was None,
    every `plan()` call fell back to `default_plan()`, and all eight `IntentType`
    values collapsed to `FEATURE_FLOW`. Measured before the fix, 8 textbook queries
    (one per intent, including "what does this project do", which is verbatim the
    PROJECT_OVERVIEW example) produced 1 distinct intent; after, 8 of 8.

    Since `INTENT_STRATEGIES[intent]` selects which retrieval legs run and how far
    call-graph expansion goes, an inert classifier means every query gets an identical
    plan. On this repository, activating it moved nDCG@10 by +0.027 and MRR by +0.040
    against a measured run-to-run noise floor of +/-0.005.

    The existing fallback tests all supply NO credentials, so they pass either way.
    That is the gap this class closes.
    """

    @staticmethod
    def _azure_embedder_config():  # type: ignore[no-untyped-def]
        """An EmbedderConfig carrying azure credentials, as .env loading would produce."""
        from trelix.core.config import EmbedderConfig

        return EmbedderConfig(
            provider="azure",
            azure_api_key="test-key-not-a-real-secret",
            azure_endpoint="https://example.openai.azure.com",
            azure_api_version="2025-01-01-preview",
        )

    def test_client_is_built_when_config_carries_credentials(self) -> None:
        """No process-environment variables, credentials only on the config object."""
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(self._azure_embedder_config())

        assert planner._client is not None, (
            "the planner discarded the credentials on its EmbedderConfig, so every "
            "plan() call will silently fall back to default_plan()"
        )

    def test_credentials_are_not_read_from_the_process_environment(
        self, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """The fix must work with the environment explicitly cleared.

        Otherwise the test would pass on a developer machine that happens to export
        AZURE_API_KEY and fail in CI, which is the inverse of what it is for.
        """
        from trelix.retrieval.planner.agent import QueryPlanner

        for var in ("AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_API_VERSION", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        planner = QueryPlanner(self._azure_embedder_config())

        assert planner._client is not None

    def test_missing_credentials_still_fall_back(self) -> None:
        """Carrying credentials over must not turn an unconfigured planner into a crash."""
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import QueryPlanner

        config = EmbedderConfig(provider="azure", azure_api_key=None, azure_endpoint=None)
        planner = QueryPlanner(config)

        plan = planner.plan("how does indexing work")
        assert isinstance(plan, QueryPlan)
        assert plan.intent == IntentType.FEATURE_FLOW
