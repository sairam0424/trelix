"""
Coverage-boosting tests for trelix.retrieval.planner.agent.

Targets the following previously-uncovered lines:
  116-120  AdaptiveRouter.route() exception fallback
  196-202  AdaptiveRouter._multi_step_plan decomposition fallback
  217-219  AdaptiveRouter._multi_step_plan empty sub_queries fallback
  251-256  AdaptiveRouter._decompose_via_llm — TrelixChatClient path
  276-277  AdaptiveRouter._decompose_via_llm markdown fence stripping
  281      AdaptiveRouter._decompose_via_llm JSON parse
  286      AdaptiveRouter._decompose_via_llm too-few sub-questions guard
  378-382  QueryPlanner._plan_direct exception → fallback
  390-393  QueryPlanner._build_user_message with/without context
  401-412  QueryPlanner._call_llm tool_call path
  421-446  QueryPlanner._parse_tool_response valid path
  461-496  QueryPlanner._parse_response legacy raw-client path
"""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    RoutingTier,
    SubQuery,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_config():
    from trelix.core.config import EmbedderConfig

    return EmbedderConfig(provider="local")


def _make_planner():
    from trelix.retrieval.planner.agent import QueryPlanner

    return QueryPlanner(_make_local_config())


def _make_router():
    from trelix.retrieval.planner.agent import AdaptiveRouter

    return AdaptiveRouter(_make_local_config())


def _make_tool_call_response(plan_args: dict[str, Any]):
    """Build a ToolCallResponse with the given tool_arguments dict."""
    from trelix.llm.client import ToolCallResponse

    return ToolCallResponse(
        tool_name="produce_query_plan",
        tool_arguments=plan_args,
    )


def _minimal_plan_args(
    intent: str = "feature_flow",
    execution_mode: str = "parallel",
    semantic_query: str = "how does indexing work?",
) -> dict[str, Any]:
    return {
        "intent": intent,
        "execution_mode": execution_mode,
        "sub_queries": [
            {
                "semantic_query": semantic_query,
                "hyde_snippet": "snippet",
                "bm25_tokens": ["indexing", "pipeline"],
                "grep_hints": ["embed"],
                "file_hints": ["indexer.py"],
            }
        ],
    }


# ---------------------------------------------------------------------------
# QueryPlanner._build_user_message (lines 390-393)
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    def test_without_context(self) -> None:
        planner = _make_planner()
        msg = planner._build_user_message("how does indexing work?", None)
        assert msg == "Query: how does indexing work?"

    def test_with_context(self) -> None:
        planner = _make_planner()
        ctx = {"language": "Python", "framework": "FastAPI"}
        msg = planner._build_user_message("how does auth work?", ctx)
        assert "Project context:" in msg
        assert "Python" in msg
        assert "Query: how does auth work?" in msg

    def test_context_json_serialized(self) -> None:
        planner = _make_planner()
        ctx = {"key": "value"}
        msg = planner._build_user_message("q", ctx)
        # JSON should be present in the message
        assert '"key": "value"' in msg


# ---------------------------------------------------------------------------
# QueryPlanner._parse_tool_response (lines 421-446)
# ---------------------------------------------------------------------------


class TestParseToolResponse:
    def test_valid_args_returns_query_plan(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        plan = planner._parse_tool_response(args, "how does indexing work?")
        assert isinstance(plan, QueryPlan)

    def test_intent_parsed_correctly(self) -> None:
        planner = _make_planner()
        plan = planner._parse_tool_response(_minimal_plan_args("symbol_lookup"), "q")
        assert plan.intent == IntentType.SYMBOL_LOOKUP

    def test_execution_mode_preserved(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args(execution_mode="sequential")
        plan = planner._parse_tool_response(args, "q")
        assert plan.execution_mode == "sequential"

    def test_default_execution_mode_parallel(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        del args["execution_mode"]
        plan = planner._parse_tool_response(args, "q")
        assert plan.execution_mode == "parallel"

    def test_sub_queries_populated(self) -> None:
        planner = _make_planner()
        plan = planner._parse_tool_response(_minimal_plan_args(), "q")
        assert len(plan.sub_queries) == 1
        assert isinstance(plan.sub_queries[0], SubQuery)

    def test_sub_query_fields_mapped(self) -> None:
        planner = _make_planner()
        plan = planner._parse_tool_response(_minimal_plan_args(), "q")
        sq = plan.sub_queries[0]
        assert sq.semantic_query == "how does indexing work?"
        assert sq.hyde_snippet == "snippet"
        assert "indexing" in sq.bm25_tokens
        assert "embed" in sq.grep_hints
        assert "indexer.py" in sq.file_hints

    def test_optional_depends_on_defaults_empty(self) -> None:
        planner = _make_planner()
        plan = planner._parse_tool_response(_minimal_plan_args(), "q")
        assert plan.sub_queries[0].depends_on == []

    def test_raw_query_set(self) -> None:
        planner = _make_planner()
        plan = planner._parse_tool_response(_minimal_plan_args(), "my raw query")
        assert plan.raw_query == "my raw query"

    def test_empty_sub_queries_raises(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        args["sub_queries"] = []
        with pytest.raises(ValueError, match="empty sub_queries"):
            planner._parse_tool_response(args, "q")

    def test_invalid_intent_raises(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args(intent="not_a_real_intent")
        with pytest.raises((ValueError, KeyError)):
            planner._parse_tool_response(args, "q")

    def test_multiple_sub_queries(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        args["sub_queries"].append(
            {
                "semantic_query": "second query",
                "hyde_snippet": "",
                "bm25_tokens": ["second"],
                "grep_hints": [],
                "file_hints": [],
            }
        )
        plan = planner._parse_tool_response(args, "q")
        assert len(plan.sub_queries) == 2


# ---------------------------------------------------------------------------
# QueryPlanner._call_llm — TrelixChatClient.tool_call path (lines 401-412)
# ---------------------------------------------------------------------------


class TestCallLLMToolCallPath:
    def _mock_llm_client(self, plan_args: dict[str, Any]):
        """Return a mock TrelixChatClient whose tool_call returns a ToolCallResponse."""
        from trelix.llm.client import ToolCallResponse, TrelixChatClient

        mock = MagicMock(spec=TrelixChatClient)
        mock.tool_call.return_value = ToolCallResponse(
            tool_name="produce_query_plan",
            tool_arguments=plan_args,
        )
        # Simulate the _client attr being the same as the mock itself (so _client is not None)
        mock._client = mock
        return mock

    def test_tool_call_returns_query_plan(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        planner._llm_client = self._mock_llm_client(args)
        # _client must be non-None so _plan_direct proceeds to _call_llm
        planner._client = planner._llm_client

        plan = planner._call_llm("how does indexing work?", None)
        assert isinstance(plan, QueryPlan)

    def test_tool_call_called_with_correct_tool_name(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        mock_client = self._mock_llm_client(args)
        planner._llm_client = mock_client
        planner._client = mock_client

        planner._call_llm("some query", None)
        call_kwargs = mock_client.tool_call.call_args
        # force_tool should be "produce_query_plan"
        assert call_kwargs.kwargs.get("force_tool") == "produce_query_plan" or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "produce_query_plan"
        )

    def test_tool_call_with_project_context(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        mock_client = self._mock_llm_client(args)
        planner._llm_client = mock_client
        planner._client = mock_client

        ctx = {"language": "Python"}
        plan = planner._call_llm("auth query", ctx)
        assert isinstance(plan, QueryPlan)
        # Verify tool_call was invoked
        mock_client.tool_call.assert_called_once()

    def test_tool_call_error_propagates(self) -> None:
        """_call_llm should raise so _plan_direct can catch and fall back."""
        from trelix.llm.client import TrelixChatClient

        planner = _make_planner()
        mock = MagicMock(spec=TrelixChatClient)
        mock.tool_call.side_effect = RuntimeError("API unavailable")
        mock._client = mock
        planner._llm_client = mock
        planner._client = mock

        with pytest.raises(RuntimeError, match="API unavailable"):
            planner._call_llm("q", None)


# ---------------------------------------------------------------------------
# QueryPlanner._plan_direct fallback on exception (lines 378-382)
# ---------------------------------------------------------------------------


class TestPlanDirectFallback:
    def test_fallback_on_llm_error(self) -> None:
        """_plan_direct must return default_plan when _call_llm raises."""
        from trelix.llm.client import TrelixChatClient

        planner = _make_planner()
        mock = MagicMock(spec=TrelixChatClient)
        mock.tool_call.side_effect = RuntimeError("network error")
        mock._client = mock
        planner._llm_client = mock
        planner._client = mock  # non-None → proceeds past the None-guard

        plan = planner._plan_direct("how does indexing work?", None)
        assert isinstance(plan, QueryPlan)
        assert plan.raw_query == "how does indexing work?"

    def test_fallback_returns_valid_plan_on_parse_error(self) -> None:
        """_plan_direct catches ValueError from _parse_tool_response."""
        from trelix.llm.client import ToolCallResponse, TrelixChatClient

        planner = _make_planner()
        mock = MagicMock(spec=TrelixChatClient)
        # Return a ToolCallResponse with bad args (missing 'intent' key)
        mock.tool_call.return_value = ToolCallResponse(
            tool_name="produce_query_plan",
            tool_arguments={"sub_queries": []},  # missing 'intent'
        )
        mock._client = mock
        planner._llm_client = mock
        planner._client = mock

        plan = planner._plan_direct("broken query", None)
        assert isinstance(plan, QueryPlan)

    def test_no_client_returns_default_plan(self) -> None:
        """When _client is None, _plan_direct returns default_plan immediately."""
        planner = _make_planner()
        planner._client = None

        plan = planner._plan_direct("anything", None)
        assert isinstance(plan, QueryPlan)
        assert plan.raw_query == "anything"


# ---------------------------------------------------------------------------
# AdaptiveRouter.route() exception fallback (lines 116-120)
# ---------------------------------------------------------------------------


class TestAdaptiveRouterExceptionFallback:
    def test_route_fallback_on_unexpected_exception(self) -> None:
        """If _tier1_plan or _single_step_plan raises, route() falls back to default_plan."""
        router = _make_router()

        # Patch _is_tier1 to raise so the outer try/except is exercised
        with patch.object(router, "_is_tier1", side_effect=RuntimeError("boom")):
            plan = router.route("some query")
        assert isinstance(plan, QueryPlan)
        assert plan.raw_query == "some query"

    def test_route_fallback_preserves_raw_query(self) -> None:
        router = _make_router()
        raw = "unexpected crash query"
        with patch.object(router, "_is_tier1", side_effect=RuntimeError("crash")):
            plan = router.route(raw)
        assert plan.raw_query == raw

    def test_route_fallback_returns_default_intent(self) -> None:
        router = _make_router()
        with patch.object(router, "_is_tier1", side_effect=Exception("any error")):
            plan = router.route("q")
        # default_plan uses FEATURE_FLOW
        assert plan.intent == IntentType.FEATURE_FLOW


# ---------------------------------------------------------------------------
# AdaptiveRouter._multi_step_plan decomposition fallback (lines 196-202)
# ---------------------------------------------------------------------------


class TestMultiStepPlanDecompositionFallback:
    def test_decomposition_failure_falls_back_to_single_step(self) -> None:
        """When _decompose_via_llm raises, _multi_step_plan falls back to single-step."""
        router = _make_router()
        planner = router._get_planner()

        # Provide a real-looking LLM client so planner._client is non-None
        from trelix.llm.client import ToolCallResponse, TrelixChatClient

        mock_client = MagicMock(spec=TrelixChatClient)
        mock_client._client = mock_client
        mock_client.tool_call.return_value = ToolCallResponse(
            tool_name="produce_query_plan",
            tool_arguments=_minimal_plan_args(),
        )
        planner._llm_client = mock_client
        planner._client = mock_client

        with patch.object(router, "_decompose_via_llm", side_effect=ValueError("parse fail")):
            plan = router._multi_step_plan("step by step end-to-end query", None)

        assert isinstance(plan, QueryPlan)
        assert plan.routing_tier == RoutingTier.TIER_3_MULTI

    def test_decomposition_fallback_preserves_raw_query(self) -> None:
        router = _make_router()
        planner = router._get_planner()

        from trelix.llm.client import ToolCallResponse, TrelixChatClient

        mock_client = MagicMock(spec=TrelixChatClient)
        mock_client._client = mock_client
        raw = "walk me through the full flow"
        mock_client.tool_call.return_value = ToolCallResponse(
            tool_name="produce_query_plan",
            tool_arguments=_minimal_plan_args(semantic_query=raw),
        )
        planner._llm_client = mock_client
        planner._client = mock_client

        with patch.object(router, "_decompose_via_llm", side_effect=RuntimeError("net")):
            plan = router._multi_step_plan(raw, None)

        assert plan.raw_query == raw


# ---------------------------------------------------------------------------
# AdaptiveRouter._multi_step_plan empty sub-queries fallback (lines 217-219)
# ---------------------------------------------------------------------------


class TestMultiStepPlanEmptySubqueriesFallback:
    def test_empty_decomposition_result_falls_back(self) -> None:
        """When _decompose_via_llm returns an empty list, fall back to _plan_direct."""
        router = _make_router()
        planner = router._get_planner()

        from trelix.llm.client import ToolCallResponse, TrelixChatClient

        mock_client = MagicMock(spec=TrelixChatClient)
        mock_client._client = mock_client
        mock_client.tool_call.return_value = ToolCallResponse(
            tool_name="produce_query_plan",
            tool_arguments=_minimal_plan_args(),
        )
        planner._llm_client = mock_client
        planner._client = mock_client

        # _decompose_via_llm returns [] → sub_queries will be empty → fallback
        with patch.object(router, "_decompose_via_llm", return_value=[]):
            plan = router._multi_step_plan("end-to-end step by step query", None)

        assert isinstance(plan, QueryPlan)
        assert plan.routing_tier == RoutingTier.TIER_3_MULTI


# ---------------------------------------------------------------------------
# AdaptiveRouter._decompose_via_llm — TrelixChatClient path (lines 251-256)
# ---------------------------------------------------------------------------


class TestDecomposeViaLLMTrelixClientPath:
    def _make_router_with_trelix_client(self, response_content: str):
        """Build a router/planner where _llm_client is a TrelixChatClient mock."""
        from trelix.llm.client import ChatResponse, TrelixChatClient

        router = _make_router()
        planner = router._get_planner()

        mock_client = MagicMock(spec=TrelixChatClient)
        mock_client.complete.return_value = ChatResponse(
            content=response_content,
            model="gpt-4o-mini",
            finish_reason="stop",
        )
        # _client set to something that IS the mock's _client equiv — so _use_raw is False
        mock_client._client = object()  # distinct from None
        planner._llm_client = mock_client
        # Planner._client should point to the internal raw client (a non-None sentinel)
        # so the None-guard passes; but _use_raw must be False (planner._client is mock._client)
        planner._client = mock_client._client

        return router, planner, mock_client

    def test_decompose_via_llm_calls_complete(self) -> None:
        sub_questions = ["how does parsing work?", "how does embedding work?"]
        router, planner, mock_client = self._make_router_with_trelix_client(
            json.dumps(sub_questions)
        )
        result = router._decompose_via_llm(planner, "walk me through parsing and embedding")
        mock_client.complete.assert_called_once()
        assert result == sub_questions

    def test_decompose_returns_list_of_strings(self) -> None:
        router, planner, _ = self._make_router_with_trelix_client(
            json.dumps(["sub q 1", "sub q 2"])
        )
        result = router._decompose_via_llm(planner, "end-to-end flow")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_decompose_clamps_to_three(self) -> None:
        router, planner, _ = self._make_router_with_trelix_client(
            json.dumps(["q1", "q2", "q3", "q4"])
        )
        result = router._decompose_via_llm(planner, "complex query")
        assert len(result) <= 3

    def test_decompose_strips_markdown_fences(self) -> None:
        """Lines 276-277: markdown code fences are stripped before JSON parse."""
        fenced = "```json\n" + json.dumps(["sub1", "sub2"]) + "\n```"
        router, planner, _ = self._make_router_with_trelix_client(fenced)
        result = router._decompose_via_llm(planner, "step by step flow")
        assert result == ["sub1", "sub2"]

    def test_decompose_invalid_json_raises(self) -> None:
        router, planner, _ = self._make_router_with_trelix_client("not json at all")
        with pytest.raises((ValueError, json.JSONDecodeError)):
            router._decompose_via_llm(planner, "some query")

    def test_decompose_too_few_sub_questions_raises(self) -> None:
        """Line 286: < 2 sub-questions raises ValueError."""
        router, planner, _ = self._make_router_with_trelix_client(json.dumps(["only one"]))
        with pytest.raises(ValueError, match="Too few"):
            router._decompose_via_llm(planner, "some query")

    def test_decompose_non_list_response_raises(self) -> None:
        """Line 280-281: non-list JSON raises ValueError."""
        router, planner, _ = self._make_router_with_trelix_client(json.dumps({"key": "val"}))
        with pytest.raises(ValueError, match="Unexpected"):
            router._decompose_via_llm(planner, "some query")


# ---------------------------------------------------------------------------
# AdaptiveRouter._decompose_via_llm — legacy raw-client path (lines 258-272)
# ---------------------------------------------------------------------------


class TestDecomposeViaLLMLegacyPath:
    def _make_legacy_raw_client(self, sub_questions: list[str]):
        """Simulate the old openai raw client injected via planner._client."""
        raw_client = types.SimpleNamespace()

        def create(**kwargs):
            content = json.dumps(sub_questions)
            choice = types.SimpleNamespace(message=types.SimpleNamespace(content=content))
            return types.SimpleNamespace(choices=[choice])

        raw_client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        return raw_client

    def test_legacy_path_returns_sub_questions(self) -> None:
        """When planner._client is a raw client distinct from _llm_client._client."""
        from trelix.llm.client import TrelixChatClient

        router = _make_router()
        planner = router._get_planner()

        # _llm_client is a TrelixChatClient (mock)
        mock_trelix = MagicMock(spec=TrelixChatClient)
        internal_raw = object()  # sentinel — the "real" raw client inside mock_trelix
        mock_trelix._client = internal_raw
        planner._llm_client = mock_trelix

        # planner._client points to a DIFFERENT raw client → _use_raw = True
        legacy_raw = self._make_legacy_raw_client(["legacy q1", "legacy q2"])
        planner._client = legacy_raw

        result = router._decompose_via_llm(planner, "walk me through legacy flow")
        assert result == ["legacy q1", "legacy q2"]

    def test_legacy_path_clamps_to_three(self) -> None:
        from trelix.llm.client import TrelixChatClient

        router = _make_router()
        planner = router._get_planner()

        mock_trelix = MagicMock(spec=TrelixChatClient)
        mock_trelix._client = object()
        planner._llm_client = mock_trelix

        legacy_raw = self._make_legacy_raw_client(["q1", "q2", "q3", "q4"])
        planner._client = legacy_raw

        result = router._decompose_via_llm(planner, "complex end-to-end query")
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# QueryPlanner._parse_response — legacy raw-client response (lines 461-496)
# ---------------------------------------------------------------------------


def _make_legacy_response(args: dict[str, Any]) -> Any:
    """Build a fake openai-style response object for _parse_response."""
    args_json = json.dumps(args)
    tool_call = types.SimpleNamespace(function=types.SimpleNamespace(arguments=args_json))
    message = types.SimpleNamespace(tool_calls=[tool_call])
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


class TestParseResponseLegacy:
    def test_valid_response_returns_query_plan(self) -> None:
        planner = _make_planner()
        response = _make_legacy_response(_minimal_plan_args())
        plan = planner._parse_response(response, "legacy query")
        assert isinstance(plan, QueryPlan)

    def test_intent_parsed(self) -> None:
        planner = _make_planner()
        response = _make_legacy_response(_minimal_plan_args("symbol_lookup"))
        plan = planner._parse_response(response, "q")
        assert plan.intent == IntentType.SYMBOL_LOOKUP

    def test_execution_mode_preserved(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args(execution_mode="sequential")
        plan = planner._parse_response(_make_legacy_response(args), "q")
        assert plan.execution_mode == "sequential"

    def test_sub_queries_populated(self) -> None:
        planner = _make_planner()
        response = _make_legacy_response(_minimal_plan_args())
        plan = planner._parse_response(response, "q")
        assert len(plan.sub_queries) == 1
        assert isinstance(plan.sub_queries[0], SubQuery)

    def test_raw_query_set(self) -> None:
        planner = _make_planner()
        response = _make_legacy_response(_minimal_plan_args())
        plan = planner._parse_response(response, "my legacy query")
        assert plan.raw_query == "my legacy query"

    def test_no_tool_calls_raises(self) -> None:
        planner = _make_planner()
        message = types.SimpleNamespace(tool_calls=None)
        choice = types.SimpleNamespace(message=message)
        response = types.SimpleNamespace(choices=[choice])
        with pytest.raises(ValueError, match="tool call"):
            planner._parse_response(response, "q")

    def test_empty_tool_calls_raises(self) -> None:
        planner = _make_planner()
        message = types.SimpleNamespace(tool_calls=[])
        choice = types.SimpleNamespace(message=message)
        response = types.SimpleNamespace(choices=[choice])
        with pytest.raises((ValueError, IndexError)):
            planner._parse_response(response, "q")

    def test_empty_sub_queries_raises(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args()
        args["sub_queries"] = []
        response = _make_legacy_response(args)
        with pytest.raises(ValueError, match="empty sub_queries"):
            planner._parse_response(response, "q")

    def test_invalid_intent_raises(self) -> None:
        planner = _make_planner()
        args = _minimal_plan_args(intent="bogus_intent")
        response = _make_legacy_response(args)
        with pytest.raises((ValueError, KeyError)):
            planner._parse_response(response, "q")

    def test_sub_query_optional_fields_default(self) -> None:
        """Ensure depends_on, bm25_tokens, etc. default correctly when absent."""
        planner = _make_planner()
        args = {
            "intent": "feature_flow",
            "execution_mode": "parallel",
            "sub_queries": [{"semantic_query": "minimal"}],
        }
        plan = planner._parse_response(_make_legacy_response(args), "q")
        sq = plan.sub_queries[0]
        assert sq.hyde_snippet == ""
        assert sq.bm25_tokens == []
        assert sq.grep_hints == []
        assert sq.file_hints == []
        assert sq.depends_on == []


# ---------------------------------------------------------------------------
# Full integration: QueryPlanner.plan() with mocked tool_call
# (exercises _plan_direct → _call_llm → _parse_tool_response chain)
# ---------------------------------------------------------------------------


class TestQueryPlannerEndToEndMocked:
    def _inject_mock_tool_call(self, planner, plan_args: dict[str, Any]) -> None:
        from trelix.llm.client import ToolCallResponse, TrelixChatClient

        mock_client = MagicMock(spec=TrelixChatClient)
        mock_client.tool_call.return_value = ToolCallResponse(
            tool_name="produce_query_plan",
            tool_arguments=plan_args,
        )
        mock_client._client = mock_client
        planner._llm_client = mock_client
        planner._client = mock_client

    def test_plan_returns_llm_intent(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(_make_local_config())
        self._inject_mock_tool_call(planner, _minimal_plan_args("config_lookup"))

        # Use _plan_direct to bypass AdaptiveRouter classification
        plan = planner._plan_direct("some config query", None)
        assert plan.intent == IntentType.CONFIG_LOOKUP

    def test_plan_direct_sub_queries_match_llm_output(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(_make_local_config())
        args = _minimal_plan_args(semantic_query="specific sub query text")
        self._inject_mock_tool_call(planner, args)

        plan = planner._plan_direct("how does auth work?", None)
        assert plan.sub_queries[0].semantic_query == "specific sub query text"

    def test_plan_direct_strategy_set(self) -> None:
        from trelix.retrieval.planner.agent import QueryPlanner

        planner = QueryPlanner(_make_local_config())
        self._inject_mock_tool_call(planner, _minimal_plan_args("dependency_map"))

        plan = planner._plan_direct("dependency query", None)
        assert plan.strategy is INTENT_STRATEGIES[IntentType.DEPENDENCY_MAP]
