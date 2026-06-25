"""
LLM Query Planner agent.

QueryPlanner makes a single tool-call to an LLM (OpenAI or Azure OpenAI)
to classify the query intent and decompose it into focused sub-queries with
per-retrieval-leg hints.

On ANY failure (missing API key, network error, parse error, invalid tool
call) it silently falls back to default_plan() — the retriever always gets
a valid QueryPlan.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    SubQuery,
    default_plan,
)
from trelix.retrieval.planner.prompts import PLANNER_TOOL_SCHEMA, SYSTEM_PROMPT

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

logger = logging.getLogger(__name__)

# Chat model to use for the planner (cheap + fast — we only need structured output)
_PLANNER_MODEL_OPENAI = "gpt-4o-mini"
_PLANNER_MODEL_AZURE  = "gpt-4o"   # deployment name; caller can override via config


class QueryPlanner:
    """
    LLM-backed query planner.

    Usage::

        config = EmbedderConfig()
        planner = QueryPlanner(config)
        plan = planner.plan("how does the indexing pipeline work?")

    The planner makes ONE tool-call to the chat LLM and parses the result
    into a QueryPlan.  Falls back to default_plan() on any failure.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._client = self._build_client(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, query: str, project_context: dict | None = None) -> QueryPlan:
        """
        Produce a QueryPlan for *query*.

        Args:
            query:           The raw natural-language question from the user.
            project_context: Optional dict of project-level hints passed to the
                             LLM (e.g. {"language": "Python", "framework": "FastAPI"}).
                             Currently appended to the user message as JSON.

        Returns:
            A fully populated QueryPlan.  Never raises — falls back to
            default_plan() on any error.
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
        from openai import OpenAI, AzureOpenAI  # noqa: F401 — needed for type narrowing

        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": self._build_user_message(query, project_context)},
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
            sub_queries.append(SubQuery(
                semantic_query=sq_raw["semantic_query"],
                hyde_snippet=sq_raw.get("hyde_snippet", ""),
                bm25_tokens=sq_raw.get("bm25_tokens", []),
                grep_hints=sq_raw.get("grep_hints", []),
                file_hints=sq_raw.get("file_hints", []),
                depends_on=sq_raw.get("depends_on", []),
            ))

        if not sub_queries:
            raise ValueError("LLM returned an empty sub_queries list.")

        return QueryPlan(
            intent=intent,
            execution_mode=execution_mode,
            strategy=strategy,
            sub_queries=sub_queries,
            raw_query=raw_query,
        )
