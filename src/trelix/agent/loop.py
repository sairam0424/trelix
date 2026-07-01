"""
AgentLoop — ReAct multi-turn agentic code intelligence.

Pattern: Thought -> Action (tool_call) -> Observation -> repeat until 'done'.

Research basis:
  CodeAct (Wang et al., 2024, arXiv:2402.01030) — code-as-action outperforms
  text/JSON tool calling for code understanding tasks.
  OpenHands (Xingyao et al., 2024, arXiv:2407.16741) — HistoryProcessor
  compression enables long multi-turn sessions within context limits.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig

from trelix.agent.actions import ActionType, AgentAction, Observation, Turn
from trelix.agent.history import HistoryCompressor, TurnHistory
from trelix.agent.tools import AGENT_TOOLS

logger = logging.getLogger("trelix.agent.loop")

_SYSTEM_PROMPT = """\
You are an expert code intelligence agent for a software repository.
You have access to four tools: retrieve, grep, get_symbol, and done.

Strategy:
1. Start by retrieving context relevant to the question.
2. Use grep or get_symbol to drill into specific details.
3. When you have sufficient context, call done with your final answer.
4. Never call done until you've done at least one retrieval.
5. Be concise in thoughts; be thorough in answers.
"""


class AgentLoop:
    """
    Multi-turn ReAct loop for code intelligence queries.

    Usage:
        loop = AgentLoop(config)
        answer = loop.run("how does the authentication system work?")
    """

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._retriever: Any = None  # lazy-init to avoid import cycles
        self._llm_client: Any = None  # overridable for tests

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            from trelix.retrieval.retriever import Retriever

            self._retriever = Retriever(self._config)
        return self._retriever

    def _get_client(self) -> Any:
        if self._llm_client is None:
            from trelix.llm.factory import build_chat_client

            self._llm_client = build_chat_client(self._config.llm)
        return self._llm_client

    def run(self, query: str) -> str:
        """
        Execute the ReAct loop for a user query.

        Returns the final answer string. Never raises — falls back to
        a summary of observations on any failure.
        """
        cfg = self._config.retrieval
        history = TurnHistory()
        compressor = HistoryCompressor(token_budget=cfg.agent_token_budget)

        for turn_n in range(cfg.agent_max_turns):
            try:
                action, thought = self._next_action(query, history, compressor)
            except Exception as exc:
                logger.warning("AgentLoop turn %d failed: %s", turn_n + 1, exc)
                break

            obs = self._execute_action(action)
            history.add(Turn(thought=thought, action=action, observation=obs))

            if action.action_type == ActionType.DONE:
                return str(action.arguments.get("answer", ""))

        # Max turns reached — synthesize from history
        return self._fallback_answer(query, history)

    def _next_action(
        self, query: str, history: TurnHistory, compressor: HistoryCompressor
    ) -> tuple[AgentAction, str]:
        """Ask the LLM which action to take next. Returns (action, thought)."""
        from trelix.llm.client import ChatMessage

        compressed = compressor.compress(history)
        history_text = compressed.to_text()

        user_content = f"Question: {query}\n\n"
        if history_text:
            user_content += f"Previous turns:\n{history_text}\n\n"
        user_content += (
            "What is your next action? Think step by step, then call the appropriate tool."
        )

        messages = [ChatMessage(role="user", content=user_content)]
        client = self._get_client()
        response = client.tool_call(
            messages=messages,
            tools=AGENT_TOOLS,
            max_tokens=512,
        )
        thought = f"Calling {response.tool_name}"
        action = AgentAction(
            action_type=ActionType(response.tool_name),
            arguments=response.tool_arguments,
        )
        return action, thought

    def _execute_action(self, action: AgentAction) -> Observation:
        """Dispatch the action and return an Observation."""
        try:
            match action.action_type:
                case ActionType.RETRIEVE:
                    return self._do_retrieve(action.arguments.get("query", ""))
                case ActionType.GREP:
                    return self._do_grep(
                        action.arguments.get("pattern", ""),
                        action.arguments.get("max_results", 10),
                    )
                case ActionType.GET_SYMBOL:
                    return self._do_get_symbol(action.arguments.get("qualified_name", ""))
                case ActionType.DONE:
                    return Observation(
                        content=action.arguments.get("answer", ""),
                        source="done",
                        success=True,
                    )
                case _:
                    return Observation(
                        content=f"Unknown action: {action.action_type}",
                        source="error",
                        success=False,
                    )
        except Exception as exc:
            return Observation(
                content=f"Action failed: {exc}",
                source=action.action_type.value,
                success=False,
            )

    def _do_retrieve(self, query: str) -> Observation:
        retriever = self._get_retriever()
        ctx = retriever.retrieve(query)
        if not ctx.results:
            return Observation("No results found.", "retrieve", False)
        lines = []
        for r in ctx.results[:8]:
            lines.append(f"[{r.file.rel_path}] {r.symbol.qualified_name}")
            lines.append(r.symbol.body[:300])
            lines.append("---")
        return Observation("\n".join(lines), "retrieve", True)

    def _do_grep(self, pattern: str, max_results: int = 10) -> Observation:
        from trelix.retrieval.grep_search import grep_search
        from trelix.store.db import Database

        db = Database(self._config.db_path_absolute)
        results = grep_search(db, pattern, k=min(max_results, 50))
        if not results:
            return Observation(f"No matches for '{pattern}'.", "grep", False)
        lines = [
            f"{r.file.rel_path}:{r.symbol.line_start} — {r.symbol.name}"
            for r in results[:max_results]
        ]
        return Observation("\n".join(lines), "grep", True)

    def _do_get_symbol(self, qualified_name: str) -> Observation:
        from trelix.store.db import Database

        db = Database(self._config.db_path_absolute)
        symbols = db.get_symbol_by_name(qualified_name.split(".")[-1])
        exact = [s for s in symbols if s.qualified_name == qualified_name]
        candidates = exact or symbols[:1]
        sym = candidates[0] if candidates else None
        if sym is None:
            return Observation(f"Symbol '{qualified_name}' not found.", "get_symbol", False)
        return Observation(f"```\n{sym.body}\n```", "get_symbol", True)

    def _fallback_answer(self, query: str, history: TurnHistory) -> str:
        """When max turns is reached, summarize what was found."""
        if not history.turns:
            return f"Could not find sufficient context for: {query}"
        observations = [t.observation.content for t in history.turns if t.observation.success]
        if not observations:
            return f"No relevant information found for: {query}"
        return "Based on the retrieved context:\n\n" + "\n\n---\n\n".join(observations[:3])
