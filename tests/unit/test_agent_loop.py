"""Tests for AgentLoop ReAct orchestrator."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from trelix.agent.loop import AgentLoop


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.repo_path = str(tmp_path)
    cfg.retrieval.agentic_enabled = True
    cfg.retrieval.agent_max_turns = 5
    cfg.retrieval.agent_token_budget = 4000
    cfg.llm = MagicMock()
    return cfg


class TestAgentLoopInit:
    def test_init_with_config(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        loop = AgentLoop(cfg)
        assert loop is not None


class TestAgentLoopRun:
    def test_run_returns_string(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_client.tool_call.return_value = MagicMock(
            tool_name="done",
            tool_arguments={"answer": "The answer is in auth.py"},
        )
        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = MagicMock()
        result = loop.run("how does auth work")
        assert isinstance(result, str)
        assert "auth.py" in result

    def test_run_stops_after_done_action(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        # First turn: retrieve. Second turn: done.
        mock_client.tool_call.side_effect = [
            MagicMock(tool_name="retrieve", tool_arguments={"query": "auth"}),
            MagicMock(tool_name="done", tool_arguments={"answer": "Found it in auth.py."}),
        ]
        mock_retriever = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.results = []
        mock_retriever.retrieve.return_value = mock_ctx

        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = mock_retriever
        result = loop.run("how does auth work")
        assert "Found it in auth.py" in result
        assert mock_client.tool_call.call_count == 2

    def test_run_respects_max_turns(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        cfg.retrieval.agent_max_turns = 2
        mock_client = MagicMock()
        # LLM always retrieves, never calls done
        mock_client.tool_call.return_value = MagicMock(
            tool_name="retrieve", tool_arguments={"query": "auth"}
        )
        mock_retriever = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.results = []
        mock_retriever.retrieve.return_value = mock_ctx

        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = mock_retriever
        result = loop.run("how does auth work")
        # Should stop after 2 turns and return a fallback answer
        assert isinstance(result, str)
        assert mock_client.tool_call.call_count <= 2

    def test_config_defaults(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        assert cfg.retrieval.agentic_enabled is False
        assert cfg.retrieval.agent_max_turns == 8
        assert cfg.retrieval.agent_token_budget == 6000
