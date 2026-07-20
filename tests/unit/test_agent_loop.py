"""Tests for AgentLoop ReAct orchestrator."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from trelix.agent.loop import AgentLoop


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.repo_path = str(tmp_path)
    cfg.retrieval.agentic_enabled = True
    cfg.retrieval.agent_max_turns = 5
    cfg.retrieval.agent_token_budget = 4000
    cfg.retrieval.agent_session_max_age_seconds = 604_800.0
    cfg.llm = MagicMock()
    return cfg


def _mock_db_class() -> MagicMock:
    """A Database class mock whose instances never raise and never load prior turns."""
    mock_db = MagicMock()
    mock_db.get_agent_turns.return_value = []
    mock_cls = MagicMock(return_value=mock_db)
    return mock_cls


class TestAgentLoopInit:
    def test_init_with_config(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        loop = AgentLoop(cfg)
        assert loop is not None


class TestAgentLoopRun:
    def test_run_returns_tuple(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_client.tool_call.return_value = MagicMock(
            tool_name="done",
            tool_arguments={"answer": "The answer is in auth.py"},
        )
        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = MagicMock()

        with patch("trelix.store.db.Database", _mock_db_class()):
            result, session_id = loop.run("how does auth work")

        assert isinstance(result, str)
        assert "auth.py" in result
        assert isinstance(session_id, str)

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

        with patch("trelix.store.db.Database", _mock_db_class()):
            result, _session_id = loop.run("how does auth work")

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

        with patch("trelix.store.db.Database", _mock_db_class()):
            result, _session_id = loop.run("how does auth work")

        # Should stop after 2 turns and return a fallback answer
        assert isinstance(result, str)
        assert mock_client.tool_call.call_count <= 2

    def test_run_without_session_id_generates_uuid(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_client.tool_call.return_value = MagicMock(
            tool_name="done", tool_arguments={"answer": "done"}
        )
        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = MagicMock()

        with patch("trelix.store.db.Database", _mock_db_class()):
            _answer, session_id = loop.run("q")

        # Must not raise — a valid UUID4 string
        uuid.UUID(session_id)

    def test_run_with_session_id_loads_prior_turns(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_client.tool_call.return_value = MagicMock(
            tool_name="done", tool_arguments={"answer": "done"}
        )
        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = MagicMock()

        prior_row = {
            "turn_index": 0,
            "thought": "earlier thought",
            "action_type": "retrieve",
            "action_arguments": {"query": "auth"},
            "observation_content": "earlier observation",
            "observation_source": "retrieve",
            "observation_success": True,
        }
        mock_db_cls = _mock_db_class()
        mock_db_cls.return_value.get_agent_turns.return_value = [prior_row]

        with patch("trelix.store.db.Database", mock_db_cls):
            _answer, session_id = loop.run("follow-up question", session_id="existing-session")

        assert session_id == "existing-session"
        mock_db_cls.return_value.get_agent_turns.assert_called_with("existing-session")

    def test_run_persists_each_turn(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_client.tool_call.side_effect = [
            MagicMock(tool_name="retrieve", tool_arguments={"query": "auth"}),
            MagicMock(tool_name="done", tool_arguments={"answer": "done"}),
        ]
        mock_retriever = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.results = []
        mock_retriever.retrieve.return_value = mock_ctx

        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = mock_retriever

        mock_db_cls = _mock_db_class()
        with patch("trelix.store.db.Database", mock_db_cls):
            _answer, session_id = loop.run("how does auth work")

        calls = mock_db_cls.return_value.insert_agent_turn.call_args_list
        assert len(calls) == 2
        for call in calls:
            assert call.kwargs["session_id"] == session_id
            # turn_index must NOT be passed by the caller — Database assigns
            # it atomically via MAX(turn_index)+1 (regression guard for the
            # collision bug found in pre-push audit).
            assert "turn_index" not in call.kwargs

    def test_run_persist_failure_does_not_crash_loop(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_client.tool_call.return_value = MagicMock(
            tool_name="done", tool_arguments={"answer": "still works"}
        )
        loop = AgentLoop(cfg)
        loop._llm_client = mock_client
        loop._retriever = MagicMock()

        mock_db_cls = _mock_db_class()
        mock_db_cls.return_value.insert_agent_turn.side_effect = Exception("db exploded")

        with patch("trelix.store.db.Database", mock_db_cls):
            answer, session_id = loop.run("q")

        assert answer == "still works"
        assert isinstance(session_id, str)

    def test_config_defaults(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        assert cfg.retrieval.agentic_enabled is False
        assert cfg.retrieval.agent_max_turns == 8
        assert cfg.retrieval.agent_token_budget == 6000
        assert cfg.retrieval.agent_session_max_age_seconds == 604_800.0
