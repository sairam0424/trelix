"""Tests for AgentLoop ReAct orchestrator."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
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


def _symbol(qualified_name: str, body: str) -> MagicMock:
    sym = MagicMock()
    sym.qualified_name = qualified_name
    sym.body = body
    return sym


class TestGetSymbolFencing:
    """_do_get_symbol wraps a symbol body in a fence the body cannot break."""

    def _observe(self, tmp_path: Path, body: str) -> str:
        cfg = _make_config(tmp_path)
        loop = AgentLoop(cfg)
        mock_db_cls = _mock_db_class()
        mock_db_cls.return_value.get_symbol_by_name.return_value = [_symbol("docs.readme", body)]
        with patch("trelix.store.db.Database", mock_db_cls):
            obs = loop._do_get_symbol("docs.readme")
        assert obs.success is True
        return str(obs.content)

    def test_plain_body_is_byte_identical_to_legacy_output(self, tmp_path: Path) -> None:
        """Guarantees the fence change is a no-op for bodies without backticks."""
        body = "def login(user):\n    return check(user)"
        assert self._observe(tmp_path, body) == f"```\n{body}\n```"

    def test_body_containing_a_fence_is_still_well_formed(self, tmp_path: Path) -> None:
        # A markdown symbol body — trelix indexes markdown, and 72 of the 79
        # tracked .md files in this repo contain a three-backtick run.
        body = "Verify a release:\n```bash\npip install pypi-attestations\n```"
        content = self._observe(tmp_path, body)
        lines = content.split("\n")
        assert lines[0] == "````"
        assert lines[-1] == "````"
        # The body must survive intact between the fences, and no line in it may
        # open a fence long enough to close the block early.
        assert content == f"````\n{body}\n````"
        for line in body.split("\n"):
            assert not line.lstrip().startswith("````")

    def test_body_with_a_long_run_grows_the_fence_further(self, tmp_path: Path) -> None:
        body = "outer\n`````\ninner\n`````"
        content = self._observe(tmp_path, body)
        assert content == f"``````\n{body}\n``````"


class TestGetSymbolAmbiguityIsReportedNotGuessed:
    """DEFECT (now fixed): `_do_get_symbol` used to fall back to an ARBITRARY
    bare-name match (`exact or symbols[:1]`) whenever the agent's requested
    qualified_name didn't exactly match any indexed symbol. If the bare last
    segment (e.g. "tag" out of "Alpha.Config.tag") matched a DIFFERENT
    symbol's name, that unrelated symbol's body was silently returned as if
    it were the one asked for -- a wrong answer with no error signal, unlike
    Indexer._resolve_symbol_match's deliberate "ambiguous -> unresolved,
    never guess" contract for the same underlying db.get_symbol_by_name()
    ambiguity.
    """

    def test_no_exact_match_reports_not_found_instead_of_a_wrong_symbol(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        loop = AgentLoop(cfg)
        mock_db_cls = _mock_db_class()
        # The agent asks for "Alpha.Config.tag", but the only bare-name match
        # ("tag") in the index is a DIFFERENT symbol, "Beta.Config.tag".
        mock_db_cls.return_value.get_symbol_by_name.return_value = [
            _symbol("Beta.Config.tag", "def tag(self): return self._beta_tag")
        ]
        with patch("trelix.store.db.Database", mock_db_cls):
            obs = loop._do_get_symbol("Alpha.Config.tag")
        assert obs.success is False
        assert "not found" in str(obs.content)
        assert "_beta_tag" not in str(obs.content), (
            "must not silently return a different symbol's body"
        )

    def test_exact_match_among_several_bare_name_candidates_still_resolves(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        loop = AgentLoop(cfg)
        mock_db_cls = _mock_db_class()
        mock_db_cls.return_value.get_symbol_by_name.return_value = [
            _symbol("Beta.Config.tag", "def tag(self): return self._beta_tag"),
            _symbol("Alpha.Config.tag", "def tag(self): return self._alpha_tag"),
        ]
        with patch("trelix.store.db.Database", mock_db_cls):
            obs = loop._do_get_symbol("Alpha.Config.tag")
        assert obs.success is True
        assert "_alpha_tag" in str(obs.content)


def _capture_tool_call_kwargs(tmp_path: Path, query: str = "how does auth work") -> dict[str, Any]:
    """Run one turn against a fake client and return the tool_call() kwargs."""
    cfg = _make_config(tmp_path)
    mock_client = MagicMock()
    mock_client.tool_call.return_value = MagicMock(tool_name="done", tool_arguments={"answer": "a"})
    loop = AgentLoop(cfg)
    loop._llm_client = mock_client
    loop._retriever = MagicMock()
    with patch("trelix.store.db.Database", _mock_db_class()):
        loop.run(query)
    assert mock_client.tool_call.call_count == 1
    return dict(mock_client.tool_call.call_args.kwargs)


class TestSystemPromptReachesModel:
    """_SYSTEM_PROMPT must actually be sent, not just defined."""

    def test_a_system_message_is_sent(self, tmp_path: Path) -> None:
        kwargs = _capture_tool_call_kwargs(tmp_path)
        # tool_call() declares no system= parameter on the ABC or any of the five
        # backends, so a role="system" message is the only route they honour.
        assert "system" not in kwargs
        assert [m.role for m in kwargs["messages"]] == ["system", "user"]

    def test_system_message_carries_the_strategy_rules(self, tmp_path: Path) -> None:
        kwargs = _capture_tool_call_kwargs(tmp_path)
        system = next(m.content for m in kwargs["messages"] if m.role == "system")
        assert "Never call done until you've done at least one retrieval." in system
        assert "You have access to four tools: retrieve, grep, get_symbol, and done." in system
        assert "Be concise in thoughts; be thorough in answers." in system

    def test_user_message_still_carries_the_question(self, tmp_path: Path) -> None:
        kwargs = _capture_tool_call_kwargs(tmp_path)
        user = next(m.content for m in kwargs["messages"] if m.role == "user")
        assert user.startswith("Question: how does auth work\n\n")
        assert user.endswith(
            "What is your next action? Think step by step, then call the appropriate tool."
        )
        # The rules belong in the system message, not smuggled into the user
        # turn — keeping the user turn byte-identical is what makes this a
        # wiring fix rather than a prompt rewrite.
        assert "Never call done" not in user
