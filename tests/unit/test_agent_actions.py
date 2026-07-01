"""Tests for AgentAction, ActionType, Observation, Turn dataclasses."""
from __future__ import annotations

from trelix.agent.actions import ActionType, AgentAction, Observation, Turn


class TestActionType:
    def test_all_four_types_exist(self) -> None:
        assert ActionType.RETRIEVE == "retrieve"
        assert ActionType.GREP == "grep"
        assert ActionType.GET_SYMBOL == "get_symbol"
        assert ActionType.DONE == "done"


class TestAgentAction:
    def test_retrieve_action(self) -> None:
        action = AgentAction(
            action_type=ActionType.RETRIEVE,
            arguments={"query": "how does auth work"},
        )
        assert action.action_type == ActionType.RETRIEVE
        assert action.arguments["query"] == "how does auth work"

    def test_done_action(self) -> None:
        action = AgentAction(
            action_type=ActionType.DONE,
            arguments={"answer": "The login() function handles auth."},
        )
        assert action.action_type == ActionType.DONE


class TestObservation:
    def test_success_observation(self) -> None:
        obs = Observation(content="Found auth.py", source="retrieve", success=True)
        assert obs.success is True
        assert obs.source == "retrieve"

    def test_failure_observation(self) -> None:
        obs = Observation(content="Symbol not found", source="get_symbol", success=False)
        assert obs.success is False


class TestTurn:
    def test_turn_stores_all_fields(self) -> None:
        action = AgentAction(ActionType.RETRIEVE, {"query": "auth"})
        obs = Observation("Found it", "retrieve", True)
        turn = Turn(thought="I need to find auth code", action=action, observation=obs)
        assert turn.thought == "I need to find auth code"
        assert turn.action.action_type == ActionType.RETRIEVE
        assert turn.observation.success is True
