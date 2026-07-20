"""Tests for TurnHistory and HistoryCompressor."""

from __future__ import annotations

from trelix.agent.actions import ActionType, AgentAction, Observation, Turn
from trelix.agent.history import HistoryCompressor, TurnHistory


def _make_turn(n: int) -> Turn:
    return Turn(
        thought=f"Thought {n}",
        action=AgentAction(ActionType.RETRIEVE, {"query": f"query {n}"}),
        observation=Observation(f"Result {n} " * 50, "retrieve", True),
    )


class TestTurnHistory:
    def test_add_and_len(self) -> None:
        h = TurnHistory()
        h.add(_make_turn(1))
        h.add(_make_turn(2))
        assert len(h.turns) == 2

    def test_to_text_contains_thought(self) -> None:
        h = TurnHistory()
        h.add(_make_turn(1))
        text = h.to_text()
        assert "Thought 1" in text
        assert "retrieve" in text.lower()

    def test_token_count_positive(self) -> None:
        h = TurnHistory()
        h.add(_make_turn(1))
        assert h.token_count() > 0

    def test_empty_history_to_text(self) -> None:
        h = TurnHistory()
        assert h.to_text() == ""

    def test_to_dicts_round_trip(self) -> None:
        h = TurnHistory()
        h.add(_make_turn(1))
        h.add(_make_turn(2))

        rows = h.to_dicts()

        assert len(rows) == 2
        assert rows[0]["thought"] == "Thought 1"
        assert rows[0]["action_type"] == "retrieve"
        assert rows[0]["action_arguments"] == {"query": "query 1"}
        assert rows[0]["observation_source"] == "retrieve"
        assert rows[0]["observation_success"] is True

    def test_from_dicts_reconstructs_turns(self) -> None:
        h = TurnHistory()
        h.add(_make_turn(1))
        h.add(_make_turn(2))
        rows = h.to_dicts()

        reconstructed = TurnHistory.from_dicts(rows)

        assert len(reconstructed.turns) == 2
        assert reconstructed.turns[0].thought == "Thought 1"
        assert reconstructed.turns[0].action.action_type == ActionType.RETRIEVE
        assert reconstructed.turns[0].action.arguments == {"query": "query 1"}
        assert reconstructed.turns[1].observation.success is True
        # Round-trip: re-serializing must produce the identical rows.
        assert reconstructed.to_dicts() == rows

    def test_from_dicts_empty_list(self) -> None:
        h = TurnHistory.from_dicts([])
        assert h.turns == []


class TestHistoryCompressor:
    def test_compress_within_budget_unchanged(self) -> None:
        h = TurnHistory()
        h.add(_make_turn(1))
        compressor = HistoryCompressor(token_budget=10_000)
        compressed = compressor.compress(h)
        assert len(compressed.turns) == len(h.turns)

    def test_compress_over_budget_drops_oldest(self) -> None:
        h = TurnHistory()
        for i in range(10):
            h.add(_make_turn(i))
        # Very tight budget — should drop oldest turns
        compressor = HistoryCompressor(token_budget=200)
        compressed = compressor.compress(h)
        assert len(compressed.turns) < len(h.turns)

    def test_compress_always_keeps_last_turn(self) -> None:
        h = TurnHistory()
        for i in range(5):
            h.add(_make_turn(i))
        compressor = HistoryCompressor(token_budget=10)  # almost nothing
        compressed = compressor.compress(h)
        # Must keep at least the most recent turn
        assert len(compressed.turns) >= 1
        assert compressed.turns[-1].thought == h.turns[-1].thought
