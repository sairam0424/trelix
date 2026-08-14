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


class TestObservationFencing:
    """`to_text()` must keep a fenced observation inside its own code block.

    `get_symbol` hands back an observation already wrapped by
    `trelix.llm.prompt.fenced_block()`. Two independent defects used to unwrap it
    again, and each is invisible to a test that only checks the payload appears
    somewhere in the output:

      1. The label and content shared a line, so the opening fence was mid-line.
         CommonMark only opens a fenced block at the START of a line, so the
         payload reached the model as prose and the CLOSING fence opened an empty
         block instead. This is structural — it happened for a payload with no
         backticks at all, which is why no amount of fence-length derivation in
         loop.py could fix it.
      2. The rendered block was sliced at a fixed character limit, cutting the
         closing fence off any longer body, so the block never closed and every
         later turn was swallowed into it.

    These assert on STRUCTURE via a real CommonMark parser rather than on
    substring presence, because the payload was always present — just not quoted.
    """

    @staticmethod
    def _symbol_turn(body: str) -> Turn:
        from trelix.llm.prompt import fenced_block

        return Turn(
            thought="Looking up the symbol",
            action=AgentAction(ActionType.GET_SYMBOL, {"qualified_name": "mod.fn"}),
            observation=Observation(fenced_block(body), "get_symbol", True),
        )

    @staticmethod
    def _fences(text: str) -> list[str]:
        """Fenced-block contents, per a real CommonMark parser."""
        import pytest

        md = pytest.importorskip(
            "markdown_it", reason="markdown-it-py needed to assert CommonMark structure"
        ).MarkdownIt()
        return [t.content for t in md.parse(text) if t.type == "fence"]

    def test_fence_opens_at_line_start(self) -> None:
        """The opening fence must begin a line, or it opens nothing."""
        h = TurnHistory()
        h.add(self._symbol_turn("def f():\n    return 1"))
        text = h.to_text()

        assert "**Observation [ok]:**\n```" in text, (
            "the label and the fence share a line, so the fence never opens a block"
        )
        assert any("def f():" in c for c in self._fences(text))

    def test_payload_carrying_its_own_fence_stays_quoted(self) -> None:
        """A markdown body containing ``` must not escape its block."""
        body = "Install it:\n```bash\npip install trelix\n```\nThen run it."
        h = TurnHistory()
        h.add(self._symbol_turn(body))

        fences = self._fences(h.to_text())
        assert any("pip install trelix" in c for c in fences)
        assert any("Then run it." in c for c in fences), (
            "the payload's own fence closed the block early and the tail became prose"
        )

    def test_truncated_observation_still_closes_its_fence(self) -> None:
        """A body past the char limit must not leave the block hanging open."""
        h = TurnHistory()
        h.add(self._symbol_turn("x = 1\n" * 150))  # ~900 chars, well past the limit
        text = h.to_text()

        assert text.rstrip().endswith("`"), "truncation sliced the closing fence off"
        assert any("x = 1" in c for c in self._fences(text))

    def test_long_first_turn_does_not_swallow_the_next(self) -> None:
        """The regression that made truncation expensive rather than merely untidy."""
        h = TurnHistory()
        h.add(self._symbol_turn("x = 1\n" * 150))
        h.add(self._symbol_turn("def later() -> None: ..."))
        text = h.to_text()

        fences = self._fences(text)
        assert len(fences) == 2, f"expected one block per turn, got {len(fences)}"
        assert any("def later()" in c for c in fences)
        assert not any("## Turn 2" in c for c in fences), (
            "turn 1's unclosed block swallowed turn 2's heading"
        )

    def test_plain_observation_is_untouched(self) -> None:
        """Non-fenced observations must render exactly as before."""
        h = TurnHistory()
        h.add(
            Turn(
                thought="t",
                action=AgentAction(ActionType.RETRIEVE, {"query": "q"}),
                observation=Observation("3 results found", "retrieve", True),
            )
        )
        assert "**Observation [ok]:**\n3 results found" in h.to_text()


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
