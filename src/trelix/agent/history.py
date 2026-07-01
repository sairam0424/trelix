"""Turn history management and compression for the ReAct agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from trelix.agent.actions import Turn


@dataclass
class TurnHistory:
    """Ordered sequence of ReAct turns for the current session."""

    turns: list[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def to_text(self) -> str:
        """Format turns as a numbered conversation for the LLM context."""
        if not self.turns:
            return ""
        lines: list[str] = []
        for i, turn in enumerate(self.turns, start=1):
            lines.append(f"## Turn {i}")
            lines.append(f"**Thought:** {turn.thought}")
            lines.append(f"**Action:** {turn.action.action_type} {turn.action.arguments}")
            status = "ok" if turn.observation.success else "err"
            lines.append(f"**Observation [{status}]:** {turn.observation.content[:500]}")
            lines.append("")
        return "\n".join(lines)

    def token_count(self) -> int:
        """Approximate token count using word-split heuristic (fast, no tiktoken needed)."""
        text = self.to_text()
        return len(text.split())


class HistoryCompressor:
    """
    Trims the oldest turns from history to keep the context within a token budget.

    Strategy: always keep the most recent turn; drop oldest turns one by one
    until token_count() fits within the budget.
    """

    def __init__(self, token_budget: int = 4_000) -> None:
        self._budget = token_budget

    def compress(self, history: TurnHistory) -> TurnHistory:
        """Return a new TurnHistory with oldest turns dropped to fit within budget."""
        if not history.turns:
            return TurnHistory()

        compressed = TurnHistory(turns=list(history.turns))
        # Always keep the last turn; drop from the front until under budget
        while len(compressed.turns) > 1 and compressed.token_count() > self._budget:
            compressed.turns.pop(0)

        return compressed
