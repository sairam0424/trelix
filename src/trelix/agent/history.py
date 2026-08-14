"""Turn history management and compression for the ReAct agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trelix.agent.actions import Turn

_OBSERVATION_CHAR_LIMIT = 500


def _truncate_preserving_fence(content: str, limit: int) -> str:
    """Truncate `content` without orphaning a code fence it opened.

    `get_symbol` returns its observation already wrapped by
    `trelix.llm.prompt.fenced_block()`, so slicing the rendered string at a
    fixed length cuts the CLOSING fence off any body longer than the limit. The
    block then never closes, and every later turn in the history is swallowed
    into it as if it were quoted code.

    When truncation cuts a block that opened with a bare fence line, the fence
    is re-appended. That overshoots `limit` by the fence's length plus a
    newline, which is the correct trade: a handful of characters over budget
    against a prompt whose remaining turns are all inside a code block.
    """
    if len(content) <= limit:
        return content
    truncated = content[:limit]
    opening = content.split("\n", 1)[0]
    # Only a bare run of backticks opens a block that we are responsible for
    # closing; anything else (prose, an error string) needs no repair.
    if opening and set(opening) == {"`"} and not truncated.rstrip().endswith(opening):
        truncated = truncated.rstrip("\n") + "\n" + opening
    return truncated


@dataclass
class TurnHistory:
    """Ordered sequence of ReAct turns for the current session."""

    turns: list[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all turns to plain dicts, for DB persistence.

        Field names match Database.insert_agent_turn()'s parameters so the
        persistence boundary needs no adapter.
        """
        return [
            {
                "thought": t.thought,
                "action_type": t.action.action_type.value,
                "action_arguments": t.action.arguments,
                "observation_content": t.observation.content,
                "observation_source": t.observation.source,
                "observation_success": t.observation.success,
            }
            for t in self.turns
        ]

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]]) -> TurnHistory:
        """Reconstruct a TurnHistory from rows shaped like Database.get_agent_turns()."""
        from trelix.agent.actions import ActionType, AgentAction, Observation, Turn

        turns = [
            Turn(
                thought=row["thought"],
                action=AgentAction(
                    action_type=ActionType(row["action_type"]),
                    arguments=row["action_arguments"],
                ),
                observation=Observation(
                    content=row["observation_content"],
                    source=row["observation_source"],
                    success=row["observation_success"],
                ),
            )
            for row in rows
        ]
        return cls(turns=turns)

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
            # The label goes on its own line and the content starts the next one.
            # A CommonMark fence only opens a block at the START of a line, so
            # "**Observation [ok]:** ```" opened nothing: the payload reached the
            # model as prose and the *closing* fence opened a block instead.
            # Confirmed against markdown-it-py — the payload sat inside no fence
            # at 3, 4 or 11 backticks, and for a payload with no backticks at all.
            # That is why deriving a longer fence in loop.py cannot help here; the
            # defect is structural, not payload-dependent.
            lines.append(f"**Observation [{status}]:**")
            lines.append(
                _truncate_preserving_fence(turn.observation.content, _OBSERVATION_CHAR_LIMIT)
            )
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
