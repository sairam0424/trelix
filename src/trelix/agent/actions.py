"""Agent action types, observation model, and turn record."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActionType(StrEnum):
    RETRIEVE = "retrieve"
    GREP = "grep"
    GET_SYMBOL = "get_symbol"
    DONE = "done"
    CLARIFY = "clarify"


@dataclass
class AgentAction:
    """A single action dispatched by the ReAct loop."""

    action_type: ActionType
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """The result of executing an AgentAction."""

    content: str  # text content returned to the LLM
    source: str  # which action produced this (e.g. "retrieve", "grep")
    success: bool  # False if the action errored or returned no results


@dataclass
class Turn:
    """One complete ReAct step: thought -> action -> observation."""

    thought: str
    action: AgentAction
    observation: Observation


@dataclass
class AgentResult:
    """Terminal result of AgentLoop.run().

    Either a completed answer, or (SEP-2322 InputRequiredResult) an explicit
    request for more input from the caller before the loop can continue.
    `content` carries the answer text or the clarifying question depending
    on `needs_input` — a max-turns-exhausted fallback answer is a distinct,
    separately reachable state from an explicit clarify (both set
    needs_input=False; only an explicit `clarify` action sets it True).
    """

    content: str
    session_id: str
    needs_input: bool = False
