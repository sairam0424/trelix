"""
TrelixChatClient — provider-agnostic chat interface.

All LLM call sites in trelix use this ABC. Never import provider SDKs
(openai, anthropic, boto3, google-genai) directly in business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str  # always plain text — backends convert to provider format


@dataclass
class ChatResponse:
    """Normalized response from any provider's chat completion."""

    content: str
    model: str
    finish_reason: str  # "stop" | "length" | "tool_calls" (normalized across providers)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCallResponse:
    """Normalized tool/function call result from any provider."""

    tool_name: str
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None  # provider-specific, for debugging


class TrelixChatClient(ABC):
    """
    Provider-agnostic interface for chat completions.

    Implement one backend per provider; call sites never touch SDKs directly.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        """Non-streaming chat completion. Returns full response."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> Iterator[str]:
        """Streaming chat completion. Yields text chunks as they arrive."""
        ...

    @abstractmethod
    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolCallResponse:
        """
        Forced tool/function call. tools uses OpenAI schema format.
        force_tool: name of the tool to force (None = auto-select).
        """
        ...
