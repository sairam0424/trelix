"""Anthropic Claude backend for TrelixChatClient."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.anthropic_backend")

_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class AnthropicBackend(TrelixChatClient):
    """
    TrelixChatClient backed by Anthropic Claude.

    Key differences from OpenAI:
    - max_tokens= (not max_completion_tokens)
    - system= as a separate top-level parameter (not in messages)
    - Tool schema uses input_schema instead of parameters
    - finish_reason: "end_turn" normalized to "stop"
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._model = config.model
        self._client = self._build_client(config)

    def _build_client(self, config: LLMConfig) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requires the anthropic package. "
                "Install it with: pip install 'trelix[anthropic]'"
            ) from exc
        if not config.anthropic_api_key:
            logger.debug("AnthropicBackend: ANTHROPIC_API_KEY not set.")
            return None
        return anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _extract_system(
        self, messages: list[ChatMessage], system: str | None
    ) -> tuple[str | None, list[dict[str, str]]]:
        effective = system or next((m.content for m in messages if m.role == "system"), None)
        user_msgs = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        return effective, user_msgs

    def _normalize_finish_reason(self, stop_reason: str) -> str:
        return _FINISH_REASON_MAP.get(stop_reason, "stop")

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        if self._client is None:
            return ChatResponse(
                content="[trelix] Anthropic not configured — set ANTHROPIC_API_KEY.",
                model="none",
                finish_reason="stop",
            )
        sys_prompt, user_msgs = self._extract_system(messages, system)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        response = self._client.messages.create(
            model=self._model,
            messages=user_msgs,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            **kwargs,
        )
        content = response.content[0].text if response.content else ""
        return ChatResponse(
            content=content,
            model=response.model,
            finish_reason=self._normalize_finish_reason(response.stop_reason or "end_turn"),
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> Iterator[str]:
        if self._client is None:
            yield "[trelix] Anthropic not configured — set ANTHROPIC_API_KEY."
            return
        sys_prompt, user_msgs = self._extract_system(messages, system)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        with self._client.messages.stream(
            model=self._model,
            messages=user_msgs,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            **kwargs,
        ) as stream:
            yield from stream.text_stream

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolCallResponse:
        if self._client is None:
            raise RuntimeError("Anthropic not configured — set ANTHROPIC_API_KEY.")
        # Convert OpenAI tool schema to Anthropic format
        anthropic_tools = [self._convert_tool(t) for t in tools]
        tool_choice: dict[str, Any] = (
            {"type": "tool", "name": force_tool} if force_tool else {"type": "auto"}
        )
        sys_prompt, user_msgs = self._extract_system(messages, None)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        response = self._client.messages.create(
            model=self._model,
            messages=user_msgs,
            tools=anthropic_tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens or self._config.max_tokens,
            **kwargs,
        )
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if not tool_use:
            raise RuntimeError("Anthropic did not return a tool_use block.")
        return ToolCallResponse(
            tool_name=tool_use.name,
            tool_arguments=dict(tool_use.input),
            raw_response=response,
        )

    def _convert_tool(self, openai_tool: dict[str, Any]) -> dict[str, Any]:
        fn = openai_tool["function"]
        return {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        }
