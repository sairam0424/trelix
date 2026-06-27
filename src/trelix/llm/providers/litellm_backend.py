"""LiteLLM universal backend for TrelixChatClient (100+ providers)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.litellm_backend")


class LiteLLMBackend(TrelixChatClient):
    """
    TrelixChatClient backed by LiteLLM — delegates to 100+ providers.

    Model string format: "bedrock/claude-3-5-sonnet", "gemini/gemini-2.0-flash",
    "anthropic/claude-3-5-haiku", "ollama/llama3", etc.

    LiteLLM handles all parameter normalization automatically.
    drop_params=True suppresses UnsupportedParamsError for provider-unsupported params.
    """

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        try:
            import litellm as _litellm
            _litellm.drop_params = config.litellm_drop_params
            self._litellm = _litellm
        except ImportError as exc:
            raise ImportError(
                "LiteLLM backend requires litellm. "
                "Install it with: pip install 'trelix[litellm]'"
            ) from exc
        self._model = config.litellm_model or config.model

    def _build_messages(
        self, messages: list[ChatMessage], system: Optional[str]
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        effective_system = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        if effective_system:
            result.append({"role": "system", "content": effective_system})
        result.extend(
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        )
        return result

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        response = self._litellm.completion(
            model=self._model,
            messages=self._build_messages(messages, system),
            max_completion_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
        )
        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model or self._model,
            finish_reason=choice.finish_reason or "stop",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        response = self._litellm.completion(
            model=self._model,
            messages=self._build_messages(messages, system),
            max_completion_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        import json
        tool_choice: Any = (
            {"type": "function", "function": {"name": force_tool}}
            if force_tool else "auto"
        )
        response = self._litellm.completion(
            model=self._model,
            messages=self._build_messages(messages, None),
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens or self._config.max_tokens,
            temperature=0.0,
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("LiteLLM did not return a tool call.")
        tc = tool_calls[0]
        return ToolCallResponse(
            tool_name=tc.function.name,
            tool_arguments=json.loads(tc.function.arguments),
            raw_response=response,
        )
