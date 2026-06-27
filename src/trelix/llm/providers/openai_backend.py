"""OpenAI and Azure OpenAI backend for TrelixChatClient."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.openai_backend")

# Module-level imports so patch() can target openai_backend.OpenAI / AzureOpenAI
try:
    from openai import AzureOpenAI, OpenAI  # noqa: F401
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]
    AzureOpenAI = None  # type: ignore[assignment,misc]

# Models that require the legacy max_tokens parameter (not max_completion_tokens)
_LEGACY_MAX_TOKENS_PREFIXES = ("gpt-4-", "gpt-4 ", "gpt-3.5")
_LEGACY_MAX_TOKENS_EXACT = {"gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"}


def _token_limit_param(model: str, value: int) -> dict[str, int]:
    """Return the correct token-limit kwarg for the given model name."""
    base = model.split("/")[-1].lower().strip()
    if base in _LEGACY_MAX_TOKENS_EXACT or any(
        base.startswith(p) for p in _LEGACY_MAX_TOKENS_PREFIXES
    ):
        return {"max_tokens": value}
    return {"max_completion_tokens": value}


class OpenAIBackend(TrelixChatClient):
    """
    TrelixChatClient backed by OpenAI or Azure OpenAI.

    Handles:
    - max_tokens vs max_completion_tokens based on model family
    - Azure deployment name routing
    - Streaming via SSE
    - Tool calls via tools= + tool_choice=
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._is_azure = config.provider == "azure"
        self._model = config.azure_chat_deployment if self._is_azure else config.model
        self._client = self._build_client(config)

    def _build_client(self, config: LLMConfig) -> Any | None:
        if self._is_azure:
            if not config.azure_api_key or not config.azure_endpoint:
                logger.debug("OpenAIBackend: Azure credentials not set.")
                return None
            try:
                return AzureOpenAI(
                    api_key=config.azure_api_key,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.azure_api_version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenAIBackend: could not build AzureOpenAI: %s", exc)
                return None
        else:
            if not config.openai_api_key:
                logger.debug("OpenAIBackend: OPENAI_API_KEY not set.")
                return None
            try:
                return OpenAI(api_key=config.openai_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenAIBackend: could not build OpenAI: %s", exc)
                return None

    def _build_messages(
        self, messages: list[ChatMessage], system: str | None
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        # Inject system prompt first
        effective_system = system or next((m.content for m in messages if m.role == "system"), None)
        if effective_system:
            result.append({"role": "system", "content": effective_system})
        result.extend(
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        )
        return result

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        if self._client is None:
            return ChatResponse(
                content="[trelix] LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY.",
                model="none",
                finish_reason="stop",
            )
        token_kwarg = _token_limit_param(self._model, max_tokens or self._config.max_tokens)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, system),
            temperature=temperature if temperature is not None else self._config.temperature,
            **token_kwarg,
        )
        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> Iterator[str]:
        if self._client is None:
            yield "[trelix] LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY."
            return
        token_kwarg = _token_limit_param(self._model, max_tokens or self._config.max_tokens)
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, system),
            temperature=temperature if temperature is not None else self._config.temperature,
            stream=True,
            **token_kwarg,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolCallResponse:
        if self._client is None:
            raise RuntimeError("LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY.")
        tool_choice: Any = (
            {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
        )
        token_kwarg = _token_limit_param(self._model, max_tokens or self._config.max_tokens)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, None),
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.0,
            timeout=self._config.timeout,
            **token_kwarg,
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("LLM did not return a tool call.")
        tc = tool_calls[0]
        return ToolCallResponse(
            tool_name=tc.function.name,
            tool_arguments=json.loads(tc.function.arguments),
            raw_response=response,
        )
