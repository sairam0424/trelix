"""OpenAI and Azure OpenAI backend for TrelixChatClient."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from trelix.core.retry import with_retry
from trelix.llm.client import (
    UNCONFIGURED_MODEL,
    ChatMessage,
    ChatResponse,
    ToolCallResponse,
    TrelixChatClient,
)
from trelix.llm.finish_reasons import classify_chat_choice
from trelix.llm.offline import LOCAL_PLACEHOLDER_KEY, small_model_warning

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


def _token_limit_param(model: str, value: int, *, local_server: bool = False) -> dict[str, int]:
    """Return the correct token-limit kwarg for the given model name.

    A local OpenAI-compatible server (TRELIX_LLM_BASE_URL) gets `max_tokens` whatever the
    model is called: Ollama has no `max_completion_tokens` field and ran unbounded when that
    was the only limit sent.
    """
    if local_server:
        return {"max_tokens": value}
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
        # Only the plain openai path reads TRELIX_LLM_BASE_URL; Azure has azure_endpoint.
        self._local_server = config.base_url is not None and not self._is_azure
        self._client = self._build_client(config)
        if self._local_server:
            warning = small_model_warning(self._model)
            if warning:
                logger.warning(warning)

    def _token_limit(self, max_tokens: int | None) -> dict[str, int]:
        return _token_limit_param(
            self._model, max_tokens or self._config.max_tokens, local_server=self._local_server
        )

    def _build_client(self, config: LLMConfig) -> Any | None:
        # max_retries=0: the SDK's own default retry (2 attempts) would
        # otherwise stack underneath @with_retry's 5-attempt tenacity layer
        # (each of tenacity's attempts silently absorbing up to 2 SDK-level
        # retries with the SDK's own independent backoff), multiplying
        # worst-case wall-clock time on a persistent outage far beyond what
        # max_attempts=5 implies. trelix's shared retry contract is meant to
        # be the sole retry layer.
        if self._is_azure:
            if not config.azure_api_key or not config.azure_endpoint:
                logger.debug("OpenAIBackend: Azure credentials not set.")
                return None
            try:
                return AzureOpenAI(
                    api_key=config.azure_api_key,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.azure_api_version,
                    max_retries=0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenAIBackend: could not build AzureOpenAI: %s", exc)
                return None
        else:
            if self._local_server:
                # The SDK refuses to build without a key and Ollama ignores whatever is
                # sent, so a configured key wins (llama-server --api-key, a gateway) and the
                # public placeholder stands in otherwise. No try/except: with the URL set, a
                # constructor failure propagates. That is loud for `search` and `ask` (the
                # planner and the synthesizer let it through) but not for `review`, whose
                # DiffReviewer._get_client swallows it into "LLM not configured" (exit 3), so
                # LLMConfig's validator refuses up front the shapes known to fail here (a bad
                # port, whitespace, control characters) rather than relying on this path.
                return OpenAI(
                    api_key=config.openai_api_key or LOCAL_PLACEHOLDER_KEY,
                    base_url=config.base_url,
                    max_retries=0,
                )
            if not config.openai_api_key:
                logger.debug("OpenAIBackend: OPENAI_API_KEY not set.")
                return None
            try:
                return OpenAI(api_key=config.openai_api_key, max_retries=0)
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenAIBackend: could not build OpenAI: %s", exc)
                return None

    def _build_messages(
        self, messages: list[ChatMessage], system: str | None
    ) -> list[dict[str, str]]:
        if any(m.images for m in messages):
            raise NotImplementedError(
                f"vision not yet supported for provider {self._config.provider}"
            )
        result: list[dict[str, str]] = []
        # Inject system prompt first
        effective_system = system or next((m.content for m in messages if m.role == "system"), None)
        if effective_system:
            result.append({"role": "system", "content": effective_system})
        result.extend(
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        )
        return result

    @with_retry(max_attempts=5)
    def _create(self, **kwargs: Any) -> Any:
        # Callers only reach here after their own `self._client is None` check.
        assert self._client is not None
        return self._client.chat.completions.create(**kwargs)

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> ChatResponse:
        if self._client is None:
            return ChatResponse(
                content="[trelix] LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY.",
                model=UNCONFIGURED_MODEL,
                finish_reason="stop",
            )
        token_kwarg = self._token_limit(max_tokens)
        response = self._create(
            model=self._model,
            messages=self._build_messages(messages, system),
            temperature=temperature if temperature is not None else self._config.temperature,
            **token_kwarg,
        )
        choice = response.choices[0]
        finish = classify_chat_choice(choice)
        if finish.signals:
            logger.warning(
                "LLM reply carries %s; the provider's content filter did not run on it",
                ", ".join(finish.signals),
            )
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=finish.finish_reason,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            raw_finish_reason=finish.raw_finish_reason,
            refusal=finish.refusal,
            signals=list(finish.signals),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> Iterator[str]:
        if self._client is None:
            yield "[trelix] LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY."
            return
        token_kwarg = self._token_limit(max_tokens)
        # Retry only the call that opens the stream — the connection is made
        # synchronously here (create() blocks until headers arrive), before
        # any chunk is yielded. Retrying mid-iteration over an already-open
        # generator would silently drop or duplicate chunks already consumed.
        stream = self._create(
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
        token_kwarg = self._token_limit(max_tokens)
        response = self._create(
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
