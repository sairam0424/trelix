"""Anthropic Claude backend for TrelixChatClient."""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from trelix.core.retry import with_retry
from trelix.llm.client import (
    UNCONFIGURED_MODEL,
    ChatMessage,
    ChatResponse,
    ThinkingBlock,
    ToolCallResponse,
    TrelixChatClient,
)
from trelix.llm.finish_reasons import ANTHROPIC_STOP_REASONS, normalise
from trelix.llm.thinking import ThinkingModeMemory, is_enabled_thinking_rejection

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.anthropic_backend")

# Log once per process, not once per call — a per-query retrieval/synthesis path would
# otherwise emit this on every single request once a caller starts passing temperature=.
_TEMPERATURE_DROP_WARNED = False


class AnthropicBackend(TrelixChatClient):
    """
    TrelixChatClient backed by Anthropic Claude.

    Key differences from OpenAI:
    - max_tokens= (not max_completion_tokens)
    - system= as a separate top-level parameter (not in messages)
    - Tool schema uses input_schema instead of parameters
    - finish_reason: "end_turn" normalized to "stop"; "refusal", "pause_turn" and
      "model_context_window_exceeded" are HTTP 200 responses and are reported as such
      ("refusal", "paused", "length"), never as "stop"
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._model = config.model
        # Which thinking shape this model takes: classified from the model id, then corrected
        # at runtime for an id the classifier does not know yet (see
        # _call_with_thinking_fallback and trelix.llm.thinking).
        self._thinking_modes = ThinkingModeMemory()
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
        # max_retries=0: the shared @with_retry contract (via _create()/
        # _open_stream() below) is meant to be the sole retry layer — the
        # SDK's own default (2 attempts) would otherwise stack underneath
        # tenacity's 5-attempt loop, multiplying worst-case wall-clock time
        # on a persistent outage far beyond what max_attempts=5 implies.
        return anthropic.Anthropic(api_key=config.anthropic_api_key, max_retries=0)

    def _build_message_content(self, message: ChatMessage) -> str | list[dict[str, Any]]:
        """Build the Anthropic `content` value for a single message.

        Returns `message.content` unchanged (a plain string) when `images` is
        None or empty — the common case, preserved byte-for-byte. When
        `images` has at least one entry, returns a list of content blocks:
        one `{"type": "image", ...}` block per ImageContent, followed by a
        trailing `{"type": "text", ...}` block carrying `message.content` —
        but only when `message.content` is non-empty, since Anthropic's
        Messages API rejects a text block with an empty string.
        """
        if not message.images:
            return message.content
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": base64.b64encode(image.data).decode(),
                },
            }
            for image in message.images
        ]
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        return blocks

    def _extract_system(
        self, messages: list[ChatMessage], system: str | None
    ) -> tuple[str | None, list[dict[str, Any]]]:
        effective = system or next((m.content for m in messages if m.role == "system"), None)
        user_msgs = [
            {"role": m.role, "content": self._build_message_content(m)}
            for m in messages
            if m.role != "system"
        ]
        return effective, user_msgs

    def _normalize_finish_reason(self, stop_reason: object) -> str:
        return normalise(ANTHROPIC_STOP_REASONS, stop_reason)

    def _warn_if_temperature_given(self, temperature: float | None) -> None:
        """anthropic-sdk-python v1.0.0 removed temperature from Messages.create.

        Warn once per process rather than raising or silently swallowing —
        callers that still pass temperature= get told their value is now
        ignored, instead of either crashing or silently changing behavior."""
        global _TEMPERATURE_DROP_WARNED
        if temperature is not None and not _TEMPERATURE_DROP_WARNED:
            _TEMPERATURE_DROP_WARNED = True
            logger.warning(
                "temperature=%s ignored: anthropic-sdk-python v1.0.0 removed the "
                "temperature parameter from Messages.create. AnthropicBackend no "
                "longer accepts it.",
                temperature,
            )

    def _thinking_kwargs(self, thinking: bool) -> dict[str, Any]:
        """
        Build extended thinking kwargs for Anthropic API.
        Returns empty dict when thinking is disabled or not supported.

        Models that reject a token budget (Claude 5 and newer, Opus 4.7 and newer) get
        ``{"type": "adaptive"}``: the model decides whether to think, so a response may
        carry no thinking block, and ``thinking_budget_tokens`` is not sent.
        """
        if not thinking:
            return {}
        if self._thinking_modes.mode_for(self._model) == "adaptive":
            return {"thinking": {"type": "adaptive"}}
        # Extended thinking is incompatible with a forced tool_choice
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self._config.thinking_budget_tokens,
            }
        }

    def _rejects_enabled_thinking(self, exc: Exception, thinking_param: object) -> bool:
        """True when *exc* is the API's 400 for sending ``thinking.type.enabled`` to a model
        that only takes adaptive thinking. The SDK raises ``anthropic.BadRequestError`` for a
        400; this reads the exception's ``status_code`` instead of importing that class, so
        the check needs no SDK import and holds for whatever SDK module is loaded."""
        if not isinstance(thinking_param, dict) or thinking_param.get("type") != "enabled":
            return False
        return getattr(exc, "status_code", None) == 400 and is_enabled_thinking_rejection(str(exc))

    def _call_with_thinking_fallback(self, call: Callable[..., Any], **kwargs: Any) -> Any:
        """Run ``call(**kwargs)``. If the API rejects the budget thinking shape for this
        model (an id the classifier did not recognise), switch this model to adaptive
        thinking, retry exactly once, and remember the switch.

        *call* is `_create` or `_open_stream`, so transient failures are still retried by
        their ``@with_retry`` (the sole retry layer). This adds one bounded re-send, not a
        loop: the retry is made outside the ``except`` block, carries ``adaptive`` (so the
        rejection cannot match it again) and its own errors propagate untouched. Every
        other error, and a 400 for any other reason, is re-raised.
        """
        try:
            return call(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- re-raised below unless it is the one rejection
            if not self._rejects_enabled_thinking(exc, kwargs.get("thinking")):
                raise
        if self._thinking_modes.remember_adaptive(self._model):
            logger.warning(
                "Anthropic model %r rejected budget thinking; using adaptive thinking "
                "for it from now on.",
                self._model,
            )
        return call(**{**kwargs, "thinking": {"type": "adaptive"}})

    def _split_content(self, content_blocks: list[Any]) -> tuple[str, list[ThinkingBlock]]:
        """
        Split Anthropic response content into (text, thinking_blocks).

        Handles both "thinking" blocks (readable text + signature) and
        "redacted_thinking" blocks (opaque data, no readable text) — the latter
        used to be silently dropped here, since this method only ever branched on
        "text"/"thinking".
        """
        if not content_blocks:
            return "", []
        text_parts: list[str] = []
        thinking_blocks: list[ThinkingBlock] = []
        for block in content_blocks:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_blocks.append(
                    ThinkingBlock(
                        type="thinking",
                        thinking=block.thinking,
                        signature=getattr(block, "signature", None),
                    )
                )
            elif block.type == "redacted_thinking":
                thinking_blocks.append(
                    ThinkingBlock(type="redacted_thinking", data=getattr(block, "data", None))
                )
        return "".join(text_parts), thinking_blocks

    @with_retry(max_attempts=5)
    def _create(self, **kwargs: Any) -> Any:
        assert self._client is not None
        return self._client.messages.create(**kwargs)

    @with_retry(max_attempts=5)
    def _open_stream(self, **kwargs: Any) -> tuple[Any, Any]:
        # __enter__ (not .stream() itself) makes the actual HTTP request —
        # retrying here, before any chunk is yielded, is what keeps this
        # safe: no partially-consumed generator is ever retried. The manager
        # is returned alongside the stream so the caller can still __exit__
        # it for cleanup once iteration finishes.
        assert self._client is not None
        manager = self._client.messages.stream(**kwargs)
        return manager, manager.__enter__()

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
                content="[trelix] Anthropic not configured — set ANTHROPIC_API_KEY.",
                model=UNCONFIGURED_MODEL,
                finish_reason="stop",
            )
        self._warn_if_temperature_given(temperature)
        sys_prompt, user_msgs = self._extract_system(messages, system)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        kwargs.update(self._thinking_kwargs(thinking))
        response = self._call_with_thinking_fallback(
            self._create,
            model=self._model,
            messages=user_msgs,
            max_tokens=max_tokens or self._config.max_tokens,
            **kwargs,
        )
        content, thinking_blocks = self._split_content(response.content)
        thinking_text_blocks = [b for b in thinking_blocks if b.type == "thinking"]
        thinking_text = (
            "".join(b.thinking or "" for b in thinking_text_blocks)
            if thinking_text_blocks
            else None
        )
        stop_reason = response.stop_reason
        return ChatResponse(
            content=content,
            model=response.model,
            finish_reason=self._normalize_finish_reason(stop_reason),
            raw_finish_reason=stop_reason if isinstance(stop_reason, str) else None,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            thinking=thinking_text,
            thinking_blocks=thinking_blocks,
            cache_read_tokens=response.usage.cache_read_input_tokens
            if hasattr(response.usage, "cache_read_input_tokens")
            else 0,
            cache_write_tokens=response.usage.cache_creation_input_tokens
            if hasattr(response.usage, "cache_creation_input_tokens")
            else 0,
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
            yield "[trelix] Anthropic not configured — set ANTHROPIC_API_KEY."
            return
        self._warn_if_temperature_given(temperature)
        sys_prompt, user_msgs = self._extract_system(messages, system)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        kwargs.update(self._thinking_kwargs(thinking))
        manager, stream = self._call_with_thinking_fallback(
            self._open_stream,
            model=self._model,
            messages=user_msgs,
            max_tokens=max_tokens or self._config.max_tokens,
            **kwargs,
        )
        try:
            yield from stream.text_stream
        finally:
            manager.__exit__(None, None, None)

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
        response = self._create(
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
