"""
TrelixChatClient — provider-agnostic chat interface.

All LLM call sites in trelix use this ABC. Never import provider SDKs
(openai, anthropic, boto3, google-genai) directly in business logic.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class ImageContent:
    """A single raster image attached to a ChatMessage."""

    data: bytes
    media_type: str  # e.g. "image/png", "image/jpeg"


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str  # plain text — backends convert to provider format
    images: list[ImageContent] | None = None  # optional raster images (Anthropic-only for now)


@dataclass
class ThinkingBlock:
    """A single reasoning/thinking content block, unified across providers.

    Field names mirror litellm's ChatCompletionThinkingBlock/
    ChatCompletionRedactedThinkingBlock TypedDicts (litellm/types/llms/openai.py) for
    interop — trelix normalizes Anthropic's thinking/redacted_thinking blocks and
    Bedrock Converse's reasoningContent block (reasoningText.{text,signature} /
    redactedContent) into this same shape.

    `signature`/`data` are opaque and must be round-tripped verbatim if ever re-sent
    in a follow-up turn — Anthropic 400s on a dropped/reordered thinking block in
    multi-turn tool-use; Bedrock's signature is a tamper-evident hash of the whole
    conversation. trelix's AgentLoop does not currently round-trip raw provider content
    blocks (HistoryCompressor/TurnHistory build a compressed plain-text history
    instead), so this isn't load-bearing yet — the fields exist so a future multi-turn
    continuation feature has somewhere to put them without another schema break.
    """

    type: Literal["thinking", "redacted_thinking"]
    thinking: str | None = None  # present only when type == "thinking"
    signature: str | None = None  # present only when type == "thinking"; still nullable
    data: str | None = None  # present only when type == "redacted_thinking"


@dataclass
class ChatResponse:
    """Normalized response from any provider's chat completion."""

    content: str
    model: str
    finish_reason: str  # "stop" | "length" | "tool_calls" (normalized across providers)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking: str | None = None  # back-compat: joined text of "thinking"-type blocks only
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
    cache_read_tokens: int = 0  # Prompt cache hits (Anthropic only)
    cache_write_tokens: int = 0  # Prompt cache writes (Anthropic only)


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

    ── Sampling seed ────────────────────────────────────────────────────────────
    A backend MAY add an optional `seed: int | None = None` parameter to any of the
    three methods below, and should forward it to its provider when the provider has
    a seed concept (OpenAI/Azure `chat.completions.create(seed=...)` does). Widening
    an override with an extra optional parameter is allowed, so a backend can opt in
    on its own schedule.

    `seed` is deliberately NOT declared on these abstract signatures. Declaring it
    here would make all five shipped backends' overrides narrower than the supertype
    — five files this interface cannot fix in one step — while changing nothing about
    what actually reaches a provider. Callers must therefore build the kwarg with
    `seed_kwargs()`, which asks the bound method whether it can take one.

    A seed is also not a reproducibility guarantee: at temperature=0.0 with no seed,
    0 of 54 golden query plans reproduced byte-for-byte. The mechanism that does
    reproduce is `RetrievalConfig.plan_cache_file`.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
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
        thinking: bool = False,
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


# Log once per (module, qualname) so a per-query planner call does not emit a
# warning per query for the life of the process.
_SEED_DROP_WARNED: set[str] = set()


def seed_kwargs(method: Callable[..., Any], seed: int | None) -> dict[str, Any]:
    """Build the `seed=` kwarg for *method*, or an empty dict.

    Returns `{}` when no seed is configured, and also when *method* cannot accept
    one. The signature check is not defensive politeness: none of the backends in
    `llm/providers/` accepts a seed yet, and passing one blindly raises TypeError
    inside `QueryPlanner._plan_direct`, whose `except Exception` converts any failure
    into `default_plan()`. That collapses all eight IntentType values to FEATURE_FLOW
    and gives every query the same retrieval legs — so a configured seed would make
    retrieval WORSE while presenting as a determinism improvement. Dropping the seed
    with a warning is the honest failure, and the warning is the thing that tells a
    user their "seeded" run was not seeded.

    A seed is necessary but nowhere near sufficient for a reproducible plan: with
    temperature=0.0 and no seed, 0 of 54 golden plans reproduced byte-for-byte, and
    no provider guarantees bit-identical sampling across fleet changes. The freeze
    that actually reproduces is the plan cache — see
    `RetrievalConfig.plan_cache_file`.
    """
    if seed is None:
        return {}
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):  # builtins / C-level callables
        parameters = {}  # type: ignore[assignment]
    accepts_seed = "seed" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if accepts_seed:
        return {"seed": seed}
    qualname = getattr(method, "__qualname__", repr(method))
    key = f"{getattr(method, '__module__', '?')}.{qualname}"
    if key not in _SEED_DROP_WARNED:
        _SEED_DROP_WARNED.add(key)
        logger.warning(
            "seed=%s dropped: %s does not accept a seed, so this run is NOT seeded. "
            "Use TRELIX_RETRIEVAL_PLAN_CACHE_FILE for a reproducible plan.",
            seed,
            key,
        )
    return {}
