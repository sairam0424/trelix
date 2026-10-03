"""Why a chat completion ended, in one vocabulary for every backend.

Each provider reports the end of generation in its own words, and several of them report a
truncated, refused or filtered reply as an ordinary successful response (HTTP 200) whose only
tell is this field. A caller that reads "the request did not raise" as "the model finished"
turns a cut-off JSON array or a refusal into a clean result. This module gives the backends one
place to translate their values, with one rule: a value nobody has classified is `UNKNOWN`,
never `STOP`.

Only `STOP` and `TOOL_CALLS` mean the model finished what it set out to say. Everything else,
including `UNKNOWN`, is a reply a caller must not treat as complete.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

STOP: Final = "stop"
LENGTH: Final = "length"  # cut off by the output cap or the context window
TOOL_CALLS: Final = "tool_calls"
REFUSAL: Final = "refusal"  # the model declined to answer
CONTENT_FILTER: Final = "content_filter"  # a provider filter or guardrail withheld the reply
PAUSED: Final = "paused"  # the turn was paused mid-way and needs a continuation
ERROR: Final = "error"  # the provider says the output is malformed
UNKNOWN: Final = "unknown"  # missing, or a value this table does not know

COMPLETE_FINISH_REASONS: Final = frozenset({STOP, TOOL_CALLS})

ANTHROPIC_STOP_REASONS: Final[Mapping[str, str]] = {
    "end_turn": STOP,
    "stop_sequence": STOP,
    "tool_use": TOOL_CALLS,
    "max_tokens": LENGTH,
    "model_context_window_exceeded": LENGTH,
    "pause_turn": PAUSED,
    "refusal": REFUSAL,
}

BEDROCK_STOP_REASONS: Final[Mapping[str, str]] = {
    "end_turn": STOP,
    "stop_sequence": STOP,
    "tool_use": TOOL_CALLS,
    "max_tokens": LENGTH,
    "model_context_window_exceeded": LENGTH,
    "guardrail_intervened": CONTENT_FILTER,
    "content_filtered": CONTENT_FILTER,
    "malformed_model_output": ERROR,
    "malformed_tool_use": ERROR,
}

# OpenAI Chat Completions, Azure OpenAI and LiteLLM. LiteLLM maps its providers onto this set
# itself and turns some provider values into "stop", through an explicit entry (Gemini
# MALFORMED_FUNCTION_CALL) or its unmapped fallback (pause_turn, Bedrock malformed_*; litellm
# 1.90.2), before trelix sees them. Through LiteLLM a "stop" is therefore weaker evidence than
# through a direct backend, and the original value cannot be recovered.
OPENAI_FINISH_REASONS: Final[Mapping[str, str]] = {
    "stop": STOP,
    "tool_calls": TOOL_CALLS,
    "function_call": TOOL_CALLS,
    "length": LENGTH,
    "content_filter": CONTENT_FILTER,
}

# google.genai FinishReason names. Only these two are known to be benign or a plain truncation;
# SAFETY, RECITATION, LANGUAGE, OTHER, BLOCKLIST, PROHIBITED_CONTENT, SPII,
# MALFORMED_FUNCTION_CALL and anything newer are UNKNOWN until each has been verified.
VERTEX_FINISH_REASONS: Final[Mapping[str, str]] = {
    "STOP": STOP,
    "MAX_TOKENS": LENGTH,
}


def normalise(table: Mapping[str, str], raw: object) -> str:
    """Translate a provider's own value through `table`; anything unlisted is `UNKNOWN`."""
    if not isinstance(raw, str):
        return UNKNOWN
    return table.get(raw, UNKNOWN)


def is_complete(finish_reason: str) -> bool:
    """True only when the model finished its reply (a normal stop, or a tool call it asked for)."""
    return finish_reason in COMPLETE_FINISH_REASONS


@dataclass(frozen=True)
class FinishInfo:
    """A normalised end-of-generation signal plus what the provider actually said."""

    finish_reason: str
    raw_finish_reason: str | None = None
    refusal: str | None = None
    signals: tuple[str, ...] = ()


def classify_chat_choice(choice: Any) -> FinishInfo:
    """Classify one OpenAI-shaped `choices[i]` (OpenAI, Azure OpenAI, LiteLLM).

    Reads duck-typed attributes and trusts a value only if it has the expected type, so a
    partially populated object classifies as `UNKNOWN` instead of raising.

    - `message.refusal` is a separate field: a refusal arrives with `finish_reason == "stop"`,
      so a non-empty refusal overrides a clean finish reason. LiteLLM's response object has no
      `refusal` attribute; it moves the value to `message.provider_specific_fields["refusal"]`,
      which is read as well.
    - Azure's content filter fails open: on a filter outage the reply is still HTTP 200 with a
      normal finish reason, and the only sign is an `error` object in `content_filter_results`.
      That is recorded as a signal; the reply itself is complete.
    """
    raw = getattr(choice, "finish_reason", None)
    raw_text = raw if isinstance(raw, str) else None
    finish = normalise(OPENAI_FINISH_REASONS, raw_text)

    message = getattr(choice, "message", None)
    refusal_field = getattr(message, "refusal", None)
    if not isinstance(refusal_field, str):
        extras = getattr(message, "provider_specific_fields", None)
        refusal_field = extras.get("refusal") if isinstance(extras, dict) else None
    refusal = (
        refusal_field.strip() if isinstance(refusal_field, str) and refusal_field.strip() else None
    )
    if refusal is not None and is_complete(finish):
        finish = REFUSAL

    signals: list[str] = []
    extra = getattr(choice, "model_extra", None)
    if isinstance(extra, dict):
        filter_results = extra.get("content_filter_results")
        if isinstance(filter_results, dict) and filter_results.get("error"):
            signals.append("content_filter_error")

    return FinishInfo(
        finish_reason=finish,
        raw_finish_reason=raw_text,
        refusal=refusal,
        signals=tuple(signals),
    )
