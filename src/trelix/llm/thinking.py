"""Which extended-thinking request shape a Claude model accepts, shared by the
Anthropic and Bedrock backends.

Two shapes exist. Older models take a token budget
(``thinking={"type": "enabled", "budget_tokens": N}``; Bedrock spells it
``reasoning_config``). Newer ones reject that with a 400 / ValidationException and take
``thinking={"type": "adaptive"}`` instead, where the model decides whether and how much to
think. Confirmed live on Bedrock Converse (us-east-1, 2026-10-04) for
``us.anthropic.claude-sonnet-5`` and ``us.anthropic.claude-sonnet-5-5``: the budget shape
fails with ``"thinking.type.enabled" is not supported for this model. Use
"thinking.type.adaptive" and "output_config.effort" to control thinking behavior.`` and
``{"thinking": {"type": "adaptive"}}`` succeeds. For the direct Anthropic API this follows
the API reference (documented, not measured).

The classifier is a pure function of the model id and is deliberately conservative: only an
id it positively recognises as adaptive-only returns ``"adaptive"``. Everything else,
including an id it has never seen, returns ``"budget"``, so every model that works today
keeps the request it gets today. A future model the table does not know yet is rescued at
runtime instead: the backend retries once in adaptive mode when the provider rejects the
budget shape with the message :func:`is_enabled_thinking_rejection` recognises, and
:class:`ThinkingModeMemory` remembers the outcome.
"""

from __future__ import annotations

import re
import threading
from typing import Final, Literal

ThinkingMode = Literal["budget", "adaptive"]

# claude-<family>-<major>[-<minor>][-<yyyymmdd>][-v<N>[:<M>]], optionally behind the Bedrock
# "anthropic." or "<region>.anthropic." prefix. Everything is anchored by fullmatch(), never
# searched, so "xclaude-sonnet-5", "claude-sonnet-5x" and "claude-sonnet-5\n" do not match.
# The version is read as numbers: the major is one or two digits with no leading zero, the
# minor likewise, and nothing after the version may start with a digit, so an eight-digit date
# can only ever be consumed as a date ("claude-sonnet-4-20250514" is 4.0, and
# "claude-sonnet-20250514" has no version at all).
# re.ASCII keeps \d to 0-9: int() reads other scripts' digits as numbers, so without it
# "claude-sonnet-5" followed by an Arabic-Indic zero would be version 50.
# Legacy ids that put the family after the version ("claude-3-7-sonnet-...") do not match.
_CLAUDE_MODEL_ID: Final = re.compile(
    r"(?:(?:[a-z]{2,10}(?:-[a-z]{2,10})?\.)?anthropic\.)?"
    r"claude-(?P<family>fable|mythos|opus|sonnet|haiku)"
    r"-(?P<major>[1-9]\d?)"
    r"(?:-(?P<minor>0|[1-9]\d?))?"
    r"(?:-\d{8})?"
    r"(?:-v\d{1,3}(?::\d{1,3})?)?",
    re.ASCII,
)

# First version of each family that accepts only adaptive thinking. Claude 5 and newer are
# adaptive-only for every family; Opus got there earlier, at 4.7 (per the Claude API
# reference: budget_tokens is a 400 on Opus 4.7 and 4.8 while Opus 4.6 and Sonnet 4.6 still
# accept it, deprecated).
_FIRST_ADAPTIVE_ONLY_VERSION: Final = (5, 0)
_FIRST_ADAPTIVE_ONLY_VERSION_BY_FAMILY: Final = {"opus": (4, 7)}

_ENABLED_THINKING_REJECTION_MARKERS: Final = ("thinking.type.enabled", "not supported")


def thinking_mode_for_model(model_id: str) -> ThinkingMode:
    """Classify *model_id* as ``"adaptive"`` (adaptive-only) or ``"budget"``.

    Returns ``"budget"`` for anything that is not positively recognised as a Claude
    fable/mythos/opus/sonnet/haiku id at an adaptive-only version, which includes the empty
    string, unknown providers' ids and malformed input.
    """
    match = _CLAUDE_MODEL_ID.fullmatch(model_id)
    if match is None:
        return "budget"
    version = (int(match["major"]), int(match["minor"] or 0))
    first_adaptive = _FIRST_ADAPTIVE_ONLY_VERSION_BY_FAMILY.get(
        match["family"], _FIRST_ADAPTIVE_ONLY_VERSION
    )
    return "adaptive" if version >= first_adaptive else "budget"


def is_enabled_thinking_rejection(message: str) -> bool:
    """True when *message* is the provider telling us this model refuses
    ``thinking.type.enabled``. Only the text is checked; each backend adds its own check on
    the exception type (HTTP 400 for Anthropic, ValidationException for Bedrock)."""
    return all(marker in message for marker in _ENABLED_THINKING_REJECTION_MARKERS)


class ThinkingModeMemory:
    """Per-backend-instance record of models that rejected the budget shape at runtime.

    Thread-safe the same way the backends' other learned state is: writes take a lock, and
    the set is replaced rather than mutated, so an unlocked read in a request builder sees
    either the old or the new value and never a torn one. The memory is keyed by model id
    because BedrockBackend can swap to its fallback model mid-life, and what one model
    rejected says nothing about the other.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._adaptive_models: frozenset[str] = frozenset()

    def mode_for(self, model_id: str) -> ThinkingMode:
        """The mode to use for *model_id*: adaptive if it was learned at runtime,
        otherwise whatever :func:`thinking_mode_for_model` says."""
        if model_id in self._adaptive_models:
            return "adaptive"
        return thinking_mode_for_model(model_id)

    def remember_adaptive(self, model_id: str) -> bool:
        """Record that *model_id* needs adaptive thinking. Returns True only for the call
        that first records it, so concurrent threads that hit the same rejection log one
        warning between them."""
        with self._lock:
            if model_id in self._adaptive_models:
                return False
            self._adaptive_models = self._adaptive_models | {model_id}
            return True
