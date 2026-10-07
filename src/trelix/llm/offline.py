"""Helpers for running the ``openai`` backend against a local OpenAI-compatible server.

``TRELIX_LLM_BASE_URL`` (``LLMConfig.base_url``) points the backend at Ollama, llama-server or
any other server that speaks the OpenAI chat-completions API. Everything here is a pure
function over configuration strings and message lists; the only I/O is the lazy load of the
cl100k_base encoder that the prompt-truncation check counts with.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeGuard

logger = logging.getLogger("trelix.llm.offline")

# Sent as the bearer when no OPENAI_API_KEY is configured. The SDK refuses to build a client
# without a key, Ollama accepts any value and ignores it. A public constant, not a secret.
LOCAL_PLACEHOLDER_KEY: Final = "trelix-local"

# The `ChatResponse.signals` value, and the `HunkResult.detail`, for a prompt the server cut.
PROMPT_TRUNCATED: Final = "prompt_truncated"

# A server that reports fewer prompt tokens than this share of what trelix sent threw part of
# the prompt away: Ollama truncates an over-long prompt to about half its context length and
# answers HTTP 200 with a normal finish reason. A tokenizer more efficient than cl100k_base on
# code is a few percent under 1.0, not 15. Exactly at the floor is not truncated.
TRUNCATION_FLOOR: Final = 0.85

_ENCODING_NAME: Final = "cl100k_base"
_ENCODER_UNAVAILABLE_WARNING: Final = (
    "tiktoken cl100k_base is not available (%s); the prompt-truncation check is off for this "
    "run (docs/OFFLINE.md: prefetch)"
)

# The encoder is loaded on first use and kept. After one failure (no cache file and no network
# on an air-gapped box) nothing is retried and the warning is not repeated, so a review does
# not pay a connect timeout per hunk.
_encoder: Any | None = None
_ENCODER_UNAVAILABLE = False

# Below this many parameters a local model is warned about once at backend construction.
MODEL_SIZE_FLOOR_BILLIONS: Final = 20.0

# A size token is a number followed by `b` or `m`, not glued to a letter, digit or dot on
# either side: `7b` in `qwen2.5-coder:7b`, `270m` in `gemma3:270m`, `30b` in `qwen3-coder:30b-a3b`.
# The `7b` in `mixtral-8x7b` is glued to the `x`, so a mixture tag has no size (reported as
# unparseable, not as N x M).
_SIZE_TOKEN: Final = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)([bBmM])(?![A-Za-z0-9.])")


def model_size_billions(model: str) -> float | None:
    """Largest parameter count named in a model tag, in billions; ``None`` when there is none."""
    sizes = [
        float(number) / (1000.0 if unit.lower() == "m" else 1.0)
        for number, unit in _SIZE_TOKEN.findall(model)
    ]
    return max(sizes) if sizes else None


def small_model_warning(model: str) -> str | None:
    """The one-line warning for a model tag under the size floor, or with no readable size.

    ``None`` when the tag names a size at or above the floor. The text carries the tag
    (operator configuration), never a URL or a key.
    """
    floor = f"{MODEL_SIZE_FLOOR_BILLIONS:g}B"
    size = model_size_billions(model)
    if size is None:
        return (
            f"Local model '{model}': no parameter count in the tag, so trelix cannot tell "
            f"whether it meets the {floor} floor docs/OFFLINE.md assumes."
        )
    if size < MODEL_SIZE_FLOOR_BILLIONS:
        return (
            f"Local model '{model}' is about {size:g}B parameters, under the {floor} floor "
            f"docs/OFFLINE.md assumes for trelix review; expect more unreviewed hunks and "
            f"weaker findings."
        )
    return None


def _load_encoder() -> Any | None:
    global _encoder, _ENCODER_UNAVAILABLE
    if _encoder is not None:
        return _encoder
    if _ENCODER_UNAVAILABLE:
        return None
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    except (OSError, RuntimeError, ValueError) as exc:
        _ENCODER_UNAVAILABLE = True
        logger.warning(_ENCODER_UNAVAILABLE_WARNING, type(exc).__name__)
        return None
    return _encoder


def estimate_prompt_tokens(messages: Sequence[Mapping[str, Any]]) -> int | None:
    """cl100k_base count of every message's ``content``; ``None`` when the encoder cannot load.

    A special token in the text (``<|endoftext|>`` inside a reviewed file) is counted as plain
    text, not refused: ``disallowed_special=()``.
    """
    encoder = _load_encoder()
    if encoder is None:
        return None
    return sum(
        len(encoder.encode(str(message.get("content") or ""), disallowed_special=()))
        for message in messages
    )


def is_token_count(value: object) -> TypeGuard[int]:
    """True for a positive ``int`` that is not a ``bool``: the only usage count worth comparing."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def prompt_truncation_signal(estimated: int | None, reported: int | None) -> str | None:
    """``PROMPT_TRUNCATED`` when the server's ``reported`` prompt tokens fall under the floor.

    ``None`` when either side is not a positive int (no usage, a zero, a bool, an encoder that
    could not load): an unknown count is never read as truncated.
    """
    if not is_token_count(estimated):
        return None
    if not is_token_count(reported):
        return None
    if reported < TRUNCATION_FLOOR * estimated:
        return PROMPT_TRUNCATED
    return None
