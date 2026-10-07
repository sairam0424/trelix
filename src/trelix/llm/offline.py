"""Helpers for running the ``openai`` backend against a local OpenAI-compatible server.

``TRELIX_LLM_BASE_URL`` (``LLMConfig.base_url``) points the backend at Ollama, llama-server or
any other server that speaks the OpenAI chat-completions API. Everything here is a pure
function over configuration strings; nothing does I/O.
"""

from __future__ import annotations

import re
from typing import Final

# Sent as the bearer when no OPENAI_API_KEY is configured. The SDK refuses to build a client
# without a key, Ollama accepts any value and ignores it. A public constant, not a secret.
LOCAL_PLACEHOLDER_KEY: Final = "trelix-local"

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
