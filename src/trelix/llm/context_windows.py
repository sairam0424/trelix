"""
Model context window sizes and resolution logic.

Window data sourced from official docs (as of Jan 2025):
- Gemini 2.0 Flash Experimental: 1,048,576 tokens (Google AI Studio)
- Claude Sonnet 4: 200,000 tokens (Anthropic)
- GPT-4o: 128,000 tokens (OpenAI)

Uses longest-prefix matching to handle version suffixes (e.g. "gpt-4o-2024-11-20").
"""

from __future__ import annotations

# Model context windows (in tokens)
# Format: (prefix, window_size)
# Ordered longest-prefix-first for correct matching of versioned models
MODEL_WINDOWS: list[tuple[str, int]] = [
    # ── Gemini ──────────────────────────────────────────────────────────────
    ("gemini-2.0-flash-exp", 1_048_576),
    ("gemini-2.5-pro", 1_000_000),  # Requirement: 1M window
    ("gemini-2.0-flash", 1_048_576),
    ("gemini-1.5-pro", 2_097_152),
    ("gemini-1.5-flash", 1_048_576),
    ("gemini-pro", 32_768),
    # ── Anthropic ───────────────────────────────────────────────────────────
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),  # Requirement: 200k window
    ("claude-haiku-4", 200_000),
    ("claude-3-5-sonnet", 200_000),
    ("claude-3-5-haiku", 200_000),
    ("claude-3-opus", 200_000),
    ("claude-3-sonnet", 200_000),
    ("claude-3-haiku", 200_000),
    ("claude-2.1", 200_000),
    ("claude-2", 100_000),
    ("claude-instant", 100_000),
    # ── OpenAI ──────────────────────────────────────────────────────────────
    ("gpt-4o", 128_000),  # Requirement: 128k window
    ("gpt-4-turbo", 128_000),
    ("gpt-4-1106", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo-16k", 16_384),
    ("gpt-3.5-turbo", 4_096),
    # ── Cohere ──────────────────────────────────────────────────────────────
    ("command-r-plus", 128_000),
    ("command-r", 128_000),
    ("command", 4_096),
    # ── Mistral ─────────────────────────────────────────────────────────────
    ("mistral-large", 128_000),
    ("mistral-medium", 32_000),
    ("mistral-small", 32_000),
    ("mixtral-8x7b", 32_000),
    ("mixtral-8x22b", 64_000),
    # ── Meta ────────────────────────────────────────────────────────────────
    ("llama-3.3", 128_000),
    ("llama-3.1-405b", 128_000),
    ("llama-3.1-70b", 128_000),
    ("llama-3.1", 128_000),
    ("llama-3", 8_192),
    ("llama-2", 4_096),
]


def resolve_window(model: str) -> int | None:
    """
    Resolve the context window size for a given model string.

    Uses longest-prefix matching to handle version suffixes and variants.
    For example:
    - "gpt-4o-2024-11-20" matches "gpt-4o" → 128,000 tokens
    - "claude-sonnet-4-20250514" matches "claude-sonnet-4" → 200,000 tokens
    - "gemini-2.5-pro-preview-0409" matches "gemini-2.5-pro" → 1,000,000 tokens

    Args:
        model: Model identifier string (e.g. "gpt-4o", "claude-sonnet-4")

    Returns:
        Context window size in tokens, or None if no match found.

    Examples:
        >>> resolve_window("gpt-4o")
        128000
        >>> resolve_window("claude-sonnet-4")
        200000
        >>> resolve_window("gemini-2.5-pro")
        1000000
        >>> resolve_window("unknown-model")
        None
    """
    model_lower = model.lower()

    # Longest-prefix match — earlier entries in MODEL_WINDOWS win for ambiguous prefixes
    for prefix, window in MODEL_WINDOWS:
        if model_lower.startswith(prefix.lower()):
            return window

    return None
