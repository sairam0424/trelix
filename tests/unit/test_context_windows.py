"""
Unit tests for trelix.llm.context_windows.

Verifies model window table coverage and longest-prefix matching logic.
"""

from trelix.llm.context_windows import resolve_window


def test_resolve_window_gemini_2_5_pro():
    """Gemini 2.5 Pro: 1M window (design requirement)."""
    assert resolve_window("gemini-2.5-pro") == 1_000_000


def test_resolve_window_gemini_2_5_pro_versioned():
    """Gemini 2.5 Pro with version suffix: matches prefix."""
    assert resolve_window("gemini-2.5-pro-preview-0409") == 1_000_000


def test_resolve_window_claude_sonnet_4():
    """Claude Sonnet 4: 200k window (design requirement)."""
    assert resolve_window("claude-sonnet-4") == 200_000


def test_resolve_window_claude_sonnet_4_versioned():
    """Claude Sonnet 4 with date suffix: matches prefix."""
    assert resolve_window("claude-sonnet-4-20250514") == 200_000


def test_resolve_window_gpt_4o():
    """GPT-4o: 128k window (design requirement)."""
    assert resolve_window("gpt-4o") == 128_000


def test_resolve_window_gpt_4o_versioned():
    """GPT-4o with date suffix: matches prefix."""
    assert resolve_window("gpt-4o-2024-11-20") == 128_000


def test_resolve_window_unknown_model():
    """Unknown models return None (no match)."""
    assert resolve_window("unknown-llm-9000") is None


def test_resolve_window_case_insensitive():
    """Model matching is case-insensitive."""
    assert resolve_window("GPT-4O") == 128_000
    assert resolve_window("Claude-Sonnet-4") == 200_000
    assert resolve_window("GEMINI-2.5-PRO") == 1_000_000


def test_resolve_window_longest_prefix_wins():
    """
    Longest-prefix matching ensures versioned models don't collide.

    "gpt-4-1106" should match "gpt-4-1106" (128k) not "gpt-4" (8k).
    Table is ordered longest-first to guarantee this.
    """
    assert resolve_window("gpt-4-1106") == 128_000  # not 8192
    assert resolve_window("gpt-4") == 8_192


def test_resolve_window_coverage():
    """Spot-check coverage across providers."""
    # Anthropic family
    assert resolve_window("claude-opus-4") == 200_000
    assert resolve_window("claude-haiku-4") == 200_000
    assert resolve_window("claude-3-5-sonnet") == 200_000

    # OpenAI family
    assert resolve_window("gpt-4-turbo") == 128_000
    assert resolve_window("gpt-3.5-turbo") == 4_096

    # Gemini family
    assert resolve_window("gemini-1.5-pro") == 2_097_152
    assert resolve_window("gemini-1.5-flash") == 1_048_576

    # Cohere
    assert resolve_window("command-r-plus") == 128_000

    # Mistral
    assert resolve_window("mistral-large") == 128_000

    # Meta
    assert resolve_window("llama-3.3") == 128_000
    assert resolve_window("llama-3.1-405b") == 128_000
