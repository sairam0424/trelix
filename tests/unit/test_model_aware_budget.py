"""
Unit tests for Feature 2: Model-Aware Context Budget.

Tests:
1. Config validation (int|None, fraction bounds)
2. Budget resolution (explicit vs auto-derived)
3. Top-k scaling behavior
4. Backward compatibility (default=12000)
"""

import os
import tempfile
from pathlib import Path

import pytest

from trelix.core.config import IndexConfig, RetrievalConfig


def test_context_token_budget_accepts_int():
    """Explicit int budget (v2.12.0 behavior)."""
    cfg = RetrievalConfig(context_token_budget=15_000)
    assert cfg.context_token_budget == 15_000


def test_context_token_budget_accepts_none():
    """None triggers auto-derivation from model window."""
    cfg = RetrievalConfig(context_token_budget=None)
    assert cfg.context_token_budget is None


def test_context_token_budget_default_is_12000():
    """Default 12000 preserves exact v2.12.0 behavior."""
    cfg = RetrievalConfig()
    assert cfg.context_token_budget == 12_000


def test_context_window_fraction_default():
    """Default fraction is 0.5 (half the window)."""
    cfg = RetrievalConfig()
    assert cfg.context_window_fraction == 0.5


def test_context_window_fraction_bounds():
    """Fraction must be in [0.1, 0.9]."""
    # Valid range
    cfg = RetrievalConfig(context_window_fraction=0.3)
    assert cfg.context_window_fraction == 0.3

    # Out of range — pydantic raises validation error
    with pytest.raises(Exception):  # ValidationError
        RetrievalConfig(context_window_fraction=0.05)

    with pytest.raises(Exception):
        RetrievalConfig(context_window_fraction=0.95)


def test_scale_top_k_to_budget_default_false():
    """Scaling is opt-in (default False)."""
    cfg = RetrievalConfig()
    assert cfg.scale_top_k_to_budget is False


def test_scale_top_k_to_budget_can_enable():
    """Scaling can be enabled via config."""
    cfg = RetrievalConfig(scale_top_k_to_budget=True)
    assert cfg.scale_top_k_to_budget is True


def test_env_var_parsing():
    """Config fields can be set via environment variables."""
    os.environ["TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION"] = "0.7"
    os.environ["TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET"] = "true"
    try:
        cfg = RetrievalConfig()
        assert cfg.context_window_fraction == 0.7
        assert cfg.scale_top_k_to_budget is True
        # Default budget is still 12000 unless explicitly set to None in code
        assert cfg.context_token_budget == 12_000
    finally:
        os.environ.pop("TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION", None)
        os.environ.pop("TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET", None)


def _make_test_config(model: str, budget: int | None = 12_000) -> IndexConfig:
    """Helper to create a minimal IndexConfig with LLM model set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "repo"
        repo.mkdir()
        cfg = IndexConfig(repo_path=str(repo))
        cfg.llm.model = model
        cfg.retrieval.context_token_budget = budget
        return cfg


def test_budget_resolution_explicit_int():
    """Explicit budget bypasses auto-derivation."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gpt-4o", budget=20_000)
    r = Retriever(cfg)
    assert r._effective_budget == 20_000


def test_budget_resolution_auto_gpt4o():
    """gpt-4o (128k window) × 0.5 = 64,000 tokens."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gpt-4o", budget=None)
    r = Retriever(cfg)
    assert r._effective_budget == 64_000


def test_budget_resolution_auto_claude_sonnet_4():
    """claude-sonnet-4 (200k window) × 0.5 = 100,000 tokens."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="claude-sonnet-4", budget=None)
    r = Retriever(cfg)
    assert r._effective_budget == 100_000


def test_budget_resolution_auto_gemini_2_5_pro():
    """gemini-2.5-pro (1M window) × 0.5 = 500,000 tokens."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gemini-2.5-pro", budget=None)
    r = Retriever(cfg)
    assert r._effective_budget == 500_000


def test_budget_resolution_auto_custom_fraction():
    """Custom fraction (0.3) applies correctly."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gpt-4o", budget=None)
    cfg.retrieval.context_window_fraction = 0.3
    r = Retriever(cfg)
    # 128k × 0.3 = 38,400
    assert r._effective_budget == 38_400


def test_budget_resolution_unknown_model_fallback():
    """Unknown models fall back to 12,000."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="mystery-llm-x1", budget=None)
    r = Retriever(cfg)
    assert r._effective_budget == 12_000


def test_top_k_scaling_disabled():
    """When scale_top_k_to_budget=False, top_k values are unchanged."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gpt-4o", budget=None)
    cfg.retrieval.scale_top_k_to_budget = False
    r = Retriever(cfg)
    # Default top_k_vector=20, rerank_top_n=15 — unchanged
    assert r._effective_top_k_vector == 20
    assert r._effective_rerank_top_n == 15


def test_top_k_scaling_enabled():
    """When scale_top_k_to_budget=True, top_k values scale with budget."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gpt-4o", budget=None)
    cfg.retrieval.scale_top_k_to_budget = True
    r = Retriever(cfg)
    # scale_factor = 64000 / 12000 = 5.333...
    # top_k_vector = 20 * 5.333... = 106.666... → 106
    # rerank_top_n = 15 * 5.333... = 79.999... → 80
    assert r._effective_top_k_vector == 106
    assert r._effective_rerank_top_n == 80


def test_top_k_scaling_only_when_budget_none():
    """Scaling only applies when context_token_budget=None (auto mode)."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gpt-4o", budget=20_000)  # explicit
    cfg.retrieval.scale_top_k_to_budget = True
    r = Retriever(cfg)
    # Explicit budget → no scaling
    assert r._effective_top_k_vector == 20
    assert r._effective_rerank_top_n == 15


def test_backward_compat_exact():
    """Default config (budget=12000) produces identical behavior to v2.12.0."""
    from trelix.retrieval.retriever import Retriever

    cfg = _make_test_config(model="gpt-4o", budget=12_000)
    r = Retriever(cfg)
    assert r._effective_budget == 12_000
    assert r._effective_top_k_vector == 20
    assert r._effective_rerank_top_n == 15
