"""
Integration tests for Feature 1 (Extended Thinking) + Feature 2 (Model-Aware Budget).

This test suite verifies that both features work together without conflicts.
"""

import os
import tempfile

import pytest

from trelix.core.config import IndexConfig, LLMConfig


def test_thinking_and_auto_budget_config_coexist():
    """Both features can be enabled in config without conflict."""
    cfg = IndexConfig(repo_path=tempfile.mkdtemp())
    cfg.llm.thinking_enabled = True
    cfg.llm.thinking_budget_tokens = 8192
    cfg.retrieval.context_token_budget = None
    cfg.retrieval.context_window_fraction = 0.5

    assert cfg.llm.thinking_enabled is True
    assert cfg.llm.thinking_budget_tokens == 8192
    assert cfg.retrieval.context_token_budget is None
    assert cfg.retrieval.context_window_fraction == 0.5


def test_thinking_does_not_affect_retrieval_budget():
    """Thinking budget is separate from retrieval context budget."""
    from trelix.retrieval.retriever import Retriever

    cfg = IndexConfig(repo_path=tempfile.mkdtemp())
    cfg.llm.model = "claude-sonnet-4"
    cfg.llm.thinking_enabled = True
    cfg.llm.thinking_budget_tokens = 8192
    cfg.retrieval.context_token_budget = None

    r = Retriever(cfg)
    assert r._effective_budget == 100_000
    assert cfg.llm.thinking_budget_tokens == 8192


def test_gemini_1m_auto_budget():
    """gemini-2.5-pro (1M window) derives 500k budget."""
    from trelix.retrieval.retriever import Retriever

    cfg = IndexConfig(repo_path=tempfile.mkdtemp())
    cfg.llm.model = "gemini-2.5-pro"
    cfg.retrieval.context_token_budget = None

    r = Retriever(cfg)
    assert r._effective_budget == 500_000


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="Vertex backend requires google-genai (optional dependency)",
)
def test_gemini_with_thinking_parameter():
    """Vertex backend accepts thinking parameter."""
    try:
        import inspect

        from trelix.llm.providers.vertex_backend import VertexBackend

        cfg = LLMConfig(provider="vertex", model="gemini-2.5-pro")
        cfg.google_api_key = os.getenv("GOOGLE_API_KEY", "test-placeholder")
        backend = VertexBackend(cfg)

        sig_complete = inspect.signature(backend.complete)
        sig_stream = inspect.signature(backend.stream)

        assert "thinking" in sig_complete.parameters
        assert "thinking" in sig_stream.parameters
    except ImportError:
        pytest.skip("google-genai not installed")


def test_all_target_models_auto_budget():
    """All 3 target models auto-derive correct budgets."""
    from trelix.retrieval.retriever import Retriever

    models_and_budgets = [
        ("gpt-4o", 64_000),
        ("claude-sonnet-4", 100_000),
        ("gemini-2.5-pro", 500_000),
    ]

    for model, expected_budget in models_and_budgets:
        cfg = IndexConfig(repo_path=tempfile.mkdtemp())
        cfg.llm.model = model
        cfg.retrieval.context_token_budget = None

        r = Retriever(cfg)
        assert r._effective_budget == expected_budget


def test_thinking_does_not_affect_budget_scaling():
    """Thinking config doesn't interfere with top_k scaling."""
    from trelix.retrieval.retriever import Retriever

    cfg = IndexConfig(repo_path=tempfile.mkdtemp())
    cfg.llm.model = "gpt-4o"
    cfg.llm.thinking_enabled = True
    cfg.llm.thinking_budget_tokens = 8192
    cfg.retrieval.context_token_budget = None
    cfg.retrieval.scale_top_k_to_budget = True

    r = Retriever(cfg)
    assert r._effective_top_k_vector == 106
    assert r._effective_budget == 64_000


def test_no_namespace_collision():
    """Config namespaces don't collide."""
    cfg = IndexConfig(repo_path=tempfile.mkdtemp())

    assert hasattr(cfg.llm, "thinking_enabled")
    assert hasattr(cfg.retrieval, "context_token_budget")
    assert not hasattr(cfg.llm, "context_token_budget")
    assert not hasattr(cfg.retrieval, "thinking_enabled")


@pytest.mark.parametrize(
    ("provider", "module_path", "class_name", "required_sdk"),
    [
        # openai>=1.35.0 is a CORE dependency, so this case always runs.
        ("openai", "trelix.llm.providers.openai_backend", "OpenAIBackend", None),
        # anthropic is an optional extra. CI installs .[local,otel,sso,dev], which
        # does not include it, so this case skips there rather than failing.
        ("anthropic", "trelix.llm.providers.anthropic_backend", "AnthropicBackend", "anthropic"),
    ],
)
def test_backends_accept_thinking(provider, module_path, class_name, required_sdk):
    """Each backend's complete/stream accepts the thinking parameter.

    Parametrized rather than looping over both backends in one test on purpose:
    constructing a backend imports its provider SDK, so a single test covering
    both is only as available as its least-available SDK. Splitting them keeps
    the openai case running in CI instead of skipping the assertion entirely.
    """
    if required_sdk:
        pytest.importorskip(
            required_sdk,
            reason=f"{required_sdk} extra not installed (pip install 'trelix[{required_sdk}]')",
        )

    import importlib
    import inspect

    backend_class = getattr(importlib.import_module(module_path), class_name)
    backend = backend_class(LLMConfig(provider=provider))

    for method_name in ("complete", "stream"):
        sig = inspect.signature(getattr(backend, method_name))
        assert "thinking" in sig.parameters, f"{class_name}.{method_name} lacks 'thinking'"


def test_config_overrides_compose():
    """Explicit config values for both features compose correctly."""
    cfg = IndexConfig(repo_path=tempfile.mkdtemp())
    cfg.llm.thinking_enabled = True
    cfg.llm.thinking_budget_tokens = 10000
    cfg.retrieval.context_token_budget = 50000

    assert cfg.llm.thinking_enabled is True
    assert cfg.llm.thinking_budget_tokens == 10000
    assert cfg.retrieval.context_token_budget == 50000


def test_mixed_explicit_and_auto():
    """Explicit thinking budget + auto context budget."""
    from trelix.retrieval.retriever import Retriever

    cfg = IndexConfig(repo_path=tempfile.mkdtemp())
    cfg.llm.model = "gpt-4o"
    cfg.llm.thinking_budget_tokens = 6000
    cfg.retrieval.context_token_budget = None

    r = Retriever(cfg)
    assert cfg.llm.thinking_budget_tokens == 6000
    assert r._effective_budget == 64_000
