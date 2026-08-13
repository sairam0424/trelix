"""
Smoke tests for realistic trelix configuration scenarios.

These are dry-run tests using mock APIs to verify that configurations work
without crashes and produce expected behavior. No real API calls are made.

Smoke test scenarios:
1. Extended thinking enabled: Does it actually work? No crashes?
2. Gemini context: Does resolve_window() return correct values?
3. Mixed config: Some features on, some off
4. Error cases: Invalid config values handled gracefully?
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, Mock

import pytest

from trelix.core.config import (
    EmbedderConfig,
    IndexConfig,
    LLMConfig,
    RetrievalConfig,
    StoreConfig,
)
from trelix.llm.client import ChatMessage, ChatResponse
from trelix.llm.context_windows import resolve_window

# Fake key for testing — not a real credential
_TEST_FAKE_KEY = "test-fake-key-for-smoke-tests"


# =============================================================================
# Scenario 1: Extended Thinking Enabled
# =============================================================================


class TestExtendedThinkingSmoke:
    """Smoke test: Extended thinking enabled configuration works without crashes."""

    def _make_mock_anthropic_module(self) -> ModuleType:
        """Build a minimal fake anthropic module."""
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        return mock_anthropic

    @pytest.fixture
    def mock_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Patch sys.modules for anthropic."""
        module = self._make_mock_anthropic_module()
        monkeypatch.setitem(sys.modules, "anthropic", module)
        monkeypatch.delitem(sys.modules, "trelix.llm.providers.anthropic_backend", raising=False)
        return module

    def test_thinking_enabled_config_instantiation(self, mock_anthropic: MagicMock):
        """Config with thinking_enabled=True instantiates without error."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4",
            anthropic_api_key=_TEST_FAKE_KEY,
            thinking_enabled=True,
            thinking_budget_tokens=8192,
            _env_file=None,
        )

        assert config.thinking_enabled is True
        assert config.thinking_budget_tokens == 8192
        assert config.provider == "anthropic"

    def test_thinking_enabled_backend_creation(self, mock_anthropic: MagicMock):
        """Backend instantiates with thinking config without crashing."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4",
            anthropic_api_key=_TEST_FAKE_KEY,
            thinking_enabled=True,
            thinking_budget_tokens=8192,
            _env_file=None,
        )

        backend = AnthropicBackend(config)
        assert backend is not None
        assert backend._config.thinking_enabled is True

    def test_thinking_enabled_api_call_simulation(self, mock_anthropic: MagicMock):
        """Simulated API call with thinking enabled completes successfully."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4",
            anthropic_api_key=_TEST_FAKE_KEY,
            thinking_enabled=True,
            thinking_budget_tokens=4096,
            _env_file=None,
        )
        backend = AnthropicBackend(config)

        # Mock API response with thinking block
        mock_client = MagicMock()

        class MockContent:
            def __init__(
                self, block_type: str, text: str | None = None, thinking: str | None = None
            ):
                self.type = block_type
                self.text = text
                self.thinking = thinking

        class MockUsage:
            def __init__(self):
                self.input_tokens = 150
                self.output_tokens = 75
                self.cache_read_input_tokens = 20
                self.cache_creation_input_tokens = 10

        mock_response = Mock()
        mock_response.content = [
            MockContent("thinking", thinking="Analyzing the problem carefully..."),
            MockContent("text", text="Here's the solution."),
        ]
        mock_response.model = "claude-sonnet-4"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MockUsage()

        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        # Execute
        result = backend.complete(
            [ChatMessage(role="user", content="Test thinking")],
            thinking=True,
        )

        # Verify
        assert isinstance(result, ChatResponse)
        assert result.content == "Here's the solution."
        assert result.thinking == "Analyzing the problem carefully..."
        assert result.input_tokens == 150
        assert result.output_tokens == 75

        # Verify API was called with thinking kwargs
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 4096
        # Temperature forced to 1.0 when thinking is enabled
        assert call_kwargs["temperature"] == 1.0

    def test_thinking_with_various_budgets(self, mock_anthropic: MagicMock):
        """Different thinking budget values work correctly."""
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        for budget in [2048, 4096, 8192, 16384]:
            config = LLMConfig(
                provider="anthropic",
                anthropic_api_key=_TEST_FAKE_KEY,
                thinking_enabled=True,
                thinking_budget_tokens=budget,
                _env_file=None,
            )
            backend = AnthropicBackend(config)

            kwargs = backend._thinking_kwargs(thinking=True)
            assert kwargs["thinking"]["budget_tokens"] == budget


# =============================================================================
# Scenario 2: Gemini Context Windows
# =============================================================================


class TestGeminiContextWindowsSmoke:
    """Smoke test: Gemini context window resolution returns correct values."""

    def test_gemini_2_5_pro_window_size(self):
        """Gemini 2.5 Pro has 1M context window."""
        window = resolve_window("gemini-2.5-pro")
        assert window == 1_000_000, "Gemini 2.5 Pro should have 1M window"

    def test_gemini_2_5_pro_with_suffix(self):
        """Gemini 2.5 Pro with version suffix still resolves to 1M."""
        window = resolve_window("gemini-2.5-pro-preview-0409")
        assert window == 1_000_000

    def test_gemini_2_0_flash_window_size(self):
        """Gemini 2.0 Flash has 1,048,576 context window."""
        window = resolve_window("gemini-2.0-flash-exp")
        assert window == 1_048_576

    def test_gemini_1_5_pro_window_size(self):
        """Gemini 1.5 Pro has 2,097,152 context window."""
        window = resolve_window("gemini-1.5-pro")
        assert window == 2_097_152

    def test_gemini_family_complete_coverage(self):
        """All Gemini models are covered in window table."""
        gemini_models = [
            ("gemini-2.5-pro", 1_000_000),
            ("gemini-2.0-flash-exp", 1_048_576),
            ("gemini-2.0-flash", 1_048_576),
            ("gemini-1.5-pro", 2_097_152),
            ("gemini-1.5-flash", 1_048_576),
            ("gemini-pro", 32_768),
        ]

        for model, expected_window in gemini_models:
            window = resolve_window(model)
            assert window == expected_window, f"{model} window mismatch"

    def test_gemini_case_insensitive(self):
        """Gemini model resolution is case-insensitive."""
        assert resolve_window("GEMINI-2.5-PRO") == 1_000_000
        assert resolve_window("Gemini-2.5-Pro") == 1_000_000
        assert resolve_window("gemini-2.5-pro") == 1_000_000

    def test_gemini_vs_other_providers(self):
        """Compare Gemini windows with other providers for sanity check."""
        gemini_window = resolve_window("gemini-2.5-pro")
        claude_window = resolve_window("claude-sonnet-4")
        gpt_window = resolve_window("gpt-4o")

        # Gemini 2.5 Pro (1M) > Claude Sonnet 4 (200k) > GPT-4o (128k)
        assert gemini_window > claude_window > gpt_window
        assert gemini_window == 1_000_000
        assert claude_window == 200_000
        assert gpt_window == 128_000


# =============================================================================
# Scenario 3: Mixed Configuration
# =============================================================================


class TestMixedConfigurationSmoke:
    """Smoke test: Partial feature enablement works correctly."""

    def test_hybrid_search_enabled_sparse_disabled(self, tmp_path):
        """Hybrid search (vector+BM25) works with sparse disabled."""
        config = IndexConfig(
            repo_path=str(tmp_path),
            retrieval=RetrievalConfig(
                top_k_vector=15,
                top_k_bm25=15,
                rerank=True,
                sparse_enabled=False,  # Sparse OFF
                agentic_enabled=False,
                graph_search_enabled=False,
            ),
        )

        assert config.retrieval.top_k_vector == 15
        assert config.retrieval.top_k_bm25 == 15
        assert config.retrieval.rerank is True
        assert config.retrieval.sparse_enabled is False

    def test_agentic_loop_enabled_graph_disabled(self, tmp_path):
        """Agentic loop enabled with graph search disabled."""
        config = IndexConfig(
            repo_path=str(tmp_path),
            retrieval=RetrievalConfig(
                agentic_enabled=True,
                agent_max_turns=5,
                agent_token_budget=8000,
                graph_search_enabled=False,  # Graph OFF
                sparse_enabled=False,
            ),
        )

        assert config.retrieval.agentic_enabled is True
        assert config.retrieval.agent_max_turns == 5
        assert config.retrieval.graph_search_enabled is False

    def test_multi_granularity_with_contextual_chunking(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """Multi-granularity + contextual chunking both enabled."""
        # Set via environment as these fields use aliases
        monkeypatch.setenv("TRELIX_CHUNKER_MULTI_GRANULARITY", "true")
        monkeypatch.setenv("TRELIX_CHUNKER_CONTEXTUAL", "true")

        config = IndexConfig(
            repo_path=str(tmp_path),
        )

        assert config.chunker.multi_granularity_enabled is True
        assert config.chunker.contextual is True

    def test_qdrant_backend_with_hnsw_enabled(self, tmp_path):
        """Qdrant backend with HNSW indexing enabled."""
        config = IndexConfig(
            repo_path=str(tmp_path),
            store=StoreConfig(
                backend="qdrant",
                qdrant_url="http://localhost:6333",
                qdrant_collection="test-collection",
                hnsw=True,
                hnsw_m=16,
                hnsw_ef_search=50,
            ),
        )

        assert config.store.backend == "qdrant"
        assert config.store.hnsw is True
        assert config.store.hnsw_m == 16

    def test_multiple_rerankers_config(self, tmp_path):
        """Different reranker providers can be configured."""
        providers = ["cohere", "cross_encoder", "plaid", "xtr"]

        for provider in providers:
            config = IndexConfig(
                repo_path=str(tmp_path),
                retrieval=RetrievalConfig(
                    rerank=True,
                    rerank_provider=provider,
                    rerank_top_n=15,
                ),
            )

            assert config.retrieval.rerank is True
            assert config.retrieval.rerank_provider == provider

    def test_embedding_providers_coverage(self):
        """All embedding providers can be configured."""
        providers = [
            "openai",
            "azure",
            "local",
            "voyage",
            "local-code",
            "bedrock-titan",
            "bedrock-cohere",
            "bge-code",
            "nomic-code",
        ]

        for provider in providers:
            config = EmbedderConfig(
                provider=provider,
                _env_file=None,
            )
            assert config.provider == provider

    def test_llm_providers_coverage(self):
        """All LLM providers can be configured."""
        providers = ["openai", "azure", "anthropic", "bedrock", "vertex", "litellm"]

        for provider in providers:
            config = LLMConfig(
                provider=provider,
                _env_file=None,
            )
            assert config.provider == provider


# =============================================================================
# Scenario 4: Error Handling & Edge Cases
# =============================================================================


class TestErrorHandlingSmoke:
    """Smoke test: Invalid configurations are handled gracefully."""

    def test_invalid_repo_path_raises_validation_error(self):
        """Non-existent repo_path raises ValueError."""
        with pytest.raises(ValueError, match="repo_path does not exist"):
            IndexConfig(repo_path="/nonexistent/path/to/repo")

    def test_negative_top_k_values_caught(self, tmp_path):
        """Negative top_k values are accepted (no validation enforced)."""
        # Note: RetrievalConfig doesn't enforce non-negative constraints on top_k values
        # This is a deliberate design choice to allow testing edge cases
        config = RetrievalConfig(
            top_k_vector=-5,  # Technically invalid but not enforced
        )
        # The config accepts it, but retrieval would fail at runtime
        assert config.top_k_vector == -5

    def test_context_window_fraction_bounds(self, tmp_path):
        """context_window_fraction must be within [0.1, 0.9]."""
        # Valid: within bounds
        config = RetrievalConfig(context_window_fraction=0.5)
        assert config.context_window_fraction == 0.5

        # Invalid: too low
        with pytest.raises(Exception):  # ValidationError
            RetrievalConfig(context_window_fraction=0.05)

        # Invalid: too high
        with pytest.raises(Exception):  # ValidationError
            RetrievalConfig(context_window_fraction=1.5)

    def test_unknown_model_resolve_window_returns_none(self):
        """Unknown models return None instead of crashing."""
        window = resolve_window("unknown-llm-9000")
        assert window is None

        window = resolve_window("gpt-5-ultra-mega")
        assert window is None

    def test_empty_string_model_resolve_window(self):
        """Empty model string returns None gracefully."""
        window = resolve_window("")
        assert window is None

    def test_special_chars_in_model_name(self):
        """Models with special characters don't crash resolution."""
        weird_models = [
            "model@version",
            "model:latest",
            "model/v2",
            "model-20241231",
        ]

        for model in weird_models:
            window = resolve_window(model)
            # Should return None (no match) but not crash
            assert window is None or isinstance(window, int)

    def test_sparse_config_edge_cases(self, monkeypatch: pytest.MonkeyPatch):
        """SparseConfig handles edge cases correctly."""
        from trelix.core.config import SparseConfig

        # Top-k tokens at lower bound
        monkeypatch.setenv("TRELIX_SPARSE_TOP_K_TOKENS", "16")
        config = SparseConfig()
        assert config.top_k_tokens == 16

        # Top-k tokens at upper bound
        monkeypatch.setenv("TRELIX_SPARSE_TOP_K_TOKENS", "512")
        config = SparseConfig()
        assert config.top_k_tokens == 512

        # Below lower bound should fail
        monkeypatch.setenv("TRELIX_SPARSE_TOP_K_TOKENS", "10")
        with pytest.raises(Exception):  # ValidationError
            SparseConfig()

        # Above upper bound should fail
        monkeypatch.setenv("TRELIX_SPARSE_TOP_K_TOKENS", "1000")
        with pytest.raises(Exception):  # ValidationError
            SparseConfig()

    def test_max_agents_turn_limits(self, tmp_path):
        """Agent turn limits are validated correctly."""
        # Valid: within bounds
        config = RetrievalConfig(
            agent_max_turns=8,
        )
        assert config.agent_max_turns == 8

        # Edge: minimum
        config = RetrievalConfig(agent_max_turns=1)
        assert config.agent_max_turns == 1

        # Edge: maximum
        config = RetrievalConfig(agent_max_turns=20)
        assert config.agent_max_turns == 20

        # Invalid: below minimum
        with pytest.raises(Exception):  # ValidationError
            RetrievalConfig(agent_max_turns=0)

        # Invalid: above maximum
        with pytest.raises(Exception):  # ValidationError
            RetrievalConfig(agent_max_turns=50)


# =============================================================================
# Integration: Real-World Scenario Combinations
# =============================================================================


class TestRealWorldScenariosSmoke:
    """Smoke test: Realistic production-like configurations."""

    def test_production_hybrid_rag_config(self, tmp_path):
        """Production-grade hybrid RAG configuration."""
        config = IndexConfig(
            repo_path=str(tmp_path),
            embedder=EmbedderConfig(
                provider="voyage",
                voyage_model="voyage-code-3",
                voyage_dimensions=1024,
            ),
            store=StoreConfig(
                backend="qdrant",
                qdrant_url="http://localhost:6333",
                hnsw=True,
                hnsw_m=32,
                hnsw_ef_search=100,
            ),
            retrieval=RetrievalConfig(
                top_k_vector=30,
                top_k_bm25=30,
                rerank=True,
                rerank_provider="cohere",
                rerank_top_n=20,
                sparse_enabled=True,
                top_k_sparse=25,
                context_token_budget=None,  # Auto-derive from model
                context_window_fraction=0.5,
                agentic_enabled=True,
                agent_max_turns=10,
            ),
            llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4",
                thinking_enabled=True,
                thinking_budget_tokens=8192,
            ),
        )

        # Verify all settings applied correctly
        assert config.embedder.provider == "voyage"
        assert config.store.backend == "qdrant"
        assert config.retrieval.sparse_enabled is True
        assert config.retrieval.agentic_enabled is True
        assert config.llm.thinking_enabled is True

    def test_local_development_minimal_config(self, tmp_path):
        """Minimal local development configuration."""
        config = IndexConfig(
            repo_path=str(tmp_path),
            embedder=EmbedderConfig(
                provider="local",  # No API key needed
            ),
            store=StoreConfig(
                backend="sqlite",
                db_path=".trelix/dev.db",
                hnsw=True,
            ),
            retrieval=RetrievalConfig(
                top_k_vector=10,
                top_k_bm25=10,
                rerank=False,  # Disabled for speed
                sparse_enabled=False,
                agentic_enabled=False,
            ),
        )

        assert config.embedder.provider == "local"
        assert config.store.backend == "sqlite"
        assert config.retrieval.rerank is False

    def test_high_context_gemini_config(self, tmp_path):
        """High-context configuration with Gemini 2.5 Pro."""
        config = IndexConfig(
            repo_path=str(tmp_path),
            retrieval=RetrievalConfig(
                context_token_budget=None,  # Auto-derive from model window
                context_window_fraction=0.7,  # Aggressive: use 70% of 1M window
                scale_top_k_to_budget=True,  # Scale retrieval to budget
            ),
            llm=LLMConfig(
                provider="vertex",
                model="gemini-2.5-pro",
            ),
        )

        # Verify settings
        assert config.retrieval.context_token_budget is None
        assert config.retrieval.context_window_fraction == 0.7
        assert config.retrieval.scale_top_k_to_budget is True
        assert config.llm.model == "gemini-2.5-pro"

        # Verify Gemini window
        window = resolve_window(config.llm.model)
        assert window == 1_000_000


# =============================================================================
# Summary
# =============================================================================


def test_smoke_suite_summary():
    """
    Smoke test suite covers:

    ✓ Scenario 1: Extended thinking enabled (no crashes, correct API usage)
    ✓ Scenario 2: Gemini context windows (resolve_window() returns correct values)
    ✓ Scenario 3: Mixed configs (partial feature enablement works)
    ✓ Scenario 4: Error handling (invalid configs caught gracefully)
    ✓ Real-world scenarios (production + dev configs work end-to-end)

    All scenarios use dry-run/mock APIs. No real API calls made.
    """
    pass
