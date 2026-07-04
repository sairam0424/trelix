"""Tests for HyDE and multi-query expansion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trelix.retrieval.query_expansion import ExpandResult, HyDEExpander, MultiQueryExpander


class TestHyDEExpander:
    def test_returns_empty_string_when_no_llm(self) -> None:
        expander = HyDEExpander(llm_config=None)
        result = expander.expand("how does authentication work")
        assert result == ""

    def test_returns_snippet_when_llm_available(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content="def authenticate(user): ...")
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig

            expander = HyDEExpander(llm_config=LLMConfig())
            result = expander.expand("how does authentication work")
        assert "def authenticate" in result

    def test_returns_empty_on_llm_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("API down")
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig

            expander = HyDEExpander(llm_config=LLMConfig())
            result = expander.expand("how does X work")
        assert result == ""


class TestMultiQueryExpander:
    def test_returns_original_when_no_llm(self) -> None:
        expander = MultiQueryExpander(llm_config=None, n=2)
        result = expander.expand("what handles JWT tokens")
        assert isinstance(result, ExpandResult)
        assert result.queries == ["what handles JWT tokens"]
        assert result.llm_used is False

    def test_returns_n_plus_original_when_llm_available(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content="how is JWT verified\nwhere is token decoded"
        )
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig

            expander = MultiQueryExpander(llm_config=LLMConfig(), n=2)
            result = expander.expand("what handles JWT tokens")
        # Original always included, plus up to n variants
        assert isinstance(result, ExpandResult)
        assert "what handles JWT tokens" in result.queries
        assert len(result.queries) >= 2
        assert result.llm_used is True

    def test_deduplicates_variants(self) -> None:
        mock_client = MagicMock()
        # LLM returns same query as original — should deduplicate
        mock_client.complete.return_value = MagicMock(
            content="what handles JWT tokens\nwhat handles JWT tokens"
        )
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig

            expander = MultiQueryExpander(llm_config=LLMConfig(), n=2)
            result = expander.expand("what handles JWT tokens")
        assert isinstance(result, ExpandResult)
        assert len(result.queries) == len(set(result.queries))  # no duplicates

    def test_returns_original_on_llm_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("API down")
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig

            expander = MultiQueryExpander(llm_config=LLMConfig(), n=2)
            result = expander.expand("what handles JWT tokens")
        assert isinstance(result, ExpandResult)
        assert result.queries == ["what handles JWT tokens"]
        assert result.llm_used is False

    def test_config_flags_default_off(self, tmp_path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path))
        assert cfg.retrieval.hyde_fallback_enabled is False
        assert cfg.retrieval.multi_query_enabled is False
        assert cfg.retrieval.multi_query_count == 2
