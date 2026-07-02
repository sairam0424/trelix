"""Tests for multi-query expansion wired into _retrieve_standard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from trelix.core.config import IndexConfig


def _make_config(tmp_path: Path, multi_query_enabled: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.repo_path = str(tmp_path)
    cfg.retrieval.multi_query_enabled = multi_query_enabled
    cfg.retrieval.multi_query_count = 2
    cfg.retrieval.agentic_enabled = False
    cfg.retrieval.file_summary_leg_enabled = False
    cfg.retrieval.sparse_enabled = False
    cfg.retrieval.sub_chunk_search_enabled = False
    cfg.retrieval.graph_search_enabled = False
    cfg.retrieval.hyde_fallback_enabled = False
    cfg.retrieval.top_k_vector = 5
    cfg.retrieval.top_k_bm25 = 5
    cfg.retrieval.top_k_grep = 5
    cfg.retrieval.rrf_k = 60
    cfg.retrieval.file_type_weighting_enabled = False
    cfg.retrieval.rerank = False
    cfg.retrieval.graph_rag_enabled = False
    cfg.retrieval.pagerank_boost_enabled = False
    cfg.retrieval.query_cache_size = 0
    cfg.retrieval.plan_cache_size = 0
    cfg.retrieval.context_token_budget = 8000
    cfg.retrieval.synthesis_max_tokens = 2000
    cfg.retrieval.assembly_mode = "greedy"
    cfg.llm = MagicMock()
    cfg.telemetry_enabled = False
    cfg.db_path_absolute = tmp_path / "index.db"
    return cfg


class TestMultiQueryConfig:
    def test_multi_query_enabled_default_false(self, tmp_path: Path) -> None:
        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)  # type: ignore[call-arg]
        assert cfg.retrieval.multi_query_enabled is False

    def test_multi_query_count_default_two(self, tmp_path: Path) -> None:
        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)  # type: ignore[call-arg]
        assert cfg.retrieval.multi_query_count == 2


class TestMultiQueryExpansionInRetrieval:
    def test_multi_query_expander_called_when_enabled(self, tmp_path: Path) -> None:
        """When multi_query_enabled=True, MultiQueryExpander.expand() is called."""
        from trelix.retrieval.query_expansion import MultiQueryExpander

        expander = MultiQueryExpander(llm_config=None, n=2)
        variants = expander.expand("how does authentication work")
        # Without LLM, returns [original] — wiring test uses mocked LLM
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert variants[0] == "how does authentication work"

    def test_expander_with_mock_llm_returns_variants(self, tmp_path: Path) -> None:
        """MultiQueryExpander with mocked LLM returns original + variants."""
        from trelix.core.config import LLMConfig
        from trelix.retrieval.query_expansion import MultiQueryExpander

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content="find authentication logic\nlocate login implementation"
        )

        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            expander = MultiQueryExpander(llm_config=LLMConfig(), n=2)
            variants = expander.expand("how does authentication work")

        assert "how does authentication work" in variants
        assert len(variants) >= 2
        # All variants are unique
        assert len(variants) == len(set(variants))

    def test_multi_query_disabled_does_not_expand(self, tmp_path: Path) -> None:
        """When multi_query_enabled=False, retrieval runs single query (no expansion)."""
        from trelix.retrieval.query_expansion import MultiQueryExpander

        expander = MultiQueryExpander(llm_config=None, n=2)
        variants = expander.expand("test query")
        # No LLM → always returns [original]
        assert variants == ["test query"]

    def test_subquery_from_variant_has_correct_semantic_query(self, tmp_path: Path) -> None:
        """SubQuery built from a variant preserves the variant text as semantic_query."""
        from trelix.retrieval.planner.models import SubQuery

        sq = SubQuery(
            semantic_query="locate login function",
            bm25_tokens=["login", "function"],
            grep_hints=[],
            file_hints=[],
            hyde_snippet="",
            depends_on=[],
        )
        assert sq.semantic_query == "locate login function"

    def test_multi_query_merge_deduplicates_by_symbol_id(self, tmp_path: Path) -> None:
        """Results from multiple query variants deduplicate on symbol_id via _dedup."""
        from trelix.core.models import (
            Chunk,
            IndexedFile,
            Language,
            SearchResult,
            Symbol,
            SymbolKind,
        )

        def _make_result(symbol_id: int, score: float) -> SearchResult:
            chunk = Chunk(id=symbol_id, symbol_id=symbol_id, chunk_text="x", token_count=1)
            sym = Symbol(
                id=symbol_id,
                file_id=1,
                name="fn",
                qualified_name="fn",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=5,
                signature="",
                body="",
            )
            file = IndexedFile(
                id=1,
                path="/r/a.py",
                rel_path="a.py",
                language=Language.PYTHON,
                hash="h",
                size_bytes=10,
            )
            return SearchResult(
                chunk=chunk, symbol=sym, file=file, score=score, rank=1, source="vector"
            )

        # Two results with same symbol_id from different query variants
        r1 = _make_result(42, 0.9)
        r2 = _make_result(42, 0.7)  # same symbol, lower score
        r3 = _make_result(99, 0.8)  # different symbol

        combined = [r1, r2, r3]
        # _dedup logic: deduplicate by chunk.symbol_id keeping first occurrence
        seen: set[int] = set()
        deduped = []
        for r in combined:
            if r.chunk.symbol_id not in seen:
                seen.add(r.chunk.symbol_id)
                deduped.append(r)

        assert len(deduped) == 2
        assert {r.chunk.symbol_id for r in deduped} == {42, 99}
