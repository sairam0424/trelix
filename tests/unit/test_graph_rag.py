"""Unit tests for GraphRAG map-reduce synthesis (U8)."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig, RetrievalConfig
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    RetrievedContext,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.graph_rag import GraphRAGSynthesizer
from trelix.retrieval.synthesizer import Synthesizer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_file(file_id: int, rel_path: str = "src/foo.py") -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash=f"sha-{file_id}",
        size_bytes=1000,
        id=file_id,
        indexed_at=datetime(2024, 1, 1),
    )


def _make_symbol(sym_id: int, file_id: int, name: str = "func") -> Symbol:
    return Symbol(
        file_id=file_id,
        name=name,
        qualified_name=f"module.{name}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def {name}()",
        body=f"def {name}():\n    pass",
        id=sym_id,
    )


def _make_chunk(sym_id: int, text: str = "def func(): pass") -> Chunk:
    return Chunk(
        symbol_id=sym_id,
        chunk_text=text,
        token_count=len(text.split()),
        id=sym_id,
    )


def _make_result(idx: int, tokens: int = 100) -> SearchResult:
    file = _make_file(idx)
    sym = _make_symbol(idx, idx, name=f"func_{idx}")
    chunk = _make_chunk(idx, text=f"def func_{idx}(): pass  " + "x " * tokens)
    return SearchResult(chunk=chunk, symbol=sym, file=file, score=0.9, rank=idx, source="vector")


def _make_context(num_results: int = 5, total_tokens: int = 1000) -> RetrievedContext:
    results = [_make_result(i) for i in range(num_results)]
    return RetrievedContext(
        query="how does authentication work?",
        results=results,
        context_text="some context text",
        total_tokens=total_tokens,
        intent="feature_flow",
    )


def _make_mock_llm_client(response_text: str = "Partial answer text") -> MagicMock:
    """Build a mock OpenAI client whose .chat.completions.create() returns response_text."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _make_embedder_config(provider: str = "openai") -> EmbedderConfig:
    """Return an EmbedderConfig for testing (provider selected by argument)."""
    # Use environment patching to avoid hardcoded secret values
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}):
        return EmbedderConfig(provider=provider)  # type: ignore[arg-type]


def _make_retrieval_config(
    graph_rag_enabled: bool = True,
    graph_rag_threshold_tokens: int = 8000,
    graph_rag_threshold_results: int = 20,
) -> RetrievalConfig:
    """Build a RetrievalConfig with GraphRAG settings, using env var aliasing."""
    env_patch = {
        "TRELIX_RETRIEVAL_GRAPH_RAG": str(graph_rag_enabled).lower(),
        "TRELIX_RETRIEVAL_GRAPH_RAG_THRESHOLD_TOKENS": str(graph_rag_threshold_tokens),
        "TRELIX_RETRIEVAL_GRAPH_RAG_THRESHOLD_RESULTS": str(graph_rag_threshold_results),
    }
    with patch.dict(os.environ, env_patch):
        return RetrievalConfig()


# ---------------------------------------------------------------------------
# should_use() tests
# ---------------------------------------------------------------------------


class TestShouldUse:
    def _make_synthesizer(
        self,
        graph_rag_enabled: bool = True,
        graph_rag_threshold_tokens: int = 8000,
        graph_rag_threshold_results: int = 20,
    ) -> GraphRAGSynthesizer:
        ec = _make_embedder_config()
        rc = _make_retrieval_config(
            graph_rag_enabled=graph_rag_enabled,
            graph_rag_threshold_tokens=graph_rag_threshold_tokens,
            graph_rag_threshold_results=graph_rag_threshold_results,
        )
        synth = GraphRAGSynthesizer(ec, rc)
        # Inject a mock client so should_use doesn't bail on missing creds
        synth._client = _make_mock_llm_client()
        return synth

    def test_true_when_results_exceed_threshold(self) -> None:
        """should_use() returns True when results > graph_rag_threshold_results."""
        synth = self._make_synthesizer(graph_rag_threshold_results=20)
        context = _make_context(num_results=21, total_tokens=100)
        assert synth.should_use(context) is True

    def test_true_when_tokens_exceed_threshold(self) -> None:
        """should_use() returns True when total_tokens > graph_rag_threshold_tokens."""
        synth = self._make_synthesizer(graph_rag_threshold_tokens=8000)
        context = _make_context(num_results=3, total_tokens=8001)
        assert synth.should_use(context) is True

    def test_false_for_small_context(self) -> None:
        """should_use() returns False when context is below both thresholds."""
        synth = self._make_synthesizer(
            graph_rag_threshold_tokens=8000,
            graph_rag_threshold_results=20,
        )
        context = _make_context(num_results=5, total_tokens=1000)
        assert synth.should_use(context) is False

    def test_false_when_disabled(self) -> None:
        """should_use() returns False when graph_rag_enabled=False."""
        synth = self._make_synthesizer(
            graph_rag_enabled=False,
            graph_rag_threshold_tokens=8000,
            graph_rag_threshold_results=20,
        )
        # Context that would normally trigger GraphRAG
        context = _make_context(num_results=25, total_tokens=9000)
        assert synth.should_use(context) is False

    def test_false_when_no_client(self) -> None:
        """should_use() returns False when LLM client is None (no creds)."""
        ec = _make_embedder_config(provider="local")
        rc = _make_retrieval_config()
        synth = GraphRAGSynthesizer(ec, rc)
        # local provider -> _client is None
        context = _make_context(num_results=25, total_tokens=9000)
        assert synth.should_use(context) is False

    def test_false_at_exact_token_threshold(self) -> None:
        """should_use() returns False when total_tokens == threshold (not >)."""
        synth = self._make_synthesizer(graph_rag_threshold_tokens=8000)
        context = _make_context(num_results=3, total_tokens=8000)
        assert synth.should_use(context) is False

    def test_true_one_above_token_threshold(self) -> None:
        """should_use() returns True one token above threshold."""
        synth = self._make_synthesizer(graph_rag_threshold_tokens=8000)
        context = _make_context(num_results=3, total_tokens=8001)
        assert synth.should_use(context) is True

    def test_false_at_exact_results_threshold(self) -> None:
        """should_use() returns False when results == threshold (not >)."""
        synth = self._make_synthesizer(graph_rag_threshold_results=20)
        context = _make_context(num_results=20, total_tokens=100)
        assert synth.should_use(context) is False

    def test_true_one_above_results_threshold(self) -> None:
        """should_use() returns True one result above threshold."""
        synth = self._make_synthesizer(graph_rag_threshold_results=20)
        context = _make_context(num_results=21, total_tokens=100)
        assert synth.should_use(context) is True


# ---------------------------------------------------------------------------
# MAP phase tests
# ---------------------------------------------------------------------------


class TestMapPhase:
    def _make_synthesizer_with_mock(
        self, response_text: str = "partial answer"
    ) -> tuple[GraphRAGSynthesizer, MagicMock]:
        ec = _make_embedder_config()
        rc = _make_retrieval_config()
        synth = GraphRAGSynthesizer(ec, rc)
        mock_client = _make_mock_llm_client(response_text)
        synth._client = mock_client
        return synth, mock_client

    def test_map_calls_llm_once_per_group_of_10(self) -> None:
        """MAP phase calls LLM once per group of 10 results."""
        synth, mock_client = self._make_synthesizer_with_mock("partial answer")
        # 25 results -> 3 groups (10, 10, 5)
        context = _make_context(num_results=25, total_tokens=9000)

        synth.synthesize("query", context, "feature_flow")

        # 3 map calls + 1 reduce call = 4 total
        assert mock_client.chat.completions.create.call_count == 4

    def test_map_calls_llm_once_for_single_group(self) -> None:
        """MAP phase calls LLM once when all results fit in one group."""
        synth, mock_client = self._make_synthesizer_with_mock("partial answer")
        # 8 results -> 1 group
        context = _make_context(num_results=8, total_tokens=9000)

        synth.synthesize("query", context, "symbol_lookup")

        # 1 map call + 1 reduce call = 2 total
        assert mock_client.chat.completions.create.call_count == 2

    def test_map_uses_max_400_tokens(self) -> None:
        """MAP calls use max_completion_tokens=400."""
        synth, mock_client = self._make_synthesizer_with_mock("partial")
        context = _make_context(num_results=5, total_tokens=9000)

        synth.synthesize("query", context, "feature_flow")

        # First call is the MAP call
        first_call_kwargs = mock_client.chat.completions.create.call_args_list[0].kwargs
        assert first_call_kwargs.get("max_completion_tokens") == 400

    def test_split_into_groups_10_per_group(self) -> None:
        """_split_into_groups produces groups of at most 10."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config()
        synth = GraphRAGSynthesizer(ec, rc)

        results = [_make_result(i) for i in range(25)]
        groups = synth._split_into_groups(results)

        assert len(groups) == 3
        assert len(groups[0]) == 10
        assert len(groups[1]) == 10
        assert len(groups[2]) == 5


# ---------------------------------------------------------------------------
# REDUCE phase tests
# ---------------------------------------------------------------------------


class TestReducePhase:
    def test_reduce_combines_partial_answers(self) -> None:
        """REDUCE phase calls LLM with all partial answers joined."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config()
        synth = GraphRAGSynthesizer(ec, rc)

        call_count = 0
        received_prompts: list[str] = []

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            user_msg = kwargs["messages"][-1]["content"]
            received_prompts.append(user_msg)
            mock_message = MagicMock()
            mock_message.content = f"response_{call_count}"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create
        synth._client = mock_client

        # 15 results -> 2 MAP groups + 1 REDUCE
        context = _make_context(num_results=15, total_tokens=9000)
        synth.synthesize("explain auth", context, "feature_flow")

        # Should have 2 MAP + 1 REDUCE = 3 calls
        assert call_count == 3
        # The REDUCE prompt should reference both partial answers
        reduce_prompt = received_prompts[-1]
        assert "[Partial 1]" in reduce_prompt
        assert "[Partial 2]" in reduce_prompt
        assert "explain auth" in reduce_prompt

    def test_reduce_uses_max_1500_tokens(self) -> None:
        """REDUCE call uses max_completion_tokens=1500."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config()
        synth = GraphRAGSynthesizer(ec, rc)
        mock_client = _make_mock_llm_client("final answer")
        synth._client = mock_client

        context = _make_context(num_results=5, total_tokens=9000)
        synth.synthesize("query", context, "feature_flow")

        last_call_kwargs = mock_client.chat.completions.create.call_args_list[-1].kwargs
        assert last_call_kwargs.get("max_completion_tokens") == 1500

    def test_reduce_returns_final_answer(self) -> None:
        """synthesize() returns the REDUCE phase output."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config()
        synth = GraphRAGSynthesizer(ec, rc)

        responses = ["partial_1", "partial_2", "FINAL SYNTHESIZED ANSWER"]
        call_count = [0]

        def fake_create(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            mock_message = MagicMock()
            mock_message.content = responses[idx] if idx < len(responses) else "extra"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create
        synth._client = mock_client

        # 15 results -> 2 MAP groups
        context = _make_context(num_results=15, total_tokens=9000)
        result = synth.synthesize("query", context, "feature_flow")
        assert result == "FINAL SYNTHESIZED ANSWER"


# ---------------------------------------------------------------------------
# Synthesizer delegation tests
# ---------------------------------------------------------------------------


class TestSynthesizerDelegation:
    def test_delegates_to_graph_rag_when_threshold_exceeded(self) -> None:
        """Synthesizer delegates to GraphRAGSynthesizer for large contexts."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config(
            graph_rag_enabled=True,
            graph_rag_threshold_tokens=8000,
            graph_rag_threshold_results=20,
        )
        synth = Synthesizer(ec, rc)
        synth._client = _make_mock_llm_client("streamed output")

        large_context = _make_context(num_results=25, total_tokens=9000)

        # Patch the class in the graph_rag module (where it's defined)
        # because synthesizer imports it with
        # `from trelix.retrieval.graph_rag import GraphRAGSynthesizer`
        with patch("trelix.retrieval.graph_rag.GraphRAGSynthesizer") as mock_graph_rag_cls:
            mock_graph_rag_instance = MagicMock()
            mock_graph_rag_instance.should_use.return_value = True
            mock_graph_rag_instance.synthesize.return_value = "GraphRAG answer"
            mock_graph_rag_cls.return_value = mock_graph_rag_instance

            result = synth.synthesize(large_context)

        mock_graph_rag_instance.should_use.assert_called_once_with(large_context)
        mock_graph_rag_instance.synthesize.assert_called_once_with(
            large_context.query, large_context, large_context.intent
        )
        assert result == "GraphRAG answer"

    def test_uses_normal_path_for_small_context(self) -> None:
        """Synthesizer uses the standard streaming path for small contexts."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config(
            graph_rag_enabled=True,
            graph_rag_threshold_tokens=8000,
            graph_rag_threshold_results=20,
        )
        synth = Synthesizer(ec, rc)

        # Build a streaming mock client
        mock_delta = MagicMock()
        mock_delta.content = "streamed token "
        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [mock_chunk]
        synth._client = mock_client

        small_context = _make_context(num_results=5, total_tokens=500)

        with patch("trelix.retrieval.graph_rag.GraphRAGSynthesizer") as mock_graph_rag_cls:
            mock_graph_rag_instance = MagicMock()
            mock_graph_rag_instance.should_use.return_value = False
            mock_graph_rag_cls.return_value = mock_graph_rag_instance

            result = synth.synthesize(small_context)

        # GraphRAG was checked but not invoked for synthesis
        mock_graph_rag_instance.should_use.assert_called_once_with(small_context)
        mock_graph_rag_instance.synthesize.assert_not_called()
        # Standard streaming path was used
        mock_client.chat.completions.create.assert_called_once()
        assert "streamed token" in result

    def test_graph_rag_not_checked_when_no_results(self) -> None:
        """Synthesizer returns early when context.results is empty — no GraphRAG check."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config()
        synth = Synthesizer(ec, rc)
        synth._client = _make_mock_llm_client("answer")

        empty_context = _make_context(num_results=0, total_tokens=0)

        with patch("trelix.retrieval.graph_rag.GraphRAGSynthesizer") as mock_cls:
            result = synth.synthesize(empty_context)

        mock_cls.assert_not_called()
        assert "No relevant code found" in result

    def test_synthesizer_accepts_retrieval_config_in_constructor(self) -> None:
        """Synthesizer accepts an optional RetrievalConfig in its constructor."""
        ec = _make_embedder_config()
        rc = _make_retrieval_config(
            graph_rag_enabled=False,
            graph_rag_threshold_tokens=5000,
            graph_rag_threshold_results=10,
        )
        synth = Synthesizer(ec, rc)
        assert synth._retrieval_config.graph_rag_enabled is False
        assert synth._retrieval_config.graph_rag_threshold_tokens == 5000

    def test_synthesizer_defaults_retrieval_config_when_not_supplied(self) -> None:
        """Synthesizer creates a default RetrievalConfig when none is passed."""
        ec = _make_embedder_config()
        synth = Synthesizer(ec)
        assert synth._retrieval_config is not None
        assert synth._retrieval_config.graph_rag_threshold_tokens == 8000
        assert synth._retrieval_config.graph_rag_threshold_results == 20
