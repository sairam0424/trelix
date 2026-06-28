"""Tests for streaming synthesis."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
from trelix.core.models import RetrievedContext, SearchResult


def _make_context() -> RetrievedContext:
    return RetrievedContext(
        query="how does auth work",
        results=[],
        context_text="def authenticate(user): ...",
        total_tokens=10,
    )


class TestSynthesizerStream:
    def test_stream_method_exists(self) -> None:
        from trelix.retrieval.synthesizer import Synthesizer
        assert hasattr(Synthesizer, "stream")

    def test_stream_yields_strings(self) -> None:
        from trelix.retrieval.synthesizer import Synthesizer
        from trelix.core.config import EmbedderConfig, RetrievalConfig
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(["The ", "auth ", "flow ", "is..."])

        with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=mock_client):
            cfg_emb = EmbedderConfig(_env_file=None)
            cfg_ret = RetrievalConfig()
            synth = Synthesizer(cfg_emb)
            tokens = list(synth.stream(_make_context(), cfg_ret))
            assert len(tokens) > 0
            assert all(isinstance(t, str) for t in tokens)

    def test_stream_falls_back_on_no_api_key(self) -> None:
        from trelix.retrieval.synthesizer import Synthesizer
        from trelix.core.config import EmbedderConfig, RetrievalConfig
        mock_client = MagicMock()
        mock_client.stream.side_effect = Exception("No API key")

        with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=mock_client):
            synth = Synthesizer(EmbedderConfig(_env_file=None))
            tokens = list(synth.stream(_make_context(), RetrievalConfig()))
            # Should yield one error message token, not raise
            assert len(tokens) >= 1
