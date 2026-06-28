"""Tests for NomicCodeEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig


class TestNomicCodeEmbedder:
    def test_importable(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        assert NomicCodeEmbedder is not None

    def test_is_base_embedder(self) -> None:
        from trelix.embedder.base import BaseEmbedder
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        assert issubclass(NomicCodeEmbedder, BaseEmbedder)

    def test_embed_prepends_doc_task_prefix(self) -> None:
        from trelix.embedder.nomic_code import _DOC_PREFIX, NomicCodeEmbedder

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=mock_model):
            cfg = EmbedderConfig(provider="nomic-code", _env_file=None)
            emb = NomicCodeEmbedder(cfg)
            emb.embed(["def foo(): pass"])
            called_texts = mock_model.encode.call_args[0][0]
            assert called_texts[0].startswith(_DOC_PREFIX)

    def test_embed_query_prepends_query_prefix(self) -> None:
        from trelix.embedder.nomic_code import _QUERY_PREFIX, NomicCodeEmbedder

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=mock_model):
            cfg = EmbedderConfig(provider="nomic-code", _env_file=None)
            emb = NomicCodeEmbedder(cfg)
            emb.embed_query("authentication logic")
            called_texts = mock_model.encode.call_args[0][0]
            assert called_texts[0].startswith(_QUERY_PREFIX)

    def test_dimension(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=mock_model):
            cfg = EmbedderConfig(provider="nomic-code", nomic_code_dimensions=768, _env_file=None)
            emb = NomicCodeEmbedder(cfg)
            assert emb.dimension == 768

    def test_config_effective_dimension(self) -> None:
        cfg = EmbedderConfig(provider="nomic-code", nomic_code_dimensions=768, _env_file=None)
        assert cfg.effective_dimension == 768
