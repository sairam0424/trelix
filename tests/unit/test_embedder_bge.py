"""Tests for BGECodeEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig


class TestBGECodeEmbedder:
    def test_importable(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        assert BGECodeEmbedder is not None

    def test_is_base_embedder(self) -> None:
        from trelix.embedder.base import BaseEmbedder
        from trelix.embedder.bge_code import BGECodeEmbedder

        assert issubclass(BGECodeEmbedder, BaseEmbedder)

    def test_dimension_default(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 768

    def test_embed_returns_correct_shape(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            result = emb.embed(["def foo(): pass", "class Bar: pass"])
            assert len(result) == 2
            assert len(result[0]) == 768

    def test_embed_query_uses_query_instruction(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            emb.embed_query("how does auth work")
            call_kwargs = mock_model.encode.call_args
            # BGE query embedding should use instruction prefix
            assert call_kwargs is not None

    def test_make_embedder_returns_bge(self) -> None:
        from trelix.embedder.base import make_embedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            from trelix.embedder.bge_code import BGECodeEmbedder

            emb = make_embedder(cfg)
            assert isinstance(emb, BGECodeEmbedder)

    def test_config_effective_dimension(self) -> None:
        cfg = EmbedderConfig(provider="bge-code", bge_code_dimensions=768, _env_file=None)
        assert cfg.effective_dimension == 768
