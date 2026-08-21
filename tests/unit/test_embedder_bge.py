"""Tests for BGECodeEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig


class _FakeHFConfig:
    hidden_size = 1536  # BAAI/bge-code-v1 config.json hidden_size


class _FakeHFModel:
    config = _FakeHFConfig()


class _FakeFlagModel:
    """FlagModel's real attribute surface — and nothing more.

    Checked against the FlagEmbedding 1.3.5 sdist: `inference.embedder.encoder_only.base.
    BaseEmbedder(AbsEmbedder)` defines only __init__ / encode / encode_corpus /
    encode_queries / encode_single_device / pooling, on top of AbsEmbedder's __del__,
    encode*, get_detailed_instruct, get_target_devices and multi-process helpers.
    `get_sentence_embedding_dimension` is in neither class, in no release (1.2.11, 1.3.5,
    1.4.0), and neither defines `__getattr__`. Both bases assign `self.model =
    AutoModel.from_pretrained(...)`, so `.model.config.hidden_size` is where the width is.

    A plain class, deliberately: `MagicMock` grows whatever attribute it is asked for,
    which is how a property that raised `AttributeError` in production passed its test.
    """

    def __init__(
        self, model_name_or_path: str, truncate_dim: int | None = None, **_: object
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.truncate_dim = truncate_dim
        self.model = _FakeHFModel()
        self.tokenizer = object()

    def _vecs(self, n: int) -> list[list[float]]:
        return [[0.01] * (self.truncate_dim or _FakeHFConfig.hidden_size) for _ in range(n)]

    def encode(self, sentences: list[str], batch_size: int = 256, **_: object) -> list[list[float]]:
        return self._vecs(len(sentences))

    encode_corpus = encode

    def encode_queries(self, queries: list[str], **_: object) -> list[list[float]]:
        return self._vecs(len(queries))


class TestBGECodeEmbedder:
    def test_importable(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        assert BGECodeEmbedder is not None

    def test_is_base_embedder(self) -> None:
        from trelix.embedder.base import BaseEmbedder
        from trelix.embedder.bge_code import BGECodeEmbedder

        assert issubclass(BGECodeEmbedder, BaseEmbedder)

    def test_dimension_is_the_width_that_actually_comes_back(self) -> None:
        """1536, and read off the model rather than from a method FlagEmbedding lacks."""
        from trelix.embedder.bge_code import BGECodeEmbedder

        with patch("trelix.embedder.bge_code.FlagModel", _FakeFlagModel):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 1536
            assert len(emb.embed(["def foo(): pass"])[0]) == emb.dimension
            assert len(emb.embed_query("how does auth work")) == emb.dimension

    def test_dimension_honours_matryoshka_truncation(self) -> None:
        """`truncate_dim` narrows the output, so it must win over `hidden_size`."""
        from trelix.embedder.bge_code import BGECodeEmbedder

        def _truncated(name: str, **kwargs: object) -> _FakeFlagModel:
            return _FakeFlagModel(name, truncate_dim=512, **kwargs)

        with patch("trelix.embedder.bge_code.FlagModel", _truncated):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 512
            assert len(emb.embed(["def foo(): pass"])[0]) == 512

    def test_dimension_falls_back_to_config_and_never_raises(self) -> None:
        """No readable HF config — return the configured width, do not raise."""
        from trelix.embedder.bge_code import BGECodeEmbedder

        class _Opaque(_FakeFlagModel):
            def __init__(self, name: str, **kwargs: object) -> None:
                super().__init__(name, **kwargs)
                self.model = None

        with patch("trelix.embedder.bge_code.FlagModel", _Opaque):
            cfg = EmbedderConfig(provider="bge-code", bge_code_dimensions=4096, _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 4096

    def test_embed_returns_correct_shape(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        mock_model = MagicMock()
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
        mock_model.encode_queries.return_value = [[0.1] * 768]
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            emb.embed_query("how does auth work")
            assert mock_model.encode_queries.called
            called_texts = mock_model.encode_queries.call_args[0][0]
            # no manual prefix — encode_queries adds it internally
            assert called_texts == ["how does auth work"]

    def test_make_embedder_returns_bge(self) -> None:
        from trelix.embedder.base import make_embedder

        mock_model = MagicMock()
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            from trelix.embedder.bge_code import BGECodeEmbedder

            emb = make_embedder(cfg)
            assert isinstance(emb, BGECodeEmbedder)

    def test_config_effective_dimension(self) -> None:
        """The default, not an argument echoed back — the default is what was wrong."""
        cfg = EmbedderConfig(provider="bge-code", _env_file=None)
        assert cfg.bge_code_dimensions == 1536
        assert cfg.effective_dimension == 1536
