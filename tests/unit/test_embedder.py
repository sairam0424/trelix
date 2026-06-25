"""Unit tests for the embedder abstraction (Phase 3 + U2 code-specialised providers)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import EmbedderConfig
from trelix.embedder.base import (
    AzureOpenAIEmbedder,
    BaseEmbedder,
    LocalCodeEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    make_embedder,
)

# Fake credentials used ONLY in tests — these are not real secrets.
_FAKE_OPENAI_KEY = "openai-test-key-not-real"
_FAKE_AZURE_KEY = "azure-test-key-not-real"
_FAKE_AZURE_ENDPOINT = "https://test.openai.azure.com/"


# ---------------------------------------------------------------------------
# BaseEmbedder is abstract
# ---------------------------------------------------------------------------

class TestBaseEmbedderIsAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        """BaseEmbedder must be abstract — direct instantiation must raise TypeError."""
        with pytest.raises(TypeError, match="abstract"):
            BaseEmbedder()  # type: ignore[abstract]

    def test_has_abstract_methods(self) -> None:
        assert len(BaseEmbedder.__abstractmethods__) > 0

    def test_abstract_methods_include_embed_and_embed_query(self) -> None:
        assert "embed" in BaseEmbedder.__abstractmethods__
        assert "embed_query" in BaseEmbedder.__abstractmethods__

    def test_abstract_property_dimension(self) -> None:
        assert "dimension" in BaseEmbedder.__abstractmethods__


# ---------------------------------------------------------------------------
# make_embedder factory
# ---------------------------------------------------------------------------

class TestMakeEmbedderFactory:
    def test_local_provider_returns_local_embedder(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping local provider test",
        )
        config = EmbedderConfig(provider="local")
        embedder = make_embedder(config)
        assert isinstance(embedder, LocalEmbedder)

    def test_openai_provider_returns_openai_embedder(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            embedder = make_embedder(config)
        assert isinstance(embedder, OpenAIEmbedder)

    def test_azure_provider_returns_azure_embedder(self) -> None:
        config = EmbedderConfig(
            provider="azure",
            azure_api_key=_FAKE_AZURE_KEY,
            azure_endpoint=_FAKE_AZURE_ENDPOINT,
        )
        with patch("openai.AzureOpenAI") as mock_azure:
            mock_azure.return_value = MagicMock()
            embedder = make_embedder(config)
        assert isinstance(embedder, AzureOpenAIEmbedder)

    def test_unknown_provider_raises(self) -> None:
        """An unrecognised provider must raise ValueError or similar."""
        config = EmbedderConfig(provider="local")
        # Bypass pydantic validation to inject a bad provider value at runtime.
        object.__setattr__(config, "provider", "nonexistent_provider")
        with pytest.raises((ValueError, Exception)):
            make_embedder(config)  # type: ignore[arg-type]

    def test_factory_returns_base_embedder_subclass(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            embedder = make_embedder(config)
        assert isinstance(embedder, BaseEmbedder)


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------

class TestOpenAIEmbedder:
    def _make(self) -> OpenAIEmbedder:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            return OpenAIEmbedder(config)

    def test_dimension_property(self) -> None:
        embedder = self._make()
        assert embedder.dimension == 3072

    def test_dimension_matches_config(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            embedder = OpenAIEmbedder(config)
        assert embedder.dimension == config.openai_dimensions

    def test_is_base_embedder(self) -> None:
        embedder = self._make()
        assert isinstance(embedder, BaseEmbedder)

    def test_embed_calls_openai_api(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 3072
        mock_client.embeddings.create.return_value.data = [mock_item]

        with patch("openai.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(config)

        result = embedder.embed(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 3072

    def test_embed_query_returns_single_vector(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.5] * 3072
        mock_client.embeddings.create.return_value.data = [mock_item]

        with patch("openai.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(config)

        result = embedder.embed_query("search query")
        assert isinstance(result, list)
        assert len(result) == 3072


# ---------------------------------------------------------------------------
# AzureOpenAIEmbedder
# ---------------------------------------------------------------------------

class TestAzureOpenAIEmbedder:
    def _make(self) -> AzureOpenAIEmbedder:
        config = EmbedderConfig(
            provider="azure",
            azure_api_key=_FAKE_AZURE_KEY,
            azure_endpoint=_FAKE_AZURE_ENDPOINT,
        )
        with patch("openai.AzureOpenAI") as mock_azure:
            mock_azure.return_value = MagicMock()
            return AzureOpenAIEmbedder(config)

    def test_dimension_property(self) -> None:
        embedder = self._make()
        assert embedder.dimension == 3072

    def test_dimension_matches_config(self) -> None:
        config = EmbedderConfig(
            provider="azure",
            azure_api_key=_FAKE_AZURE_KEY,
            azure_endpoint=_FAKE_AZURE_ENDPOINT,
        )
        with patch("openai.AzureOpenAI") as mock_azure:
            mock_azure.return_value = MagicMock()
            embedder = AzureOpenAIEmbedder(config)
        assert embedder.dimension == config.azure_dimensions

    def test_is_base_embedder(self) -> None:
        embedder = self._make()
        assert isinstance(embedder, BaseEmbedder)

    def test_embed_calls_azure_api(self) -> None:
        config = EmbedderConfig(
            provider="azure",
            azure_api_key=_FAKE_AZURE_KEY,
            azure_endpoint=_FAKE_AZURE_ENDPOINT,
        )
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.2] * 3072
        mock_client.embeddings.create.return_value.data = [mock_item]

        with patch("openai.AzureOpenAI", return_value=mock_client):
            embedder = AzureOpenAIEmbedder(config)

        result = embedder.embed(["hello azure"])
        assert len(result) == 1
        assert len(result[0]) == 3072

    def test_embed_query_returns_single_vector(self) -> None:
        config = EmbedderConfig(
            provider="azure",
            azure_api_key=_FAKE_AZURE_KEY,
            azure_endpoint=_FAKE_AZURE_ENDPOINT,
        )
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.3] * 3072
        mock_client.embeddings.create.return_value.data = [mock_item]

        with patch("openai.AzureOpenAI", return_value=mock_client):
            embedder = AzureOpenAIEmbedder(config)

        result = embedder.embed_query("azure query")
        assert isinstance(result, list)
        assert len(result) == 3072


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------

class TestLocalEmbedder:
    def test_dimension_is_384(self) -> None:
        """LocalEmbedder with all-MiniLM-L6-v2 must report 384 dimensions."""
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping LocalEmbedder tests",
        )
        config = EmbedderConfig(provider="local")
        embedder = LocalEmbedder(config)
        assert embedder.dimension == 384

    def test_is_base_embedder(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping LocalEmbedder tests",
        )
        config = EmbedderConfig(provider="local")
        embedder = LocalEmbedder(config)
        assert isinstance(embedder, BaseEmbedder)

    def test_import_error_when_sentence_transformers_missing(self) -> None:
        """When sentence-transformers is not installed, LocalEmbedder must raise
        ImportError with a helpful pip install message."""
        config = EmbedderConfig(provider="local")
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="pip install"):
                LocalEmbedder(config)

    def test_embed_returns_list_of_vectors(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping LocalEmbedder tests",
        )
        config = EmbedderConfig(provider="local")
        embedder = LocalEmbedder(config)
        results = embedder.embed(["hello", "world"])
        assert len(results) == 2
        assert all(len(v) == 384 for v in results)

    def test_embed_query_returns_single_vector(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping LocalEmbedder tests",
        )
        config = EmbedderConfig(provider="local")
        embedder = LocalEmbedder(config)
        result = embedder.embed_query("a single query")
        assert isinstance(result, list)
        assert len(result) == 384


# ---------------------------------------------------------------------------
# VoyageEmbedder
# ---------------------------------------------------------------------------

_FAKE_VOYAGE_KEY = "voyage-test-key-not-real"


class TestVoyageEmbedder:
    """Tests for the Voyage AI code-optimised embedder (voyage-code-3)."""

    def _make_client_mock(self, dim: int = 1024) -> MagicMock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * dim]
        mock_client.embed.return_value = mock_response
        return mock_client

    def _make(self, dim: int = 1024) -> tuple[VoyageEmbedder, MagicMock]:
        config = EmbedderConfig(provider="voyage", voyage_api_key=_FAKE_VOYAGE_KEY)
        mock_client = self._make_client_mock(dim)
        mock_voyage_module = MagicMock()
        mock_voyage_module.Client.return_value = mock_client
        with patch.dict(sys.modules, {"voyageai": mock_voyage_module}):
            embedder = VoyageEmbedder(config)
        # Replace client with mock for assertion purposes
        embedder._client = mock_client  # noqa: SLF001
        return embedder, mock_client

    def test_dimension_property(self) -> None:
        embedder, _ = self._make()
        assert embedder.dimension == 1024

    def test_is_base_embedder(self) -> None:
        embedder, _ = self._make()
        assert isinstance(embedder, BaseEmbedder)

    def test_embed_uses_document_input_type(self) -> None:
        embedder, mock_client = self._make()
        embedder.embed(["def foo(): pass"])
        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args
        assert call_kwargs.kwargs.get("input_type") == "document" or \
               call_kwargs.args[2] == "document" if len(call_kwargs.args) > 2 \
               else call_kwargs.kwargs["input_type"] == "document"

    def test_embed_query_uses_query_input_type(self) -> None:
        embedder, mock_client = self._make()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.5] * 1024]
        mock_client.embed.return_value = mock_response
        embedder.embed_query("find all async functions")
        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args
        assert call_kwargs.kwargs.get("input_type") == "query" or \
               call_kwargs.args[2] == "query" if len(call_kwargs.args) > 2 \
               else call_kwargs.kwargs["input_type"] == "query"

    def test_embed_returns_list_of_vectors(self) -> None:
        embedder, mock_client = self._make()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_client.embed.return_value = mock_response
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert all(len(v) == 1024 for v in result)

    def test_embed_query_returns_single_vector(self) -> None:
        embedder, mock_client = self._make()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.7] * 1024]
        mock_client.embed.return_value = mock_response
        result = embedder.embed_query("search query")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_embed_batches_at_128(self) -> None:
        """VoyageEmbedder must split inputs into chunks of 128."""
        config = EmbedderConfig(provider="voyage", voyage_api_key=_FAKE_VOYAGE_KEY)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.0] * 1024] * 128
        mock_client.embed.return_value = mock_response
        mock_voyage_module = MagicMock()
        mock_voyage_module.Client.return_value = mock_client
        with patch.dict(sys.modules, {"voyageai": mock_voyage_module}):
            embedder = VoyageEmbedder(config)
        embedder._client = mock_client  # noqa: SLF001
        texts = ["text"] * 256
        embedder.embed(texts)
        assert mock_client.embed.call_count == 2

    def test_factory_returns_voyage_embedder(self) -> None:
        config = EmbedderConfig(provider="voyage", voyage_api_key=_FAKE_VOYAGE_KEY)
        mock_voyage_module = MagicMock()
        mock_voyage_module.Client.return_value = MagicMock()
        with patch.dict(sys.modules, {"voyageai": mock_voyage_module}):
            embedder = make_embedder(config)
        assert isinstance(embedder, VoyageEmbedder)

    def test_import_error_with_helpful_message_if_voyageai_missing(self) -> None:
        config = EmbedderConfig(provider="voyage", voyage_api_key=_FAKE_VOYAGE_KEY)
        with patch.dict(sys.modules, {"voyageai": None}):
            with pytest.raises(ImportError, match="pip install"):
                VoyageEmbedder(config)


# ---------------------------------------------------------------------------
# LocalCodeEmbedder
# ---------------------------------------------------------------------------

class TestLocalCodeEmbedder:
    """Tests for the SFR-Embedding-Code-2B_R local code embedder."""

    def test_factory_returns_local_code_embedder(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping local-code provider test",
        )
        config = EmbedderConfig(provider="local-code")
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 4096
        mock_st_module.SentenceTransformer.return_value = mock_model
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            embedder = make_embedder(config)
        assert isinstance(embedder, LocalCodeEmbedder)

    def test_is_base_embedder(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping local-code provider test",
        )
        config = EmbedderConfig(provider="local-code")
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 4096
        mock_st_module.SentenceTransformer.return_value = mock_model
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            embedder = LocalCodeEmbedder(config)
        assert isinstance(embedder, BaseEmbedder)

    def test_dimension_uses_model_method(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping local-code provider test",
        )
        config = EmbedderConfig(provider="local-code")
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 4096
        mock_st_module.SentenceTransformer.return_value = mock_model
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            embedder = LocalCodeEmbedder(config)
        assert embedder.dimension == 4096

    def test_dimension_fallback_is_4096(self) -> None:
        """If model has no dimension method, fallback must be 4096."""
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping local-code provider test",
        )
        config = EmbedderConfig(provider="local-code")
        mock_st_module = MagicMock()
        mock_model = MagicMock(spec=[])  # no methods
        mock_st_module.SentenceTransformer.return_value = mock_model
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            embedder = LocalCodeEmbedder(config)
        assert embedder.dimension == 4096

    def test_import_error_when_sentence_transformers_missing(self) -> None:
        config = EmbedderConfig(provider="local-code")
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="pip install"):
                LocalCodeEmbedder(config)

    def test_trust_remote_code_true(self) -> None:
        """SentenceTransformer must be called with trust_remote_code=True."""
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping local-code provider test",
        )
        config = EmbedderConfig(provider="local-code")
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 4096
        mock_st_module.SentenceTransformer.return_value = mock_model
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            LocalCodeEmbedder(config)
        call_kwargs = mock_st_module.SentenceTransformer.call_args
        assert call_kwargs.kwargs.get("trust_remote_code") is True or \
               (len(call_kwargs.args) > 1 and call_kwargs.args[1] is True)

