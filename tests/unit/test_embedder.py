"""Unit tests for the embedder abstraction (Phase 3)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import EmbedderConfig
from trelix.embedder.base import (
    AavaPlatformEmbedder,
    AzureOpenAIEmbedder,
    BaseEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
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
# AavaPlatformEmbedder — VS Code plugin compatibility
# ---------------------------------------------------------------------------

# Placeholder values used only in tests — not real credentials.
_TEST_AAVA_AUTH = "aava-test-placeholder"
_TEST_AAVA_URL = "https://aava-test.avateam.io"


class TestAavaPlatformEmbedder:
    def _make_config(self) -> EmbedderConfig:
        return EmbedderConfig(
            provider="aava",
            embedding_bearer_token=_TEST_AAVA_AUTH,
            embedding_base_url=_TEST_AAVA_URL,
        )

    def test_is_base_embedder(self) -> None:
        embedder = AavaPlatformEmbedder(self._make_config())
        assert isinstance(embedder, BaseEmbedder)

    def test_dimension_is_3072(self) -> None:
        embedder = AavaPlatformEmbedder(self._make_config())
        assert embedder.dimension == 3072

    def test_factory_returns_aava_embedder(self) -> None:
        embedder = make_embedder(self._make_config())
        assert isinstance(embedder, AavaPlatformEmbedder)

    def test_embed_posts_to_api(self) -> None:
        """embed() must POST to the Aava embedding endpoint and parse the response."""
        import json
        from unittest.mock import MagicMock, patch

        embedder = AavaPlatformEmbedder(self._make_config())

        fake_response_data = {
            "data": {"embeddings": [{"vector": [0.1] * 3072}]}
        }
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = embedder.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 3072

    def test_embed_query_returns_single_vector(self) -> None:
        """embed_query() must return a single flat list of floats."""
        import json
        from unittest.mock import MagicMock, patch

        embedder = AavaPlatformEmbedder(self._make_config())

        fake_response_data = {
            "data": {"embeddings": [{"vector": [0.5] * 3072}]}
        }
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = embedder.embed_query("search query")

        assert isinstance(result, list)
        assert len(result) == 3072
