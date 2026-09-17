"""Unit tests for the embedder abstraction (Phase 3 + U2 code-specialised providers)."""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from trelix.core.config import EmbedderConfig
from trelix.embedder.base import (
    REMOTE_MODEL_CODE_ENV_VAR,
    AzureOpenAIEmbedder,
    BaseEmbedder,
    BatchJobIncompleteError,
    BatchJobTerminalError,
    BedrockCohereEmbedder,
    BedrockTitanEmbedder,
    LocalCodeEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    make_embedder,
)
from trelix.embedder.cohere import CohereEmbedder


def _status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError(f"{status_code} error", response=response, body=None)


# Fake credentials used ONLY in tests — these are not real secrets.
_FAKE_OPENAI_KEY = "openai-test-key-not-real"
_FAKE_AZURE_KEY = "azure-test-key-not-real"
_FAKE_AZURE_ENDPOINT = "https://test.openai.azure.com/"


def _mock_boto3_modules(mock_boto3: MagicMock) -> dict[str, MagicMock]:
    """sys.modules patch dict covering both boto3 and botocore.config —
    _BedrockEmbedderBase._make_boto3_client() imports both (`import boto3`
    + `from botocore.config import Config`), but botocore isn't installed
    in every environment that runs this test suite (it's a transitive
    dependency of the optional 'bedrock' extra, not a base dependency) —
    CI's own test job never installs it. Mocking boto3 alone is not
    enough."""
    botocore_config_mock = MagicMock()
    botocore_mock = MagicMock()
    botocore_mock.config = botocore_config_mock
    return {
        "boto3": mock_boto3,
        "botocore": botocore_mock,
        "botocore.config": botocore_config_mock,
    }


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
# OpenAIEmbedder — Batch API (submit_batch / poll_batch)
#
# The OpenAI Batch API (24h completion window, 50% cost discount) is
# architecturally distinct from the ordinary request batching embed() already
# does via config.batch_size — it is trelix's first long-running external job
# pattern: submit a job, poll it later, retrieve results from an output file.
# Scoped to OpenAIEmbedder only — voyageai==0.5.0's Client has no batch-job
# mechanism at all (verified against the installed SDK), so VoyageEmbedder is
# deliberately untouched.
# ---------------------------------------------------------------------------


class TestOpenAIEmbedderBatchAPI:
    def test_submit_batch_returns_job_id(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.files.create.return_value = MagicMock(id="file_xyz")
            mock_client.batches.create.return_value = MagicMock(id="batch_xyz")
            mock_openai_cls.return_value = mock_client
            embedder = OpenAIEmbedder(config)
            job_id = embedder.submit_batch(["def foo(): pass", "def bar(): pass"])

        assert job_id == "batch_xyz"
        mock_client.batches.create.assert_called_once()

        # The input file must be JSONL, one line per text, custom_id = index,
        # each line a full /v1/embeddings request body — OpenAI's documented
        # batch input format.
        uploaded_file = mock_client.files.create.call_args.kwargs["file"]
        uploaded_lines = uploaded_file.getvalue().decode("utf-8").strip().split("\n")
        assert len(uploaded_lines) == 2
        first = json.loads(uploaded_lines[0])
        assert first == {
            "custom_id": "0",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": config.openai_model,
                "input": "def foo(): pass",
                "dimensions": config.openai_dimensions,
            },
        }
        assert mock_client.files.create.call_args.kwargs["purpose"] == "batch"

        create_kwargs = mock_client.batches.create.call_args.kwargs
        assert create_kwargs["completion_window"] == "24h"
        assert create_kwargs["endpoint"] == "/v1/embeddings"
        assert create_kwargs["input_file_id"] == "file_xyz"

    def test_poll_batch_returns_none_while_processing(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.batches.retrieve.return_value = MagicMock(status="in_progress")
            mock_openai_cls.return_value = mock_client
            embedder = OpenAIEmbedder(config)
            assert embedder.poll_batch("batch_xyz") is None

        mock_client.batches.retrieve.assert_called_once_with("batch_xyz")

    def test_poll_batch_returns_vectors_when_completed(self) -> None:
        """Output file lines are not guaranteed to preserve input order, so
        poll_batch() must re-sort on custom_id — this test deliberately
        supplies the completed-job lines out of order to prove that."""
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.batches.retrieve.return_value = MagicMock(
                status="completed", output_file_id="file_out_123"
            )
            jsonl_lines = "\n".join(
                [
                    json.dumps(
                        {
                            "custom_id": "1",
                            "response": {"body": {"data": [{"embedding": [0.2] * 3072}]}},
                        }
                    ),
                    json.dumps(
                        {
                            "custom_id": "0",
                            "response": {"body": {"data": [{"embedding": [0.1] * 3072}]}},
                        }
                    ),
                ]
            )
            mock_client.files.content.return_value = MagicMock(text=jsonl_lines)
            mock_openai_cls.return_value = mock_client
            embedder = OpenAIEmbedder(config)
            vectors = embedder.poll_batch("batch_xyz")

        assert vectors == [[0.1] * 3072, [0.2] * 3072]
        mock_client.files.content.assert_called_once_with("file_out_123")

    def test_poll_batch_raises_on_failed_status(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.batches.retrieve.return_value = MagicMock(status="failed")
            mock_openai_cls.return_value = mock_client
            embedder = OpenAIEmbedder(config)
            with pytest.raises(BatchJobTerminalError, match="failed"):
                embedder.poll_batch("batch_xyz")

    def test_poll_batch_raises_incomplete_error_when_a_request_failed(self) -> None:
        """OpenAI can report overall status "completed" while an individual
        request inside the batch failed -- its output line has response=None
        rather than embedding data. A positional zip downstream would silently
        mis-pair every later chunk with the wrong vector, so poll_batch must
        raise loudly instead of returning a short/wrong list."""
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.batches.retrieve.return_value = MagicMock(
                status="completed", output_file_id="file_out_123"
            )
            jsonl_lines = "\n".join(
                [
                    json.dumps(
                        {
                            "custom_id": "0",
                            "response": {"body": {"data": [{"embedding": [0.1] * 3072}]}},
                        }
                    ),
                    json.dumps({"custom_id": "1", "response": None, "error": {"code": "x"}}),
                ]
            )
            mock_client.files.content.return_value = MagicMock(text=jsonl_lines)
            mock_openai_cls.return_value = mock_client
            embedder = OpenAIEmbedder(config)
            with pytest.raises(BatchJobIncompleteError, match="1/2"):
                embedder.poll_batch("batch_xyz", expected_count=2)

    def test_poll_batch_raises_incomplete_error_when_count_short_but_no_response_null(
        self,
    ) -> None:
        """Even if no line explicitly has response=None, a short output file
        (e.g. a failed request landed only in error_file_id, never in
        output_file_id at all) must still be caught via expected_count."""
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.batches.retrieve.return_value = MagicMock(
                status="completed", output_file_id="file_out_123"
            )
            jsonl_lines = json.dumps(
                {
                    "custom_id": "0",
                    "response": {"body": {"data": [{"embedding": [0.1] * 3072}]}},
                }
            )
            mock_client.files.content.return_value = MagicMock(text=jsonl_lines)
            mock_openai_cls.return_value = mock_client
            embedder = OpenAIEmbedder(config)
            with pytest.raises(BatchJobIncompleteError, match="1/3"):
                embedder.poll_batch("batch_xyz", expected_count=3)

    def test_poll_batch_raises_incomplete_error_when_output_file_id_is_none(self) -> None:
        """OpenAI can report status "completed" with output_file_id=None when
        every request in the batch failed -- must not crash on a None file id
        or silently return an empty list."""
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.batches.retrieve.return_value = MagicMock(
                status="completed", output_file_id=None
            )
            mock_openai_cls.return_value = mock_client
            embedder = OpenAIEmbedder(config)
            with pytest.raises(BatchJobIncompleteError):
                embedder.poll_batch("batch_xyz", expected_count=1)
            mock_client.files.content.assert_not_called()


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

    def test_frozen_binary_import_error_message_when_sentence_transformers_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under a PyInstaller frozen binary (sys.frozen=True), the message must
        say pip install has NO EFFECT on this binary — a frozen process never
        consults the host's Python/pip environment, so the normal "pip install
        'trelix[local]'" advice is actively misleading there."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        config = EmbedderConfig(provider="local")
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                LocalEmbedder(config)
        message = str(exc_info.value)
        assert "does not bundle it" in message
        assert "no effect on this binary" in message
        # The frozen message must not contain the pip-install-fixes-this-process
        # instruction — that is the exact thing this fix removes for this case.
        assert "pip install" not in message

    def test_import_error_message_unchanged_when_not_frozen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outside a frozen binary (normal pip-installed usage, sys.frozen absent
        or False), the original pip-install message must be unchanged — that
        advice IS correct for a regular Python install."""
        config = EmbedderConfig(provider="local")

        monkeypatch.delattr(sys, "frozen", raising=False)  # absent, the common case
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="pip install") as exc_info:
                LocalEmbedder(config)
        assert "does not bundle it" not in str(exc_info.value)

        monkeypatch.setattr(sys, "frozen", False, raising=False)  # explicit False
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="pip install") as exc_info:
                LocalEmbedder(config)
        assert "does not bundle it" not in str(exc_info.value)

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

    def test_pins_device_to_cpu_rather_than_auto_detecting(self) -> None:
        """MPS auto-detection crashes when this embedder is constructed inside a
        child process spawned deep inside a sandboxed host (e.g. an Electron
        extension host spawning trelix-mcp over stdio) -- the crash is native,
        so it never surfaces as a catchable Python exception, only a closed
        pipe. device="cpu" must always be passed explicitly, never left to
        sentence-transformers' own auto-detection."""
        config = EmbedderConfig(provider="local")
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_module.SentenceTransformer.return_value = mock_model
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            LocalEmbedder(config)
        call_kwargs = mock_st_module.SentenceTransformer.call_args
        assert call_kwargs.kwargs.get("device") == "cpu"


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
        assert (
            call_kwargs.kwargs.get("input_type") == "document" or call_kwargs.args[2] == "document"
            if len(call_kwargs.args) > 2
            else call_kwargs.kwargs["input_type"] == "document"
        )

    def test_embed_query_uses_query_input_type(self) -> None:
        embedder, mock_client = self._make()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.5] * 1024]
        mock_client.embed.return_value = mock_response
        embedder.embed_query("find all async functions")
        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args
        assert (
            call_kwargs.kwargs.get("input_type") == "query" or call_kwargs.args[2] == "query"
            if len(call_kwargs.args) > 2
            else call_kwargs.kwargs["input_type"] == "query"
        )

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
# TestVoyageMatryoshka — Voyage Matryoshka dimension support
# ---------------------------------------------------------------------------


class TestVoyageMatryoshka:
    """Tests for Voyage Matryoshka dimension support (voyage-code-3)."""

    def test_embed_passes_output_dimension(self) -> None:
        """VoyageEmbedder passes output_dimension to API when set."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 512]
        mock_client.embed.return_value = mock_response
        mock_voyage_module = MagicMock()
        mock_voyage_module.Client.return_value = mock_client
        with patch.dict(sys.modules, {"voyageai": mock_voyage_module}):
            cfg = EmbedderConfig(
                provider="voyage",
                voyage_api_key=_FAKE_VOYAGE_KEY,
                voyage_output_dimensions=512,
                _env_file=None,
            )
            emb = VoyageEmbedder(cfg)
        emb._client = mock_client  # noqa: SLF001
        emb.embed(["def foo(): pass"])
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs.get("output_dimension") == 512

    def test_effective_dimension_with_output_dimensions(self) -> None:
        cfg = EmbedderConfig(
            provider="voyage",
            voyage_api_key=_FAKE_VOYAGE_KEY,
            voyage_output_dimensions=256,
            _env_file=None,
        )
        assert cfg.effective_dimension == 256

    def test_effective_dimension_without_output_dimensions(self) -> None:
        cfg = EmbedderConfig(
            provider="voyage",
            voyage_api_key=_FAKE_VOYAGE_KEY,
            _env_file=None,
        )
        assert cfg.effective_dimension == 1024  # voyage_dimensions default


# ---------------------------------------------------------------------------
# LocalCodeEmbedder
# ---------------------------------------------------------------------------


class TestLocalCodeEmbedder:
    """Tests for the SFR-Embedding-Code-2B_R local code embedder."""

    @pytest.fixture(autouse=True)
    def _remote_model_code_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Grant SEC-03c's opt-in for this class only.

        local-code loads its model with remote code trusted, which the gate in
        embedder/base.py refuses unless the operator set the variable in the real
        process environment. Every construction below therefore has to opt in.
        Scoped to this class deliberately: an autouse fixture in conftest.py would
        mean no test in the suite could ever observe the gate regressing. The
        refusal side lives in tests/unit/test_remote_model_code_gate.py.
        """
        monkeypatch.setenv(REMOTE_MODEL_CODE_ENV_VAR, "1")

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

    def test_dimension_fallback_is_2304(self) -> None:
        """If model has no dimension method, fallback must be the model's real width.

        The expected value's provenance is the MODEL REPOSITORY, not this file and not
        the constant under test: Salesforce/SFR-Embedding-Code-2B_R states its width
        twice — `hidden_size: 2304` in config.json and `word_embedding_dimension: 2304`
        in 1_Pooling/config.json — with modules.json declaring no Dense module to widen
        it. This test previously asserted 4096, i.e. the constant against itself through
        a mock that cannot disagree, and 4096 is that model's max_seq_length.
        """
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
        assert embedder.dimension == 2304

    def test_import_error_when_sentence_transformers_missing(self) -> None:
        config = EmbedderConfig(provider="local-code")
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="pip install"):
                LocalCodeEmbedder(config)

    def test_frozen_binary_import_error_message_when_sentence_transformers_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same frozen-binary distinction as LocalEmbedder — see its test's
        docstring for why the standard pip-install advice is wrong here."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        config = EmbedderConfig(provider="local-code")
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                LocalCodeEmbedder(config)
        message = str(exc_info.value)
        assert "does not bundle it" in message
        assert "no effect on this binary" in message
        assert "pip install" not in message

    def test_import_error_message_unchanged_when_not_frozen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outside a frozen binary, the original pip-install message is unchanged."""
        config = EmbedderConfig(provider="local-code")

        monkeypatch.delattr(sys, "frozen", raising=False)  # absent, the common case
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="pip install") as exc_info:
                LocalCodeEmbedder(config)
        assert "does not bundle it" not in str(exc_info.value)

        monkeypatch.setattr(sys, "frozen", False, raising=False)  # explicit False
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="pip install") as exc_info:
                LocalCodeEmbedder(config)
        assert "does not bundle it" not in str(exc_info.value)

    def test_trust_remote_code_true(self) -> None:
        """With SEC-03c's opt-in granted (class fixture), the kwarg must still arrive.

        The SFR architecture does not load without it, so the gate is an opt-in and
        not a removal — this is the half that proves the provider still works. Before
        the gate this test passed with no opt-in anywhere, which is what made a
        `.env` in an indexed repository sufficient to run its own Python here.
        """
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
        assert call_kwargs.kwargs.get("trust_remote_code") is True or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] is True
        )


# ---------------------------------------------------------------------------
# BedrockTitanEmbedder (mocked — no real AWS calls)
# ---------------------------------------------------------------------------


def _make_mock_boto3(vectors: list[list[float]]) -> MagicMock:
    """Return a mock boto3 bedrock-runtime client that yields preset vectors."""
    import json

    mock_client = MagicMock()
    responses = iter(vectors)

    def invoke_model(**kwargs: object) -> dict:
        vec = next(responses, [0.0] * 1024)
        body_bytes = json.dumps({"embedding": vec}).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = body_bytes
        return {"body": mock_body}

    mock_client.invoke_model.side_effect = invoke_model
    return mock_client


class TestBedrockTitanEmbedder:
    def _make(self, dims: int = 1024) -> BedrockTitanEmbedder:
        config = EmbedderConfig(
            provider="bedrock-titan",
            bedrock_titan_dimensions=dims,
        )
        embedder = BedrockTitanEmbedder.__new__(BedrockTitanEmbedder)
        embedder._model = config.bedrock_titan_model
        embedder._dims = dims
        embedder._normalize = config.bedrock_titan_normalize
        embedder._client = _make_mock_boto3([[float(i) / dims for i in range(dims)]] * 10)
        return embedder

    def test_is_base_embedder(self) -> None:
        assert issubclass(BedrockTitanEmbedder, BaseEmbedder)

    def test_dimension_1024(self) -> None:
        assert self._make(1024).dimension == 1024

    def test_dimension_512(self) -> None:
        assert self._make(512).dimension == 512

    def test_dimension_256(self) -> None:
        assert self._make(256).dimension == 256

    def test_embed_returns_list_of_vectors(self) -> None:
        embedder = self._make(1024)
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 1024
        assert len(result[1]) == 1024

    def test_embed_query_returns_single_vector(self) -> None:
        embedder = self._make(1024)
        vec = embedder.embed_query("def login():")
        assert isinstance(vec, list)
        assert len(vec) == 1024

    def test_invoke_model_called_per_text(self) -> None:
        embedder = self._make(1024)
        embedder.embed(["a", "b", "c"])
        assert embedder._client.invoke_model.call_count == 3

    def test_invoke_model_passes_correct_dims(self) -> None:
        embedder = self._make(512)
        embedder.embed(["test"])
        import json

        call_body = json.loads(embedder._client.invoke_model.call_args[1]["body"])
        assert call_body["dimensions"] == 512

    def test_invoke_model_passes_normalize_true(self) -> None:
        embedder = self._make(1024)
        embedder.embed(["test"])
        import json

        call_body = json.loads(embedder._client.invoke_model.call_args[1]["body"])
        assert call_body["normalize"] is True

    def test_factory_returns_titan_embedder(self) -> None:
        config = EmbedderConfig(provider="bedrock-titan", bedrock_aws_region="us-east-1")
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        with patch.dict(sys.modules, _mock_boto3_modules(mock_boto3)):
            embedder = make_embedder(config)
        assert isinstance(embedder, BedrockTitanEmbedder)

    def test_import_error_when_boto3_missing(self) -> None:
        config = EmbedderConfig(provider="bedrock-titan")
        with patch.dict(sys.modules, {"boto3": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="pip install"):
                BedrockTitanEmbedder(config)

    def test_raises_without_region_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mirrors BedrockBackend's v3.3.0 region requirement (see
        test_llm_bedrock_backend.py::test_raises_without_region_configured) —
        anthropic-sdk-python v1.0.0 made AnthropicBedrock raise if no region is
        set, instead of silently defaulting to us-east-1."""
        monkeypatch.delenv("AWS_REGION", raising=False)
        config = EmbedderConfig(provider="bedrock-titan", _env_file=None)  # type: ignore[call-arg]
        assert config.bedrock_aws_region is None
        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, _mock_boto3_modules(mock_boto3)):
            with pytest.raises(ValueError, match="AWS_REGION"):
                BedrockTitanEmbedder(config)

    def test_effective_dimension_in_config(self) -> None:
        config = EmbedderConfig(provider="bedrock-titan", bedrock_titan_dimensions=512)
        assert config.effective_dimension == 512


def _s3_output_body(records: list[dict]) -> bytes:
    """Build a Bedrock batch inference output JSONL body from record dicts."""
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")


def _make_batch_titan(
    *,
    bucket: str | None = "trelix-bucket",
    role_arn: str | None = "arn:aws:iam::123456789012:role/BedrockBatch",
    dims: int = 2,
) -> BedrockTitanEmbedder:
    """BedrockTitanEmbedder with mocked bedrock-runtime, bedrock (control-plane),
    and S3 clients — bypasses __init__ (and therefore real boto3 client
    construction) the same way TestBedrockTitanEmbedder._make does above."""
    config = EmbedderConfig(
        provider="bedrock-titan",
        bedrock_titan_dimensions=dims,
        bedrock_batch_s3_bucket=bucket,
        bedrock_batch_role_arn=role_arn,
    )
    embedder = BedrockTitanEmbedder.__new__(BedrockTitanEmbedder)
    embedder._model = config.bedrock_titan_model
    embedder._dims = dims
    embedder._normalize = config.bedrock_titan_normalize
    embedder._config = config
    embedder._client = MagicMock()
    embedder._bedrock_control_client = MagicMock()
    embedder._s3_client = MagicMock()
    embedder.last_batch_s3_uris = None
    return embedder


class TestBedrockTitanBatchApi:
    """submit_batch/poll_batch on BedrockTitanEmbedder — the Bedrock analogue of
    OpenAIEmbedder's Batch API strategy. Deliberately Titan-only: Cohere Embed
    is not in AWS's supported-models list for Bedrock batch inference at all,
    so BedrockCohereEmbedder must structurally lack these methods (see
    TestBedrockCohereEmbedderStructurallyLacksBatchApi below)."""

    # -- submit_batch: config validation -----------------------------------

    def test_submit_batch_raises_without_bucket(self) -> None:
        embedder = _make_batch_titan(bucket=None)
        with pytest.raises(ValueError, match="TRELIX_BEDROCK_BATCH_S3_BUCKET"):
            embedder.submit_batch(["hello"])

    def test_submit_batch_raises_without_role_arn(self) -> None:
        embedder = _make_batch_titan(role_arn=None)
        with pytest.raises(ValueError, match="TRELIX_BEDROCK_BATCH_ROLE_ARN"):
            embedder.submit_batch(["hello"])

    def test_submit_batch_error_names_both_required_vars(self) -> None:
        """Naming only the one that happens to be missing would still leave the
        operator to discover the second requirement via a second failed run."""
        embedder = _make_batch_titan(bucket=None, role_arn=None)
        with pytest.raises(ValueError) as exc_info:
            embedder.submit_batch(["hello"])
        message = str(exc_info.value)
        assert "TRELIX_BEDROCK_BATCH_S3_BUCKET" in message
        assert "TRELIX_BEDROCK_BATCH_ROLE_ARN" in message

    # -- submit_batch: request shape ---------------------------------------

    def test_submit_batch_uploads_correct_jsonl(self) -> None:
        """Input JSONL must be {"recordId": "<index>", "modelInput": {"inputText":
        ...}} per record, reusing the exact real-time invoke_model body shape
        (dimensions/normalize included) rather than a second, possibly-drifted
        copy of it."""
        embedder = _make_batch_titan(dims=512)
        embedder._bedrock_control_client.create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/abc"
        }

        embedder.submit_batch(["hello", "world"])

        put_call = embedder._s3_client.put_object.call_args
        assert put_call.kwargs["Bucket"] == "trelix-bucket"
        body = put_call.kwargs["Body"].decode("utf-8")
        lines = [json.loads(line) for line in body.splitlines() if line.strip()]
        assert lines == [
            {
                "recordId": "0",
                "modelInput": {"inputText": "hello", "dimensions": 512, "normalize": True},
            },
            {
                "recordId": "1",
                "modelInput": {"inputText": "world", "dimensions": 512, "normalize": True},
            },
        ]

    def test_submit_batch_calls_create_model_invocation_job_correctly(self) -> None:
        embedder = _make_batch_titan()
        embedder._bedrock_control_client.create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/abc"
        }

        job_id = embedder.submit_batch(["hello", "world"])

        assert job_id == "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/abc"
        create_call = embedder._bedrock_control_client.create_model_invocation_job.call_args
        assert create_call.kwargs["roleArn"] == "arn:aws:iam::123456789012:role/BedrockBatch"
        assert create_call.kwargs["modelId"] == embedder._model
        input_uri = create_call.kwargs["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
        output_uri = create_call.kwargs["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        assert input_uri.startswith("s3://trelix-bucket/trelix-batch/")
        assert input_uri.endswith("/input.jsonl")
        assert output_uri.startswith("s3://trelix-bucket/trelix-batch/")
        assert output_uri.endswith("/output/")
        # submit_batch does not return the S3 URIs (its return type is pinned to
        # `str`, matching OpenAIEmbedder.submit_batch) -- the indexer needs them
        # to persist via db.insert_batch_job, so they are exposed here instead.
        assert embedder.last_batch_s3_uris == (input_uri, output_uri)

    # -- poll_batch: non-terminal statuses -> None --------------------------

    @pytest.mark.parametrize(
        "status", ["Submitted", "Validating", "Scheduled", "InProgress", "Stopping"]
    )
    def test_poll_batch_returns_none_while_non_terminal(self, status: str) -> None:
        embedder = _make_batch_titan()
        embedder._bedrock_control_client.get_model_invocation_job.return_value = {"status": status}
        assert embedder.poll_batch("job-arn") is None

    # -- poll_batch: terminal failure statuses -> BatchJobTerminalError -----

    @pytest.mark.parametrize("status", ["Failed", "Expired", "Stopped"])
    def test_poll_batch_raises_on_terminal_failure(self, status: str) -> None:
        embedder = _make_batch_titan()
        embedder._bedrock_control_client.get_model_invocation_job.return_value = {"status": status}
        with pytest.raises(BatchJobTerminalError):
            embedder.poll_batch("job-arn")

    # -- poll_batch: Completed -----------------------------------------------

    def test_poll_batch_completed_returns_ordered_vectors_never_positional(self) -> None:
        """Output records arrive out of S3-file order and don't start at index
        0 -- a positional zip/index assumption would return them in the wrong
        order (or crash); recordId-keyed lookup must not."""
        embedder = _make_batch_titan()
        embedder._bedrock_control_client.get_model_invocation_job.return_value = {
            "status": "Completed",
            "outputDataConfig": {
                "s3OutputDataConfig": {"s3Uri": "s3://trelix-bucket/trelix-batch/xyz/output/"}
            },
        }
        embedder._s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "trelix-batch/xyz/output/manifest.json.out"},
                {"Key": "trelix-batch/xyz/output/abc123/input.jsonl.out"},
            ]
        }
        records = [
            {"recordId": "2", "modelOutput": {"embedding": [2.0, 2.0]}},
            {"recordId": "0", "modelOutput": {"embedding": [0.0, 0.0]}},
            {"recordId": "1", "modelOutput": {"embedding": [1.0, 1.0]}},
        ]
        mock_body = MagicMock()
        mock_body.read.return_value = _s3_output_body(records)
        embedder._s3_client.get_object.return_value = {"Body": mock_body}

        vectors = embedder.poll_batch("job-arn", expected_count=3)

        assert vectors == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        get_call = embedder._s3_client.get_object.call_args
        assert get_call.kwargs["Bucket"] == "trelix-bucket"
        assert get_call.kwargs["Key"] == "trelix-batch/xyz/output/abc123/input.jsonl.out"
        list_call = embedder._s3_client.list_objects_v2.call_args
        assert list_call.kwargs["Prefix"] == "trelix-batch/xyz/output/"

    # -- poll_batch: PartiallyCompleted (success, not an error) -------------

    def test_poll_batch_partially_completed_is_not_an_error(self) -> None:
        """PartiallyCompleted is a SUCCESS path per the design decision matching
        trelix's own reconciliation story -- must NOT raise BatchJobIncompleteError
        (or anything else) purely because len(vectors) < expected_count."""
        embedder = _make_batch_titan()
        embedder._bedrock_control_client.get_model_invocation_job.return_value = {
            "status": "PartiallyCompleted",
            "outputDataConfig": {
                "s3OutputDataConfig": {"s3Uri": "s3://trelix-bucket/trelix-batch/xyz/output/"}
            },
        }
        embedder._s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "trelix-batch/xyz/output/abc123/input.jsonl.out"}]
        }
        # Middle record (index 1) failed.
        records = [
            {"recordId": "0", "modelOutput": {"embedding": [0.0, 0.0]}},
            {"recordId": "1", "error": {"errorCode": "ModelError", "errorMessage": "boom"}},
            {"recordId": "2", "modelOutput": {"embedding": [2.0, 2.0]}},
        ]
        mock_body = MagicMock()
        mock_body.read.return_value = _s3_output_body(records)
        embedder._s3_client.get_object.return_value = {"Body": mock_body}

        vectors = embedder.poll_batch("job-arn", expected_count=3)

        # Failed record correctly OMITTED, not misaligned: [0, 2]'s vectors,
        # never [0, 2]'s vectors with a garbage/duplicate third slot, and never
        # index 2's vector shifted down into index 1's slot.
        assert vectors == [[0.0, 0.0], [2.0, 2.0]]

    def test_poll_batch_records_preserves_original_positions_for_partial(self) -> None:
        """poll_batch_records (the indexer-facing, position-preserving variant)
        must keep index 1 visibly MISSING rather than shifting index 2's vector
        down to slot 1 -- this is what lets the indexer pair vectors back to
        the correct chunk_ids without a positional zip/index assumption."""
        embedder = _make_batch_titan()
        embedder._bedrock_control_client.get_model_invocation_job.return_value = {
            "status": "PartiallyCompleted",
            "outputDataConfig": {
                "s3OutputDataConfig": {"s3Uri": "s3://trelix-bucket/trelix-batch/xyz/output/"}
            },
        }
        embedder._s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "trelix-batch/xyz/output/abc123/input.jsonl.out"}]
        }
        records = [
            {"recordId": "0", "modelOutput": {"embedding": [0.0, 0.0]}},
            {"recordId": "1", "error": {"errorCode": "ModelError"}},
            {"recordId": "2", "modelOutput": {"embedding": [2.0, 2.0]}},
        ]
        mock_body = MagicMock()
        mock_body.read.return_value = _s3_output_body(records)
        embedder._s3_client.get_object.return_value = {"Body": mock_body}

        by_position = embedder.poll_batch_records("job-arn")

        assert by_position == {0: [0.0, 0.0], 2: [2.0, 2.0]}
        assert 1 not in by_position

    def test_poll_batch_records_merges_multiple_output_shards(self) -> None:
        """Bedrock's own docs state it generates one output JSONL file PER
        INPUT JSONL file -- if a job's input was ever split across multiple
        files, ListObjectsV2 returns multiple matching .jsonl.out keys.
        Reading only the first (an earlier version of this method did) would
        silently drop every record in every shard after it, with no error."""
        embedder = _make_batch_titan()
        embedder._bedrock_control_client.get_model_invocation_job.return_value = {
            "status": "Completed",
            "outputDataConfig": {
                "s3OutputDataConfig": {"s3Uri": "s3://trelix-bucket/trelix-batch/xyz/output/"}
            },
        }
        embedder._s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "trelix-batch/xyz/output/manifest.json.out"},
                {"Key": "trelix-batch/xyz/output/shard-0/input.jsonl.out"},
                {"Key": "trelix-batch/xyz/output/shard-1/input.jsonl.out"},
            ]
        }
        shard_bodies = {
            "trelix-batch/xyz/output/shard-0/input.jsonl.out": _s3_output_body(
                [{"recordId": "0", "modelOutput": {"embedding": [0.0, 0.0]}}]
            ),
            "trelix-batch/xyz/output/shard-1/input.jsonl.out": _s3_output_body(
                [{"recordId": "1", "modelOutput": {"embedding": [1.0, 1.0]}}]
            ),
        }

        def fake_get_object(Bucket: str, Key: str) -> dict:
            mock_body = MagicMock()
            mock_body.read.return_value = shard_bodies[Key]
            return {"Body": mock_body}

        embedder._s3_client.get_object.side_effect = fake_get_object

        by_position = embedder.poll_batch_records("job-arn")

        assert by_position == {0: [0.0, 0.0], 1: [1.0, 1.0]}


class TestBedrockCohereEmbedderStructurallyLacksBatchApi:
    """Cohere Embed is NOT in AWS's supported-models list for Bedrock batch
    inference at all -- BedrockCohereEmbedder must structurally lack
    submit_batch/poll_batch, exactly like VoyageEmbedder does today."""

    def test_no_submit_batch(self) -> None:
        assert not hasattr(BedrockCohereEmbedder, "submit_batch")

    def test_no_poll_batch(self) -> None:
        assert not hasattr(BedrockCohereEmbedder, "poll_batch")


# ---------------------------------------------------------------------------
# BedrockCohereEmbedder (mocked — no real AWS calls)
# ---------------------------------------------------------------------------


def _make_mock_cohere_boto3(num_texts: int = 2) -> MagicMock:
    """Return a mock boto3 client that yields Cohere-style embeddings response."""
    import json

    mock_client = MagicMock()

    def invoke_model(**kwargs: object) -> dict:
        body = json.loads(kwargs["body"])
        n = len(body.get("texts", []))
        vecs = [[float(i) / 1024 for i in range(1024)] for _ in range(n)]
        body_bytes = json.dumps({"embeddings": vecs}).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = body_bytes
        return {"body": mock_body}

    mock_client.invoke_model.side_effect = invoke_model
    return mock_client


class TestBedrockCohereEmbedder:
    def _make(self) -> BedrockCohereEmbedder:
        config = EmbedderConfig(provider="bedrock-cohere")
        embedder = BedrockCohereEmbedder.__new__(BedrockCohereEmbedder)
        embedder._model = config.bedrock_cohere_model
        embedder._client = _make_mock_cohere_boto3()
        return embedder

    def test_is_base_embedder(self) -> None:
        assert issubclass(BedrockCohereEmbedder, BaseEmbedder)

    def test_dimension_is_1024(self) -> None:
        assert self._make().dimension == 1024

    def test_embed_returns_correct_count(self) -> None:
        embedder = self._make()
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert all(len(v) == 1024 for v in result)

    def test_embed_uses_search_document_input_type(self) -> None:
        embedder = self._make()
        embedder.embed(["some code"])
        import json

        call_body = json.loads(embedder._client.invoke_model.call_args[1]["body"])
        assert call_body["input_type"] == "search_document"

    def test_embed_query_uses_search_query_input_type(self) -> None:
        embedder = self._make()
        embedder.embed_query("find authentication")
        import json

        call_body = json.loads(embedder._client.invoke_model.call_args[1]["body"])
        assert call_body["input_type"] == "search_query"

    def test_embed_query_returns_single_vector(self) -> None:
        embedder = self._make()
        vec = embedder.embed_query("find auth")
        assert isinstance(vec, list)
        assert len(vec) == 1024

    def test_large_batch_splits_at_96(self) -> None:
        embedder = self._make()
        texts = [f"text {i}" for i in range(200)]
        embedder.embed(texts)
        # 200 texts → ceil(200/96) = 3 invoke_model calls
        assert embedder._client.invoke_model.call_count == 3

    def test_texts_pre_truncated_to_2048_chars(self) -> None:
        # Bedrock validates length before truncation — must pre-truncate client-side.
        embedder = self._make()
        long_text = "x" * 5000
        embedder.embed([long_text, "short"])
        import json

        call_body = json.loads(embedder._client.invoke_model.call_args[1]["body"])
        assert len(call_body["texts"][0]) == 2048, "Long text must be pre-truncated to 2048 chars"
        assert call_body["texts"][1] == "short", "Short text must pass through unchanged"

    def test_factory_returns_cohere_embedder(self) -> None:
        config = EmbedderConfig(provider="bedrock-cohere", bedrock_aws_region="us-east-1")
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        with patch.dict(sys.modules, _mock_boto3_modules(mock_boto3)):
            embedder = make_embedder(config)
        assert isinstance(embedder, BedrockCohereEmbedder)

    def test_import_error_when_boto3_missing(self) -> None:
        config = EmbedderConfig(provider="bedrock-cohere")
        with patch.dict(sys.modules, {"boto3": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="pip install"):
                BedrockCohereEmbedder(config)

    def test_effective_dimension_in_config(self) -> None:
        config = EmbedderConfig(provider="bedrock-cohere")
        assert config.effective_dimension == 1024


# ---------------------------------------------------------------------------
# CohereEmbedder — direct Cohere API (cohere.ClientV2), not the Bedrock envelope
# ---------------------------------------------------------------------------

_FAKE_COHERE_KEY = "cohere-test-key-not-real"


class TestCohereEmbedder:
    """Tests for the direct Cohere API embedder (cohere.ClientV2)."""

    def _make_response(self, dim: int = 1024, n: int = 1, tokens: float | None = 7.0) -> MagicMock:
        response = MagicMock()
        response.embeddings.float_ = [[0.1] * dim for _ in range(n)]
        response.meta.billed_units.input_tokens = tokens
        return response

    def _make_client_mock(self, dim: int = 1024, n: int = 1) -> MagicMock:
        mock_client = MagicMock()
        mock_client.embed.return_value = self._make_response(dim, n)
        return mock_client

    def _make(self, dim: int = 1024) -> tuple[CohereEmbedder, MagicMock]:
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        mock_client = self._make_client_mock(dim)
        with patch("trelix.embedder.cohere.ClientV2", return_value=mock_client):
            embedder = CohereEmbedder(config)
        return embedder, mock_client

    def test_is_base_embedder(self) -> None:
        embedder, _ = self._make()
        assert isinstance(embedder, BaseEmbedder)

    def test_dimension_property_returns_configured_dimensions(self) -> None:
        config = EmbedderConfig(
            provider="cohere", cohere_api_key=_FAKE_COHERE_KEY, cohere_dimensions=999
        )
        mock_client = self._make_client_mock(dim=999)
        with patch("trelix.embedder.cohere.ClientV2", return_value=mock_client):
            embedder = CohereEmbedder(config)
        assert embedder.dimension == 999

    def test_embed_uses_search_document_input_type_and_float_embedding_type(self) -> None:
        embedder, mock_client = self._make()
        embedder.embed(["def foo(): pass"])
        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_document"
        assert call_kwargs["embedding_types"] == ["float"]
        assert call_kwargs["texts"] == ["def foo(): pass"]

    def test_embed_query_uses_search_query_input_type(self) -> None:
        embedder, mock_client = self._make()
        embedder.embed_query("find all async functions")
        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_query"

    def test_embed_returns_vectors_from_response_embeddings_float(self) -> None:
        embedder, mock_client = self._make()
        mock_client.embed.return_value = self._make_response(dim=1024, n=2)
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert all(len(v) == 1024 for v in result)

    def test_embed_query_returns_single_vector(self) -> None:
        embedder, mock_client = self._make()
        mock_client.embed.return_value = self._make_response(dim=1024, n=1)
        result = embedder.embed_query("search query")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_large_batch_splits_at_96(self) -> None:
        """Mirrors BedrockCohereEmbedder's real client-side 96-text batch limit."""
        embedder, mock_client = self._make()

        def _side_effect(**kwargs: object) -> MagicMock:
            texts = kwargs["texts"]
            assert isinstance(texts, list)
            return self._make_response(dim=1024, n=len(texts))

        mock_client.embed.side_effect = _side_effect
        texts = [f"text {i}" for i in range(200)]
        embedder.embed(texts)
        # 200 texts → ceil(200/96) = 3 calls
        assert mock_client.embed.call_count == 3

    def test_factory_returns_cohere_embedder(self) -> None:
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        mock_client = self._make_client_mock()
        with patch("trelix.embedder.cohere.ClientV2", return_value=mock_client):
            embedder = make_embedder(config)
        assert isinstance(embedder, CohereEmbedder)

    def test_import_error_with_helpful_message_if_cohere_missing(self) -> None:
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        with patch("trelix.embedder.cohere.ClientV2", None):
            with pytest.raises(ImportError, match="pip install"):
                CohereEmbedder(config)

    def test_effective_dimension_in_config(self) -> None:
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        assert config.effective_dimension == 1024

    def test_default_model_and_dimensions(self) -> None:
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        assert config.cohere_model == "embed-english-v3.0"
        assert config.cohere_dimensions == 1024

    def test_embed_records_billed_input_tokens(self) -> None:
        """Token metering reads response.meta.billed_units.input_tokens (Cohere's
        own usage shape) rather than base.py's _usage_tokens() (which reads
        response.usage.total_tokens / response.total_tokens and does not apply
        here)."""
        embedder, mock_client = self._make()
        mock_client.embed.return_value = self._make_response(dim=1024, n=1, tokens=42.0)
        with (
            patch("trelix.embedder.base.otel_tracing.metrics_enabled", return_value=True),
            patch("trelix.embedder.base.otel_tracing.record_embedding_call") as mock_record,
        ):
            embedder.embed(["hello"])
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["tokens"] == 42

    def test_import_error_message_names_cohere_extra(self) -> None:
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        with patch("trelix.embedder.cohere.ClientV2", None):
            with pytest.raises(ImportError, match=r"trelix\[cohere\]"):
                CohereEmbedder(config)


# ---------------------------------------------------------------------------
# Shared retry contract — sync + true-async remote embedder paths
# ---------------------------------------------------------------------------


class TestEmbedderRetryContract:
    def test_openai_embed_retries_on_503_then_succeeds(self) -> None:
        """A transient 5xx must be retried, not surfaced immediately —
        confirms the shared retry contract is wired into OpenAIEmbedder.embed()."""
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 3072
        success = MagicMock()
        success.data = [mock_item]
        mock_client.embeddings.create.side_effect = [_status_error(503), success]

        with patch("openai.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(config)

        with patch("tenacity.nap.time.sleep"):
            result = embedder.embed(["hello world"])

        assert len(result) == 1
        assert mock_client.embeddings.create.call_count == 2

    def test_openai_embed_400_is_not_retried(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = _status_error(400)

        with patch("openai.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(config)

        with pytest.raises(openai.APIStatusError):
            embedder.embed(["hello world"])

        assert mock_client.embeddings.create.call_count == 1

    async def test_openai_embed_async_retries_on_503_then_succeeds(self) -> None:
        """The TRUE async path (AsyncOpenAI) must also honor the shared
        retry contract — tenacity auto-dispatches sync vs. async decoration."""
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        with patch("openai.OpenAI", return_value=MagicMock()):
            embedder = OpenAIEmbedder(config)

        mock_item = MagicMock()
        mock_item.embedding = [0.2] * 3072
        success = MagicMock()
        success.data = [mock_item]
        mock_async_client = MagicMock()
        mock_async_client.embeddings.create = AsyncMock(side_effect=[_status_error(503), success])
        mock_async_client.close = AsyncMock()

        with (
            patch.object(embedder, "_get_async_client", return_value=mock_async_client),
            patch("tenacity.nap.time.sleep"),
        ):
            result = await embedder.embed_async(["hello world"])

        assert len(result) == 1
        assert mock_async_client.embeddings.create.call_count == 2

    def test_bedrock_titan_invoke_model_retries_on_throttling_then_succeeds(self) -> None:
        """Bedrock's boto3 invoke_model surfaces ThrottlingException as
        botocore.exceptions.ClientError — must retry like any other
        429-shaped failure, not just OpenAI's own exception types."""
        botocore = pytest.importorskip("botocore.exceptions")
        config = EmbedderConfig(provider="bedrock-titan", bedrock_aws_region="us-east-1")
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            embedder = BedrockTitanEmbedder(config)

        throttled = botocore.ClientError(
            {
                "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
                "ResponseMetadata": {"HTTPStatusCode": 429},
            },
            "InvokeModel",
        )
        success_response = {"body": MagicMock()}
        success_response["body"].read.return_value = b'{"embedding": [0.1, 0.2]}'
        mock_client.invoke_model.side_effect = [throttled, success_response]

        with patch("tenacity.nap.time.sleep"):
            result = embedder.embed_query("hello")

        assert result == [0.1, 0.2]

    def test_cohere_embed_retries_on_real_too_many_requests_error_then_succeeds(self) -> None:
        """The installed cohere==7.1.1 SDK does NOT raise httpx.HTTPStatusError
        for API errors — it raises typed subclasses of
        cohere.core.api_error.ApiError with a plain `.status_code` int
        attribute. Must retry like any other 429-shaped failure. Uses a REAL
        cohere.errors.TooManyRequestsError (not a MagicMock) because a
        MagicMock instance would pass isinstance checks it shouldn't and
        mask this exact bug."""
        cohere_errors = pytest.importorskip("cohere.errors")
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        mock_client = MagicMock()
        success = MagicMock()
        success.embeddings.float_ = [[0.1] * 1024]
        success.meta.billed_units.input_tokens = 7.0
        throttled = cohere_errors.TooManyRequestsError(body={"message": "rate limited"})
        mock_client.embed.side_effect = [throttled, success]

        with patch("trelix.embedder.cohere.ClientV2", return_value=mock_client):
            embedder = CohereEmbedder(config)

        with patch("tenacity.nap.time.sleep"):
            result = embedder.embed_query("hello")

        assert result == [0.1] * 1024
        assert mock_client.embed.call_count == 2


# ---------------------------------------------------------------------------
# Embedder client retry configuration — SDK's own retry must be disabled
# ---------------------------------------------------------------------------


class TestEmbedderClientRetryConfiguration:
    """Each embedder's own SDK client (openai/boto3) has a default retry
    layer that would otherwise stack underneath @with_retry's 5-attempt
    tenacity loop, multiplying worst-case wall-clock time on a persistent
    outage far beyond what max_attempts=5 implies. Uses the REAL SDK
    client constructors (no mocking) so this actually proves what config
    reaches the client, not what trelix's own code believes it passed."""

    def test_openai_embedder_client_has_sdk_retries_disabled(self) -> None:
        config = EmbedderConfig(provider="openai", openai_api_key=_FAKE_OPENAI_KEY)
        embedder = OpenAIEmbedder(config)
        assert embedder._client.max_retries == 0

    def test_azure_embedder_client_has_sdk_retries_disabled(self) -> None:
        config = EmbedderConfig(
            provider="azure",
            azure_api_key=_FAKE_AZURE_KEY,
            azure_endpoint=_FAKE_AZURE_ENDPOINT,
        )
        embedder = AzureOpenAIEmbedder(config)
        assert embedder._client.max_retries == 0

    def test_bedrock_titan_embedder_client_has_sdk_retries_disabled(self) -> None:
        pytest.importorskip("boto3")
        config = EmbedderConfig(provider="bedrock-titan", bedrock_aws_region="us-east-1")
        embedder = BedrockTitanEmbedder(config)
        retries = embedder._client.meta.config.retries
        assert retries["total_max_attempts"] == 1

    def test_bedrock_cohere_embedder_client_has_sdk_retries_disabled(self) -> None:
        pytest.importorskip("boto3")
        config = EmbedderConfig(provider="bedrock-cohere", bedrock_aws_region="us-east-1")
        embedder = BedrockCohereEmbedder(config)
        retries = embedder._client.meta.config.retries
        assert retries["total_max_attempts"] == 1

    def test_cohere_embedder_client_has_sdk_retries_disabled(self) -> None:
        """cohere's base_client.py defaults max_retries to 2 when the caller
        doesn't pass it (`_defaulted_max_retries = max_retries if max_retries
        is not None else 2`) — CohereEmbedder must pass max_retries=0
        explicitly so it isn't stacked underneath @with_retry's 5-attempt
        tenacity loop. Real SDK client, no mocking, so this proves what
        config actually reaches the client — the exact class of test whose
        absence for Cohere let this gap through."""
        pytest.importorskip("cohere")
        config = EmbedderConfig(provider="cohere", cohere_api_key=_FAKE_COHERE_KEY)
        embedder = CohereEmbedder(config)
        assert embedder._client._client_wrapper.get_max_retries() == 0
