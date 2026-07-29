"""Tests for VertexBackend (mocked — google-genai not required)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse

_FAKE_GKEY = "test-google-api-key-placeholder"


def _google_genai_modules():
    """Return a minimal sys.modules patch for google.genai."""
    mock_genai = MagicMock()
    mock_genai.Client = MagicMock(return_value=MagicMock())

    mock_types_mod = MagicMock()
    mock_types_mod.GenerateContentConfig = MagicMock(return_value=MagicMock())
    mock_genai.types = mock_types_mod

    mock_google = MagicMock()
    mock_google.genai = mock_genai

    return {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types_mod,
    }


class TestVertexBackend:
    def _make_backend(self, extra_mods=None):
        mods = _google_genai_modules()
        if extra_mods:
            mods.update(extra_mods)
        with patch.dict("sys.modules", mods):
            from trelix.llm.providers.vertex_backend import VertexBackend

            cfg = LLMConfig(
                provider="vertex",
                model="gemini-2.0-flash",
                google_api_key=_FAKE_GKEY,
                _env_file=None,  # type: ignore[call-arg]
            )
            return VertexBackend(cfg)

    def test_complete_returns_chat_response(self) -> None:
        mods = _google_genai_modules()
        with patch.dict("sys.modules", mods):
            from trelix.llm.providers.vertex_backend import VertexBackend

            cfg = LLMConfig(
                provider="vertex",
                model="gemini-2.0-flash",
                google_api_key=_FAKE_GKEY,
                _env_file=None,  # type: ignore[call-arg]
            )
            backend = VertexBackend(cfg)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "hello from gemini"
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_client.models.generate_content.return_value = mock_response
        backend._client = mock_client

        with patch.dict("sys.modules", mods):
            result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello from gemini"
        assert result.finish_reason == "stop"

    def test_import_error_when_google_genai_not_installed(self) -> None:
        # Remove any cached vertex_backend module to force fresh import
        for key in list(sys.modules.keys()):
            if "vertex_backend" in key:
                del sys.modules[key]
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            from trelix.llm.providers.vertex_backend import VertexBackend

            cfg = LLMConfig(provider="vertex", _env_file=None)  # type: ignore[call-arg]
            with pytest.raises(ImportError, match="pip install"):
                VertexBackend(cfg)

    def test_complete_retries_on_503_then_succeeds(self) -> None:
        """A transient server error must be retried, not surfaced
        immediately — confirms the shared retry contract is wired in."""
        genai_errors = pytest.importorskip("google.genai.errors")
        mods = _google_genai_modules()
        with patch.dict("sys.modules", mods):
            from trelix.llm.providers.vertex_backend import VertexBackend

            cfg = LLMConfig(
                provider="vertex",
                model="gemini-2.0-flash",
                google_api_key=_FAKE_GKEY,
                _env_file=None,  # type: ignore[call-arg]
            )
            backend = VertexBackend(cfg)

        mock_client = MagicMock()
        success = MagicMock()
        success.text = "ok"
        success.candidates[0].finish_reason.name = "STOP"
        success.usage_metadata.prompt_token_count = 1
        success.usage_metadata.candidates_token_count = 1
        server_error = genai_errors.ServerError(503, {"message": "unavailable"}, None)
        mock_client.models.generate_content.side_effect = [server_error, success]
        backend._client = mock_client

        # Only google.genai.types needs mocking here (complete() imports it
        # inline) — google.genai.errors must stay the REAL module so
        # trelix.core.retry's isinstance(exc, genai_errors.APIError) check
        # (also a real import) sees the same class the test constructed.
        with (
            patch.dict("sys.modules", {"google.genai.types": mods["google.genai.types"]}),
            patch("tenacity.nap.time.sleep"),
        ):
            result = backend.complete([ChatMessage(role="user", content="hi")])

        assert result.content == "ok"
        assert mock_client.models.generate_content.call_count == 2

    def test_complete_400_is_not_retried(self) -> None:
        """A non-retryable client error must fail on the first attempt."""
        genai_errors = pytest.importorskip("google.genai.errors")
        mods = _google_genai_modules()
        with patch.dict("sys.modules", mods):
            from trelix.llm.providers.vertex_backend import VertexBackend

            cfg = LLMConfig(
                provider="vertex",
                model="gemini-2.0-flash",
                google_api_key=_FAKE_GKEY,
                _env_file=None,  # type: ignore[call-arg]
            )
            backend = VertexBackend(cfg)

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.ClientError(
            400, {"message": "bad request"}, None
        )
        backend._client = mock_client

        with patch.dict("sys.modules", {"google.genai.types": mods["google.genai.types"]}):
            with pytest.raises(genai_errors.ClientError):
                backend.complete([ChatMessage(role="user", content="hi")])

        assert mock_client.models.generate_content.call_count == 1
