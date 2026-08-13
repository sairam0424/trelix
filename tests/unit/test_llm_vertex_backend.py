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


class TestVertexToolCallSystemInstruction:
    """`tool_call()` must forward a system message, like `complete()` already does.

    `_build_contents()` filters out every `role="system"` message and never
    re-injects it, so a system prompt sent through `tool_call()` was discarded:
    the model received the tool declarations and the user turn with no
    instructions. `complete()` and `stream()` avoid this by computing
    `effective_system` and passing `system_instruction=`; `tool_call()` did not.

    This mattered for two callers. `QueryPlanner._call_llm` passes its planning
    rules as a system message, so planning silently degraded on Vertex only. And
    the ReAct loop's `_SYSTEM_PROMPT` is delivered as a `role="system"` message
    because `tool_call()` takes no `system=` parameter on the ABC — so the agent
    would have run instruction-free on Vertex while working on every other
    provider, which is the hardest kind of gap to notice.
    """

    def _backend_and_config_mock(self):
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

        # A realistic function-call response, so tool_call() returns normally and
        # the assertion is on a completed call rather than on a raised path.
        mock_client = MagicMock()
        part = MagicMock()
        part.function_call.name = "done"
        part.function_call.args = {"answer": "ok"}
        response = MagicMock()
        response.candidates = [MagicMock(content=MagicMock(parts=[part]))]
        mock_client.models.generate_content.return_value = response
        backend._client = mock_client
        return backend, mods

    _TOOLS = [{"function": {"name": "done", "description": "finish", "parameters": {}}}]

    def test_system_message_becomes_system_instruction(self) -> None:
        backend, mods = self._backend_and_config_mock()
        config_ctor = mods["google.genai.types"].GenerateContentConfig
        config_ctor.reset_mock()

        with patch.dict("sys.modules", mods):
            backend.tool_call(
                [
                    ChatMessage(role="system", content="Never call done before retrieving."),
                    ChatMessage(role="user", content="how does auth work?"),
                ],
                tools=self._TOOLS,
            )

        assert config_ctor.call_args is not None, "GenerateContentConfig was never built"
        kwargs = config_ctor.call_args.kwargs
        assert kwargs.get("system_instruction") == "Never call done before retrieving.", (
            "the system message was dropped — the model would receive no instructions"
        )

    def test_no_system_message_passes_none(self) -> None:
        """Absent a system message the behaviour is unchanged from before the fix."""
        backend, mods = self._backend_and_config_mock()
        config_ctor = mods["google.genai.types"].GenerateContentConfig
        config_ctor.reset_mock()

        with patch.dict("sys.modules", mods):
            backend.tool_call([ChatMessage(role="user", content="hi")], tools=self._TOOLS)

        assert config_ctor.call_args.kwargs.get("system_instruction") is None
