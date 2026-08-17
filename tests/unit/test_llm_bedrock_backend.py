"""Tests for BedrockBackend (mocked — no real AWS calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse


class _ValidationException(Exception):
    """Simulated botocore ValidationException for fallback tests."""


def _make_boto3_mock() -> MagicMock:
    """Return a minimal boto3 mock that satisfies _build_client."""
    boto3_mock = MagicMock()
    session_mock = MagicMock()
    boto3_mock.Session.return_value = session_mock
    session_mock.client.return_value = MagicMock()
    return boto3_mock


def _mock_boto3_modules(boto3_mock: MagicMock) -> dict[str, MagicMock]:
    """sys.modules patch dict covering both boto3 and botocore.config —
    _build_client() imports both (`import boto3` + `from botocore.config
    import Config`), but botocore isn't installed in every environment
    that runs this test suite (it's a transitive dependency of the
    optional 'bedrock' extra, not a base dependency) — CI's own test job
    never installs it. Mocking boto3 alone is not enough."""
    botocore_config_mock = MagicMock()
    botocore_mock = MagicMock()
    botocore_mock.config = botocore_config_mock
    return {
        "boto3": boto3_mock,
        "botocore": botocore_mock,
        "botocore.config": botocore_config_mock,
    }


class TestBedrockBackend:
    def _make_backend(self, model: str = "us.anthropic.claude-sonnet-4-6"):
        from trelix.llm.providers.bedrock_backend import BedrockBackend

        cfg = LLMConfig(
            provider="bedrock",
            model=model,
            _env_file=None,  # type: ignore[call-arg]
        )
        boto3_mock = _make_boto3_mock()
        with patch.dict("sys.modules", _mock_boto3_modules(boto3_mock)):
            backend = BedrockBackend(cfg)
        return backend

    def test_complete_returns_chat_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = {
            "output": {"message": {"content": [{"text": "hello"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        mock_client.converse.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"
        assert result.finish_reason == "stop"

    def test_uses_inference_config_max_tokens(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hi")], max_tokens=100)

        call_kwargs = mock_client.converse.call_args[1]
        assert "inferenceConfig" in call_kwargs
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 100
        # Must NOT have top-level max_tokens
        assert "max_tokens" not in call_kwargs
        assert "maxTokens" not in call_kwargs

    def test_system_as_top_level_list(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        backend.complete(
            [ChatMessage(role="user", content="hi")],
            system="You are helpful.",
        )
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs.get("system") == [{"text": "You are helpful."}]

    def test_content_always_list_of_dicts(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hello")])

        call_kwargs = mock_client.converse.call_args[1]
        messages = call_kwargs["messages"]
        for msg in messages:
            assert isinstance(msg["content"], list)
            assert all(isinstance(block, dict) for block in msg["content"])

    def test_tool_choice_auto_format(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"toolUse": {"toolUseId": "1", "name": "fn", "input": {"x": 1}}}],
                    "role": "assistant",
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        tools = [
            {
                "type": "function",
                "function": {"name": "fn", "parameters": {"type": "object", "properties": {}}},
            }
        ]
        backend.tool_call([ChatMessage(role="user", content="hi")], tools=tools)

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_import_error_when_boto3_not_installed(self) -> None:
        from trelix.llm.providers.bedrock_backend import BedrockBackend

        cfg = LLMConfig(provider="bedrock", _env_file=None)  # type: ignore[call-arg]
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="pip install"):
                BedrockBackend(cfg)


class TestBedrockDefaultModels:
    """Default model resolution and fallback behaviour."""

    @pytest.fixture(autouse=True)
    def _unconfigured_bedrock_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scrub the three vars that decide which models "the default" resolves to.

        Every test in this class asserts on the SHIPPED default pair
        (sonnet-4-6 primary, haiku-4-5 fallback) — three of them by name, and
        the fallback tests by matching `modelId` inside `converse()`. But
        `LLMConfig(..., _env_file=None)` only silences the ./.env FILE source;
        the process environment is a separate, higher-precedence
        pydantic-settings source, and all three of
        TRELIX_LLM_BEDROCK_{PRIMARY,FALLBACK}_MODEL and TRELIX_LLM_MODEL live in
        it. TRELIX_LLM_MODEL matters because `_resolve_bedrock_model()` treats any
        `config.model` other than the literal "gpt-4o" as an explicit override
        that becomes the primary — so `TRELIX_LLM_MODEL=claude-3-opus` in a
        developer's .env takes over the primary for this whole class, including
        `test_config_fields_override_defaults`.

        Importing `litellm` (tests/unit/test_retry.py does, for its
        error-classification checks) runs load_dotenv() at import time and
        publishes this repo's root .env into os.environ for the rest of the
        process. That .env pins the primary to claude-opus-4-8, so once that
        import had happened, `_primary_model` was opus, the ValidationException
        side-effects keyed on "us.anthropic.claude-sonnet-4-6" never fired, and
        exactly three tests here failed. Reproduce without litellm by exporting
        the same value the .env carries:
            TRELIX_LLM_BEDROCK_PRIMARY_MODEL=us.anthropic.claude-opus-4-8 \
              pytest tests/unit/test_llm_bedrock_backend.py
        The .env fallback happens to equal the shipped fallback, so
        test_default_fallback_is_haiku_4_5 passed by coincidence, not by testing.

        test_explicit_model_overrides_primary passes `model=` as a kwarg, which
        outranks env, so it still exercises the override path.
        """
        monkeypatch.delenv("TRELIX_LLM_BEDROCK_PRIMARY_MODEL", raising=False)
        monkeypatch.delenv("TRELIX_LLM_BEDROCK_FALLBACK_MODEL", raising=False)
        monkeypatch.delenv("TRELIX_LLM_MODEL", raising=False)

    def _make_backend_with_mock_client(self, model: str | None = None):
        """Build a BedrockBackend with a mock boto3 client. model=None uses config default."""
        from trelix.llm.providers.bedrock_backend import BedrockBackend

        kwargs: dict = {"provider": "bedrock", "_env_file": None}
        if model is not None:
            kwargs["model"] = model
        cfg = LLMConfig(**kwargs)  # type: ignore[arg-type]
        boto3_mock = _make_boto3_mock()
        with patch.dict("sys.modules", _mock_boto3_modules(boto3_mock)):
            backend = BedrockBackend(cfg)
        return backend

    def test_default_primary_is_sonnet_4_6(self) -> None:
        backend = self._make_backend_with_mock_client()
        assert backend._primary_model == "us.anthropic.claude-sonnet-4-6"

    def test_default_fallback_is_haiku_4_5(self) -> None:
        backend = self._make_backend_with_mock_client()
        assert backend._fallback_model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def test_default_active_model_is_primary(self) -> None:
        backend = self._make_backend_with_mock_client()
        assert backend._model == backend._primary_model

    def test_explicit_model_overrides_primary(self) -> None:
        backend = self._make_backend_with_mock_client(model="us.anthropic.claude-opus-4-8")
        assert backend._primary_model == "us.anthropic.claude-opus-4-8"
        assert backend._model == "us.anthropic.claude-opus-4-8"

    def test_fallback_triggered_on_validation_exception(self) -> None:
        """complete() falls back to haiku when primary raises ValidationException."""
        backend = self._make_backend_with_mock_client()
        mock_client = MagicMock()

        fallback_resp = {
            "output": {"message": {"content": [{"text": "haiku reply"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3},
        }

        call_count = 0

        def converse_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("modelId") == "us.anthropic.claude-sonnet-4-6":
                raise _ValidationException(
                    "ValidationException: Invocation of model ID "
                    "with on-demand throughput not supported"
                )
            return fallback_resp

        mock_client.converse.side_effect = converse_side_effect
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert call_count == 2, f"Expected 2 calls (primary + fallback), got {call_count}"
        assert result.content == "haiku reply"
        assert backend._model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def test_fallback_not_triggered_for_non_validation_error(self) -> None:
        """Non-availability errors (auth, network) must propagate — not fall back."""
        backend = self._make_backend_with_mock_client()
        mock_client = MagicMock()
        mock_client.converse.side_effect = RuntimeError("Connection refused")
        backend._client = mock_client

        with pytest.raises(RuntimeError, match="Connection refused"):
            backend.complete([ChatMessage(role="user", content="hi")])

        assert mock_client.converse.call_count == 1, "Should not retry on non-availability errors"

    def test_fallback_not_retried_if_already_on_fallback(self) -> None:
        """If already on the fallback model, raise — don't loop."""
        backend = self._make_backend_with_mock_client()
        backend._model = backend._fallback_model  # simulate already fallen back
        mock_client = MagicMock()

        exc = _ValidationException("ValidationException: on-demand throughput not supported")
        mock_client.converse.side_effect = exc
        backend._client = mock_client

        with pytest.raises(Exception, match="ValidationException"):
            backend.complete([ChatMessage(role="user", content="hi")])

        assert mock_client.converse.call_count == 1

    def test_subsequent_calls_use_fallback_after_switch(self) -> None:
        """Once switched to fallback, all subsequent calls skip the primary."""
        backend = self._make_backend_with_mock_client()
        mock_client = MagicMock()
        call_models = []

        ok_resp = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }

        def converse_side_effect(**kwargs):
            call_models.append(kwargs.get("modelId"))
            if kwargs.get("modelId") == "us.anthropic.claude-sonnet-4-6":
                raise _ValidationException(
                    "ValidationException: on-demand throughput not supported"
                )
            return ok_resp

        mock_client.converse.side_effect = converse_side_effect
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="first")])  # triggers fallback
        backend.complete([ChatMessage(role="user", content="second")])  # uses fallback directly

        # First call: primary + fallback; second call: fallback only
        assert call_models == [
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ]

    def test_config_fields_override_defaults(self) -> None:
        """TRELIX_LLM_BEDROCK_PRIMARY_MODEL / FALLBACK_MODEL env vars override defaults."""
        from trelix.llm.providers.bedrock_backend import BedrockBackend

        cfg = LLMConfig(
            provider="bedrock",
            bedrock_primary_model="us.anthropic.claude-opus-4-8",
            bedrock_fallback_model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            _env_file=None,  # type: ignore[call-arg]
        )
        boto3_mock = _make_boto3_mock()
        with patch.dict("sys.modules", _mock_boto3_modules(boto3_mock)):
            backend = BedrockBackend(cfg)
        assert backend._primary_model == "us.anthropic.claude-opus-4-8"


class TestBedrockClientRetryConfiguration:
    """botocore's own default retry mode (legacy, up to 5 attempts per
    call) would otherwise stack underneath @with_retry's 5-attempt
    tenacity loop, multiplying worst-case wall-clock time on a persistent
    outage far beyond what max_attempts=5 implies. Uses the REAL boto3
    session (skips if the optional 'bedrock' extra isn't installed) so
    this actually proves what config reaches the client, not what
    trelix's own code believes it passed."""

    def test_bedrock_client_has_sdk_retries_disabled(self) -> None:
        pytest.importorskip("boto3")
        from trelix.llm.providers.bedrock_backend import BedrockBackend

        cfg = LLMConfig(provider="bedrock", _env_file=None)  # type: ignore[call-arg]
        backend = BedrockBackend(cfg)
        retries = backend._client.meta.config.retries
        assert retries["total_max_attempts"] == 1


class TestBedrockRetry:
    """Shared retry contract wiring — ThrottlingException/5xx must retry
    before model-fallback logic ever sees them; ValidationException must
    still fall through to fallback with zero wasted retry attempts."""

    def _client_error(self, code: str, status_code: int) -> Exception:
        botocore = pytest.importorskip("botocore.exceptions")
        return botocore.ClientError(
            {
                "Error": {"Code": code, "Message": "x"},
                "ResponseMetadata": {"HTTPStatusCode": status_code},
            },
            "Converse",
        )

    def _make_backend_with_mock_client(self):
        from trelix.llm.providers.bedrock_backend import BedrockBackend

        cfg = LLMConfig(provider="bedrock", _env_file=None)  # type: ignore[call-arg]
        boto3_mock = _make_boto3_mock()
        with patch.dict("sys.modules", _mock_boto3_modules(boto3_mock)):
            return BedrockBackend(cfg)

    def test_throttling_exception_is_retried_then_succeeds(self) -> None:
        backend = self._make_backend_with_mock_client()
        mock_client = MagicMock()
        success = {
            "output": {"message": {"content": [{"text": "ok"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        mock_client.converse.side_effect = [
            self._client_error("ThrottlingException", 429),
            success,
        ]
        backend._client = mock_client

        with patch("tenacity.nap.time.sleep"):
            result = backend.complete([ChatMessage(role="user", content="hi")])

        assert result.content == "ok"
        assert mock_client.converse.call_count == 2

    def test_validation_exception_falls_back_without_wasted_retries(self) -> None:
        """A non-retryable ClientError (no retryable status) must reach
        fallback logic on the first attempt, not exhaust retries first."""
        backend = self._make_backend_with_mock_client()
        mock_client = MagicMock()
        fallback_resp = {
            "output": {"message": {"content": [{"text": "haiku reply"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }

        def converse_side_effect(**kwargs):
            if kwargs.get("modelId") == backend._primary_model:
                raise _ValidationException(
                    "ValidationException: on-demand throughput not supported"
                )
            return fallback_resp

        mock_client.converse.side_effect = converse_side_effect
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert result.content == "haiku reply"
        assert mock_client.converse.call_count == 2  # primary (1 try) + fallback (1 try)
