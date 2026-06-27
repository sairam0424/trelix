"""
End-to-end integration tests for LLM backends (Azure + Bedrock).

Each test skips automatically when credentials are absent.

Bedrock models require inference profile IDs (us.* prefix), NOT bare model IDs.
Bare model IDs (e.g. anthropic.claude-sonnet-4-6) fail with:
  ValidationException: on-demand throughput not supported — use an inference profile ARN.
Correct IDs verified 2026-06-27:
  us.anthropic.claude-sonnet-4-6
  us.anthropic.claude-haiku-4-5-20251001-v1:0
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# Load .env from repo root so tests work when run directly (not via CI env injection)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


def _decode_credential(value: str) -> str:
    """Mirror BedrockBackend._decode_credential: transparently handle base64."""
    try:
        decoded = base64.b64decode(value).decode("utf-8")
        if decoded.isprintable() and "\n" not in decoded:
            return decoded
    except Exception:  # noqa: BLE001
        pass
    return value


def _env(key: str) -> str | None:
    """Return env var value, decoding base64 if needed."""
    raw = os.environ.get(key)
    if raw is None:
        return None
    return _decode_credential(raw)


# ---------------------------------------------------------------------------
# Credential presence checks (used by skipif markers)
# ---------------------------------------------------------------------------

_HAS_AZURE = bool(_env("AZURE_API_KEY") and _env("AZURE_ENDPOINT"))
_HAS_BEDROCK = bool(_env("AWS_ACCESS_KEY_ID") and _env("AWS_SECRET_ACCESS_KEY"))

_SKIP_AZURE = pytest.mark.skipif(
    not _HAS_AZURE,
    reason="Azure credentials not set (AZURE_API_KEY + AZURE_ENDPOINT required)",
)
_SKIP_BEDROCK = pytest.mark.skipif(
    not _HAS_BEDROCK,
    reason="AWS credentials not set (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY required)",
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_HELLO_MESSAGES = [
    {"role": "user", "content": "Reply with exactly the word: PONG"},
]

_TOOL_DEF = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

_TOOL_MESSAGES = [
    {"role": "user", "content": "What is the weather in London?"},
]

# ---------------------------------------------------------------------------
# Backend factory helpers
# ---------------------------------------------------------------------------


def _make_azure_backend() -> Any:
    from trelix.llm.factory import build_chat_client
    from trelix.core.config import LLMConfig
    # LLMConfig reads from env automatically via pydantic-settings aliases
    config = LLMConfig(provider="azure", max_tokens=256, temperature=0.0)
    return build_chat_client(config)


def _make_bedrock_backend(model_id: str) -> Any:
    from trelix.llm.factory import build_chat_client
    from trelix.core.config import LLMConfig
    config = LLMConfig(provider="bedrock", model=model_id, max_tokens=256, temperature=0.0)
    return build_chat_client(config)


def _chat_messages(raw: list[dict[str, str]]) -> list[Any]:
    from trelix.llm.client import ChatMessage

    return [ChatMessage(role=m["role"], content=m["content"]) for m in raw]


# ---------------------------------------------------------------------------
# Azure tests
# ---------------------------------------------------------------------------


@_SKIP_AZURE
def test_azure_complete() -> None:
    """Azure: complete() returns non-empty content."""
    backend = _make_azure_backend()
    response = backend.complete(_chat_messages(_HELLO_MESSAGES))
    assert response.content, f"Azure complete() returned empty content: {response!r}"


@_SKIP_AZURE
def test_azure_stream() -> None:
    """Azure: stream() yields at least one chunk."""
    backend = _make_azure_backend()
    chunks = list(backend.stream(_chat_messages(_HELLO_MESSAGES)))
    assert chunks, "Azure stream() yielded no chunks"
    full = "".join(chunks)
    assert full.strip(), f"Azure stream() chunks were all empty: {chunks!r}"


@_SKIP_AZURE
def test_azure_tool_call() -> None:
    """Azure: tool_call() returns a ToolCallResponse with get_weather."""
    backend = _make_azure_backend()
    result = backend.tool_call(
        _chat_messages(_TOOL_MESSAGES),
        tools=_TOOL_DEF,
        force_tool="get_weather",
    )
    assert result.tool_name == "get_weather", (
        f"Expected tool_name='get_weather', got {result.tool_name!r}"
    )
    assert "city" in result.tool_arguments, (
        f"Expected 'city' in tool_arguments, got {result.tool_arguments!r}"
    )


# ---------------------------------------------------------------------------
# Bedrock sonnet-4-6 tests
# Model: us.anthropic.claude-sonnet-4-6  (inference profile — verified 2026-06-27)
# ---------------------------------------------------------------------------

_BEDROCK_SONNET_MODEL = "us.anthropic.claude-sonnet-4-6"


@_SKIP_BEDROCK
def test_bedrock_sonnet_complete() -> None:
    """Bedrock sonnet-4-6: complete() returns non-empty content."""
    backend = _make_bedrock_backend(_BEDROCK_SONNET_MODEL)
    response = backend.complete(_chat_messages(_HELLO_MESSAGES))
    assert response.content, f"Bedrock sonnet complete() returned empty content: {response!r}"
    assert response.input_tokens > 0, "Expected input_tokens > 0"


@_SKIP_BEDROCK
def test_bedrock_sonnet_stream() -> None:
    """Bedrock sonnet-4-6: stream() yields at least one chunk."""
    backend = _make_bedrock_backend(_BEDROCK_SONNET_MODEL)
    chunks = list(backend.stream(_chat_messages(_HELLO_MESSAGES)))
    assert chunks, "Bedrock sonnet stream() yielded no chunks"
    assert "".join(chunks).strip(), "Bedrock sonnet stream() chunks were all empty"


@_SKIP_BEDROCK
def test_bedrock_sonnet_tool_call() -> None:
    """Bedrock sonnet-4-6: tool_call() returns get_weather tool."""
    backend = _make_bedrock_backend(_BEDROCK_SONNET_MODEL)
    result = backend.tool_call(
        _chat_messages(_TOOL_MESSAGES),
        tools=_TOOL_DEF,
        force_tool="get_weather",
    )
    assert result.tool_name == "get_weather", (
        f"Expected tool_name='get_weather', got {result.tool_name!r}"
    )
    assert "city" in result.tool_arguments, (
        f"Expected 'city' in tool_arguments, got {result.tool_arguments!r}"
    )


# ---------------------------------------------------------------------------
# Bedrock haiku-4-5 tests
# Model: us.anthropic.claude-haiku-4-5-20251001-v1:0  (inference profile — verified 2026-06-27)
# ---------------------------------------------------------------------------

_BEDROCK_HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@_SKIP_BEDROCK
def test_bedrock_haiku_complete() -> None:
    """Bedrock haiku-4-5: complete() returns non-empty content."""
    backend = _make_bedrock_backend(_BEDROCK_HAIKU_MODEL)
    response = backend.complete(_chat_messages(_HELLO_MESSAGES))
    assert response.content, f"Bedrock haiku complete() returned empty content: {response!r}"
    assert response.input_tokens > 0, "Expected input_tokens > 0"


@_SKIP_BEDROCK
def test_bedrock_haiku_stream() -> None:
    """Bedrock haiku-4-5: stream() yields at least one chunk."""
    backend = _make_bedrock_backend(_BEDROCK_HAIKU_MODEL)
    chunks = list(backend.stream(_chat_messages(_HELLO_MESSAGES)))
    assert chunks, "Bedrock haiku stream() yielded no chunks"
    assert "".join(chunks).strip(), "Bedrock haiku stream() chunks were all empty"


@_SKIP_BEDROCK
def test_bedrock_haiku_tool_call() -> None:
    """Bedrock haiku-4-5: tool_call() returns get_weather tool."""
    backend = _make_bedrock_backend(_BEDROCK_HAIKU_MODEL)
    result = backend.tool_call(
        _chat_messages(_TOOL_MESSAGES),
        tools=_TOOL_DEF,
        force_tool="get_weather",
    )
    assert result.tool_name == "get_weather", (
        f"Expected tool_name='get_weather', got {result.tool_name!r}"
    )
    assert "city" in result.tool_arguments, (
        f"Expected 'city' in tool_arguments, got {result.tool_arguments!r}"
    )
