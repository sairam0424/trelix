"""OpenAIBackend against an OpenAI-compatible local server (`TRELIX_LLM_BASE_URL`): the client
it builds, the request shape it sends, and the size-floor warning at construction.

The request-shape tests run the REAL openai SDK with only the HTTP transport faked: the
constructor is patched at `trelix.llm.providers.openai_backend.OpenAI` with a wrapper that
forwards every kwarg the backend passes and adds `http_client=`, so what the tests see on the
wire is what the SDK would send (the bearer, the path, which token-limit field the body has).
`complete()`, `stream()` and `tool_call()` each get a wire test: the query planner calls
`tool_call()` and `trelix ask` calls `stream()`, and a flag forgotten at one call site would
ship `max_completion_tokens` to Ollama, which has no such field and runs unbounded.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage
from trelix.llm.providers.openai_backend import OpenAIBackend, _token_limit_param

_LOCAL_URL = "http://127.0.0.1:11434/v1"
_MODEL = "qwen2.5-coder:7b"
_FAKE_KEY = "test-k"  # short enough not to trigger the secret scanner; never sent anywhere real
_MESSAGES = [ChatMessage(role="user", content="hi")]
_TOOLS = [
    {
        "type": "function",
        "function": {"name": "plan", "parameters": {"type": "object", "properties": {}}},
    }
]
_BACKEND_LOGGER = "trelix.llm.openai_backend"

_SSE_BODY = (
    b'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m", '
    b'"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}]}\n\n'
    b"data: [DONE]\n\n"
)


def _completion_reply(content: str) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def _tool_call_reply() -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "plan", "arguments": '{"k": 1}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


class _Wire:
    """Records every request and answers by the request's shape."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_SSE_BODY
            )
        if body.get("tools"):
            return httpx.Response(200, json=_tool_call_reply())
        return httpx.Response(200, json=_completion_reply("[]"))

    def body(self) -> dict[str, Any]:
        assert len(self.requests) == 1, [str(r.url) for r in self.requests]
        return json.loads(self.requests[0].content)

    def header(self, name: str) -> str:
        return self.requests[0].headers[name]


def _patched_constructor(wire: _Wire) -> Callable[..., OpenAI]:
    def construct(**kwargs: Any) -> OpenAI:
        return OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(wire)), **kwargs)

    return construct


def _llm(**fields: object) -> LLMConfig:
    return LLMConfig(_env_file=None, **fields)  # type: ignore[call-arg]


def _wired_backend(wire: _Wire, **fields: object) -> OpenAIBackend:
    with patch("trelix.llm.providers.openai_backend.OpenAI", new=_patched_constructor(wire)):
        return OpenAIBackend(_llm(**fields))


def _backend_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _BACKEND_LOGGER]


class TestClientConstruction:
    def test_without_a_key_the_client_is_real_with_the_placeholder_bearer(self) -> None:
        """MUTATION: drop the placeholder (the SDK refuses to build without a key); pass
        `base_url` only when a key exists (the client is None)."""
        backend = OpenAIBackend(_llm(provider="openai", base_url=_LOCAL_URL, model=_MODEL))

        client = backend._client
        assert isinstance(client, OpenAI)
        assert str(client.base_url) == "http://127.0.0.1:11434/v1/"
        assert client.api_key == "trelix-local"
        assert client.max_retries == 0
        assert backend._local_server is True

    def test_a_configured_key_wins_over_the_placeholder(self) -> None:
        backend = OpenAIBackend(
            _llm(provider="openai", base_url=_LOCAL_URL, openai_api_key=_FAKE_KEY)
        )

        assert backend._client.api_key == "test-k"
        assert str(backend._client.base_url) == "http://127.0.0.1:11434/v1/"

    def test_without_the_url_a_missing_key_still_means_no_client(self) -> None:
        """MUTATION: apply the placeholder on the hosted path too."""
        backend = OpenAIBackend(_llm(provider="openai"))

        assert backend._client is None
        assert backend._local_server is False

    def test_a_constructor_failure_with_the_url_set_propagates(self) -> None:
        """MUTATION: wrap the local-server constructor in the hosted path's
        `except Exception: return None` (a constructor error in `search`/`ask` then reads as
        "LLM not configured" instead of the real error).

        `review` is not protected by this: DiffReviewer._get_client swallows the error into
        exit 3 anyway, which is why LLMConfig's validator refuses up front the shapes known to
        fail in the SDK (tests/unit/test_llm_config_local_server.py).
        """

        def refuse(**_kwargs: Any) -> OpenAI:
            raise RuntimeError("canary")

        with (
            patch("trelix.llm.providers.openai_backend.OpenAI", new=refuse),
            pytest.raises(RuntimeError, match="canary"),
        ):
            OpenAIBackend(_llm(provider="openai", base_url=_LOCAL_URL))


class TestTokenLimitParam:
    def test_a_local_server_gets_max_tokens_for_a_local_tag(self) -> None:
        assert _token_limit_param("qwen2.5-coder:7b", 100, local_server=True) == {"max_tokens": 100}

    def test_a_local_server_gets_max_tokens_even_for_a_hosted_name(self) -> None:
        """MUTATION: route the local flag through the legacy-name table instead."""
        assert _token_limit_param("gpt-4o", 100, local_server=True) == {"max_tokens": 100}

    def test_hosted_keeps_max_completion_tokens_for_a_local_tag(self) -> None:
        assert _token_limit_param("qwen2.5-coder:7b", 100) == {"max_completion_tokens": 100}

    def test_hosted_legacy_names_are_unchanged(self) -> None:
        assert _token_limit_param("gpt-4", 100) == {"max_tokens": 100}
        assert _token_limit_param("gpt-4o", 100) == {"max_completion_tokens": 100}


class TestLocalServerRequestShape:
    """One wire test per entry point; all three must send `max_tokens` and the bearer."""

    def test_complete(self) -> None:
        """MUTATION: build the limit kwarg without `local_server=` in complete()."""
        wire = _Wire()
        backend = _wired_backend(wire, provider="openai", base_url=_LOCAL_URL, model=_MODEL)

        response = backend.complete(_MESSAGES, max_tokens=4096)

        assert response.content == "[]"
        assert response.input_tokens == 10
        assert str(wire.requests[0].url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert wire.header("authorization") == "Bearer trelix-local"
        body = wire.body()
        assert body["model"] == "qwen2.5-coder:7b"
        assert body["max_tokens"] == 4096
        assert "max_completion_tokens" not in body

    def test_stream(self) -> None:
        """MUTATION: build the limit kwarg without `local_server=` in stream()."""
        wire = _Wire()
        backend = _wired_backend(wire, provider="openai", base_url=_LOCAL_URL, model=_MODEL)

        chunks = list(backend.stream(_MESSAGES, max_tokens=256))

        assert chunks == ["x"]
        assert str(wire.requests[0].url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert wire.header("authorization") == "Bearer trelix-local"
        body = wire.body()
        assert body["stream"] is True
        assert body["model"] == "qwen2.5-coder:7b"
        assert body["max_tokens"] == 256
        assert "max_completion_tokens" not in body

    def test_tool_call(self) -> None:
        """MUTATION: build the limit kwarg without `local_server=` in tool_call().

        The planner's own 30 s timeout still goes out with the request (pinned so a later
        change to the local-server path is a deliberate one).
        """
        wire = _Wire()
        backend = _wired_backend(wire, provider="openai", base_url=_LOCAL_URL, model=_MODEL)

        result = backend.tool_call(_MESSAGES, tools=_TOOLS, max_tokens=512)

        assert result.tool_name == "plan"
        assert result.tool_arguments == {"k": 1}
        assert str(wire.requests[0].url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert wire.header("authorization") == "Bearer trelix-local"
        body = wire.body()
        assert body["model"] == "qwen2.5-coder:7b"
        assert body["max_tokens"] == 512
        assert "max_completion_tokens" not in body
        assert body["tools"] == _TOOLS
        assert wire.requests[0].extensions["timeout"] == {
            "connect": 30.0,
            "read": 30.0,
            "write": 30.0,
            "pool": 30.0,
        }

    def test_a_configured_key_is_the_bearer_on_the_wire(self) -> None:
        wire = _Wire()
        backend = _wired_backend(
            wire, provider="openai", base_url=_LOCAL_URL, model=_MODEL, openai_api_key=_FAKE_KEY
        )

        backend.complete(_MESSAGES)

        assert wire.header("authorization") == "Bearer test-k"


class TestHostedPathIsUnchanged:
    def test_hosted_gpt_4o_still_sends_max_completion_tokens_to_api_openai_com(self) -> None:
        """MUTATION: gate `_local_server` on something other than `base_url` (the hosted
        request then carries `max_tokens`, which the hosted API rejects for gpt-4o)."""
        wire = _Wire()
        backend = _wired_backend(wire, provider="openai", openai_api_key=_FAKE_KEY)

        backend.complete(_MESSAGES, max_tokens=4096)

        assert backend._local_server is False
        assert str(wire.requests[0].url) == "https://api.openai.com/v1/chat/completions"
        assert wire.header("authorization") == "Bearer test-k"
        body = wire.body()
        assert body["model"] == "gpt-4o"
        assert body["max_completion_tokens"] == 4096
        assert "max_tokens" not in body


_W1_QWEN_7B = (
    "Local model 'qwen2.5-coder:7b' is about 7B parameters, under the 20B floor "
    "docs/OFFLINE.md assumes for trelix review; expect more unreviewed hunks and weaker findings."
)
_W2_GLM = (
    "Local model 'glm-4.5-air': no parameter count in the tag, so trelix cannot tell "
    "whether it meets the 20B floor docs/OFFLINE.md assumes."
)


class TestSizeFloorWarningAtConstruction:
    def test_a_small_local_model_warns_once_with_the_pinned_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_BACKEND_LOGGER):
            OpenAIBackend(_llm(provider="openai", base_url=_LOCAL_URL, model=_MODEL))

        assert _backend_warnings(caplog) == [_W1_QWEN_7B]

    def test_a_local_model_at_the_floor_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_BACKEND_LOGGER):
            OpenAIBackend(_llm(provider="openai", base_url=_LOCAL_URL, model="gpt-oss:20b"))

        assert _backend_warnings(caplog) == []

    def test_an_unparseable_local_tag_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_BACKEND_LOGGER):
            OpenAIBackend(_llm(provider="openai", base_url=_LOCAL_URL, model="glm-4.5-air"))

        assert _backend_warnings(caplog) == [_W2_GLM]

    def test_a_hosted_model_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """MUTATION: warn regardless of `base_url` (gpt-4o is unparseable, so W2 would fire)."""
        with caplog.at_level(logging.WARNING, logger=_BACKEND_LOGGER):
            OpenAIBackend(_llm(provider="openai", openai_api_key=_FAKE_KEY))

        assert _backend_warnings(caplog) == []
