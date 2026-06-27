# LLM Client Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 5 scattered, provider-specific LLM call sites with a single `TrelixChatClient` ABC backed by pluggable provider implementations (OpenAI, Azure, Anthropic, Bedrock, Vertex AI, LiteLLM).

**Architecture:** Three-layer design — `LLMConfig` (config), `TrelixChatClient` (ABC interface), and provider backends (`src/trelix/llm/providers/`). All existing call sites are migrated to use the factory; adding a new provider requires zero changes to business logic files.

**Tech Stack:** Python 3.11+, pydantic-settings, openai>=1.35, anthropic>=0.40 (optional), boto3>=1.35 (optional), google-genai>=1.0 (optional), litellm>=1.50 (optional).

## Global Constraints

- Python ≥ 3.11 (uses `match` statement)
- `src/` layout, hatchling build — all new files under `src/trelix/llm/`
- Env prefix for `LLMConfig`: `TRELIX_LLM_` (reads same `AZURE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` as aliases)
- `TrelixChatClient.complete()` / `stream()` / `tool_call()` — these exact method names used by all call sites
- `ChatMessage(role, content)`, `ChatResponse(content, model, finish_reason, input_tokens, output_tokens)`, `ToolCallResponse(tool_name, tool_arguments, raw_response)` — exact dataclass names and fields
- All 860 existing unit tests must remain green throughout
- No new required runtime deps — all new providers are optional (`pip install trelix[bedrock]`)
- `LLMConfig` added to `IndexConfig` as `llm: LLMConfig = Field(default_factory=LLMConfig)`
- Lazy imports in every backend — `ImportError` with install hint if optional dep missing
- Repo: `/Users/sairamugge/Desktop/Not-Humans-World/trelix`
- Venv: `.venv/bin/python`

---

### Task 1: `src/trelix/llm/` package scaffold + `LLMConfig` + `TrelixChatClient` ABC

**Files:**
- Create: `src/trelix/llm/__init__.py`
- Create: `src/trelix/llm/providers/__init__.py`
- Create: `src/trelix/llm/client.py`
- Create: `src/trelix/llm/factory.py` (stub)
- Modify: `src/trelix/core/config.py` (add `LLMConfig`, add `llm` field to `IndexConfig`)
- Test: `tests/unit/test_llm_client.py`

**Interfaces:**
- Produces:
  - `ChatMessage(role: str, content: str)`
  - `ChatResponse(content: str, model: str, finish_reason: str, input_tokens: int = 0, output_tokens: int = 0)`
  - `ToolCallResponse(tool_name: str, tool_arguments: dict[str, Any], raw_response: Any)`
  - `TrelixChatClient` ABC with `complete()`, `stream()`, `tool_call()`
  - `LLMConfig` with `provider`, `model`, all credential fields
  - `build_chat_client(config: LLMConfig) -> TrelixChatClient` (stub raises NotImplementedError until Task 2)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/trelix/llm/providers
touch src/trelix/llm/__init__.py src/trelix/llm/providers/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_llm_client.py`:

```python
"""Tests for TrelixChatClient ABC and dataclasses."""
from __future__ import annotations

import pytest
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient


class TestDataclasses:
    def test_chat_message_fields(self) -> None:
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_chat_response_defaults(self) -> None:
        r = ChatResponse(content="hi", model="gpt-4o", finish_reason="stop")
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_chat_response_full(self) -> None:
        r = ChatResponse(content="hi", model="gpt-4o", finish_reason="stop",
                         input_tokens=10, output_tokens=5)
        assert r.input_tokens == 10
        assert r.output_tokens == 5

    def test_tool_call_response(self) -> None:
        t = ToolCallResponse(tool_name="fn", tool_arguments={"x": 1}, raw_response=None)
        assert t.tool_name == "fn"
        assert t.tool_arguments == {"x": 1}


class TestTrelixChatClientABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            TrelixChatClient()  # type: ignore[abstract]

    def test_abstract_methods_present(self) -> None:
        assert "complete" in TrelixChatClient.__abstractmethods__
        assert "stream" in TrelixChatClient.__abstractmethods__
        assert "tool_call" in TrelixChatClient.__abstractmethods__

    def test_concrete_subclass_must_implement_all(self) -> None:
        class Partial(TrelixChatClient):
            def complete(self, messages, max_tokens=None, temperature=None, system=None):
                return ChatResponse("", "", "stop")
            # missing stream and tool_call
        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


class TestLLMConfig:
    def test_default_provider_is_openai(self) -> None:
        from trelix.core.config import LLMConfig
        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.provider == "openai"

    def test_default_model_is_gpt4o(self) -> None:
        from trelix.core.config import LLMConfig
        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.model == "gpt-4o"

    def test_llm_field_on_index_config(self) -> None:
        from trelix.core.config import IndexConfig
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            cfg = IndexConfig(repo_path=tmp)
            assert hasattr(cfg, "llm")
            assert cfg.llm.provider == "openai"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import LLMConfig
        monkeypatch.setenv("TRELIX_LLM_PROVIDER", "anthropic")
        cfg = LLMConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.provider == "anthropic"
```

- [ ] **Step 3: Run tests — expect failure**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_client.py -v --tb=short 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'trelix.llm'`

- [ ] **Step 4: Create `src/trelix/llm/client.py`**

```python
"""
TrelixChatClient — provider-agnostic chat interface.

All LLM call sites in trelix use this ABC. Never import provider SDKs
(openai, anthropic, boto3, google-genai) directly in business logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str     # "system" | "user" | "assistant"
    content: str  # always plain text — backends convert to provider format


@dataclass
class ChatResponse:
    """Normalized response from any provider's chat completion."""
    content: str
    model: str
    finish_reason: str   # "stop" | "length" | "tool_calls" (normalized across providers)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCallResponse:
    """Normalized tool/function call result from any provider."""
    tool_name: str
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None  # provider-specific, for debugging


class TrelixChatClient(ABC):
    """
    Provider-agnostic interface for chat completions.

    Implement one backend per provider; call sites never touch SDKs directly.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        """Non-streaming chat completion. Returns full response."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Streaming chat completion. Yields text chunks as they arrive."""
        ...

    @abstractmethod
    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        """
        Forced tool/function call. tools uses OpenAI schema format.
        force_tool: name of the tool to force (None = auto-select).
        """
        ...
```

- [ ] **Step 5: Create stub `src/trelix/llm/factory.py`**

```python
"""LLM client factory — instantiates the right backend from LLMConfig."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig
    from trelix.llm.client import TrelixChatClient


def build_chat_client(config: "LLMConfig") -> "TrelixChatClient":
    """Return a TrelixChatClient for the configured provider."""
    match config.provider:
        case "openai" | "azure":
            from trelix.llm.providers.openai_backend import OpenAIBackend
            return OpenAIBackend(config)
        case "anthropic":
            from trelix.llm.providers.anthropic_backend import AnthropicBackend
            return AnthropicBackend(config)
        case "bedrock":
            from trelix.llm.providers.bedrock_backend import BedrockBackend
            return BedrockBackend(config)
        case "vertex":
            from trelix.llm.providers.vertex_backend import VertexBackend
            return VertexBackend(config)
        case "litellm":
            from trelix.llm.providers.litellm_backend import LiteLLMBackend
            return LiteLLMBackend(config)
        case _:
            raise ValueError(
                f"Unknown LLM provider: {config.provider!r}. "
                "Expected one of: openai, azure, anthropic, bedrock, vertex, litellm"
            )
```

- [ ] **Step 6: Add `LLMConfig` to `src/trelix/core/config.py`**

After the existing `RetrievalConfig` class and before `class IndexConfig`, add:

```python
class LLMConfig(BaseSettings):
    """
    Chat/synthesis LLM provider config.
    Separate from EmbedderConfig — you can embed with Azure and synthesize
    with Anthropic, for example.
    """
    model_config = SettingsConfigDict(
        env_prefix="TRELIX_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Literal["openai", "azure", "anthropic", "bedrock", "vertex", "litellm"] = "openai"
    model: str = "gpt-4o"

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    azure_api_key: Optional[str] = Field(default=None, alias="AZURE_API_KEY")
    azure_endpoint: Optional[str] = Field(default=None, alias="AZURE_ENDPOINT")
    azure_api_version: str = Field(default="2025-04-01-preview", alias="AZURE_API_VERSION")
    azure_chat_deployment: str = Field(default="gpt-4o", alias="AZURE_CHAT_MODEL")

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_profile: Optional[str] = Field(default=None, alias="AWS_PROFILE")

    # ── Vertex AI / Gemini ────────────────────────────────────────────────────
    google_project_id: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")

    # ── LiteLLM passthrough ───────────────────────────────────────────────────
    litellm_model: Optional[str] = Field(default=None, alias="TRELIX_LLM_LITELLM_MODEL")
    litellm_drop_params: bool = True

    # ── Common ────────────────────────────────────────────────────────────────
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: float = 30.0
```

Also add `llm` field to `IndexConfig` (after `retrieval:`):
```python
    llm:       LLMConfig       = Field(default_factory=LLMConfig)
```

Also add `"LLMConfig"` to the existing imports/exports in `config.py`.

- [ ] **Step 7: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_client.py -v --tb=short
```
Expected: all 9 tests PASS

- [ ] **Step 8: Run existing suite — expect no regression**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: 860+ passed

- [ ] **Step 9: Commit**

```bash
git add src/trelix/llm/ src/trelix/core/config.py tests/unit/test_llm_client.py
git commit -m "feat(llm): Task 1 — TrelixChatClient ABC + LLMConfig + package scaffold

- New src/trelix/llm/ package with TrelixChatClient ABC
- ChatMessage, ChatResponse, ToolCallResponse dataclasses
- LLMConfig: openai|azure|anthropic|bedrock|vertex|litellm providers
- LLMConfig added to IndexConfig as llm field
- Stub factory (raises ValueError for unknown provider)
- 9 unit tests all passing"
```

---

### Task 2: `OpenAIBackend` — refactor existing logic, zero behavior change

**Files:**
- Create: `src/trelix/llm/providers/openai_backend.py`
- Test: `tests/unit/test_llm_openai_backend.py`

**Interfaces:**
- Consumes: `TrelixChatClient`, `ChatMessage`, `ChatResponse`, `ToolCallResponse`, `LLMConfig` from Task 1
- Produces: `OpenAIBackend(config: LLMConfig)` — implements all 3 abstract methods

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_llm_openai_backend.py`:

```python
"""Tests for OpenAIBackend (mocked — no real API calls)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse
from trelix.llm.providers.openai_backend import OpenAIBackend, _token_limit_param


class TestTokenLimitParam:
    def test_gpt4o_uses_max_completion_tokens(self) -> None:
        result = _token_limit_param("gpt-4o", 100)
        assert result == {"max_completion_tokens": 100}

    def test_gpt4_uses_max_tokens(self) -> None:
        result = _token_limit_param("gpt-4", 100)
        assert result == {"max_tokens": 100}

    def test_gpt35_turbo_uses_max_tokens(self) -> None:
        result = _token_limit_param("gpt-3.5-turbo", 100)
        assert result == {"max_tokens": 100}

    def test_o3_uses_max_completion_tokens(self) -> None:
        result = _token_limit_param("o3", 100)
        assert result == {"max_completion_tokens": 100}

    def test_azure_deployment_name_uses_max_completion_tokens(self) -> None:
        # Azure deployment names don't start with legacy prefixes
        result = _token_limit_param("my-gpt4o-deployment", 100)
        assert result == {"max_completion_tokens": 100}


class TestOpenAIBackendComplete:
    def _make_backend(self, provider: str = "openai") -> OpenAIBackend:
        cfg = LLMConfig(provider=provider, _env_file=None)  # type: ignore[call-arg]
        return OpenAIBackend(cfg)

    def test_complete_returns_chat_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "hello"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        messages = [ChatMessage(role="user", content="hi")]
        result = backend.complete(messages)

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"
        assert result.finish_reason == "stop"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_complete_system_message_injected(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 2
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        backend.complete(
            [ChatMessage(role="user", content="hi")],
            system="You are helpful.",
        )
        call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
        assert call_messages[0]["role"] == "system"
        assert call_messages[0]["content"] == "You are helpful."

    def test_stream_yields_chunks(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()

        def make_chunk(text: str) -> MagicMock:
            chunk = MagicMock()
            chunk.choices[0].delta.content = text
            return chunk

        mock_client.chat.completions.create.return_value = iter([
            make_chunk("hel"), make_chunk("lo"), make_chunk("!")
        ])
        backend._client = mock_client

        chunks = list(backend.stream([ChatMessage(role="user", content="hi")]))
        assert "".join(chunks) == "hello!"

    def test_tool_call_returns_tool_call_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "search_code"
        mock_tool_call.function.arguments = '{"query": "auth", "repo_path": "/repo"}'
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        tools = [{"type": "function", "function": {"name": "search_code",
                   "parameters": {"type": "object", "properties": {}}}}]
        result = backend.tool_call(
            [ChatMessage(role="user", content="search for auth")],
            tools=tools,
        )
        assert isinstance(result, ToolCallResponse)
        assert result.tool_name == "search_code"
        assert result.tool_arguments == {"query": "auth", "repo_path": "/repo"}

    def test_client_is_none_when_no_key(self) -> None:
        cfg = LLMConfig(provider="openai", _env_file=None)  # type: ignore[call-arg]
        backend = OpenAIBackend(cfg)
        # No API key → _client should be None (graceful)
        assert backend._client is None

    def test_azure_uses_azure_client(self) -> None:
        cfg = LLMConfig(
            provider="azure",
            azure_api_key="fake-key",
            azure_endpoint="https://test.openai.azure.com/",
            _env_file=None,  # type: ignore[call-arg]
        )
        with patch("trelix.llm.providers.openai_backend.AzureOpenAI") as MockAzure:
            MockAzure.return_value = MagicMock()
            backend = OpenAIBackend(cfg)
            assert MockAzure.called
```

- [ ] **Step 2: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_openai_backend.py -v --tb=short 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'trelix.llm.providers.openai_backend'`

- [ ] **Step 3: Implement `src/trelix/llm/providers/openai_backend.py`**

```python
"""OpenAI and Azure OpenAI backend for TrelixChatClient."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.openai_backend")

# Models that require the legacy max_tokens parameter (not max_completion_tokens)
_LEGACY_MAX_TOKENS_PREFIXES = ("gpt-4-", "gpt-4 ", "gpt-3.5")
_LEGACY_MAX_TOKENS_EXACT = {"gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"}


def _token_limit_param(model: str, value: int) -> dict[str, int]:
    """Return the correct token-limit kwarg for the given model name."""
    base = model.split("/")[-1].lower().strip()
    if base in _LEGACY_MAX_TOKENS_EXACT or any(
        base.startswith(p) for p in _LEGACY_MAX_TOKENS_PREFIXES
    ):
        return {"max_tokens": value}
    return {"max_completion_tokens": value}


class OpenAIBackend(TrelixChatClient):
    """
    TrelixChatClient backed by OpenAI or Azure OpenAI.

    Handles:
    - max_tokens vs max_completion_tokens based on model family
    - Azure deployment name routing
    - Streaming via SSE
    - Tool calls via tools= + tool_choice=
    """

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._is_azure = config.provider == "azure"
        self._model = config.azure_chat_deployment if self._is_azure else config.model
        self._client = self._build_client(config)

    def _build_client(self, config: "LLMConfig") -> Any | None:
        if self._is_azure:
            if not config.azure_api_key or not config.azure_endpoint:
                logger.debug("OpenAIBackend: Azure credentials not set.")
                return None
            try:
                from openai import AzureOpenAI
                return AzureOpenAI(
                    api_key=config.azure_api_key,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.azure_api_version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenAIBackend: could not build AzureOpenAI: %s", exc)
                return None
        else:
            if not config.openai_api_key:
                logger.debug("OpenAIBackend: OPENAI_API_KEY not set.")
                return None
            try:
                from openai import OpenAI
                return OpenAI(api_key=config.openai_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenAIBackend: could not build OpenAI: %s", exc)
                return None

    def _build_messages(
        self, messages: list[ChatMessage], system: Optional[str]
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        # Inject system prompt first
        effective_system = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        if effective_system:
            result.append({"role": "system", "content": effective_system})
        result.extend(
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        )
        return result

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        if self._client is None:
            return ChatResponse(
                content="[trelix] LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY.",
                model="none",
                finish_reason="stop",
            )
        token_kwarg = _token_limit_param(
            self._model, max_tokens or self._config.max_tokens
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, system),
            temperature=temperature if temperature is not None else self._config.temperature,
            **token_kwarg,
        )
        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        if self._client is None:
            yield "[trelix] LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY."
            return
        token_kwarg = _token_limit_param(
            self._model, max_tokens or self._config.max_tokens
        )
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, system),
            temperature=temperature if temperature is not None else self._config.temperature,
            stream=True,
            **token_kwarg,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        if self._client is None:
            raise RuntimeError(
                "LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY."
            )
        tool_choice: Any = (
            {"type": "function", "function": {"name": force_tool}}
            if force_tool
            else "auto"
        )
        token_kwarg = _token_limit_param(
            self._model, max_tokens or self._config.max_tokens
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(messages, None),
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.0,
            timeout=self._config.timeout,
            **token_kwarg,
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("LLM did not return a tool call.")
        tc = tool_calls[0]
        return ToolCallResponse(
            tool_name=tc.function.name,
            tool_arguments=json.loads(tc.function.arguments),
            raw_response=response,
        )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_openai_backend.py -v --tb=short
```
Expected: all 8 tests PASS

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: 860+ passed

- [ ] **Step 6: Commit**

```bash
git add src/trelix/llm/providers/openai_backend.py tests/unit/test_llm_openai_backend.py
git commit -m "feat(llm): Task 2 — OpenAIBackend (OpenAI + Azure)

- _token_limit_param() detects legacy vs modern models
- complete(), stream(), tool_call() all implemented
- Graceful None client when credentials absent
- 8 unit tests passing"
```

---

### Task 3: Migrate all 5 call sites to `TrelixChatClient`

**Files:**
- Modify: `src/trelix/indexing/chunker.py` (lines ~174-184, ~260-270)
- Modify: `src/trelix/indexing/indexer.py` (lines ~218-247)
- Modify: `src/trelix/retrieval/synthesizer.py` (full client build + stream call)
- Modify: `src/trelix/retrieval/planner/agent.py` (QueryPlanner + AdaptiveRouter)
- Modify: `src/trelix/retrieval/graph_rag.py` (GraphRAGSynthesizer)
- Test: existing `tests/unit/test_chunker.py`, `tests/unit/test_planner_adaptive.py`, etc.

**Interfaces:**
- Consumes: `TrelixChatClient`, `ChatMessage`, `ChatResponse`, `ToolCallResponse` from Task 1; `OpenAIBackend`, `build_chat_client` from Tasks 1-2
- Produces: all 5 call sites use `TrelixChatClient` — no direct SDK imports in business logic

- [ ] **Step 1: Run existing tests as baseline**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: 860+ passed (record this number)

- [ ] **Step 2: Migrate `chunker.py` — `ContextualChunker`**

Change `__init__` signature from `llm_client: Any | None` to accept both `Any | None` AND `TrelixChatClient | None` (union type for backward compat):

```python
# In ContextualChunker.__init__:
from trelix.llm.client import ChatMessage, TrelixChatClient

def __init__(
    self,
    config: ChunkerConfig,
    llm_client: Any | None = None,  # openai.OpenAI, TrelixChatClient, or None
) -> None:
    super().__init__(config)
    self._llm_client = llm_client
```

Change `_generate_summary` to detect which interface is available:

```python
def _generate_summary(
    self,
    symbol: "Symbol",
    file_rel_path: str,
    language: str,
) -> str | None:
    prompt = self._CONTEXT_PROMPT.format(
        rel_path=file_rel_path,
        language=language,
        body=symbol.body[:800],
    )
    try:
        assert self._llm_client is not None
        # New path: TrelixChatClient interface
        if isinstance(self._llm_client, TrelixChatClient):
            response = self._llm_client.complete(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=self.config.contextual_max_tokens,
                temperature=0,
            )
            return response.content.strip() or None
        # Legacy path: raw openai client (backward compat)
        response = self._llm_client.chat.completions.create(
            model=self.config.contextual_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.config.contextual_max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001
        logger.warning("ContextualChunker LLM call failed: %s", exc)
        return None
```

- [ ] **Step 3: Migrate `indexer.py` — `_build_chunker`**

In `Indexer._build_chunker()`, replace the existing client-building logic with the factory:

```python
def _build_chunker(self, config: IndexConfig) -> Chunker:
    if not config.chunker.contextual:
        return Chunker(config.chunker)
    try:
        from trelix.llm.factory import build_chat_client
        llm_client = build_chat_client(config.llm)
        logger.info(
            "ContextualChunker: using %s provider, model=%s",
            config.llm.provider,
            config.llm.model,
        )
        return ContextualChunker(config.chunker, llm_client=llm_client)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ContextualChunker: could not build LLM client (%s) — falling back to base Chunker",
            exc,
        )
        return Chunker(config.chunker)
```

- [ ] **Step 4: Migrate `synthesizer.py`**

Replace `_build_client()` + `_stream_response()` with factory-based approach:

```python
# In Synthesizer.__init__:
from trelix.llm.factory import build_chat_client
from trelix.llm.client import ChatMessage

def __init__(self, config: EmbedderConfig) -> None:
    self._config = config
    # Build from LLMConfig derived from EmbedderConfig for backward compat
    from trelix.core.config import LLMConfig
    llm_cfg = LLMConfig(
        provider=config.provider if config.provider in ("openai", "azure") else "openai",
        _env_file=None,  # type: ignore[call-arg]
    )
    # Carry over credentials
    llm_cfg = llm_cfg.model_copy(update={
        "openai_api_key": config.openai_api_key,
        "azure_api_key": config.azure_api_key,
        "azure_endpoint": config.azure_endpoint,
        "azure_api_version": config.azure_api_version,
        "azure_chat_deployment": config.azure_chat_deployment,
        "model": config.openai_chat_model,
    })
    self._llm_client = build_chat_client(llm_cfg)

def _stream_response(self, context: RetrievedContext, config: EmbedderConfig) -> str:
    user_message = _USER_TEMPLATE.format(
        context_text=context.context_text,
        query=context.query,
    )
    max_tokens: int = getattr(config, "synthesis_max_tokens", 2048)
    collected: list[str] = []
    for chunk in self._llm_client.stream(
        messages=[ChatMessage(role="user", content=user_message)],
        system=self._system_prompt(context.intent),
        max_tokens=max_tokens,
        temperature=0.2,
    ):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        collected.append(chunk)
    return "".join(collected)
```

- [ ] **Step 5: Migrate `planner/agent.py` — `QueryPlanner._call_llm` + `AdaptiveRouter._decompose_via_llm`**

In `QueryPlanner.__init__`:
```python
from trelix.llm.factory import build_chat_client
from trelix.llm.client import ChatMessage

# Replace _build_client() call:
from trelix.core.config import LLMConfig
llm_cfg = LLMConfig(
    provider=config.provider if config.provider in ("openai", "azure") else "openai",
    _env_file=None,  # type: ignore[call-arg]
)
self._llm_client = build_chat_client(llm_cfg)
```

In `QueryPlanner._call_llm()`:
```python
def _call_llm(self, query: str, project_context: dict[str, Any] | None) -> QueryPlan:
    result = self._llm_client.tool_call(
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=self._build_user_message(query, project_context)),
        ],
        tools=[PLANNER_TOOL_SCHEMA],
        force_tool="produce_query_plan",
        max_tokens=512,
    )
    return self._parse_tool_response(result.tool_arguments)
```

In `AdaptiveRouter._decompose_via_llm()`:
```python
def _decompose_via_llm(self, planner: QueryPlanner, query: str) -> list[str]:
    response = planner._llm_client.complete(
        messages=[ChatMessage(role="user", content=self._decomposition_prompt(query))],
        max_tokens=256,
        temperature=0.0,
    )
    return self._parse_decomposition(response.content)
```

- [ ] **Step 6: Migrate `graph_rag.py` — `GraphRAGSynthesizer._call_llm`**

In `GraphRAGSynthesizer.__init__`:
```python
from trelix.llm.factory import build_chat_client
from trelix.llm.client import ChatMessage

from trelix.core.config import LLMConfig
llm_cfg = LLMConfig(
    provider=embedder_config.provider if embedder_config.provider in ("openai","azure") else "openai",
    _env_file=None,  # type: ignore[call-arg]
)
self._llm_client = build_chat_client(llm_cfg)
```

In `GraphRAGSynthesizer._call_llm()`:
```python
def _call_llm(self, prompt: str, max_tokens: int) -> str:
    response = self._llm_client.complete(
        messages=[ChatMessage(role="user", content=prompt)],
        system=(
            "You are an expert software engineer answering questions about a "
            "codebase. Base your answer strictly on the provided code context. "
            "Be concise and precise."
        ),
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return response.content
```

- [ ] **Step 7: Run full test suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```
Expected: same number as Step 1 baseline (860+), no regressions

- [ ] **Step 8: Commit**

```bash
git add src/trelix/indexing/chunker.py src/trelix/indexing/indexer.py \
        src/trelix/retrieval/synthesizer.py src/trelix/retrieval/planner/agent.py \
        src/trelix/retrieval/graph_rag.py
git commit -m "feat(llm): Task 3 — migrate all 5 call sites to TrelixChatClient

- chunker.py: ContextualChunker._generate_summary uses TrelixChatClient
- indexer.py: _build_chunker uses factory instead of raw client
- synthesizer.py: _stream_response uses client.stream()
- planner/agent.py: QueryPlanner._call_llm uses client.tool_call()
- planner/agent.py: AdaptiveRouter._decompose_via_llm uses client.complete()
- graph_rag.py: _call_llm uses client.complete()
All 860+ unit tests passing"
```

---

### Task 4: `AnthropicBackend`

**Files:**
- Create: `src/trelix/llm/providers/anthropic_backend.py`
- Modify: `pyproject.toml` (add `anthropic` optional dep)
- Test: `tests/unit/test_llm_anthropic_backend.py`

**Interfaces:**
- Consumes: `TrelixChatClient`, `ChatMessage`, `ChatResponse`, `ToolCallResponse`, `LLMConfig`
- Produces: `AnthropicBackend(config: LLMConfig)` — `max_tokens=` (NOT `max_completion_tokens`), system as separate param, tool schema uses `input_schema`

- [ ] **Step 1: Add optional dep to `pyproject.toml`**

Add to `[project.optional-dependencies]`:
```toml
anthropic = ["anthropic>=0.40.0"]
bedrock   = ["boto3>=1.35.0"]
vertex    = ["google-genai>=1.0.0"]
litellm   = ["litellm>=1.50.0"]
llm-all   = ["trelix[anthropic,bedrock,vertex,litellm]"]
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_llm_anthropic_backend.py`:

```python
"""Tests for AnthropicBackend (mocked — no real API calls)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse


class TestAnthropicBackend:
    def _make_backend(self):
        from trelix.llm.providers.anthropic_backend import AnthropicBackend
        cfg = LLMConfig(
            provider="anthropic",
            anthropic_api_key="sk-ant-fake",
            model="claude-3-5-sonnet-20241022",
            _env_file=None,  # type: ignore[call-arg]
        )
        return AnthropicBackend(cfg)

    def test_complete_returns_chat_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"
        assert result.finish_reason == "stop"  # normalized from "end_turn"

    def test_uses_max_tokens_not_max_completion_tokens(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete([ChatMessage(role="user", content="hi")], max_tokens=100)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs
        assert call_kwargs["max_tokens"] == 100

    def test_system_as_separate_param(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        backend.complete(
            [ChatMessage(role="user", content="hi")],
            system="You are a bot.",
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "You are a bot."
        # system must NOT appear in messages list
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    def test_finish_reason_normalization(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.model = "claude"
        mock_response.stop_reason = "max_tokens"  # Anthropic name
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])
        assert result.finish_reason == "length"  # normalized

    def test_import_error_when_anthropic_not_installed(self) -> None:
        from trelix.llm.providers.anthropic_backend import AnthropicBackend
        cfg = LLMConfig(
            provider="anthropic",
            anthropic_api_key="sk-ant-fake",
            _env_file=None,  # type: ignore[call-arg]
        )
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install"):
                AnthropicBackend(cfg)
```

- [ ] **Step 3: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_anthropic_backend.py -v --tb=short 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'trelix.llm.providers.anthropic_backend'`

- [ ] **Step 4: Implement `src/trelix/llm/providers/anthropic_backend.py`**

```python
"""Anthropic Claude backend for TrelixChatClient."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.anthropic_backend")

_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class AnthropicBackend(TrelixChatClient):
    """
    TrelixChatClient backed by Anthropic Claude.

    Key differences from OpenAI:
    - max_tokens= (not max_completion_tokens)
    - system= as a separate top-level parameter (not in messages)
    - Tool schema uses input_schema instead of parameters
    - finish_reason: "end_turn" normalized to "stop"
    """

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._model = config.model
        self._client = self._build_client(config)

    def _build_client(self, config: "LLMConfig") -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requires the anthropic package. "
                "Install it with: pip install 'trelix[anthropic]'"
            ) from exc
        if not config.anthropic_api_key:
            logger.debug("AnthropicBackend: ANTHROPIC_API_KEY not set.")
            return None
        return anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _extract_system(
        self, messages: list[ChatMessage], system: Optional[str]
    ) -> tuple[Optional[str], list[dict[str, str]]]:
        effective = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        user_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]
        return effective, user_msgs

    def _normalize_finish_reason(self, stop_reason: str) -> str:
        return _FINISH_REASON_MAP.get(stop_reason, "stop")

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        if self._client is None:
            return ChatResponse(
                content="[trelix] Anthropic not configured — set ANTHROPIC_API_KEY.",
                model="none",
                finish_reason="stop",
            )
        sys_prompt, user_msgs = self._extract_system(messages, system)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        response = self._client.messages.create(
            model=self._model,
            messages=user_msgs,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            **kwargs,
        )
        content = response.content[0].text if response.content else ""
        return ChatResponse(
            content=content,
            model=response.model,
            finish_reason=self._normalize_finish_reason(response.stop_reason or "end_turn"),
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        if self._client is None:
            yield "[trelix] Anthropic not configured — set ANTHROPIC_API_KEY."
            return
        sys_prompt, user_msgs = self._extract_system(messages, system)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        with self._client.messages.stream(
            model=self._model,
            messages=user_msgs,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        if self._client is None:
            raise RuntimeError("Anthropic not configured — set ANTHROPIC_API_KEY.")
        # Convert OpenAI tool schema to Anthropic format
        anthropic_tools = [self._convert_tool(t) for t in tools]
        tool_choice: dict[str, Any] = (
            {"type": "tool", "name": force_tool} if force_tool else {"type": "auto"}
        )
        sys_prompt, user_msgs = self._extract_system(messages, None)
        kwargs: dict[str, Any] = {}
        if sys_prompt:
            kwargs["system"] = sys_prompt
        response = self._client.messages.create(
            model=self._model,
            messages=user_msgs,
            tools=anthropic_tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens or self._config.max_tokens,
            **kwargs,
        )
        tool_use = next(
            (block for block in response.content if block.type == "tool_use"), None
        )
        if not tool_use:
            raise RuntimeError("Anthropic did not return a tool_use block.")
        return ToolCallResponse(
            tool_name=tool_use.name,
            tool_arguments=dict(tool_use.input),
            raw_response=response,
        )

    def _convert_tool(self, openai_tool: dict[str, Any]) -> dict[str, Any]:
        fn = openai_tool["function"]
        return {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        }
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_anthropic_backend.py -v --tb=short
```
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/trelix/llm/providers/anthropic_backend.py tests/unit/test_llm_anthropic_backend.py pyproject.toml
git commit -m "feat(llm): Task 4 — AnthropicBackend

- max_tokens= (not max_completion_tokens)
- system= as separate top-level param
- Tool schema input_schema conversion
- finish_reason normalization (end_turn → stop)
- Lazy import with helpful pip install hint
- 5 unit tests passing"
```

---

### Task 5: `BedrockBackend`

**Files:**
- Create: `src/trelix/llm/providers/bedrock_backend.py`
- Test: `tests/unit/test_llm_bedrock_backend.py`

**Interfaces:**
- Consumes: `TrelixChatClient`, `LLMConfig`
- Produces: `BedrockBackend(config: LLMConfig)` — uses Bedrock Converse API with camelCase params

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_llm_bedrock_backend.py`:

```python
"""Tests for BedrockBackend (mocked — no real AWS calls)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse


class TestBedrockBackend:
    def _make_backend(self):
        from trelix.llm.providers.bedrock_backend import BedrockBackend
        cfg = LLMConfig(
            provider="bedrock",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region="us-east-1",
            _env_file=None,  # type: ignore[call-arg]
        )
        return BedrockBackend(cfg)

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
            "output": {"message": {
                "content": [{"toolUse": {"toolUseId": "1", "name": "fn", "input": {"x": 1}}}],
                "role": "assistant",
            }},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        backend._client = mock_client

        tools = [{"type": "function", "function": {"name": "fn",
                   "parameters": {"type": "object", "properties": {}}}}]
        backend.tool_call([ChatMessage(role="user", content="hi")], tools=tools)

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_import_error_when_boto3_not_installed(self) -> None:
        from trelix.llm.providers.bedrock_backend import BedrockBackend
        cfg = LLMConfig(provider="bedrock", _env_file=None)  # type: ignore[call-arg]
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="pip install"):
                BedrockBackend(cfg)
```

- [ ] **Step 2: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_bedrock_backend.py -v --tb=short 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'trelix.llm.providers.bedrock_backend'`

- [ ] **Step 3: Implement `src/trelix/llm/providers/bedrock_backend.py`**

```python
"""AWS Bedrock Converse API backend for TrelixChatClient."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.bedrock_backend")

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class BedrockBackend(TrelixChatClient):
    """
    TrelixChatClient backed by AWS Bedrock Converse API.

    Key differences (research-verified, 3-0 vote):
    - Token limit: inferenceConfig.maxTokens (camelCase, nested)
    - System prompt: system=[{"text": "..."}] at top level
    - Message content: always list-of-dicts [{"text": "..."}]
    - Tool choice: {"auto": {}} / {"any": {}} / {"tool": {"name": "fn"}}
    """

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._model = config.model
        self._client = self._build_client(config)

    def _build_client(self, config: "LLMConfig") -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Bedrock backend requires boto3. "
                "Install it with: pip install 'trelix[bedrock]'"
            ) from exc
        session_kwargs: dict[str, Any] = {}
        if config.aws_profile:
            session_kwargs["profile_name"] = config.aws_profile
        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {"region_name": config.aws_region}
        if config.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = config.aws_access_key_id
        if config.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = config.aws_secret_access_key
        return session.client("bedrock-runtime", **client_kwargs)

    def _build_request(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int],
        system: Optional[str],
        tools: Optional[list[dict[str, Any]]] = None,
        force_tool: Optional[str] = None,
    ) -> dict[str, Any]:
        effective_system = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        request: dict[str, Any] = {
            "modelId": self._model,
            "inferenceConfig": {
                "maxTokens": max_tokens or self._config.max_tokens,
                "temperature": self._config.temperature,
            },
            "messages": [
                {
                    "role": m.role,
                    "content": [{"text": m.content}],  # always list-of-dicts
                }
                for m in messages if m.role != "system"
            ],
        }
        if effective_system:
            request["system"] = [{"text": effective_system}]
        if tools:
            request["toolConfig"] = {
                "tools": [self._convert_tool(t) for t in tools],
                "toolChoice": (
                    {"tool": {"name": force_tool}} if force_tool else {"auto": {}}
                ),
            }
        return request

    def _convert_tool(self, openai_tool: dict[str, Any]) -> dict[str, Any]:
        fn = openai_tool["function"]
        return {
            "toolSpec": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "inputSchema": {"json": fn.get("parameters", {})},
            }
        }

    def _normalize_finish_reason(self, stop_reason: str) -> str:
        return _STOP_REASON_MAP.get(stop_reason, "stop")

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        request = self._build_request(messages, max_tokens, system)
        if temperature is not None:
            request["inferenceConfig"]["temperature"] = temperature
        response = self._client.converse(**request)
        output_msg = response["output"]["message"]
        content = next(
            (block["text"] for block in output_msg["content"] if "text" in block), ""
        )
        usage = response.get("usage", {})
        return ChatResponse(
            content=content,
            model=self._model,
            finish_reason=self._normalize_finish_reason(response.get("stopReason", "end_turn")),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        request = self._build_request(messages, max_tokens, system)
        if temperature is not None:
            request["inferenceConfig"]["temperature"] = temperature
        response = self._client.converse_stream(**request)
        stream = response.get("stream")
        if stream:
            for event in stream:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        request = self._build_request(messages, max_tokens, None, tools, force_tool)
        response = self._client.converse(**request)
        output_msg = response["output"]["message"]
        tool_use = next(
            (block["toolUse"] for block in output_msg["content"] if "toolUse" in block),
            None,
        )
        if not tool_use:
            raise RuntimeError("Bedrock did not return a toolUse block.")
        return ToolCallResponse(
            tool_name=tool_use["name"],
            tool_arguments=dict(tool_use.get("input", {})),
            raw_response=response,
        )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_bedrock_backend.py -v --tb=short
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trelix/llm/providers/bedrock_backend.py tests/unit/test_llm_bedrock_backend.py
git commit -m "feat(llm): Task 5 — BedrockBackend (AWS Bedrock Converse API)

- inferenceConfig.maxTokens (camelCase, nested — NOT top-level max_tokens)
- system=[{'text': '...'}] at top level
- Content always list-of-dicts
- toolChoice: {'auto': {}} format
- Lazy import with pip install hint
- 6 unit tests passing"
```

---

### Task 6: `VertexBackend` + `LiteLLMBackend`

**Files:**
- Create: `src/trelix/llm/providers/vertex_backend.py`
- Create: `src/trelix/llm/providers/litellm_backend.py`
- Test: `tests/unit/test_llm_vertex_backend.py`
- Test: `tests/unit/test_llm_litellm_backend.py`

**Interfaces:**
- Consumes: `TrelixChatClient`, `LLMConfig`
- Produces: `VertexBackend`, `LiteLLMBackend` — both implement all 3 abstract methods

- [ ] **Step 1: Write failing tests for VertexBackend**

Create `tests/unit/test_llm_vertex_backend.py`:

```python
"""Tests for VertexBackend (mocked)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse


class TestVertexBackend:
    def _make_backend(self):
        from trelix.llm.providers.vertex_backend import VertexBackend
        cfg = LLMConfig(
            provider="vertex",
            model="gemini-2.0-flash",
            google_api_key="fake-key",
            _env_file=None,  # type: ignore[call-arg]
        )
        return VertexBackend(cfg)

    def test_complete_returns_chat_response(self) -> None:
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "hello from gemini"
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_client.models.generate_content.return_value = mock_response
        backend._client = mock_client

        result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello from gemini"
        assert result.finish_reason == "stop"

    def test_import_error_when_google_genai_not_installed(self) -> None:
        from trelix.llm.providers.vertex_backend import VertexBackend
        cfg = LLMConfig(provider="vertex", _env_file=None)  # type: ignore[call-arg]
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="pip install"):
                VertexBackend(cfg)
```

- [ ] **Step 2: Write failing tests for LiteLLMBackend**

Create `tests/unit/test_llm_litellm_backend.py`:

```python
"""Tests for LiteLLMBackend (mocked)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage, ChatResponse


class TestLiteLLMBackend:
    def _make_backend(self):
        from trelix.llm.providers.litellm_backend import LiteLLMBackend
        cfg = LLMConfig(
            provider="litellm",
            litellm_model="bedrock/claude-3-5-sonnet",
            _env_file=None,  # type: ignore[call-arg]
        )
        return LiteLLMBackend(cfg)

    def test_complete_calls_litellm_completion(self) -> None:
        backend = self._make_backend()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "bedrock/claude-3-5-sonnet"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with patch("trelix.llm.providers.litellm_backend.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response
            result = backend.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"

    def test_uses_litellm_model_string(self) -> None:
        backend = self._make_backend()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "bedrock/claude-3-5-sonnet"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1

        with patch("trelix.llm.providers.litellm_backend.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response
            backend.complete([ChatMessage(role="user", content="hi")])
            call_kwargs = mock_litellm.completion.call_args[1]
            assert call_kwargs["model"] == "bedrock/claude-3-5-sonnet"

    def test_import_error_when_litellm_not_installed(self) -> None:
        from trelix.llm.providers.litellm_backend import LiteLLMBackend
        cfg = LLMConfig(provider="litellm", _env_file=None)  # type: ignore[call-arg]
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="pip install"):
                LiteLLMBackend(cfg)
```

- [ ] **Step 3: Implement `src/trelix/llm/providers/vertex_backend.py`**

```python
"""Google Vertex AI / Gemini backend for TrelixChatClient."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.vertex_backend")


class VertexBackend(TrelixChatClient):
    """
    TrelixChatClient backed by Google Vertex AI / Gemini via google-genai SDK.

    Key differences from OpenAI:
    - max_output_tokens in generation_config
    - system_instruction= separate param
    - google.genai.types.Tool for function definitions
    """

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._model = config.model
        self._client = self._build_client(config)

    def _build_client(self, config: "LLMConfig") -> Any:
        try:
            from google import genai
        except (ImportError, TypeError) as exc:
            raise ImportError(
                "Vertex backend requires google-genai. "
                "Install it with: pip install 'trelix[vertex]'"
            ) from exc
        if config.google_api_key:
            return genai.Client(api_key=config.google_api_key)
        if config.google_project_id:
            return genai.Client(
                vertexai=True,
                project=config.google_project_id,
                location=config.google_location,
            )
        logger.debug("VertexBackend: no Google credentials set.")
        return None

    def _build_contents(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [
            {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]}
            for m in messages if m.role != "system"
        ]

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        if self._client is None:
            return ChatResponse(
                content="[trelix] Vertex AI not configured.",
                model="none",
                finish_reason="stop",
            )
        from google.genai import types  # type: ignore[import]
        effective_system = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            system_instruction=effective_system,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._build_contents(messages),
            config=gen_config,
        )
        finish = response.candidates[0].finish_reason.name.lower() if response.candidates else "stop"
        normalized = "stop" if finish in ("stop", "1") else "length" if finish in ("max_tokens", "2") else "stop"
        return ChatResponse(
            content=response.text or "",
            model=self._model,
            finish_reason=normalized,
            input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
            output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        if self._client is None:
            yield "[trelix] Vertex AI not configured."
            return
        from google.genai import types  # type: ignore[import]
        effective_system = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            system_instruction=effective_system,
        )
        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=self._build_contents(messages),
            config=gen_config,
        ):
            if chunk.text:
                yield chunk.text

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        if self._client is None:
            raise RuntimeError("Vertex AI not configured.")
        from google.genai import types  # type: ignore[import]
        vertex_tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["function"]["name"],
                    description=t["function"].get("description", ""),
                    parameters=t["function"].get("parameters", {}),
                )
            ])
            for t in tools
        ]
        gen_config = types.GenerateContentConfig(
            tools=vertex_tools,
            max_output_tokens=max_tokens or self._config.max_tokens,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._build_contents(messages),
            config=gen_config,
        )
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if part.function_call:
                return ToolCallResponse(
                    tool_name=part.function_call.name,
                    tool_arguments=dict(part.function_call.args),
                    raw_response=response,
                )
        raise RuntimeError("Vertex AI did not return a function call.")
```

- [ ] **Step 4: Implement `src/trelix/llm/providers/litellm_backend.py`**

```python
"""LiteLLM universal backend for TrelixChatClient (100+ providers)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.litellm_backend")


class LiteLLMBackend(TrelixChatClient):
    """
    TrelixChatClient backed by LiteLLM — delegates to 100+ providers.

    Model string format: "bedrock/claude-3-5-sonnet", "gemini/gemini-2.0-flash",
    "anthropic/claude-3-5-haiku", "ollama/llama3", etc.

    LiteLLM handles all parameter normalization automatically.
    drop_params=True suppresses UnsupportedParamsError for provider-unsupported params.
    """

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        try:
            import litellm as _litellm
            _litellm.drop_params = config.litellm_drop_params
            self._litellm = _litellm
        except ImportError as exc:
            raise ImportError(
                "LiteLLM backend requires litellm. "
                "Install it with: pip install 'trelix[litellm]'"
            ) from exc
        self._model = config.litellm_model or config.model

    def _build_messages(
        self, messages: list[ChatMessage], system: Optional[str]
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        effective_system = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        if effective_system:
            result.append({"role": "system", "content": effective_system})
        result.extend(
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        )
        return result

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        response = self._litellm.completion(
            model=self._model,
            messages=self._build_messages(messages, system),
            max_completion_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
        )
        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model or self._model,
            finish_reason=choice.finish_reason or "stop",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        response = self._litellm.completion(
            model=self._model,
            messages=self._build_messages(messages, system),
            max_completion_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        import json
        tool_choice: Any = (
            {"type": "function", "function": {"name": force_tool}}
            if force_tool else "auto"
        )
        response = self._litellm.completion(
            model=self._model,
            messages=self._build_messages(messages, None),
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens or self._config.max_tokens,
            temperature=0.0,
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("LiteLLM did not return a tool call.")
        tc = tool_calls[0]
        return ToolCallResponse(
            tool_name=tc.function.name,
            tool_arguments=json.loads(tc.function.arguments),
            raw_response=response,
        )
```

- [ ] **Step 5: Run all new backend tests**

```bash
.venv/bin/python -m pytest tests/unit/test_llm_vertex_backend.py tests/unit/test_llm_litellm_backend.py -v --tb=short
```
Expected: all 5 tests PASS

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: 860+ passed

- [ ] **Step 7: Commit**

```bash
git add src/trelix/llm/providers/vertex_backend.py \
        src/trelix/llm/providers/litellm_backend.py \
        tests/unit/test_llm_vertex_backend.py \
        tests/unit/test_llm_litellm_backend.py
git commit -m "feat(llm): Task 6 — VertexBackend + LiteLLMBackend

- VertexBackend: max_output_tokens in GenerateContentConfig, system_instruction param
- LiteLLMBackend: drop_params=True, routes to 100+ providers via model string
- 5 unit tests passing"
```

---

### Task 7: `__init__.py` exports, `.env.example`, CHANGELOG, version bump

**Files:**
- Modify: `src/trelix/llm/__init__.py`
- Modify: `.env.example`
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml` (version 0.6.0 → 0.7.0)
- Modify: `src/trelix/__init__.py` (__version__)

- [ ] **Step 1: Update `src/trelix/llm/__init__.py`**

```python
"""trelix LLM client factory — provider-agnostic chat interface."""
from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse
from trelix.llm.factory import build_chat_client

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "TrelixChatClient",
    "ToolCallResponse",
    "build_chat_client",
]
```

- [ ] **Step 2: Add new env vars to `.env.example`**

Append after the existing content:
```bash
# ---------------------------------------------------------------------------
# LLM provider for chat/synthesis (separate from embedder)
# Default: reads OPENAI_API_KEY / AZURE_API_KEY from above
# ---------------------------------------------------------------------------
# TRELIX_LLM_PROVIDER=openai        # openai | azure | anthropic | bedrock | vertex | litellm
# TRELIX_LLM_MODEL=gpt-4o

# Anthropic (pip install trelix[anthropic])
# ANTHROPIC_API_KEY=sk-ant-...

# AWS Bedrock (pip install trelix[bedrock])
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_PROFILE=my-profile

# Vertex AI / Gemini (pip install trelix[vertex])
# GOOGLE_CLOUD_PROJECT=my-project
# GOOGLE_CLOUD_LOCATION=us-central1
# GOOGLE_API_KEY=...  (for AI Studio)

# LiteLLM passthrough — 100+ providers (pip install trelix[litellm])
# TRELIX_LLM_PROVIDER=litellm
# TRELIX_LLM_LITELLM_MODEL=bedrock/claude-3-5-sonnet
```

- [ ] **Step 3: Update CHANGELOG.md** — add v0.7.0 entry before v0.6.0:

```markdown
## [0.7.0] — 2026-06-27

### Overview
Universal LLM client factory — all 5 chat call sites migrated to a provider-agnostic
`TrelixChatClient` ABC. Adding any new provider requires zero changes to business logic.

### Added
- **`src/trelix/llm/` package** — `TrelixChatClient` ABC, `ChatMessage`, `ChatResponse`,
  `ToolCallResponse` dataclasses, `build_chat_client()` factory
- **`LLMConfig`** — new config class for chat providers (separate from `EmbedderConfig`).
  Added as `IndexConfig.llm` field.
- **`OpenAIBackend`** — OpenAI + Azure. Auto-detects `max_completion_tokens` vs `max_tokens`
  based on model family (gpt-4o→max_completion_tokens; gpt-4/gpt-3.5→max_tokens)
- **`AnthropicBackend`** — Anthropic Claude direct. `max_tokens=`, `system=` separate param,
  `input_schema` tool format, `end_turn`→`stop` normalization. `pip install trelix[anthropic]`
- **`BedrockBackend`** — AWS Bedrock Converse API. `inferenceConfig.maxTokens` (nested camelCase),
  `system=[{"text":...}]` top-level, content always list-of-dicts, `{"auto":{}}` tool choice.
  `pip install trelix[bedrock]`
- **`VertexBackend`** — Google Vertex AI / Gemini via google-genai SDK. `max_output_tokens` in
  `GenerateContentConfig`, `system_instruction=` param. `pip install trelix[vertex]`
- **`LiteLLMBackend`** — universal delegate for 100+ providers. `drop_params=True` suppresses
  UnsupportedParamsError. Model strings: `"bedrock/claude-3-5-sonnet"`, `"gemini/gemini-2.0-flash"`.
  `pip install trelix[litellm]`
- New optional dep groups: `[anthropic]`, `[bedrock]`, `[vertex]`, `[litellm]`, `[llm-all]`

### Changed
- All 5 LLM call sites now use `TrelixChatClient` via factory — never import provider SDKs directly
- `ContextualChunker` accepts `TrelixChatClient` (new) or raw openai client (backward compat)

### Fixed
- `_token_limit_param()` in OpenAIBackend correctly routes legacy models to `max_tokens=`
  and modern models to `max_completion_tokens=` — eliminates the recurring parameter bug
```

- [ ] **Step 4: Bump version**

```bash
sed -i '' 's/^version = "0.6.0"/version = "0.7.0"/' pyproject.toml
sed -i '' 's/__version__ = "0.6.0"/__version__ = "0.7.0"/' src/trelix/__init__.py
```

- [ ] **Step 5: Run full suite one final time**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
.venv/bin/ruff check src/ tests/ --ignore E501 2>&1 | tail -2
.venv/bin/ruff format --check src/ tests/ 2>&1 | tail -2
```
Expected: 860+ passed, ruff clean, format clean

- [ ] **Step 6: Commit and tag**

```bash
git add src/trelix/llm/__init__.py .env.example CHANGELOG.md pyproject.toml src/trelix/__init__.py
git commit -m "feat(llm): Task 7 — v0.7.0 exports, docs, CHANGELOG, version bump

Universal LLM client factory complete:
- 5 backends: openai, azure, anthropic, bedrock, vertex, litellm
- All 5 call sites migrated to TrelixChatClient
- max_tokens/max_completion_tokens/maxTokens/max_output_tokens handled per backend
- 860+ tests passing"

git checkout develop && git merge --no-ff main -m "sync: v0.7.0 llm factory"
git checkout main
git tag -a v0.7.0 -m "v0.7.0: Universal LLM client factory"
```
