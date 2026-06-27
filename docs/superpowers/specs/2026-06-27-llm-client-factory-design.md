# LLM Client Factory — Design Spec

**Date:** 2026-06-27  
**Status:** Approved  
**Research basis:** 105-agent deep research, 1,200 tool uses, adversarial verification

---

## Problem

trelix has 5 LLM chat call sites across 4 files. Each one builds its own client and hardcodes OpenAI/Azure-specific payload parameters. The `max_tokens` → `max_completion_tokens` bug (hit twice: synthesizer.py, chunker.py) proves this pattern breaks every time a new model or provider is added.

### Provider payload incompatibilities (research-verified)

| Parameter | OpenAI Chat | OpenAI Responses API | Anthropic | AWS Bedrock | Vertex AI/Gemini |
|---|---|---|---|---|---|
| Token limit | `max_completion_tokens` | `max_output_tokens` | `max_tokens` | `inferenceConfig.maxTokens` | `generation_config.max_output_tokens` |
| System prompt | In `messages` as `role:system` | In `messages` | Separate `system=` param | Separate `system=[{"text":"..."}]` | Separate `system_instruction=` |
| Content type | string OR list-of-dicts | string OR list-of-dicts | list-of-dicts | **always** list-of-dicts | string OR list |
| Tool choice | `"auto"` / `"required"` / `{"type":"function",...}` | same | `{"type":"auto"}` | `{"auto":{}}` / `{"any":{}}` | `tool_config` object |
| Import | `from openai import OpenAI` | same | `from anthropic import Anthropic` | `import boto3` | `from google import genai` |

---

## Architecture

### Three layers

```
EmbedderConfig / LLMConfig
        ↓
LLMClientFactory                 ← src/trelix/llm/factory.py
  build_chat_client(config)
        ↓
TrelixChatClient  (interface)    ← src/trelix/llm/client.py
  complete(messages, ...)
  stream(messages, ...)
  tool_call(messages, tools, ...)
        ↓
Provider backends                ← src/trelix/llm/providers/
  openai_backend.py
  anthropic_backend.py
  bedrock_backend.py
  vertex_backend.py
  litellm_backend.py  ← optional delegate for 100+ providers
```

**Key principle:** All 5 call sites use `TrelixChatClient`. They never touch provider SDKs directly. Adding Bedrock requires zero changes to chunker.py, synthesizer.py, graph_rag.py, or planner/agent.py.

---

## Component Specifications

### 1. `LLMConfig` — new config dataclass

**File:** `src/trelix/core/config.py` (add alongside `EmbedderConfig`)

```python
class LLMConfig(BaseSettings):
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

    # ── Anthropic Claude ──────────────────────────────────────────────────────
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
    # model string in LiteLLM format: "bedrock/claude-3-5-sonnet", "gemini/gemini-2.0-flash"
    litellm_model: Optional[str] = Field(default=None, alias="TRELIX_LLM_LITELLM_MODEL")
    litellm_drop_params: bool = True   # suppress UnsupportedParamsError

    # ── Common ────────────────────────────────────────────────────────────────
    max_tokens: int = 2048             # used as max_completion_tokens / maxTokens / etc.
    temperature: float = 0.0
    timeout: float = 30.0
```

**Backward compat:** existing `EmbedderConfig.azure_chat_deployment` and `EmbedderConfig.openai_chat_model` continue to work. `LLMConfig` defaults to reading the same env vars (`AZURE_API_KEY`, `OPENAI_API_KEY`) so existing `.env` files need zero changes.

---

### 2. `TrelixChatClient` — unified interface

**File:** `src/trelix/llm/client.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional

@dataclass
class ChatMessage:
    role: str               # "system" | "user" | "assistant"
    content: str            # always plain text at this interface — backend converts

@dataclass
class ChatResponse:
    content: str
    model: str
    finish_reason: str      # "stop" | "length" | "tool_calls" | "end_turn" (normalized)
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class ToolCallResponse:
    tool_name: str
    tool_arguments: dict[str, Any]
    raw_response: Any       # provider-specific for debugging

class TrelixChatClient(ABC):
    """
    Provider-agnostic chat interface.

    All call sites (chunker, synthesizer, graph_rag, planner) use this.
    Never import provider SDKs directly in call sites.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,   # provider-specific system prompt handling
    ) -> ChatResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]: ...             # yields text chunks

    @abstractmethod
    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],    # OpenAI tool schema format
        force_tool: Optional[str] = None,  # force specific tool by name
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse: ...
```

---

### 3. Provider backends

#### `openai_backend.py` — OpenAI + Azure

**Handles:**
- Detects model family: gpt-4o/gpt-4.1 → `max_completion_tokens`; gpt-4/gpt-3.5 → `max_tokens`; o-series → `max_completion_tokens` only
- Azure: uses `AzureOpenAI`, deployment name as model
- Streaming via SSE iteration
- Tool calls via `tools=` + `tool_choice=`

```python
# Token limit detection
_LEGACY_MAX_TOKENS_MODELS = {"gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"}

def _token_limit_param(model: str, value: int) -> dict:
    base = model.split("/")[-1].lower()
    if any(base.startswith(m) for m in _LEGACY_MAX_TOKENS_MODELS):
        return {"max_tokens": value}
    return {"max_completion_tokens": value}
```

#### `anthropic_backend.py` — Anthropic Claude direct

**Handles:**
- `from anthropic import Anthropic, AsyncAnthropic`
- `max_tokens=` (not max_completion_tokens)
- System as separate `system=` param (not in messages)
- Tool schema uses `input_schema` not `parameters`
- Streaming via `with client.messages.stream(...)` context manager
- finish_reason: `"end_turn"` → normalize to `"stop"`

```python
# Anthropic tool schema format differs from OpenAI
def _convert_tool_schema(openai_tool: dict) -> dict:
    fn = openai_tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }

# System prompt handling
def complete(self, messages, max_tokens=None, system=None):
    # Extract system from messages if not passed separately
    sys_prompt = system or next(
        (m.content for m in messages if m.role == "system"), None
    )
    user_messages = [m for m in messages if m.role != "system"]
    kwargs = {"system": sys_prompt} if sys_prompt else {}
    response = self._client.messages.create(
        model=self._model,
        messages=[{"role": m.role, "content": m.content} for m in user_messages],
        max_tokens=max_tokens or self._config.max_tokens,
        **kwargs,
    )
```

#### `bedrock_backend.py` — AWS Bedrock Converse API

**Handles (research-verified, all 3-0):**
- `import boto3` → `bedrock_runtime = boto3.client("bedrock-runtime", region_name=...)`
- `maxTokens` nested in `inferenceConfig` (NOT top-level `max_tokens`)
- System as `system=[{"text": "..."}]` at top level
- Content always as list-of-dicts: `[{"text": "..."}]`
- `toolChoice`: `{"auto": {}}` / `{"any": {}}` / `{"tool": {"name": "fn"}}`
- Streaming via `converse_stream()` → iterate `EventStream`

```python
def _build_request(self, messages, max_tokens, system, tools, force_tool):
    request = {
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
    if system:
        request["system"] = [{"text": system}]
    if tools:
        request["toolConfig"] = {
            "tools": [self._convert_tool(t) for t in tools],
            "toolChoice": {"tool": {"name": force_tool}} if force_tool else {"auto": {}},
        }
    return request

def _convert_tool(self, openai_tool):
    fn = openai_tool["function"]
    return {
        "toolSpec": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "inputSchema": {"json": fn.get("parameters", {})},
        }
    }
```

#### `vertex_backend.py` — Google Vertex AI / Gemini

**Handles:**
- `from google import genai` (new unified SDK, 2025+)
- `max_output_tokens` in `generation_config`
- `system_instruction=` separate param
- Tool definitions use `google.genai.types.Tool`
- Streaming via `generate_content_stream()`

#### `litellm_backend.py` — LiteLLM universal delegate

**Handles (research-verified, 3-0):**
- `import litellm; litellm.drop_params = True`
- `litellm.completion(model=..., messages=..., max_tokens=..., stream=...)`
- Automatically routes to Responses API for reasoning models
- `drop_params=True` suppresses `UnsupportedParamsError` for provider-unsupported params
- Model format: `"bedrock/claude-3-5-sonnet"`, `"gemini/gemini-2.0-flash"`, `"anthropic/claude-3-5-haiku"`

```python
import litellm
litellm.drop_params = True  # suppress UnsupportedParamsError globally

def complete(self, messages, max_tokens=None, system=None, **kwargs):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend({"role": m.role, "content": m.content} for m in messages)
    response = litellm.completion(
        model=self._litellm_model,
        messages=msgs,
        max_completion_tokens=max_tokens or self._config.max_tokens,
        temperature=self._config.temperature,
    )
    return ChatResponse(
        content=response.choices[0].message.content,
        model=response.model,
        finish_reason=response.choices[0].finish_reason,
    )
```

---

### 4. `LLMClientFactory`

**File:** `src/trelix/llm/factory.py`

```python
def build_chat_client(config: LLMConfig) -> TrelixChatClient:
    match config.provider:
        case "openai":   return OpenAIBackend(config)
        case "azure":    return OpenAIBackend(config, azure=True)
        case "anthropic": return AnthropicBackend(config)
        case "bedrock":  return BedrockBackend(config)
        case "vertex":   return VertexBackend(config)
        case "litellm":  return LiteLLMBackend(config)
        case _: raise ValueError(f"Unknown LLM provider: {config.provider}")
```

---

### 5. Call site migration

Each of the 5 call sites is updated to use `TrelixChatClient`:

| Call site | File | Change |
|---|---|---|
| `_generate_summary` | `chunker.py` | `self._llm_client: TrelixChatClient` → call `complete()` |
| `_stream_response` | `synthesizer.py` | build via factory → call `stream()` |
| `_call_llm` (planner) | `planner/agent.py` | build via factory → call `tool_call()` |
| `_decompose_via_llm` | `planner/agent.py` | build via factory → call `complete()` |
| `_call_llm` (graph_rag) | `graph_rag.py` | build via factory → call `complete()` |

**All 5 sites become provider-agnostic.** Adding Bedrock or Vertex touches zero call site code.

---

### 6. Backward compatibility

- Existing `EmbedderConfig` unchanged — embedder providers (openai, azure, voyage, local) unaffected
- New `LLMConfig` with `provider="openai"` as default reads same env vars
- Callers that pass `llm_client=...` directly (e.g., `ContextualChunker(config, llm_client=my_client)`) continue to work — the factory is the preferred path but not required
- `pip install trelix` continues to work with just OpenAI/Azure (other providers are optional imports)

---

## New optional dependency groups

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40.0"]
bedrock   = ["boto3>=1.35.0"]
vertex    = ["google-genai>=1.0.0"]
litellm   = ["litellm>=1.50.0"]
llm-all   = ["trelix[anthropic,bedrock,vertex,litellm]"]
```

Each backend does a lazy import and raises `ImportError` with a helpful `pip install trelix[bedrock]` message if the dep is missing.

---

## New environment variables

```bash
# Select LLM provider for chat/synthesis (separate from embedder)
TRELIX_LLM_PROVIDER=openai        # openai | azure | anthropic | bedrock | vertex | litellm

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# AWS Bedrock  
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
# OR: AWS_PROFILE=my-profile (uses ~/.aws/credentials)

# Vertex AI
GOOGLE_CLOUD_PROJECT=my-project
GOOGLE_CLOUD_LOCATION=us-central1
# OR: GOOGLE_API_KEY=... (for AI Studio)

# LiteLLM passthrough
TRELIX_LLM_LITELLM_MODEL=bedrock/claude-3-5-sonnet  # any LiteLLM model string
```

---

## File structure

```
src/trelix/llm/
├── __init__.py
├── client.py          # TrelixChatClient ABC + ChatMessage, ChatResponse, ToolCallResponse
├── factory.py         # LLMClientFactory.build_chat_client()
└── providers/
    ├── __init__.py
    ├── openai_backend.py     # OpenAI + Azure (existing logic refactored here)
    ├── anthropic_backend.py  # Anthropic Claude direct
    ├── bedrock_backend.py    # AWS Bedrock Converse API
    ├── vertex_backend.py     # Google Vertex AI / Gemini
    └── litellm_backend.py    # LiteLLM universal delegate
```

Touches to existing files:
- `src/trelix/core/config.py` — add `LLMConfig`, add to `IndexConfig`
- `src/trelix/indexing/chunker.py` — migrate `_generate_summary` to use `TrelixChatClient`
- `src/trelix/retrieval/synthesizer.py` — migrate to factory + `stream()`
- `src/trelix/retrieval/planner/agent.py` — migrate both call sites
- `src/trelix/retrieval/graph_rag.py` — migrate `_call_llm`
- `pyproject.toml` — add optional dep groups

---

## Testing strategy

- Unit tests: mock `TrelixChatClient` — call sites are trivially testable
- Backend tests: one test per backend with mocked SDK (no real API calls)
- Integration test: `test_llm_factory.py` — instantiates each backend, calls `complete()` with a minimal message, checks `ChatResponse` shape
- Existing tests: zero changes needed — factory falls back to OpenAI backend by default

---

## Implementation order

1. `LLMConfig` + `TrelixChatClient` + `ChatMessage/Response` dataclasses
2. `OpenAIBackend` (refactor existing logic — no behavior change)
3. Migrate all 5 call sites to use factory + client
4. Tests for existing behavior (regression)
5. `AnthropicBackend`
6. `BedrockBackend`
7. `VertexBackend`
8. `LiteLLMBackend` (optional, adds 100+ providers at once)
9. Docs + CHANGELOG v0.7.0
