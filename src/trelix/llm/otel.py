"""
GenAI ``chat`` spans around every TrelixChatClient call — opt-in, see docs/OBSERVABILITY.md.

`build_chat_client()` returns a `TracedChatClient` only when `TRELIX_OTEL_ENABLED` resolves
true from the environment (`otel_tracing.is_enabled_from_env()`), so this module is never
imported on the off path and imports nothing from `opentelemetry` at module level itself:
the library is reached lazily, and when it is missing the factory gets the bare backend plus
one WARNING per process (owner decision D8).

One span per `complete()`, `stream()` and `tool_call()`, built with
`opentelemetry-util-genai`'s `TelemetryHandler.inference()` (floor 1.2b0: `suspend()` and the
`gen_ai.usage.cache_write.input_tokens` name). Request attributes on every span; usage,
finish reasons and the response model only on `complete()` (`ToolCallResponse` carries no
usage and a stream is not buffered). Prompt, system instruction and reply text are handed
to the library only when `TRELIX_OTEL_CAPTURE_CONTENT=true`, and the library emits them only
under its own `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. Images on a
`ChatMessage` are never serialised. Instrumentation never changes a result: every span step
is try/except -> debug, and a backend exception is recorded and re-raised unchanged.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final

from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient
from trelix.retrieval import otel_tracing

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.otel")

_unavailable_warned = False

# `gen_ai.provider.name` per LLMConfig.provider. Well-known values from the GenAI registry
# where one exists; `litellm` is a custom value because the router hides the real provider
# (owner decision D7). `vertex` is resolved by provider_name() from the credentials.
PROVIDER_NAMES: Final[Mapping[str, str]] = {
    "openai": "openai",
    "azure": "azure.ai.openai",
    "anthropic": "anthropic",
    "bedrock": "aws.bedrock",
    "litellm": "litellm",
}


def provider_name(config: LLMConfig) -> str:
    """`gen_ai.provider.name` for *config*.

    Vertex follows VertexBackend._build_client: an API key selects the Gemini API
    (`gcp.gemini`), otherwise the project selects Vertex AI (`gcp.vertex_ai`).
    """
    if config.provider == "vertex":
        return "gcp.gemini" if config.google_api_key else "gcp.vertex_ai"
    return PROVIDER_NAMES[config.provider]


def request_model(backend: Any, config: LLMConfig) -> str:
    """The model id the backend sends: its `_model` when that is a non-empty string, else
    `config.model`.

    `_model` is the Azure deployment, `litellm_model`, or Bedrock's ACTIVE id, which
    `_try_with_fallback` may have swapped to the fallback on an earlier call of the same
    instance (docs/OBSERVABILITY.md).
    """
    model = getattr(backend, "_model", None)
    return model if isinstance(model, str) and model else config.model


def _genai_types() -> Any:
    """util-genai's message and error types (`InputMessage`, `OutputMessage`, `TextPart`,
    `Error`), typed Any on purpose: the installed library may be older or newer than the
    1.2b0 floor these calls target, and a static type must not depend on which one it is.
    Missing names fail inside the span step that uses them, which logs at debug."""
    return importlib.import_module("opentelemetry.util.genai.types")


def traced(backend: TrelixChatClient, config: LLMConfig) -> TrelixChatClient:
    """Wrap *backend* in chat spans; on any failure warn once (D8) and return it bare."""
    try:
        # An ImportError here is the usual cause, and its text is what the operator reads.
        _genai_types()
        handler = otel_tracing.handler_from_env()
        if handler is None:
            raise RuntimeError("the TelemetryHandler could not be built (see the debug log)")
    except Exception as exc:
        _warn_unavailable_once(exc)
        return backend
    return TracedChatClient(backend, config, handler)


def _warn_unavailable_once(exc: Exception) -> None:
    """WARNING, not debug (like the metrics path): the operator asked for spans and would
    otherwise read an empty LLM dashboard as "no calls were made". Once per process."""
    global _unavailable_warned
    if _unavailable_warned:
        return
    _unavailable_warned = True
    logger.warning(
        "TRELIX_OTEL_ENABLED is set but OpenTelemetry GenAI spans are unavailable (%s) — "
        "LLM calls will NOT be traced. Install: pip install 'trelix[otel]'",
        exc,
    )


def _error_type(exc: BaseException) -> str:
    """`error.type` as util-genai derives it: the qualified class name, bare for builtins."""
    cls = type(exc)
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


class TracedChatClient(TrelixChatClient):
    """One GenAI `chat` span per call of the wrapped backend.

    The three methods carry the ABC's exact signatures: no `**kwargs`, no `seed`, so
    `seed_kwargs()` keeps returning `{}` for them (a widened signature would make the
    planner pass `seed=` and collapse every plan to `default_plan()`). Nothing else of the
    backend is forwarded (no `__getattr__`): a method the wrapper does not define cannot
    bypass tracing. The one exception is the `_client` property below.
    """

    def __init__(self, backend: TrelixChatClient, config: LLMConfig, handler: Any) -> None:
        self._backend = backend
        self._config = config
        self._handler = handler

    @property
    def _client(self) -> Any:
        """The backend's provider client, by identity, for the three consumers that read it.

        synthesizer.py, retrieval/planner/agent.py and retrieval/graph_rag.py do
        `self._llm_client._client if hasattr(self._llm_client, "_client") else self._llm_client`
        and later compare that object with `getattr(self._llm_client, "_client", None)` to
        choose the TrelixChatClient path over a legacy raw-OpenAI path. A wrapper without
        `_client` sends all three down the legacy path. AttributeError propagates when the
        backend has no `_client` (LiteLLMBackend), so `hasattr` stays False exactly as today.
        """
        return getattr(self._backend, "_client")  # noqa: B009 -- must raise, not default

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> ChatResponse:
        inv = self._start(messages, system, max_tokens)
        try:
            response = self._backend.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                thinking=thinking,
            )
        except BaseException as exc:
            self._fail(inv, exc)
            raise
        self._record(inv, response)
        return response

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> Iterator[str]:
        # A generator: the span starts at the first next(), not at the call, and a generator
        # that is never iterated produces no span. suspend() detaches the span from the
        # current context while the consumer drains the stream, so its own work is not
        # parented under this span; stop()/fail() end it on exhaustion, early close or error.
        inv = self._start(messages, system, max_tokens)
        self._suspend(inv)
        try:
            yield from self._backend.stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                thinking=thinking,
            )
        except GeneratorExit:
            self._stop(inv)
            raise
        except BaseException as exc:
            self._fail(inv, exc)
            raise
        else:
            self._stop(inv)

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolCallResponse:
        # Request attributes only: ToolCallResponse carries no usage (owner decision D5).
        inv = self._start(messages, None, max_tokens)
        try:
            response = self._backend.tool_call(
                messages, tools, force_tool=force_tool, max_tokens=max_tokens
            )
        except BaseException as exc:
            self._fail(inv, exc)
            raise
        self._stop(inv)
        return response

    # ------------------------------------------------------------------
    # Span steps -- each never raises; a backend exception always propagates
    # ------------------------------------------------------------------

    def _start(
        self, messages: list[ChatMessage], system: str | None, max_tokens: int | None
    ) -> Any | None:
        """Open the span with the request attributes. None (and no span) if the library
        refuses; a failure after the span opened keeps it, so it is still ended later."""
        try:
            inv = self._handler.inference(
                provider_name(self._config),
                request_model=request_model(self._backend, self._config),
            )
        except Exception as exc:
            logger.debug("Failed to start the chat span: %s", exc)
            return None
        try:
            # The effective cap, as every backend applies it: the argument or the config.
            inv.max_tokens = max_tokens or self._config.max_tokens
            if otel_tracing.capture_content_enabled():
                self._capture_request(inv, messages, system)
        except Exception as exc:
            logger.debug("Failed to set the chat span's request attributes: %s", exc)
        return inv

    def _capture_request(self, inv: Any, messages: list[ChatMessage], system: str | None) -> None:
        """Hand the prompt to the library (its own content mode decides whether it is
        emitted). The system instruction is what the backends send: `system`, else the
        first system-role message. Images are not serialised."""
        types = _genai_types()
        inv.input_messages = [
            types.InputMessage(role=m.role, parts=[types.TextPart(content=m.content)])
            for m in messages
            if m.role != "system"
        ]
        instruction = system or next((m.content for m in messages if m.role == "system"), None)
        if instruction is not None:
            inv.system_instruction = [types.TextPart(content=instruction)]
        otel_tracing._warn_if_capture_is_inert(self._handler)

    def _record(self, inv: Any, response: ChatResponse) -> None:
        """Fill the response, usage and finish attributes from a complete() reply, then end
        the span. The unconfigured placeholder (`model == "none"`) is recorded like any
        reply (owner decision D4)."""
        if inv is None:
            return
        try:
            inv.response_model_name = response.model
            # The GenAI Anthropic page: input tokens INCLUDE the cache read and write counts
            # (Anthropic's own usage.input_tokens is the non-cached part). The library drops
            # a zero cache value itself, so the two cache fields are assigned as they are.
            inv.input_tokens = (
                response.input_tokens + response.cache_read_tokens + response.cache_write_tokens
            )
            inv.output_tokens = response.output_tokens
            inv.cache_read_input_tokens = response.cache_read_tokens
            inv.cache_write_input_tokens = response.cache_write_tokens
            # The provider's own word (R-C6-02); trelix's normalised value rides alongside.
            inv.finish_reasons = (
                [response.raw_finish_reason] if response.raw_finish_reason else None
            )
            inv.attributes = {
                **(inv.attributes or {}),
                "trelix.finish_reason": response.finish_reason,
            }
            if otel_tracing.capture_content_enabled():
                self._capture_reply(inv, response.content)
        except Exception as exc:
            logger.debug("Failed to record the reply on the chat span: %s", exc)
        self._stop(inv)

    @staticmethod
    def _capture_reply(inv: Any, content: str) -> None:
        types = _genai_types()
        inv.output_messages = [
            types.OutputMessage(role="assistant", parts=[types.TextPart(content=content)])
        ]

    @staticmethod
    def _stop(inv: Any) -> None:
        if inv is None:
            return
        try:
            inv.stop()
        except Exception as exc:
            logger.debug("Failed to end the chat span: %s", exc)

    @staticmethod
    def _suspend(inv: Any) -> None:
        if inv is None:
            return
        try:
            inv.suspend()
        except Exception as exc:
            logger.debug("Failed to suspend the chat span: %s", exc)

    @staticmethod
    def _fail(inv: Any, exc: BaseException) -> None:
        """End the span with ERROR status. The class name only, as `error.type` and as the
        status description (owner decision D10): `str(exc)` can echo a provider error body."""
        if inv is None:
            return
        try:
            name = _error_type(exc)
            inv.fail(_genai_types().Error(message=name, type=name))
        except Exception as inner_exc:
            logger.debug("Failed to record the error on the chat span: %s", inner_exc)
