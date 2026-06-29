"""
LLM Synthesizer: turns a RetrievedContext into a natural-language answer.

Usage::

    from trelix.retrieval.synthesizer import Synthesizer
    from trelix.core.config import EmbedderConfig

    synth = Synthesizer(EmbedderConfig())
    synth.synthesize(context, config)   # streams answer to stdout

Design principles:
- Streams tokens to stdout so the user sees output immediately.
- Adapts to provider: openai, azure, or local (no-op with a clear message).
- Falls back gracefully when no API key is present.
- Uses per-intent system prompts to guide the response shape.
- Never raises — all errors are caught and printed as messages.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig, LLMConfig, RetrievalConfig
    from trelix.core.models import RetrievedContext

# Module-level import so tests can patch "trelix.retrieval.synthesizer.build_chat_client"
from trelix.llm.factory import build_chat_client  # noqa: E402

logger = logging.getLogger("trelix.retrieval.synthesizer")

# ---------------------------------------------------------------------------
# Per-intent system prompts
# ---------------------------------------------------------------------------

_INTENT_PROMPTS: dict[str, str] = {
    "symbol_lookup": (
        "You are a precise code documentation assistant. "
        "Explain exactly what the identified symbol does: its purpose, parameters, "
        "return values, and any side effects. Be concise and technical."
    ),
    "file_overview": (
        "You are a code tour guide. Given the full contents of a source file, "
        "provide a structured overview: the file's purpose, its main classes and "
        "functions, and how they relate to each other. Use a table-of-contents style."
    ),
    "feature_flow": (
        "You are a senior engineer explaining a feature's end-to-end implementation. "
        "Trace the flow from entry point to final output, naming the key functions and "
        "data transformations at each step. Show the call chain clearly."
    ),
    "project_overview": (
        "You are a technical writer producing a codebase orientation doc. "
        "Explain the project's architecture, its main modules, how data flows "
        "between them, and what problem the project solves."
    ),
    "comparison": (
        "You are a code reviewer comparing two or more implementations. "
        "Highlight key similarities, differences, trade-offs, and when to prefer each."
    ),
    "config_lookup": (
        "You are a configuration expert. Explain each configuration key found, "
        "its purpose, accepted values, and defaults."
    ),
    "dependency_map": (
        "You are a dependency analyst. List what each component depends on, "
        "explain why, and note any circular or problematic dependencies."
    ),
    "blast_radius": (
        "You are a change-impact analyst. Explain what would break if the target "
        "symbol or file were changed, listing affected callers, importers, and "
        "downstream services."
    ),
}

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software engineer answering questions about a codebase. "
    "Base your answer strictly on the provided code context. "
    "Be precise, cite the relevant file and function names, and avoid speculation."
)

_USER_TEMPLATE = """\
## Code Context
{context_text}

## Question
{query}

Answer based solely on the code shown above."""


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class Synthesizer:
    """
    Wraps an LLM chat client to synthesize a natural-language answer from
    a RetrievedContext.

    Streams output to stdout so the user sees tokens arrive in real time.
    Falls back silently when no API key / provider is available.

    For large contexts (>20 results or >8k tokens), delegates to
    GraphRAGSynthesizer which runs map-reduce synthesis.
    """

    def __init__(
        self,
        config: EmbedderConfig,
        retrieval_config: RetrievalConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self._config = config
        from trelix.llm.client import ChatMessage as _ChatMessage  # noqa: F401 – ensure import

        if llm_config is not None:
            # Use the explicitly supplied LLMConfig (e.g. IndexConfig.llm).
            # This is the correct path for non-OpenAI providers such as
            # Anthropic, Bedrock, and Vertex.
            self._llm_client = build_chat_client(llm_config)
        else:
            # Backward-compat shim: rebuild LLMConfig from EmbedderConfig.
            # Only valid when the embedder provider is openai or azure; all
            # other providers silently fell back to provider="openai" before
            # this fix, which caused failures without OPENAI_API_KEY.
            from trelix.core.config import LLMConfig

            shim_cfg = LLMConfig(
                provider=config.provider if config.provider in ("openai", "azure") else "openai",
                _env_file=None,  # type: ignore[call-arg]
            )
            shim_cfg = shim_cfg.model_copy(
                update={
                    "openai_api_key": config.openai_api_key,
                    "azure_api_key": config.azure_api_key,
                    "azure_endpoint": config.azure_endpoint,
                    "azure_api_version": config.azure_api_version,
                    "azure_chat_deployment": config.azure_chat_deployment,
                    "model": config.openai_chat_model,
                }
            )
            self._llm_client = build_chat_client(shim_cfg)

        # Keep _client for the None check used by synthesize()
        self._client = (
            self._llm_client._client if hasattr(self._llm_client, "_client") else self._llm_client
        )
        # Lazy-import to avoid circular deps; default to RetrievalConfig() if not supplied.
        if retrieval_config is None:
            from trelix.core.config import RetrievalConfig as _RC

            retrieval_config = _RC()
        self._retrieval_config = retrieval_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, context: RetrievedContext, config: EmbedderConfig | None = None) -> str:
        """
        Synthesize an answer from the retrieved context, streaming to stdout.

        Args:
            context: Output of Retriever.retrieve() — contains context_text,
                     query, and intent.
            config:  Optional override EmbedderConfig. Uses the one from __init__
                     when not provided.

        Returns:
            The full synthesized text (same content that was streamed).
            Returns an empty string when no client is available.
        """
        cfg = config or self._config

        if self._client is None:
            msg = (
                "[trelix] No LLM API key configured — skipping synthesis. "
                "Set OPENAI_API_KEY (or AZURE_API_KEY + AZURE_ENDPOINT) to enable answers."
            )
            print(msg, flush=True)
            return ""

        if not context.results:
            msg = "[trelix] No relevant code found — cannot synthesize an answer."
            print(msg, flush=True)
            return msg

        # Delegate to GraphRAG map-reduce for large contexts.
        try:
            from trelix.retrieval.graph_rag import GraphRAGSynthesizer

            graph_rag = GraphRAGSynthesizer(cfg, self._retrieval_config)
            if graph_rag.should_use(context):
                logger.info(
                    "Delegating to GraphRAG map-reduce (results=%d, tokens=%d)",
                    len(context.results),
                    context.total_tokens,
                )
                return graph_rag.synthesize(context.query, context, context.intent)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphRAG check/dispatch failed, falling back to standard: %s", exc)

        try:
            return self._stream_response(context, cfg)
        except Exception as exc:  # noqa: BLE001
            msg = f"[trelix] Synthesis failed: {exc}"
            logger.warning(msg)
            print(f"\n{msg}", flush=True)
            return ""

    def stream(
        self,
        context: RetrievedContext,
        config: RetrievalConfig,
    ) -> Iterator[str]:
        """
        Stream synthesis tokens to the caller.

        Yields str tokens as they arrive from the LLM.
        Yields a single error message string on failure (never raises).

        Usage::
            for token in synth.stream(context, config):
                print(token, end="", flush=True)
        """
        intent = getattr(context, "intent", None) or "feature_flow"
        system_prompt = _INTENT_PROMPTS.get(intent, _DEFAULT_SYSTEM_PROMPT)

        user_message = _USER_TEMPLATE.format(
            context_text=context.context_text,
            query=context.query,
        )
        max_tokens: int = getattr(config, "synthesis_max_tokens", 2048)

        try:
            from trelix.llm.client import ChatMessage

            yield from self._llm_client.stream(
                messages=[ChatMessage(role="user", content=user_message)],
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Streaming synthesis failed: %s", exc)
            yield f"\n[trelix: synthesis unavailable — {exc}]"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _system_prompt(self, intent: str) -> str:
        return _INTENT_PROMPTS.get(intent, _DEFAULT_SYSTEM_PROMPT)

    def _stream_response(self, context: RetrievedContext, config: EmbedderConfig) -> str:
        """
        Call the chat API with streaming, print tokens to stdout, and return
        the full assembled text.

        Uses TrelixChatClient when available; falls back to raw _client for
        backward compat with tests that inject mock._client directly.
        """
        from trelix.llm.client import ChatMessage, TrelixChatClient

        user_message = _USER_TEMPLATE.format(
            context_text=context.context_text,
            query=context.query,
        )
        max_tokens: int = getattr(config, "synthesis_max_tokens", 2048)
        collected: list[str] = []

        # Detect if a raw client was injected directly (e.g. by tests) by checking
        # whether _client is the same object as the backend's internal _client.
        _backend_internal = (
            getattr(self._llm_client, "_client", None)
            if isinstance(self._llm_client, TrelixChatClient)
            else None
        )
        _use_raw = self._client is not None and self._client is not _backend_internal

        if isinstance(self._llm_client, TrelixChatClient) and not _use_raw:
            for chunk in self._llm_client.stream(
                messages=[ChatMessage(role="user", content=user_message)],
                system=self._system_prompt(context.intent),
                max_tokens=max_tokens,
                temperature=0.2,
            ):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                collected.append(chunk)
        else:
            # Legacy path: raw openai client (backward compat / test injection via _client)
            assert self._client is not None  # guaranteed by synthesize() None check
            model = (
                config.azure_chat_deployment
                if config.provider == "azure"
                else config.openai_chat_model
            )
            stream = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=[
                    {"role": "system", "content": self._system_prompt(context.intent)},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=max_tokens,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()
                    collected.append(delta.content)

        # Ensure we end on a newline
        if collected and not collected[-1].endswith("\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()

        return "".join(collected)
