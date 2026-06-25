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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig, RetrievalConfig
    from trelix.core.models import RetrievedContext

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
    ) -> None:
        self._config = config
        self._client = self._build_client(config)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self, config: EmbedderConfig) -> object | None:
        """
        Instantiate the appropriate OpenAI client.
        Returns None for provider=local or when credentials are missing.
        """
        if config.provider == "azure":
            if not config.azure_api_key or not config.azure_endpoint:
                logger.debug("Synthesizer: Azure credentials not set.")
                return None
            try:
                from openai import AzureOpenAI

                return AzureOpenAI(
                    api_key=config.azure_api_key,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.azure_api_version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Synthesizer: could not create AzureOpenAI client: %s", exc)
                return None

        if config.provider == "openai":
            if not config.openai_api_key:
                logger.debug("Synthesizer: OPENAI_API_KEY not set.")
                return None
            try:
                from openai import OpenAI

                return OpenAI(api_key=config.openai_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Synthesizer: could not create OpenAI client: %s", exc)
                return None

        # provider == "local" — no chat API
        return None

    def _model_name(self) -> str:
        if self._config.provider == "azure":
            return self._config.azure_chat_deployment
        return self._config.openai_chat_model

    def _system_prompt(self, intent: str) -> str:
        return _INTENT_PROMPTS.get(intent, _DEFAULT_SYSTEM_PROMPT)

    def _stream_response(self, context: RetrievedContext, config: EmbedderConfig) -> str:
        """
        Call the chat API with streaming, print tokens to stdout, and return
        the full assembled text.
        """
        user_message = _USER_TEMPLATE.format(
            context_text=context.context_text,
            query=context.query,
        )

        max_tokens: int = getattr(config, "synthesis_max_tokens", 2048)

        # max_completion_tokens is required for newer OpenAI/Azure models (o-series, gpt-4o);
        # older deployments use max_tokens. Try max_completion_tokens first, fall back on error.
        stream = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model_name(),
            messages=[
                {"role": "system", "content": self._system_prompt(context.intent)},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.2,
            stream=True,
        )

        collected: list[str] = []
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
