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
- Never raises — all errors are caught and printed as messages. The failure is also
  recorded in ``last_error`` so a caller (``trelix ask``) can exit non-zero without
  parsing the message text; the token stream itself is unchanged.
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
from trelix.retrieval.citations import NO_RESULTS_MESSAGE, AbstainReason, is_abstention

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

# Appended to the system prompt only when the context carries citation tags
# (RetrievedContext.citation_sources is non-empty, i.e. TRELIX_RETRIEVAL_CITATIONS is on).
# The last sentence is the abstention protocol; is_abstention() recognises the reply it asks for.
_CITATION_INSTRUCTION = (
    "\n\nEvery block of the code context starts with a tag such as [C3]. After each sentence "
    "that relies on a block, write that block's tag, for example "
    "`validate_token checks the signature [C3].` Cite only tags that appear in the context. "
    "If the context does not contain what the question needs, reply with exactly one line "
    "starting with INSUFFICIENT_EVIDENCE: followed by what is missing, and nothing else."
)

_USER_TEMPLATE = """\
## Code Context
{context_text}

## Question
{query}

Answer based solely on the code shown above."""


NOT_CONFIGURED_MESSAGE = (
    "LLM not configured — set OPENAI_API_KEY (or AZURE_API_KEY + AZURE_ENDPOINT), "
    "or choose another provider with TRELIX_LLM_PROVIDER."
)
GRAPH_RAG_EMPTY_MESSAGE = (
    "GraphRAG synthesis produced no answer (every LLM call failed; check the API key "
    "and connectivity)."
)
EMPTY_ANSWER_MESSAGE = (
    "The LLM returned no answer: the endpoint may not be an OpenAI-compatible chat API, "
    "OPENAI_BASE_URL (or a proxy in front of it) may be wrong, or the model returned nothing."
)

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
            self._llm_config = llm_config
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
            self._llm_config = shim_cfg
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
        # Why the last synthesize()/stream() call produced no answer; None when it did.
        self.last_error: str | None = None
        # Why the last synthesize()/stream() call abstained instead of answering; None when
        # it answered (or failed: an abstention is not an error, so the two never both hold).
        self.last_abstain_reason: AbstainReason | None = None

    @property
    def is_configured(self) -> bool:
        """False when the backend has no credentials and only returns a placeholder."""
        return self._client is not None

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
            Returns an empty string when no client is available. An empty or
            whitespace-only answer is returned as is, with ``last_error`` set.
        """
        cfg = config or self._config
        self.last_error = None
        self.last_abstain_reason = None

        if self._client is None:
            self.last_error = NOT_CONFIGURED_MESSAGE
            msg = (
                "[trelix] No LLM API key configured — skipping synthesis. "
                "Set OPENAI_API_KEY (or AZURE_API_KEY + AZURE_ENDPOINT) to enable answers."
            )
            print(msg, flush=True)
            return ""

        if not context.results:
            self.last_abstain_reason = "no_results"
            print(NO_RESULTS_MESSAGE, flush=True)
            return NO_RESULTS_MESSAGE

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
                answer = graph_rag.synthesize(context.query, context, context.intent)
                if not answer.strip():
                    # Map-reduce swallows every LLM error and returns "", so an empty
                    # result is the only failure signal it gives.
                    self.last_error = GRAPH_RAG_EMPTY_MESSAGE
                self._record_abstention(answer)
                return answer
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphRAG check/dispatch failed, falling back to standard: %s", exc)

        try:
            answer = self._stream_response(context, cfg)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            msg = f"[trelix] Synthesis failed: {exc}"
            logger.warning(msg)
            print(f"\n{msg}", flush=True)
            return ""
        if not answer.strip():
            # The call succeeded but carried nothing readable: an endpoint that ignores
            # stream=True, an HTML error page behind a 200, or a model that said nothing.
            logger.warning("Synthesis returned an empty answer")
            self.last_error = EMPTY_ANSWER_MESSAGE
        self._record_abstention(answer)
        return answer

    def stream(
        self,
        context: RetrievedContext,
        config: RetrievalConfig,
    ) -> Iterator[str]:
        """
        Stream synthesis tokens to the caller.

        Yields str tokens as they arrive from the LLM.
        Yields a single error message string on failure (never raises); the same
        failure is recorded in ``last_error`` *before* that token is yielded.
        A stream that ends with no non-whitespace text is a failure too: it is recorded
        in ``last_error`` when the stream ends, with no banner token after it.
        An empty retrieval yields ``NO_RESULTS_MESSAGE`` alone, without calling the model,
        and records ``last_abstain_reason == "no_results"`` (a missing LLM key is reported
        first, as ``synthesize()`` does: that stream is the backend's placeholder with
        ``last_error`` set, whatever retrieval found); an answer that opens with
        ``INSUFFICIENT_EVIDENCE:`` streams as any answer does and records
        ``"insufficient_evidence"`` when the stream ends (a stream closed early records nothing).

        Usage::
            for token in synth.stream(context, config):
                print(token, end="", flush=True)
        """
        self.last_abstain_reason = None
        self.last_error = None if self.is_configured else NOT_CONFIGURED_MESSAGE
        if self.is_configured and not context.results:
            # Nothing to ground an answer in: "No relevant code found." as the whole code
            # context only invites the model to answer from its own knowledge. The key is
            # checked first, as in synthesize(): a missing one is a failure to report, not
            # something the notice should hide until a question retrieves code.
            self.last_abstain_reason = "no_results"
            yield NO_RESULTS_MESSAGE
            return

        intent = getattr(context, "intent", None) or "feature_flow"
        system_prompt = self._system_prompt(intent, cited=bool(context.citation_sources))

        user_message = _USER_TEMPLATE.format(
            context_text=context.context_text,
            query=context.query,
        )
        max_tokens: int = getattr(config, "synthesis_max_tokens", 2048)

        has_answer = False
        collected: list[str] = []
        try:
            from trelix.llm.client import ChatMessage

            for token in self._llm_client.stream(
                messages=[ChatMessage(role="user", content=user_message)],
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                thinking=self._llm_config.thinking_enabled,
            ):
                has_answer = has_answer or bool(token.strip())
                collected.append(token)
                yield token
        except Exception as exc:
            logger.warning("Streaming synthesis failed: %s", exc)
            self.last_error = str(exc)
            yield f"\n[trelix: synthesis unavailable — {exc}]"
        else:
            if not has_answer and self.last_error is None:
                # No banner token here: unlike an exception there is no error text to show,
                # and the caller reads last_error once the stream has ended.
                logger.warning("Streaming synthesis returned an empty answer")
                self.last_error = EMPTY_ANSWER_MESSAGE
            # On the whole answer, once it is complete: a marker split across tokens is
            # still found, and a reader that closed the stream early never reaches here.
            if self.last_error is None:
                self._record_abstention("".join(collected))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_abstention(self, answer: str) -> None:
        """Note an ``INSUFFICIENT_EVIDENCE:`` answer; it is an answer, so ``last_error`` stays."""
        if is_abstention(answer):
            self.last_abstain_reason = "insufficient_evidence"

    def _system_prompt(self, intent: str, *, cited: bool) -> str:
        """The per-intent system prompt, plus the citation instruction for a tagged context.

        The intent fallback is the caller's: ``stream()`` passes ``"feature_flow"`` for a
        missing intent and ``_stream_response()`` passes ``context.intent`` as is, so an
        empty intent still resolves to the same base prompt it did before tags existed.
        """
        prompt = _INTENT_PROMPTS.get(intent, _DEFAULT_SYSTEM_PROMPT)
        return f"{prompt}{_CITATION_INSTRUCTION}" if cited else prompt

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
        system_prompt = self._system_prompt(context.intent, cited=bool(context.citation_sources))
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
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=0.2,
                thinking=self._llm_config.thinking_enabled,
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
                    {"role": "system", "content": system_prompt},
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
