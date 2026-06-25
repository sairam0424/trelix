"""
GraphRAG Map-Reduce Synthesizer.

For large result sets (>20 results OR >8k tokens), splits the retrieved context
into groups, runs a MAP phase (partial answers per group), then a REDUCE phase
to combine them into a single coherent answer.

This handles the 1M-token corpus case where a single LLM call would exceed
context limits or produce degraded results.

Usage::

    from trelix.retrieval.graph_rag import GraphRAGSynthesizer
    from trelix.core.config import EmbedderConfig, RetrievalConfig

    synth = GraphRAGSynthesizer(EmbedderConfig(), RetrievalConfig())
    if synth.should_use(context):
        answer = synth.synthesize(query, context, intent)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig, RetrievalConfig
    from trelix.core.models import RetrievedContext, SearchResult

logger = logging.getLogger("trelix.retrieval.graph_rag")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAP_RESULTS_PER_GROUP = 10
_MAP_MAX_TOKENS = 400
_REDUCE_MAX_TOKENS = 1500

_MAP_PROMPT_TEMPLATE = """\
Partially answer the following question using ONLY the code context provided.
Be concise. Focus on what this specific code reveals about the question.

Question: {query}

Code context:
{group_context}

Partial answer:"""

_REDUCE_PROMPT_TEMPLATE = """\
You are synthesizing partial answers about a codebase query into a single coherent response.
Each partial answer was derived from a different subset of the relevant code.

Question: {query}

Partial answers:
{partial_answers}

Provide a complete, synthesized answer that integrates all relevant information above.
Cite specific file and function names where possible. Be precise and technical."""


# ---------------------------------------------------------------------------
# GraphRAGSynthesizer
# ---------------------------------------------------------------------------

class GraphRAGSynthesizer:
    """
    Map-reduce synthesis for large retrieved contexts.

    MAP phase: each group of ~10 results is independently summarised by the LLM.
    REDUCE phase: partial answers are combined into a final answer.

    Groups are processed sequentially to avoid rate-limit issues.
    """

    def __init__(self, embedder_config: EmbedderConfig, retrieval_config: RetrievalConfig) -> None:
        self._embedder_config = embedder_config
        self._retrieval_config = retrieval_config
        self._client = self._build_client(embedder_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_use(self, context: RetrievedContext) -> bool:
        """
        Return True when the context is large enough to warrant map-reduce.

        Triggers when:
        - total_tokens > graph_rag_threshold_tokens (default: 8000), OR
        - len(results) > graph_rag_threshold_results (default: 20)

        Also requires graph_rag_enabled=True (default) and a live LLM client.
        """
        if not self._retrieval_config.graph_rag_enabled:
            return False
        if self._client is None:
            return False
        token_threshold = self._retrieval_config.graph_rag_threshold_tokens
        results_threshold = self._retrieval_config.graph_rag_threshold_results
        return (
            context.total_tokens > token_threshold
            or len(context.results) > results_threshold
        )

    def synthesize(self, query: str, context: RetrievedContext, intent: str) -> str:
        """
        Run map-reduce synthesis over the context.

        MAP: split results into groups of ~10, get a partial answer per group.
        REDUCE: combine partial answers into a final coherent response.

        Returns the final synthesized string.
        Falls back to an empty string on client/API errors.
        """
        if self._client is None:
            logger.warning("GraphRAGSynthesizer.synthesize called with no LLM client.")
            return ""

        groups = self._split_into_groups(context.results)
        logger.info(
            "GraphRAG map-reduce: %d results -> %d groups (total_tokens=%d)",
            len(context.results),
            len(groups),
            context.total_tokens,
        )

        # --- MAP phase ---
        partial_answers: list[str] = []
        for idx, group in enumerate(groups):
            group_context = self._format_group(group)
            prompt = _MAP_PROMPT_TEMPLATE.format(
                query=query,
                group_context=group_context,
            )
            try:
                partial = self._call_llm(prompt, max_tokens=_MAP_MAX_TOKENS)
                if partial.strip():
                    partial_answers.append(partial.strip())
                    logger.debug("MAP group %d/%d: %d chars", idx + 1, len(groups), len(partial))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GraphRAG MAP group %d failed: %s", idx + 1, exc)

        if not partial_answers:
            logger.warning("GraphRAG MAP produced no partial answers.")
            return ""

        # --- REDUCE phase ---
        numbered = "\n\n".join(
            f"[Partial {i + 1}]\n{ans}" for i, ans in enumerate(partial_answers)
        )
        reduce_prompt = _REDUCE_PROMPT_TEMPLATE.format(
            query=query,
            partial_answers=numbered,
        )
        try:
            final = self._call_llm(reduce_prompt, max_tokens=_REDUCE_MAX_TOKENS)
            logger.info("GraphRAG REDUCE complete: %d chars", len(final))
            return final.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphRAG REDUCE failed: %s", exc)
            # Fall back to concatenated partial answers
            return "\n\n".join(partial_answers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_groups(self, results: list[SearchResult]) -> list[list[SearchResult]]:
        """Split results into groups of _MAP_RESULTS_PER_GROUP."""
        groups: list[list[SearchResult]] = []
        for i in range(0, len(results), _MAP_RESULTS_PER_GROUP):
            groups.append(results[i : i + _MAP_RESULTS_PER_GROUP])
        return groups

    def _format_group(self, results: list[SearchResult]) -> str:
        """Format a group of SearchResults into a context block for the MAP prompt."""
        parts: list[str] = []
        for r in results:
            header = f"# {r.file.rel_path} — {r.symbol.name} ({r.symbol.kind})"
            body = r.chunk.chunk_text.strip()
            parts.append(f"{header}\n{body}")
        return "\n\n".join(parts)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """
        Make a single non-streaming chat completion call.
        Returns the response text or raises on error.
        """
        model = self._model_name()
        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert software engineer answering questions about a "
                        "codebase. Base your answer strictly on the provided code context. "
                        "Be concise and precise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.1,
            stream=False,
        )
        content = response.choices[0].message.content or ""
        return content

    def _model_name(self) -> str:
        if self._embedder_config.provider == "azure":
            return self._embedder_config.azure_chat_deployment
        return self._embedder_config.openai_chat_model

    def _build_client(self, config: EmbedderConfig) -> object | None:
        """
        Instantiate the appropriate OpenAI client.
        Mirrors the pattern used in Synthesizer._build_client().
        Returns None for provider=local or missing credentials.
        """
        if config.provider == "azure":
            if not config.azure_api_key or not config.azure_endpoint:
                logger.debug("GraphRAGSynthesizer: Azure credentials not set.")
                return None
            try:
                from openai import AzureOpenAI
                return AzureOpenAI(
                    api_key=config.azure_api_key,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.azure_api_version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("GraphRAGSynthesizer: could not create AzureOpenAI client: %s", exc)
                return None

        if config.provider == "openai":
            if not config.openai_api_key:
                logger.debug("GraphRAGSynthesizer: OPENAI_API_KEY not set.")
                return None
            try:
                from openai import OpenAI
                return OpenAI(api_key=config.openai_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("GraphRAGSynthesizer: could not create OpenAI client: %s", exc)
                return None

        # provider == "local" — no chat API
        return None
