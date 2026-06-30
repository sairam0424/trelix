"""
HyDE and multi-query expansion — zero-shot retrieval improvements.

HyDE (Hypothetical Document Embeddings, Gao et al. 2022, arXiv:2212.10496):
  Instead of embedding the user's NL query, ask an LLM to write a hypothetical
  code snippet that would answer the question, then embed that snippet.
  The encoder's bottleneck filters hallucinated details while preserving semantics.

Multi-query expansion:
  Ask an LLM to rephrase the query N ways. Run each variant as a separate
  sub-query, then RRF-merge. Increases recall on varied vocabulary.

Both are opt-in via RetrievalConfig flags. Both return empty/original on any
failure — the pipeline always has a fallback.
"""

from __future__ import annotations

import logging

from trelix.core.config import LLMConfig
from trelix.llm.client import TrelixChatClient
from trelix.llm.factory import build_chat_client

logger = logging.getLogger("trelix.retrieval.query_expansion")

_HYDE_SYSTEM = (
    "You are a senior software engineer. Given a question about a codebase, "
    "write a SHORT hypothetical code snippet (3-8 lines) that would directly answer it. "
    "Output ONLY the code snippet, no explanation, no markdown fences."
)

_MULTI_QUERY_SYSTEM = (
    "You are a search query expert. Given a code search query, write {n} alternative "
    "phrasings that cover different vocabulary but have the same intent. "
    "Output one query per line, no numbering, no explanation."
)


class HyDEExpander:
    """Generate a hypothetical code snippet to use as the vector query (HyDE)."""

    def __init__(self, llm_config: LLMConfig | None) -> None:
        self._llm_config = llm_config
        self._client: TrelixChatClient | None = None

    def _get_client(self) -> TrelixChatClient | None:
        if self._llm_config is None:
            return None
        if self._client is None:
            try:
                self._client = build_chat_client(self._llm_config)
            except Exception as exc:
                logger.debug("HyDEExpander: could not build LLM client: %s", exc)
                return None
        return self._client

    def expand(self, query: str) -> str:
        """Return a synthetic code snippet, or '' on any failure."""
        client = self._get_client()
        if client is None:
            return ""
        try:
            from trelix.llm.client import ChatMessage
            resp = client.complete(
                messages=[ChatMessage(role="user", content=query)],
                max_tokens=150,
                temperature=0.1,
                system=_HYDE_SYSTEM,
            )
            return str(resp.content).strip()
        except Exception as exc:
            logger.debug("HyDE expansion failed for query %r: %s", query, exc)
            return ""


class MultiQueryExpander:
    """Generate N rephrased variants of the query to increase retrieval recall."""

    def __init__(self, llm_config: LLMConfig | None, n: int = 2) -> None:
        self._llm_config = llm_config
        self._n = n
        self._client: TrelixChatClient | None = None

    def _get_client(self) -> TrelixChatClient | None:
        if self._llm_config is None:
            return None
        if self._client is None:
            try:
                self._client = build_chat_client(self._llm_config)
            except Exception as exc:
                logger.debug("MultiQueryExpander: could not build LLM client: %s", exc)
                return None
        return self._client

    def expand(self, query: str) -> list[str]:
        """Return [original] + up to N variants. Deduplicates. Never raises."""
        client = self._get_client()
        if client is None:
            return [query]
        try:
            from trelix.llm.client import ChatMessage
            system = _MULTI_QUERY_SYSTEM.format(n=self._n)
            resp = client.complete(
                messages=[ChatMessage(role="user", content=query)],
                max_tokens=200,
                temperature=0.3,
                system=system,
            )
            variants = [line.strip() for line in resp.content.strip().splitlines() if line.strip()]
            # Deduplicate while preserving order; original always first
            seen: set[str] = {query}
            result = [query]
            for v in variants[: self._n]:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
            return result
        except Exception as exc:
            logger.debug("Multi-query expansion failed for %r: %s", query, exc)
            return [query]
