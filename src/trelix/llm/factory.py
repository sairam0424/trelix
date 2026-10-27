"""LLM client factory — instantiates the right backend from LLMConfig."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trelix.retrieval import otel_tracing

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig
    from trelix.llm.client import TrelixChatClient

logger = logging.getLogger("trelix.llm.factory")


def build_chat_client(config: LLMConfig) -> TrelixChatClient:
    """Return a TrelixChatClient for the configured provider.

    With TRELIX_OTEL_ENABLED resolved true from the environment the backend comes wrapped
    in GenAI chat spans (trelix.llm.otel); otherwise the bare backend, unchanged in object
    and type, at the cost of one memoised RetrievalConfig() read per process.
    """
    backend = _build_backend(config)
    if not otel_tracing.is_enabled_from_env():
        return backend
    from trelix.llm.otel import traced  # lazy: never imported while the flag is off

    return traced(backend, config)


def _build_backend(config: LLMConfig) -> TrelixChatClient:
    """The provider's backend for *config* -- the one place the provider SDKs are chosen."""
    if config.base_url is not None and config.provider != "openai":
        # Not an error: three retrieval shims build an LLMConfig whose provider is the
        # EMBEDDER's (`azure` for an Azure embedder) while the environment still carries
        # TRELIX_LLM_BASE_URL, and refusing here would break `search`/`ask` for a valid
        # review configuration. Only the openai backend reads the URL.
        logger.warning(
            "TRELIX_LLM_BASE_URL is set but TRELIX_LLM_PROVIDER=%s does not use it",
            config.provider,
        )
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
