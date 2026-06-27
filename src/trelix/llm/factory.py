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
