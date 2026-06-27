"""Embedder abstraction — public API."""

from trelix.embedder.base import (
    AzureOpenAIEmbedder,
    BaseEmbedder,
    BedrockCohereEmbedder,
    BedrockTitanEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    make_embedder,
)

__all__ = [
    "BaseEmbedder",
    "AzureOpenAIEmbedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "BedrockTitanEmbedder",
    "BedrockCohereEmbedder",
    "make_embedder",
]
