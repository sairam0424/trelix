"""Embedder abstraction — public API."""

from trelix.embedder.base import (
    AzureOpenAIEmbedder,
    BaseEmbedder,
    BedrockCohereEmbedder,
    BedrockTitanEmbedder,
    LocalCodeEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    make_embedder,
)

__all__ = [
    "BaseEmbedder",
    "AzureOpenAIEmbedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "VoyageEmbedder",
    "LocalCodeEmbedder",
    "BedrockTitanEmbedder",
    "BedrockCohereEmbedder",
    "make_embedder",
]
