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
from trelix.embedder.cache import CachingEmbedder

__all__ = [
    "BaseEmbedder",
    "AzureOpenAIEmbedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "VoyageEmbedder",
    "LocalCodeEmbedder",
    "BedrockTitanEmbedder",
    "BedrockCohereEmbedder",
    "CachingEmbedder",
    "make_embedder",
]
