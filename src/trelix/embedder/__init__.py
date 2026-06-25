"""Embedder abstraction — public API."""

from trelix.embedder.base import (
    AzureOpenAIEmbedder,
    BaseEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    make_embedder,
)

__all__ = [
    "BaseEmbedder",
    "AzureOpenAIEmbedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "make_embedder",
]
