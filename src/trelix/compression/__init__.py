"""Compression abstraction — public API."""

from trelix.compression.abstractive import AbstractiveCompressor
from trelix.compression.base import (
    CompressionResult,
    CompressionUnit,
    Compressor,
    make_compressor,
)
from trelix.compression.extractive import ExtractiveCompressor

__all__ = [
    "AbstractiveCompressor",
    "Compressor",
    "CompressionResult",
    "CompressionUnit",
    "ExtractiveCompressor",
    "make_compressor",
]
