"""Compression abstraction — public API."""

from trelix.compression.base import (
    CompressionResult,
    CompressionUnit,
    Compressor,
    make_compressor,
)
from trelix.compression.extractive import ExtractiveCompressor

__all__ = [
    "Compressor",
    "CompressionResult",
    "CompressionUnit",
    "ExtractiveCompressor",
    "make_compressor",
]
