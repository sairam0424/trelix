"""
Compression abstraction — shrink an oversized symbol body to fit the packing
budget WITHOUT dropping the result (result-lossless) and WITHOUT lying about
which source lines the kept text came from (citation-faithful).

House style mirrors ``embedder/``: an ABC, concrete impls, and a
``make_compressor(config, db, embedder)`` factory at the bottom.

Providers:
  extractive → ExtractiveCompressor  (no inference; scores already-stored
               sub-chunk vectors by cosine, or falls back to a zero-inference
               lexical splitter — works for any language / any index)

Additive + default-OFF by contract: the retriever only ever constructs a
compressor when compression is explicitly enabled, so today's assembled
context is byte-identical. A compressor NEVER raises into the retrieval path
(graceful-degradation, same contract as reranker) — on any internal failure it
degrades to signature-only or passthrough, never worse-without-a-trace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompressionResult:
    """
    The outcome of compressing a single symbol body.

    ``text``                 — the rendered (possibly shrunk) body. When spans
                               were elided it contains explicit
                               ``# ... N lines elided ...`` markers between the
                               kept line-range blocks so no header ever claims
                               lines the text no longer contains.
    ``token_count``          — tiktoken count of ``text``, ALWAYS recomputed
                               (never inherited from the original).
    ``original_token_count`` — tiktoken count of the pre-compression body.
    ``kept_spans``           — absolute, inclusive ``(line_start, line_end)``
                               ranges retained; every span is a subrange of the
                               unit's original ``[line_start, line_end]``.
    ``provider``             — which compressor produced this (e.g. "extractive").
    """

    text: str
    token_count: int
    original_token_count: int
    kept_spans: list[tuple[int, int]] = field(default_factory=list)
    provider: str = "extractive"


@dataclass
class CompressionUnit:
    """
    The minimal, retriever-agnostic view of a symbol a compressor needs.

    Built by the caller from a ``SearchResult`` (reusing ``symbol_id`` so the
    result stays de-dup-stable); the compressor never mutates it.
    """

    symbol_id: int
    body: str
    signature: str
    docstring: str | None
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive
    qualified_name: str


class Compressor(ABC):
    """Base class every compression provider implements."""

    @abstractmethod
    def compress(
        self,
        query: str,
        unit: CompressionUnit,
        *,
        target_ratio: float,
        query_embedding: list[float] | None = None,
    ) -> CompressionResult:
        """
        Shrink ``unit.body`` toward ``target_ratio`` of its original token count.

        ``target_ratio >= 1.0`` is a passthrough (identical text). The signature
        line and docstring are ALWAYS kept; the result is never empty and the
        SearchResult it derives from is never dropped (result-lossless).

        ``query_embedding`` is the embedding the retriever already computed for
        the query — passed in so the sub-chunk cosine path costs zero inference.
        When absent (or unavailable), the zero-inference lexical path is used.
        """
        ...

    def compress_batch(
        self,
        query: str,
        units: list[CompressionUnit],
        *,
        target_ratio: float,
        query_embedding: list[float] | None = None,
    ) -> list[CompressionResult]:
        """Compress a batch. Default: loop over :meth:`compress`."""
        return [
            self.compress(query, unit, target_ratio=target_ratio, query_embedding=query_embedding)
            for unit in units
        ]


def make_compressor(config: object, db: object, embedder: object | None = None) -> Compressor:
    """
    Factory — instantiate the configured compressor.

    Args:
        config:   IndexConfig-like object; ``config.compression.provider`` selects
                  the backend. Defaults to "extractive" when unset (forward-safe:
                  no compression config surface exists yet).
        db:       Database handle (provides ``get_sub_chunks_for_symbol`` and the
                  shared SQLite connection used to read stored sub-chunk vectors).
        embedder: The active embedder (stored for reserved future use; the
                  extractive provider makes no embedding calls).

    Returns:
        A :class:`Compressor` instance.

    Raises:
        NotImplementedError: for any provider other than "extractive"
            (abstractive / LLM-based providers are reserved for v3.4).
    """
    from trelix.compression.extractive import ExtractiveCompressor

    provider = "extractive"
    comp_cfg = getattr(config, "compression", None)
    if comp_cfg is not None:
        provider = getattr(comp_cfg, "provider", "extractive")
    else:
        # No dedicated compression section — read the flag that actually exists
        # today, RetrievalConfig.compression_provider.
        retrieval_cfg = getattr(config, "retrieval", None)
        if retrieval_cfg is not None:
            provider = getattr(retrieval_cfg, "compression_provider", "extractive")

    if provider == "extractive":
        return ExtractiveCompressor(db=db, embedder=embedder)

    raise NotImplementedError(
        f"Compression provider {provider!r} is not implemented "
        "(only 'extractive' is available; abstractive providers are reserved for v3.4)."
    )
