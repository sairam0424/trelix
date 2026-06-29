"""
BGE-Code-v1 embedder (BAAI, May 2025).

Uses FlagEmbedding library (pip install FlagEmbedding>=1.3.0).
BGE-Code-v1 self-reports 81.77 CoIR average, the highest-known score
as of mid-2025. Uses asymmetric query/document encoding:
  - Documents: encoded directly (code text)
  - Queries: encoded with instruction prefix for retrieval

Install:
    pip install 'trelix[bge-code]'

Usage:
    TRELIX_EMBEDDER_PROVIDER=bge-code trelix index ./my-repo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trelix.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

_FlagModel: Any | None
try:
    from FlagEmbedding import FlagModel as _FM_cls

    _FlagModel = _FM_cls
except ImportError:  # pragma: no cover
    _FlagModel = None

FlagModel = _FlagModel

_QUERY_INSTRUCTION = "Represent this query for searching relevant code: "


class BGECodeEmbedder(BaseEmbedder):
    """
    Embedder backed by BAAI/bge-code-v1 via FlagEmbedding.

    Asymmetric: queries use an instruction prefix; documents (code) are
    encoded directly. This matches BGE-Code-v1's training protocol.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        if FlagModel is None:
            raise ImportError(
                "FlagEmbedding is required for bge-code embedder. "
                "Install it with: pip install 'trelix[bge-code]'"
            )

        self._model = FlagModel(
            config.bge_code_model,
            query_instruction_for_retrieval=_QUERY_INSTRUCTION,
            use_fp16=True,
        )
        self._dimensions = config.bge_code_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=32)
        return [list(v) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        # encode_queries() handles the instruction prefix internally via
        # query_instruction_for_retrieval — no manual prepend needed.
        vecs = self._model.encode_queries([text])
        return list(vecs[0])

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension() or self._dimensions
