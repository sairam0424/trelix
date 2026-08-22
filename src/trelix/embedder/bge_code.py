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
        """Width of the vectors `embed()` returns.

        This used to call `self._model.get_sentence_embedding_dimension()`. FlagEmbedding
        has never had that method: it appears in no class in the 1.2.11, 1.3.5 or 1.4.0
        sdists, and neither `AbsEmbedder` nor `FlagModel` defines `__getattr__` to forward
        it, so the call raised `AttributeError` against every real install. `Indexer.
        __init__` reads this property to size the vec0 table and `Retriever.__init__` to
        arm `DimensionGuard`, so `bge-code` could not index or query at all. Its unit test
        passed only because a `MagicMock` answers for any attribute asked of it.

        The width is read off the loaded model instead. `truncate_dim` wins when set: it
        is FlagEmbedding's Matryoshka knob and the only thing that narrows the output.
        Otherwise the HF config's `hidden_size` is the width every FlagEmbedding pooling
        method (cls / mean / last_token) emits — 1536 for BAAI/bge-code-v1. Both the
        encoder-only and decoder-only bases assign `self.model = AutoModel.from_pretrained
        (...)`, so that attribute path holds whichever class is used. `bge_code_dimensions`
        is the last resort, so this property cannot raise.
        """
        truncate = getattr(self._model, "truncate_dim", None)
        if isinstance(truncate, int) and truncate > 0:
            return truncate
        hf_config = getattr(getattr(self._model, "model", None), "config", None)
        width = getattr(hf_config, "hidden_size", None)
        if isinstance(width, int) and width > 0:
            return width
        return self._dimensions
