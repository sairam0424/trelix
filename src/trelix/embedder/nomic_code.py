"""
Nomic CodeRankEmbed embedder (nomic-ai/nomic-embed-code).

Uses sentence-transformers (already a dependency via local embedder).
Nomic CodeRankEmbed uses task-prefix protocol:
  - Documents: "search_document: <code>"
  - Queries:   "search_query: <natural language>"

No extra dependencies beyond sentence-transformers (already in trelix[local]).

Usage:
    TRELIX_EMBEDDER_PROVIDER=nomic-code trelix index ./my-repo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trelix.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

_SentenceTransformer: Any | None
try:
    from sentence_transformers import SentenceTransformer as _ST_cls

    _SentenceTransformer = _ST_cls
except ImportError:  # pragma: no cover
    _SentenceTransformer = None

SentenceTransformer = _SentenceTransformer

_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class NomicCodeEmbedder(BaseEmbedder):
    """
    Embedder backed by nomic-ai/nomic-embed-code via sentence-transformers.

    Task-prefix asymmetric encoding (same protocol as Nomic text v1.5).
    Compatible with trelix[local] install — no extra dependencies.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for nomic-code embedder. "
                "Install it with: pip install 'trelix[local]'"
            )

        self._model = SentenceTransformer(
            config.nomic_code_model,
            trust_remote_code=True,
        )
        self._dimensions = config.nomic_code_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{_DOC_PREFIX}{t}" for t in texts]
        vecs = self._model.encode(prefixed, batch_size=32, normalize_embeddings=True)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        prefixed = [f"{_QUERY_PREFIX}{text}"]
        vecs = self._model.encode(prefixed, normalize_embeddings=True)
        v = vecs[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)

    @property
    def dimension(self) -> int:
        d = self._model.get_sentence_embedding_dimension()
        return d if d else self._dimensions
