"""
Nomic CodeRankEmbed embedder (nomic-ai/CodeRankEmbed) — EXPERIMENTAL.

Uses sentence-transformers (already a dependency via local embedder).

POOLING IS CORRECT HERE. CodeRankEmbed is a bidirectional encoder
(`architectures: ["NomicBertModel"]`, `causal: false`, `n_embd: 768`) publishing
`pooling_mode_cls_token: true`, and sentence-transformers' Pooling.load() reads that
file — so cls on token 0 is the published choice, not our guess. This is the
opposite of bge-code, whose library ignores the published pooling entirely.

THE PREFIX PROTOCOL BELOW IS THE WRONG MODEL'S. CodeRankEmbed publishes exactly one
prefix — a query instruction, "Represent this query for searching relevant code: ",
carried in its own config_sentence_transformers.json under `prompts.query`, and its
model card states the query prompt *must* include it. Code/documents take NO prefix.
The "search_document: "/"search_query: " pair used here belongs to nomic-embed-text
v1.5, and arrived because the original plan specified `nomic-ai/nomic-embed-code`
(docs/superpowers/plans/2026-06-28-trelix-v2-upgrade.md); the model default later
changed to CodeRankEmbed and the prefixes did not. Both sides therefore carry a
prefix the model was never trained on.

NOT fixed here, deliberately, for the same reason bge-code is not being rewritten in
3.1.7: the correct form is `encode(..., prompt_name="query")` for queries and no
prefix for documents, which changes every vector this provider has ever produced and
so needs a real-weight retrieval check plus a reindex, not a one-line edit. The unit
tests assert the constants below against a MagicMock, so they cannot detect this.

Requires the opt-in TRELIX_ALLOW_REMOTE_MODEL_CODE=1 in the process environment:
CodeRankEmbed loads with trust_remote_code=True. Without it this provider refuses to
construct, which is by design (see embedder/base.py).

No extra dependencies beyond sentence-transformers (already in trelix[local]).

Usage:
    TRELIX_ALLOW_REMOTE_MODEL_CODE=1 TRELIX_EMBEDDER_PROVIDER=nomic-code \\
        trelix index ./my-repo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trelix.embedder.base import BaseEmbedder, load_remote_code_model

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

_SentenceTransformer: Any | None
try:
    from sentence_transformers import SentenceTransformer as _ST_cls

    _SentenceTransformer = _ST_cls
except ImportError:  # pragma: no cover
    _SentenceTransformer = None

SentenceTransformer = _SentenceTransformer

# WRONG FOR THE MODEL THAT IS LOADED — see the module docstring. CodeRankEmbed's
# published protocol is: queries prefixed with
# "Represent this query for searching relevant code: " (its own
# config_sentence_transformers.json `prompts.query`), documents unprefixed. These two
# constants are nomic-embed-text-v1.5's protocol. Left in place because changing them
# invalidates every vector already indexed with this provider; the fix is tracked with
# the bge-code rewrite, not slipped into a patch release.
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class NomicCodeEmbedder(BaseEmbedder):
    """
    Embedder backed by nomic-ai/CodeRankEmbed via sentence-transformers.

    Task-prefix asymmetric encoding using Nomic text v1.5's protocol, which is NOT
    CodeRankEmbed's — see the module docstring. Treat this provider as experimental.
    Compatible with trelix[local] install — no extra dependencies.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for nomic-code embedder. "
                "Install it with: pip install 'trelix[local]'"
            )

        # CodeRankEmbed needs trust_remote_code, i.e. Python from the model repo
        # runs here — so the load is gated on TRELIX_ALLOW_REMOTE_MODEL_CODE, which
        # a `.env` in an indexed repository structurally cannot set. Same helper as
        # local-code: one gate, not one per provider.
        self._model = load_remote_code_model(
            config.nomic_code_model,
            provider="nomic-code",
            factory=SentenceTransformer,
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
