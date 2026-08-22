"""
Nomic CodeRankEmbed embedder (nomic-ai/CodeRankEmbed).

POOLING IS CORRECT HERE, and for a reason worth keeping written down. CodeRankEmbed is a
bidirectional encoder (`architectures: ["NomicBertModel"]`, `n_positions: 8192`) publishing
`pooling_mode_cls_token: true`, and sentence-transformers' `Pooling.load()` reads that file
— so cls on token 0 is the model author's published choice, not our guess. That is the
whole difference between this provider and bge-code, whose library ignores the published
pooling entirely.

THE PROMPT PROTOCOL NOW COMES FROM THE MODEL, NOT FROM US. Until 3.2.0 this module carried
`_DOC_PREFIX = "search_document: "` and `_QUERY_PREFIX = "search_query: "`. Those belong to
nomic-embed-text-v1.5, a different model; they arrived because the original plan specified
`nomic-ai/nomic-embed-code` and the default later changed to CodeRankEmbed while the
prefixes did not. CodeRankEmbed publishes exactly one prefix — a query instruction, carried
in its own `config_sentence_transformers.json` under `prompts.query` — and takes NO prefix
on documents. So both sides previously carried a prefix the model was never trained on.

The fix does not hardcode the correct string either. `encode(prompt_name="query")` makes
sentence-transformers look the prompt up in the model's own published config, exactly as
`Pooling.load()` looks up the pooling method. Applying the same principle to both means no
future model swap can leave this provider asserting a protocol its model does not declare —
which is the mechanism that produced the original bug, and the one that put CodeRankEmbed's
prompt string inside the bge-code provider (see embedder/bge_code.py). If a configured model
publishes no `prompts.query`, `encode` raises rather than silently substituting something,
and that is the intended behaviour: a loud refusal beats a plausible wrong prefix.

NO REINDEX IS REQUIRED, because no index built with this provider can exist. Until 3.2.0
this provider could not construct at all for any user: CodeRankEmbed's `config.json`
declares `auto_map.AutoModel = modeling_hf_nomic_bert.NomicBertModel`, that published module
imports `einops` at its top level, and `einops` was declared by no trelix extra — so the
load raised `ModuleNotFoundError` from inside remote model code before any protocol question
arose. `einops` is now declared in the `nomic-code` extra. That is also why correcting the
prompts costs nothing: there are no stored vectors to invalidate.

LICENCE: MIT, the most permissive of the three code-specialised providers.

Requires the opt-in TRELIX_ALLOW_REMOTE_MODEL_CODE=1 in the process environment:
CodeRankEmbed loads with trust_remote_code=True. Without it this provider refuses to
construct, which is by design (see embedder/base.py).

Usage:
    pip install 'trelix[nomic-code]'
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

# The name of the prompt to look up in the MODEL's own config_sentence_transformers.json,
# not the prompt text. sentence-transformers resolves it at encode time, so the string lives
# in exactly one place: the model repository.
_QUERY_PROMPT_NAME = "query"

# What CodeRankEmbed publishes under that name, as of this writing. Recorded ONLY so a test
# can assert the published config still says it — never read by the encode path, so a drift
# here cannot change a vector. Keeping the expected value out of the call path is what makes
# it safe to record at all.
_EXPECTED_PUBLISHED_QUERY_PROMPT = "Represent this query for searching relevant code: "


class NomicCodeEmbedder(BaseEmbedder):
    """
    Embedder backed by nomic-ai/CodeRankEmbed via sentence-transformers.

    Asymmetric encoding using the model's OWN published protocol: queries carry the
    instruction from `prompts.query`, documents carry nothing. Compatible with the
    `trelix[nomic-code]` extra.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for nomic-code embedder. "
                "Install it with: pip install 'trelix[nomic-code]'"
            )

        # CodeRankEmbed needs trust_remote_code, i.e. Python from the model repo runs here —
        # so the load is gated on TRELIX_ALLOW_REMOTE_MODEL_CODE, which a `.env` in an
        # indexed repository structurally cannot set. Same helper as local-code: one gate,
        # not one per provider.
        self._model = load_remote_code_model(
            config.nomic_code_model,
            provider="nomic-code",
            factory=SentenceTransformer,
        )
        self._dimensions = config.nomic_code_dimensions
        # Was hardcoded 32, which silently ignored TRELIX_EMBEDDER_BATCH_SIZE — the knob
        # documented for every other sentence-transformers provider. Wired up here because
        # this provider has never produced a vector, so there is no memory profile to
        # preserve; LocalEmbedder and LocalCodeEmbedder already read the same field.
        self._batch_size = config.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode documents with NO prompt — CodeRankEmbed publishes none for them.

        `normalize_embeddings=True` is OUR choice, not the model's: CodeRankEmbed's
        modules.json is [Transformer, Pooling] with no 2_Normalize stage, so nothing
        normalizes unless we ask. It is kept because the sqlite vec0 table declares no
        distance metric and sqlite-vec defaults to L2 — on unit vectors L2 is monotone in
        cosine, so the ranking is the one the model's `similarity_fn_name: cosine` intends.
        Both encode paths must pass it or queries and documents would live on different
        scales.
        """
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
        )
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        """Encode a query with the instruction the MODEL publishes for queries.

        `prompt_name` rather than a literal string: sentence-transformers reads
        `prompts[_QUERY_PROMPT_NAME]` from the model's own
        config_sentence_transformers.json, so the protocol travels with the weights. If the
        configured model declares no such prompt this raises, which is deliberate — see the
        module docstring.
        """
        vecs = self._model.encode(
            [text],
            prompt_name=_QUERY_PROMPT_NAME,
            normalize_embeddings=True,
        )
        v = vecs[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)

    @property
    def dimension(self) -> int:
        d = self._model.get_sentence_embedding_dimension()
        return d if d else self._dimensions
