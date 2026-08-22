"""
BGE-Code-v1 embedder (BAAI, May 2025). EXPERIMENTAL — DO NOT USE FOR RETRIEVAL.

This provider pools the wrong token and its vectors are degenerate. The 81.77 CoIR
average below is BAAI's number for BAAI/bge-code-v1 pooled the way BAAI publishes it;
it is NOT what this class produces.

`FlagModel` is not a dispatcher. `from FlagEmbedding import FlagModel` is the alias
`from .base import BaseEmbedder as FlagModel` (FlagEmbedding 1.4.0,
inference/embedder/encoder_only/__init__.py:1), i.e. the ENCODER-ONLY `BaseEmbedder`,
whose `DEFAULT_POOLING_METHOD` and `__init__` default are both `"cls"`
(encoder_only/base.py:38, :50) and whose `pooling()` body for that method is
`return last_hidden_state[:, 0]` (encoder_only/base.py:302). But `BAAI/bge-code-v1` is a
causal Qwen2 decoder (`config.json`: `architectures: ["Qwen2Model"]`, `model_type:
"qwen2"`) published with `1_Pooling/config.json` `pooling_mode_lasttoken: true` and
`pooling_mode_cls_token: false`. The model card's own examples use `FlagLLMModel` with
`trust_remote_code=True`, or `last_token_pool` directly.

CLS on a causal decoder reads position 0, which cannot attend forward, so the vector
depends on token 0 and nothing else. Measured against a randomly initialised
`Qwen2Model` pooled by the real `FlagModel.pooling` — two sequences identical in token 0
and different in every later token gave cosine 1.0 and max|diff| 0.0 (bitwise
identical), while the same hidden states through the real `last_token_pool` gave cosine
0.10 and through the real `mean` branch 0.65. Across 200 resamples of every token after
position 0, position 0's output never moved at all: the information loss is total, not
partial.

Two consequences. Queries: `encode_queries` prefixes every query with the same
instruction (`AbsEmbedder.get_detailed_instruct`), and `tokenizer_config.json` has
`add_bos_token: false` / `bos_token: null`, so token 0 is the first token of that shared
prefix and EVERY query embedding is identical. Documents: `embed()` passes no
instruction, so token 0 is the chunk's own first token — chunks are not all identical,
but every chunk sharing a first token collapses to one vector (measured: 5 distinct
chunks sharing token 0 -> 1 distinct vector).

Not fixable by a kwarg: `FlagModel(..., pooling_method="last_token")` raises
`NotImplementedError` (encoder_only/base.py:308). The real fix is `FlagLLMModel`, which
needs `trust_remote_code=True` and validation against real weights, and is deferred.

Install:
    pip install 'trelix[bge-code]'

Usage (experimental; prefer `local-code` or `voyage` for real retrieval):
    TRELIX_EMBEDDER_PROVIDER=bge-code trelix index ./my-repo
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from trelix.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

logger = logging.getLogger("trelix.embedder.bge_code")

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
    encoded directly. That part matches BGE-Code-v1's training protocol; the
    POOLING does not — see the EXPERIMENTAL note in the module docstring.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        if FlagModel is None:
            raise ImportError(
                "FlagEmbedding is required for bge-code embedder. "
                "Install it with: pip install 'trelix[bge-code]'"
            )

        # EXPERIMENTAL, warned at construction because there is no longer anything else to
        # notice. Until v3.1.6 the `dimension` property raised `AttributeError` and blocked
        # this provider outright; removing that raise made a degenerate pooling path
        # reachable, so v3.1.6 turned a loud failure into a silent one. `FlagModel` is
        # FlagEmbedding's ENCODER-only `BaseEmbedder` (`DEFAULT_POOLING_METHOD = "cls"`;
        # `pooling()` returns `last_hidden_state[:, 0]`) and no `pooling_method` is passed
        # below, so CLS runs — while BAAI publishes BAAI/bge-code-v1 as a causal Qwen2
        # decoder with `pooling_mode_lasttoken: true` in `1_Pooling/config.json`. A causal
        # mask makes position 0's hidden state a function of token 0 alone, and every query
        # carries the same `_QUERY_INSTRUCTION` prefix.
        #
        # WARNING, not INFO, for the reason `graph --concepts` gives (graph/builder.py:79):
        # the CLI's default level is WARNING, so an INFO line here is invisible without -v,
        # which is how this class of thing goes unnoticed. Once per construction, NOT per
        # call as `_xtr_rerank` does (retrieval/reranker.py:301) — `embed()` runs once per
        # batch, so per-call would be thousands of lines per index, whereas exactly one
        # embedder is built per index or query run. Same principle as reranker.py:300
        # though: silence here reads as "bge-code embedded your code the way BAAI trained
        # it to".
        logger.warning(
            "bge-code is EXPERIMENTAL: its pooling is unverified. trelix builds it with "
            "FlagEmbedding's FlagModel, which pools with CLS (last_hidden_state[:, 0]), "
            "but BAAI publishes %s as a causal decoder with pooling_mode_lasttoken: true. "
            "CLS pooling reads position 0 only, and every query carries the same "
            "instruction prefix, so query embeddings may be degenerate and results may not "
            "depend on the query. No retrieval-quality claim is made for this provider; any "
            "'CoIR SOTA' figure in older trelix docs describes the upstream model, not "
            "trelix's use of it. For an offline embedder without an open protocol question, "
            "use provider 'local' (all-MiniLM-L6-v2): sentence-transformers reads that "
            "model's own 1_Pooling/config.json instead of assuming a method. Do NOT "
            "substitute 'local-code' or 'nomic-code' — this same release marks both "
            "EXPERIMENTAL for their own protocol mismatches; see base.py's module docstring.",
            config.bge_code_model,
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
        #
        # EXPERIMENTAL: that prefix is exactly what makes this degenerate. encode_queries
        # -> encode(instruction=...) -> get_detailed_instruct formats every query as
        # _QUERY_INSTRUCTION + text with the default "{}{}" format, and no BOS is
        # prepended (tokenizer_config.json: add_bos_token false, bos_token null), so
        # token 0 is the same for every query. FlagModel pools CLS — position 0 only —
        # on a causal decoder, so all queries return the identical vector. See the
        # module docstring for the measurement.
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
