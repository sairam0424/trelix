"""
Direct Cohere embedder — Cohere's own API (`cohere.ClientV2`), not the Bedrock envelope.

trelix already reaches a Cohere embed model via Bedrock (`BedrockCohereEmbedder`,
base.py), but that goes through Bedrock's own `invoke_model` request/response
envelope, which reports no token usage at all. This module talks to Cohere's own
API directly instead, via the `cohere` SDK's `ClientV2`, which DOES report usage
and has its own request/response shape — verified against the installed
cohere==7.1.1 SDK's source directly (not from memory or docs):

  - `cohere.ClientV2(api_key=...).embed(model=..., input_type=..., texts=...,
    embedding_types=["float"], truncate=...)` -> `EmbedByTypeResponse`
    (cohere/v2/client.py — `texts`: "Maximum number of texts per call is `96`.")
  - Vectors: `response.embeddings.float_` — a `list[list[float]]`. The attribute
    is named `float_`, not `float`, because pydantic aliases the wire field
    "float" (a reserved word) to `float_`
    (cohere/types/embed_by_type_response_embeddings.py).
  - Usage: `response.meta.billed_units.input_tokens` — an `Optional[float]`
    (cohere/types/api_meta.py, cohere/types/api_meta_billed_units.py). `meta`
    and `billed_units` are themselves Optional, so every hop needs a None check.
  - `embed-english-v3.0` (the default model here) embeds at 1024 dims — stated
    directly in the SDK's own bundled docs table (cohere/embed_jobs/client.py:
    "- `embed-english-v3.0` : 1024"), not assumed.

Batch limit: 96 texts per call — the same client-side limit
`BedrockCohereEmbedder` already enforces for the Bedrock envelope of the same
model family (base.py's `_BATCH_LIMIT`), now confirmed independently from the
direct API's own docstring above.

Asymmetric input_type, same pattern as VoyageEmbedder / BedrockCohereEmbedder:
"search_document" for embed(), "search_query" for embed_query().

Out of scope (deliberately, not an oversight):
  - Quantized `embedding_types` (int8/uint8/binary/ubinary/base64) — embed()
    only ever requests `["float"]`. Nothing here assumes a single embedding
    type is returned (the accessor reads `.float_` specifically, not "whichever
    key is present"), so adding a quantized-type kwarg later needs no rework.
  - `output_dimension` — the SDK's own embed() docstring says this is "only
    available for `embed-v4` and newer models"; `embed-english-v3.0` (the
    default here) has a fixed 1024-dim output, so there is nothing to plumb
    for the default model. `cohere_dimensions` on EmbedderConfig is still a
    normal configured int (not hardcoded, unlike BedrockCohereEmbedder's
    fixed `1024`), so wiring a future `output_dimension` kwarg through to it
    is a small addition, not a redesign.

Install:
    pip install 'trelix[cohere]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trelix.core.retry import with_retry
from trelix.embedder.base import BaseEmbedder, _count_embed_call

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

_ClientV2: Any | None
try:
    from cohere import ClientV2 as _ClientV2_cls

    _ClientV2 = _ClientV2_cls
except ImportError:  # pragma: no cover
    _ClientV2 = None

ClientV2 = _ClientV2


def _billed_input_tokens(response: Any) -> int | None:
    """Provider-reported input token count, or None when unavailable.

    `response.meta.billed_units.input_tokens` — three Optional hops deep
    (`meta`, then `billed_units`, then `input_tokens` itself), per
    cohere/types/api_meta.py and api_meta_billed_units.py. Unlike
    `_usage_tokens()` in base.py (which reads OpenAI's/Voyage's flatter
    `response.usage.total_tokens` / `response.total_tokens` shape), Cohere's
    usage lives under `meta`, so that helper does not apply here.
    """
    meta = getattr(response, "meta", None)
    billed_units = getattr(meta, "billed_units", None) if meta is not None else None
    tokens = getattr(billed_units, "input_tokens", None) if billed_units is not None else None
    return int(tokens) if isinstance(tokens, int | float) and not isinstance(tokens, bool) else None


class CohereEmbedder(BaseEmbedder):
    """
    Direct Cohere API embedder — `cohere.ClientV2`, not the Bedrock envelope.

    Default model: embed-english-v3.0 (1024 dims) — Cohere's well-established,
    generally-available embed model; see the module docstring for how the
    1024-dim default was confirmed rather than assumed.

    Distinguishes document vs query embeddings via input_type — same
    asymmetric pattern as VoyageEmbedder / BedrockCohereEmbedder:
    "search_document" for embed(), "search_query" for embed_query().

    Batches requests in groups of 96 (Cohere embed API's per-call limit).

    Requires the optional 'cohere' extra:
        pip install 'trelix[cohere]'
    """

    _BATCH_LIMIT = 96  # Cohere embed API: max 96 texts per call (v2/client.py docstring)

    def __init__(self, config: EmbedderConfig) -> None:
        if ClientV2 is None:
            raise ImportError(
                "cohere is required for the cohere embedder. "
                "Install it with: pip install 'trelix[cohere]'"
            )
        # max_retries=0: @with_retry below is the sole retry layer — the
        # cohere SDK's own default (2 attempts, base_client.py's
        # `_defaulted_max_retries = max_retries if max_retries is not None
        # else 2`) would otherwise stack underneath tenacity's 5-attempt
        # loop, multiplying worst-case wall-clock time on a persistent
        # outage far beyond what max_attempts=5 implies — the same
        # double-retry-stacking bug OpenAIEmbedder/AzureOpenAIEmbedder's
        # max_retries=0 and Bedrock's `retries={"max_attempts": 0, ...}`
        # exist to prevent. Unlike VoyageEmbedder (whose SDK defaults to 0
        # retries on its own), cohere's SDK defaults to 2 — so this must be
        # passed explicitly, not inherited for free.
        self._client = ClientV2(api_key=config.cohere_api_key, max_retries=0)
        self._model = config.cohere_model
        self._dimensions = config.cohere_dimensions

    @with_retry(max_attempts=5)
    def _embed(self, texts: list[str], input_type: str) -> Any:
        response = self._client.embed(
            model=self._model,
            input_type=input_type,
            texts=texts,
            embedding_types=["float"],
            truncate="END",
        )
        # Single call site for both embed() and embed_query(), so counting here
        # covers document and query embeddings alike — same shape as
        # VoyageEmbedder._embed()'s _count_embed_call() call in base.py.
        _count_embed_call("cohere", self._model, texts, _billed_input_tokens(response))
        return response

    @staticmethod
    def _floats(response: Any) -> list[list[float]]:
        floats = response.embeddings.float_
        if floats is None:
            # Cannot happen with embedding_types=["float"] per the documented
            # contract above — but a None here would otherwise fail as an opaque
            # "NoneType is not iterable" deep inside list.extend()/indexing.
            raise RuntimeError(
                "Cohere embed() response has no float embeddings even though "
                "embedding_types=['float'] was requested; unexpected API response shape."
            )
        return floats  # type: ignore[no-any-return]

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_LIMIT):
            batch = texts[i : i + self._BATCH_LIMIT]
            response = self._embed(batch, "search_document")
            results.extend(self._floats(response))
        return results

    def embed_query(self, text: str) -> list[float]:
        response = self._embed([text], "search_query")
        return self._floats(response)[0]

    @property
    def dimension(self) -> int:
        return self._dimensions
