"""
Qdrant-backed vector store for trelix.

Requires the optional `qdrant` extra:
    pip install "trelix[qdrant]"
or:
    pip install qdrant-client>=1.9.0

Drop-in replacement for SQLiteVectorStore for large-scale deployments (>500k chunks).
Uses filterable HNSW with m=16 / ef_construct=200 — precision stays high without
collapsing under high cardinality, matching the research recommendation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from trelix.store.vector import BaseVectorStore

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig

logger = logging.getLogger(__name__)

_QDRANT_MISSING_MSG = (
    "qdrant-client is not installed. "
    "Install it with: pip install 'trelix[qdrant]' "
    "or: pip install qdrant-client>=1.9.0"
)

_BATCH_SIZE = 100  # Qdrant upsert batch size

# Ids per scroll page in stored_chunk_ids(). Larger than _BATCH_SIZE because a scroll page
# carries ids only (with_payload / with_vectors both off), not vectors, so the request-size
# ceiling _BATCH_SIZE exists for does not apply.
_SCROLL_PAGE_SIZE = 1000


class QdrantVectorStore(BaseVectorStore):
    """
    Vector store backed by Qdrant HNSW index.

    Collection is created automatically on first use if it does not exist.
    HNSW parameters: m=16, ef_construct=200 — good balance of recall and speed
    for >500k vectors.

    Args:
        config:    IndexConfig (provides store.qdrant_url, store.qdrant_api_key,
                   store.qdrant_collection).
        dimension: Embedding dimension; must match the embedder in use.
    """

    def __init__(self, config: IndexConfig, dimension: int) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, HnswConfigDiff, VectorParams
        except ImportError as exc:
            raise ImportError(_QDRANT_MISSING_MSG) from exc

        self._dimension = dimension
        self._collection = config.store.qdrant_collection
        self._quantization = config.store.qdrant_quantization
        self._quantization_rescore = config.store.qdrant_quantization_rescore

        self._client = QdrantClient(
            url=config.store.qdrant_url,
            api_key=config.store.qdrant_api_key,
            prefer_grpc=config.store.qdrant_prefer_grpc,
            timeout=int(config.store.qdrant_timeout),
        )

        quantization_config = self._build_quantization_config()
        self._ensure_collection(VectorParams, HnswConfigDiff, Distance, quantization_config)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _build_quantization_config(self) -> Any:
        """Build the qdrant-client `quantization_config` object for collection creation.

        Returns None when `qdrant_quantization` is unset -- today's unquantized
        behavior, unchanged. The returned value is only ever consulted by
        `_ensure_collection` at collection-CREATION time (or by `recreate()`,
        which deletes and re-creates): Qdrant fixes a collection's quantization
        at creation and does not expose a way to change it via update.

        Typed `Any` rather than a real qdrant-client union type: those classes
        are only reachable behind the deferred, try/except-guarded import this
        method itself does (`qdrant_client` is an optional extra -- see
        `_QDRANT_MISSING_MSG`), so a module-level import for annotation
        purposes alone would defeat that. `object | None` was tried first and
        rejected: mypy still requires the *exact* SDK union at the
        `create_collection(quantization_config=...)` call site, and `object`
        does not satisfy it.
        """
        if self._quantization is None:
            return None
        try:
            from qdrant_client.models import (
                BinaryQuantization,
                BinaryQuantizationConfig,
                ScalarQuantization,
                ScalarQuantizationConfig,
                ScalarType,
            )
        except ImportError as exc:
            raise ImportError(_QDRANT_MISSING_MSG) from exc

        if self._quantization == "int8":
            return ScalarQuantization(scalar=ScalarQuantizationConfig(type=ScalarType.INT8))
        if self._quantization == "binary":
            return BinaryQuantization(binary=BinaryQuantizationConfig())
        # Unreachable: config.py's Literal["int8", "binary"] | None already rejects
        # anything else at settings-load time. Kept as a loud failure rather than an
        # `else: return None` so a future third literal value can't silently skip
        # quantization instead of raising here to be implemented.
        raise ValueError(f"Unsupported qdrant_quantization value: {self._quantization!r}")

    @staticmethod
    def _quantization_kind(quantization_config: object | None) -> str | None:
        """Map a real `CollectionConfig.quantization_config` value to our vocabulary.

        Returns None (no quantization), "int8", "binary", or "other" for a
        quantization kind trelix never configures itself (Qdrant's
        ProductQuantization / TurboQuantization) but that could exist on a
        collection created or modified outside trelix. Checked via `hasattr`
        rather than `isinstance` so this needs no import of the real qdrant
        model classes -- it only has to distinguish the *shapes* trelix itself
        ever writes (`scalar=...` vs `binary=...`).
        """
        if quantization_config is None:
            return None
        if hasattr(quantization_config, "scalar"):
            return "int8"
        if hasattr(quantization_config, "binary"):
            return "binary"
        return "other"

    def _warn_if_quantization_mismatch(self) -> None:
        """Log a warning if this store's configured quantization does not match
        what the EXISTING collection actually has.

        `_ensure_collection` no-ops when the collection already exists --
        Qdrant fixes `quantization_config` at collection-creation time, so a
        user who changes `qdrant_quantization` for an existing collection would
        otherwise see the setting take zero effect with nothing telling them
        why. This does not fix that (only `recreate()` can); it only makes the
        mismatch visible instead of silent. Callers must only invoke this when
        `self._quantization is not None` -- see `_ensure_collection`.

        `get_collection()` failures (timeout, transient outage, an API key
        scoped to write-only, a race where the collection was just deleted)
        are caught and logged rather than propagated: this is a best-effort
        diagnostic check, and letting it fail store construction would turn
        "collection already exists" from a guaranteed no-op success into a
        new failure mode.
        """
        try:
            info = self._client.get_collection(self._collection)
        except Exception:
            logger.warning(
                "Could not verify quantization for existing collection '%s' -- "
                "get_collection() failed, skipping the quantization mismatch check.",
                self._collection,
                exc_info=True,
            )
            return
        actual = self._quantization_kind(info.config.quantization_config)
        if actual != self._quantization:
            logger.warning(
                "Collection '%s' already exists with quantization=%r, but "
                "qdrant_quantization=%r is configured. Qdrant fixes "
                "quantization_config at collection-creation time -- this setting "
                "has NO effect on an existing collection. Call recreate() to "
                "rebuild it with the new setting (this permanently deletes every "
                "vector currently stored in this collection).",
                self._collection,
                actual,
                self._quantization,
            )

    def _ensure_collection(
        self,
        VectorParams: type,  # noqa: N803
        HnswConfigDiff: type,  # noqa: N803
        Distance: type,
        quantization_config: Any,
    ) -> None:
        """Create the Qdrant collection if it does not already exist.

        `quantization_config` is only applied on the create path: it is fixed
        for the lifetime of a collection once created, so setting/changing
        `qdrant_quantization` has NO effect on a collection that already
        exists -- see `_warn_if_quantization_mismatch` for how that is
        surfaced, and `recreate()` for the only way to actually change it.

        The mismatch check itself is skipped entirely when `qdrant_quantization`
        is unset: with nothing configured there is nothing to mismatch, so
        there is no reason to pay for a `get_collection()` network round-trip
        on every `__init__()` against an already-created collection -- the
        common case for essentially every `trelix search`/index invocation
        against a Qdrant backend, quantization or not.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            vectors_config = VectorParams(
                size=self._dimension,
                distance=Distance.COSINE,  # type: ignore[attr-defined]
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                ),
            )
            # `quantization_config` kwarg omitted entirely (rather than passed
            # as `quantization_config=None`) when unset, so this call stays
            # byte-for-byte identical to the pre-quantization call when the
            # feature is off. Two explicit calls rather than one call built
            # from a conditionally-populated kwargs dict: unpacking a
            # `dict[str, object]` into `create_collection`'s precisely-typed
            # kwargs erases every parameter's real type for mypy.
            if quantization_config is not None:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=vectors_config,
                    quantization_config=quantization_config,
                )
            else:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=vectors_config,
                )
            return

        if self._quantization is not None:
            self._warn_if_quantization_mismatch()

    def recreate(self) -> None:
        """Delete this Qdrant collection and recreate it fresh.

        The only way trelix exposes to adopt (or drop) `qdrant_quantization` on
        a collection that already has data: quantization is fixed at
        collection-creation time (see `_ensure_collection`), so the config
        alone has no effect on an existing collection. Also usable the same
        way `SQLiteVectorStore.recreate()` is -- e.g. to recover from an
        embedding-provider dimension change.

        Matches the `BaseVectorStore.recreate()` contract: discards EVERY
        stored vector (real chunks, file summaries, sub-chunks -- there is no
        partial recreate) and leaves the collection usable at
        `self._dimension` immediately afterward, now created with whichever
        quantization setting is currently configured. Callers are responsible
        for invalidating whatever tracks "already embedded" state (see
        `Database.clear_all_embeddings`) -- this only rebuilds the vector
        store itself, it has no notion of chunks or content hashes.
        """
        try:
            from qdrant_client.models import Distance, HnswConfigDiff, VectorParams
        except ImportError as exc:
            raise ImportError(_QDRANT_MISSING_MSG) from exc

        self._client.delete_collection(collection_name=self._collection)
        quantization_config = self._build_quantization_config()
        self._ensure_collection(VectorParams, HnswConfigDiff, Distance, quantization_config)

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        """
        Upsert embeddings in batches of _BATCH_SIZE to stay within Qdrant
        request-size limits.
        """
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise ImportError(_QDRANT_MISSING_MSG) from exc

        for start in range(0, len(pairs), _BATCH_SIZE):
            batch = pairs[start : start + _BATCH_SIZE]
            points = [
                PointStruct(id=chunk_id, vector=embedding, payload={})
                for chunk_id, embedding in batch
            ]
            self._client.upsert(
                collection_name=self._collection,
                points=points,
            )

    def _build_search_params(self) -> Any:
        """Build the `SearchParams` to pass to `query_points()`, or None.

        None (rather than an empty `SearchParams()`) when `qdrant_quantization`
        is unset, so `search()` sends `query_points()` NO `search_params` kwarg
        at all when the feature is off -- byte-for-byte identical to the
        pre-quantization call, not merely functionally equivalent.

        `rescore=True` (the default, see config.py) re-checks the quantized
        candidates against full-precision vectors -- how the ~99.99%
        (int8) / 90-98% (binary) recall numbers this feature exists to
        capture are actually achieved. Disabling it trades recall for the
        maximum possible speed gain.

        Typed `Any` for the same reason as `_build_quantization_config`: the
        real return type is only reachable behind a deferred, optional import.
        """
        if self._quantization is None:
            return None
        try:
            from qdrant_client.models import QuantizationSearchParams, SearchParams
        except ImportError as exc:
            raise ImportError(_QDRANT_MISSING_MSG) from exc

        return SearchParams(
            quantization=QuantizationSearchParams(rescore=self._quantization_rescore)
        )

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """
        Return top-k (chunk_id, score) pairs using cosine similarity.

        Note: Qdrant cosine search returns higher scores for more similar
        vectors (unlike sqlite-vec which returns L2 distance — lower is closer).
        Callers in retriever.py compute `max(0.0, 1.0 - distance)` on the
        result; since Qdrant already returns similarity scores in [0, 1],
        results pass through correctly.
        """
        search_params = self._build_search_params()
        # `search_params` kwarg omitted entirely (rather than passed as
        # `search_params=None`) when quantization is off, so this call stays
        # byte-for-byte identical to the pre-quantization call. Two explicit
        # calls rather than one built from a conditionally-populated kwargs
        # dict: unpacking a `dict[str, object]` into query_points' precisely
        # -typed kwargs erases every parameter's real type for mypy.
        if search_params is not None:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query,
                limit=k,
                search_params=search_params,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query,
                limit=k,
            )
        return [(int(hit.id), hit.score) for hit in response.points]

    def delete_batch(self, chunk_ids: list[int]) -> None:
        """Delete embeddings for the given chunk_ids. No-op for empty list."""
        if not chunk_ids:
            return
        try:
            from qdrant_client.models import PointIdsList
        except ImportError as exc:
            raise ImportError(_QDRANT_MISSING_MSG) from exc

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=list(chunk_ids)),
        )

    def count(self) -> int:
        """Points in the collection, sentinels included — see `BaseVectorStore.count`.

        Not comparable against the SQLite `chunks` table for that reason; use
        `stored_chunk_ids()`.
        """
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def stored_chunk_ids(self) -> set[int]:
        """Every real chunk_id with a vector in this collection. See the base method.

        Pages through `scroll()` on its own `next_offset` cursor, which is the only way to
        enumerate: Qdrant has no "select ids" primitive, and `count()` cannot distinguish a
        sentinel from a chunk. `with_payload=False, with_vectors=False` keeps each page to
        ids — the vectors are the expensive part and nothing here needs them.

        Filtered in Python via `_is_chunk_id` rather than with a server-side `Filter`: the
        sentinel encoding lives in the point ID, and Qdrant filters on payload fields, which
        `upsert_batch` does not populate (it writes `payload={}`).

        Exercised end-to-end against qdrant-client 1.18.0's local mode, which runs the real
        client and pagination path with no server: 184 ids over 4 pages of 50, sentinels
        excluded, holes recovered exactly. NOT verified against a Qdrant *server*, and one
        difference is known to matter: `upsert_file_summary_embedding` writes `id=-(file_id)`,
        local mode accepts it, but the server requires unsigned 64-bit numeric ids. If it
        rejects them, file-summary vectors never land on this backend at all — a separate
        pre-existing bug, and this method would simply find no negative ids to exclude.
        """
        seen: set[int] = set()
        # `object | None` because the cursor is opaque to us: we only ever hand back what
        # scroll() returned. Both ignore codes are load-bearing across the mypy range
        # pyproject allows (>=1.10.0). Older mypy rejects `object | None` against scroll's
        # declared `int | str | UUID | PointId | None`, so `arg-type` is required; mypy 2.3
        # accepts it and then flags the ignore itself as unnecessary, so `unused-ignore` is
        # required too. Verified both directions: on 2.1.0, dropping `arg-type` fails; on
        # CI's 2.3.1, keeping only `arg-type` fails. Listing both is the only form that
        # passes the whole supported range.
        offset: object | None = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=_SCROLL_PAGE_SIZE,
                offset=offset,  # type: ignore[arg-type, unused-ignore]
                with_payload=False,
                with_vectors=False,
            )
            seen.update(int(r.id) for r in records if self._is_chunk_id(int(r.id)))
            if offset is None:
                return seen

    def upsert_file_summary_embedding(self, file_id: int, embedding: list[float]) -> None:
        """
        Insert or replace a file-level summary embedding.

        Uses point_id = -(file_id) as a negative sentinel to distinguish
        file-summary entries from regular chunk entries — same convention as
        SQLiteVectorStore.  Payload carries type and file_id for filtering.
        """
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise ImportError(_QDRANT_MISSING_MSG) from exc

        point = PointStruct(
            id=-(file_id),
            vector=embedding,
            payload={"type": "file_summary", "file_id": file_id},
        )
        self._client.upsert(
            collection_name=self._collection,
            points=[point],
        )

    def search_file_summaries(
        self, query_embedding: list[float], k: int
    ) -> list[tuple[int, float]]:
        """Search file-summary rows (negative point IDs). Returns (file_id, score) pairs."""
        results = self.search(query_embedding, k=k * 5)
        return [(-cid, score) for cid, score in results if cid < 0][:k]

    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        """Store sub-chunk embedding using point_id = sub_chunk_id + _SUB_CHUNK_OFFSET.

        Offset inherited from `BaseVectorStore` — the identical copy that shadowed it here
        is gone, because `stored_chunk_ids()` above now keys its sentinel filter off it.
        """
        self.upsert_batch([(sub_chunk_id + self._SUB_CHUNK_OFFSET, embedding)])

    def search_sub_chunks(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        """Search sub-chunk embeddings only. Returns (sub_chunk_id, score) pairs."""
        results = self.search(query_embedding, k=k * 5)
        return [
            (cid - self._SUB_CHUNK_OFFSET, score)
            for cid, score in results
            if cid >= self._SUB_CHUNK_OFFSET
        ][:k]
