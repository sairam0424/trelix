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

from typing import TYPE_CHECKING

from trelix.store.vector import BaseVectorStore

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig

_QDRANT_MISSING_MSG = (
    "qdrant-client is not installed. "
    "Install it with: pip install 'trelix[qdrant]' "
    "or: pip install qdrant-client>=1.9.0"
)

_BATCH_SIZE = 100  # Qdrant upsert batch size


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

        self._client = QdrantClient(
            url=config.store.qdrant_url,
            api_key=config.store.qdrant_api_key,
            prefer_grpc=config.store.qdrant_prefer_grpc,
            timeout=config.store.qdrant_timeout,
        )

        self._ensure_collection(VectorParams, HnswConfigDiff, Distance)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(
        self,
        VectorParams: type,  # noqa: N803
        HnswConfigDiff: type,  # noqa: N803
        Distance: type,
    ) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,  # type: ignore[attr-defined]
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=200,
                    ),
                ),
            )

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

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """
        Return top-k (chunk_id, score) pairs using cosine similarity.

        Note: Qdrant cosine search returns higher scores for more similar
        vectors (unlike sqlite-vec which returns L2 distance — lower is closer).
        Callers in retriever.py compute `max(0.0, 1.0 - distance)` on the
        result; since Qdrant already returns similarity scores in [0, 1],
        results pass through correctly.
        """
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query,
            limit=k,
        )
        return [(int(hit.id), hit.score) for hit in hits]

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
            points_selector=PointIdsList(points=chunk_ids),
        )

    def count(self) -> int:
        """Return the number of vectors stored in the collection."""
        info = self._client.get_collection(self._collection)
        return info.vectors_count or 0

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

    _SUB_CHUNK_OFFSET = 10_000_000

    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        """Store sub-chunk embedding using point_id = sub_chunk_id + _SUB_CHUNK_OFFSET."""
        self.upsert_batch([(sub_chunk_id + self._SUB_CHUNK_OFFSET, embedding)])

    def search_sub_chunks(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        """Search sub-chunk embeddings only. Returns (sub_chunk_id, score) pairs."""
        results = self.search(query_embedding, k=k * 5)
        return [
            (cid - self._SUB_CHUNK_OFFSET, score)
            for cid, score in results
            if cid >= self._SUB_CHUNK_OFFSET
        ][:k]
