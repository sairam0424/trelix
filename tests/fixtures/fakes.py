"""Hand-written, ABC-checked fakes for the two provider interfaces trelix's
retrieval pipeline is built on: ``BaseEmbedder`` (src/trelix/embedder/base.py)
and ``BaseVectorStore`` (src/trelix/store/vector.py).

Deliberately real ``ABC`` subclasses, not ``unittest.mock.Mock`` (test-quality
rule 3): a ``Mock`` answers ANY method call with another ``Mock``, so a test
built on one cannot tell "the interface under test doesn't have this method"
from "the interface has it and returns a mock". Subclassing the real ABC
makes a missing required method a ``TypeError`` at CONSTRUCTION time --
before the test body ever runs -- because ``abc.ABCMeta`` refuses to
instantiate a class that has not overridden every ``@abstractmethod``. See
``tests/fixtures/test_fakes_fidelity.py`` for the proof (delete a method,
show the ``TypeError``, restore it).

Neither fake talks to a real model or a real database. Both are in-memory,
deterministic, and exist so retrieval-layer tests can exercise real control
flow (ranking, filtering, sentinel-id arithmetic) without a GPU, an API key,
or sqlite.
"""

from __future__ import annotations

import hashlib
import math

from trelix.embedder.base import BaseEmbedder
from trelix.store.vector import BaseVectorStore

# ---------------------------------------------------------------------------
# FakeEmbedder
# ---------------------------------------------------------------------------


class FakeEmbedder(BaseEmbedder):
    """Deterministic, hash-based embedder. No model, no I/O.

    Two calls with the SAME text always produce the SAME vector (this is
    what "deterministic" buys a test: reproducible assertions without
    pinning literal floats copied from a real model's output). Two calls
    with DIFFERENT text produce different vectors with high probability
    (each of `dimension` components is one byte of that text's sha256
    digest, scaled to [0, 1)) -- enough to exercise ranking/similarity code
    paths without claiming any resemblance to a real embedding space.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    def _vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    @property
    def dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# FakeVectorStore
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class FakeVectorStore(BaseVectorStore):
    """In-memory vector store. No sqlite, no disk.

    Uses the SAME sentinel convention every real backend uses (see
    ``BaseVectorStore._SUB_CHUNK_OFFSET`` / ``_is_chunk_id``): file summaries
    at negative ``-file_id`` keys, sub-chunks at
    ``sub_chunk_id + _SUB_CHUNK_OFFSET``, real chunks at positive ids below
    that offset -- stored in ONE dict, mirroring the real backends' single
    table, so ``count()`` (sentinel-inclusive, per the real contract) and
    ``stored_chunk_ids()`` (sentinel-EXCLUSIVE, per the real contract) can
    share one code path instead of drifting into two.

    Ranking: cosine similarity, HIGHEST first -- an explicit, stated
    convention rather than an accident of insertion order.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self._vectors: dict[int, list[float]] = {}

    def _require_dimension(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimension:
            raise ValueError(
                f"FakeVectorStore configured for dimension {self._dimension}, "
                f"got a vector of length {len(embedding)}"
            )

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        for chunk_id, embedding in pairs:
            self._require_dimension(embedding)
        for chunk_id, embedding in pairs:
            self._vectors[chunk_id] = embedding

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        self._require_dimension(query)
        candidates = [
            (chunk_id, _cosine_similarity(query, vec))
            for chunk_id, vec in self._vectors.items()
            if self._is_chunk_id(chunk_id)
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:k]

    def delete_batch(self, chunk_ids: list[int]) -> None:
        for chunk_id in chunk_ids:
            self._vectors.pop(chunk_id, None)

    def count(self) -> int:
        return len(self._vectors)

    def stored_chunk_ids(self) -> set[int]:
        return {chunk_id for chunk_id in self._vectors if self._is_chunk_id(chunk_id)}

    def upsert_file_summary_embedding(self, file_id: int, embedding: list[float]) -> None:
        self._require_dimension(embedding)
        self._vectors[-file_id] = embedding

    def search_file_summaries(
        self, query_embedding: list[float], k: int
    ) -> list[tuple[int, float]]:
        self._require_dimension(query_embedding)
        candidates = [
            (-chunk_id, _cosine_similarity(query_embedding, vec))
            for chunk_id, vec in self._vectors.items()
            if chunk_id < 0
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:k]

    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        self._require_dimension(embedding)
        self._vectors[sub_chunk_id + self._SUB_CHUNK_OFFSET] = embedding

    def search_sub_chunks(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        self._require_dimension(query_embedding)
        candidates = [
            (chunk_id - self._SUB_CHUNK_OFFSET, _cosine_similarity(query_embedding, vec))
            for chunk_id, vec in self._vectors.items()
            if chunk_id >= self._SUB_CHUNK_OFFSET
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:k]
