"""
Unit tests for QdrantVectorStore and the make_vector_store factory.

qdrant_client is an optional dependency. All tests inject a fake module via
sys.modules so no live Qdrant instance or installed package is required.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake qdrant_client module injected before every test that needs it
# ---------------------------------------------------------------------------


def _build_fake_qdrant_module() -> tuple[types.ModuleType, MagicMock]:
    """
    Build a minimal fake `qdrant_client` package + models submodule that is
    sufficient for QdrantVectorStore to import and run against.

    Returns (fake_pkg, mock_client_instance) so tests can assert on the latter.
    """
    mock_client_instance = MagicMock()
    mock_client_instance.get_collections.return_value = MagicMock(collections=[])
    mock_client_instance.create_collection.return_value = None
    mock_client_instance.upsert.return_value = None
    mock_client_instance.search.return_value = []

    # PointStruct — store id/vector/payload as plain attributes
    class PointStruct:
        def __init__(self, id: int, vector: list, payload: dict) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    # PointIdsList
    class PointIdsList:
        def __init__(self, points: list[int]) -> None:
            self.points = points

    class Distance:
        COSINE = "Cosine"

    class HnswConfigDiff:
        def __init__(self, m: int, ef_construct: int) -> None:
            self.m = m
            self.ef_construct = ef_construct

    class VectorParams:
        def __init__(self, size: int, distance: str, hnsw_config: Any = None) -> None:
            self.size = size
            self.distance = distance
            self.hnsw_config = hnsw_config

    # models submodule
    fake_models = types.ModuleType("qdrant_client.models")
    fake_models.PointStruct = PointStruct  # type: ignore[attr-defined]
    fake_models.PointIdsList = PointIdsList  # type: ignore[attr-defined]
    fake_models.Distance = Distance  # type: ignore[attr-defined]
    fake_models.HnswConfigDiff = HnswConfigDiff  # type: ignore[attr-defined]
    fake_models.VectorParams = VectorParams  # type: ignore[attr-defined]

    # root package
    fake_pkg = types.ModuleType("qdrant_client")

    def _make_client(*args: Any, **kwargs: Any) -> MagicMock:
        return mock_client_instance

    fake_pkg.QdrantClient = _make_client  # type: ignore[attr-defined]
    fake_pkg.models = fake_models  # type: ignore[attr-defined]

    return fake_pkg, mock_client_instance


def _inject_fake_qdrant() -> MagicMock:
    """Inject fake qdrant_client into sys.modules and return mock client instance."""
    # Remove any previously cached modules so the store re-imports cleanly
    for key in list(sys.modules):
        if key.startswith("trelix.store.vector_qdrant"):
            del sys.modules[key]

    fake_pkg, mock_instance = _build_fake_qdrant_module()
    sys.modules["qdrant_client"] = fake_pkg
    sys.modules["qdrant_client.models"] = fake_pkg.models
    return mock_instance


def _remove_fake_qdrant() -> None:
    """Remove injected fake qdrant_client from sys.modules."""
    sys.modules.pop("qdrant_client", None)
    sys.modules.pop("qdrant_client.models", None)
    for key in list(sys.modules):
        if key.startswith("trelix.store.vector_qdrant"):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    backend: str = "qdrant",
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str | None = None,
    qdrant_collection: str = "trelix",
) -> Any:
    """Build a minimal IndexConfig-like object for testing."""
    from trelix.core.config import IndexConfig, StoreConfig

    store = StoreConfig(db_path=".trelix/index.db")  # type: ignore[call-arg]
    # Override via direct attribute assignment to avoid pydantic env-prefix issues
    object.__setattr__(store, "backend", backend)
    object.__setattr__(store, "qdrant_url", qdrant_url)
    object.__setattr__(store, "qdrant_api_key", qdrant_api_key)
    object.__setattr__(store, "qdrant_collection", qdrant_collection)

    repo_root = str(Path(__file__).parent.parent.parent.resolve())
    config = IndexConfig(repo_path=repo_root)
    object.__setattr__(config, "store", store)
    return config


# ---------------------------------------------------------------------------
# QdrantVectorStore — upsert_batch
# ---------------------------------------------------------------------------


class TestQdrantUpsertBatch:
    def setup_method(self) -> None:
        self.mock_client = _inject_fake_qdrant()

    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def _make_store(self, collection: str = "trelix") -> Any:
        from trelix.store.vector_qdrant import QdrantVectorStore
        return QdrantVectorStore(_make_config(qdrant_collection=collection), dimension=4)

    def test_upsert_batch_calls_upsert_with_point_structs(self) -> None:
        """upsert_batch should call client.upsert with PointStruct objects."""
        store = self._make_store()
        pairs = [(1, [0.1, 0.2, 0.3, 0.4]), (2, [0.5, 0.6, 0.7, 0.8])]
        store.upsert_batch(pairs)

        self.mock_client.upsert.assert_called_once()
        _, kwargs = self.mock_client.upsert.call_args
        points = kwargs.get("points") or self.mock_client.upsert.call_args[1]["points"]
        assert len(points) == 2

    def test_upsert_batch_point_ids_match(self) -> None:
        """PointStruct IDs must match the chunk_ids from pairs."""
        store = self._make_store()
        pairs = [(10, [1.0, 0.0, 0.0, 0.0]), (20, [0.0, 1.0, 0.0, 0.0])]
        store.upsert_batch(pairs)

        _, kwargs = self.mock_client.upsert.call_args
        points = kwargs["points"]
        ids = {p.id for p in points}
        assert ids == {10, 20}

    def test_upsert_batch_vectors_match(self) -> None:
        """PointStruct vectors must match the embeddings from pairs."""
        store = self._make_store()
        emb = [0.1, 0.2, 0.3, 0.4]
        store.upsert_batch([(7, emb)])

        _, kwargs = self.mock_client.upsert.call_args
        points = kwargs["points"]
        assert points[0].vector == emb

    def test_upsert_batch_splits_into_batches_of_100(self) -> None:
        """Pairs exceeding 100 should be split into multiple upsert calls."""
        store = self._make_store()
        pairs = [(i, [float(i % 4 == j) for j in range(4)]) for i in range(250)]
        store.upsert_batch(pairs)

        # 250 items → 3 batches (100 + 100 + 50)
        assert self.mock_client.upsert.call_count == 3

    def test_upsert_batch_empty_is_noop(self) -> None:
        """Empty pairs list should not call client.upsert."""
        store = self._make_store()
        store.upsert_batch([])

        self.mock_client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# QdrantVectorStore — search
# ---------------------------------------------------------------------------


class TestQdrantSearch:
    def setup_method(self) -> None:
        self.mock_client = _inject_fake_qdrant()

    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def _make_store(self, collection: str = "trelix") -> Any:
        from trelix.store.vector_qdrant import QdrantVectorStore
        return QdrantVectorStore(_make_config(qdrant_collection=collection), dimension=4)

    def test_search_returns_id_score_tuples(self) -> None:
        """search() must return list[tuple[int, float]]."""
        hit1 = MagicMock(id=42, score=0.95)
        hit2 = MagicMock(id=7, score=0.80)
        self.mock_client.search.return_value = [hit1, hit2]

        store = self._make_store()
        results = store.search([0.1, 0.2, 0.3, 0.4], k=5)

        assert results == [(42, 0.95), (7, 0.80)]

    def test_search_passes_k_as_limit(self) -> None:
        """search() must forward k as `limit` to client.search."""
        store = self._make_store()
        store.search([0.0, 0.0, 1.0, 0.0], k=17)

        _, kwargs = self.mock_client.search.call_args
        assert kwargs.get("limit") == 17

    def test_search_passes_collection_name(self) -> None:
        store = self._make_store(collection="my_collection")
        store.search([0.0, 1.0, 0.0, 0.0], k=5)

        _, kwargs = self.mock_client.search.call_args
        assert kwargs.get("collection_name") == "my_collection"

    def test_search_empty_result(self) -> None:
        store = self._make_store()
        self.mock_client.search.return_value = []

        results = store.search([0.0, 0.0, 0.0, 1.0], k=5)
        assert results == []


# ---------------------------------------------------------------------------
# make_vector_store factory
# ---------------------------------------------------------------------------


class TestMakeVectorStoreFactory:
    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def test_factory_returns_sqlite_by_default(self) -> None:
        """Default backend should be SQLiteVectorStore."""
        from trelix.core.config import IndexConfig
        from trelix.store.vector import SQLiteVectorStore, make_vector_store

        repo_root = str(Path(__file__).parent.parent.parent.resolve())
        config = IndexConfig(repo_path=repo_root)
        store = make_vector_store(config, dimension=4)
        assert isinstance(store, SQLiteVectorStore)

    def test_factory_returns_qdrant_when_backend_qdrant(self) -> None:
        """backend='qdrant' should return QdrantVectorStore."""
        _inject_fake_qdrant()

        from trelix.store.vector import make_vector_store
        from trelix.store.vector_qdrant import QdrantVectorStore

        config = _make_config(backend="qdrant")
        store = make_vector_store(config, dimension=4)
        assert isinstance(store, QdrantVectorStore)

    def test_factory_sqlite_config_unchanged(self) -> None:
        """Explicitly setting backend='sqlite' should still give SQLiteVectorStore."""
        from trelix.core.config import IndexConfig, StoreConfig
        from trelix.store.vector import SQLiteVectorStore, make_vector_store

        repo_root = str(Path(__file__).parent.parent.parent.resolve())
        store_cfg = StoreConfig(db_path=".trelix/index.db")  # type: ignore[call-arg]
        object.__setattr__(store_cfg, "backend", "sqlite")

        config = IndexConfig(repo_path=repo_root)
        object.__setattr__(config, "store", store_cfg)

        store = make_vector_store(config, dimension=4)
        assert isinstance(store, SQLiteVectorStore)


# ---------------------------------------------------------------------------
# Import error when qdrant_client is not installed
# ---------------------------------------------------------------------------


class TestQdrantImportError:
    def setup_method(self) -> None:
        # Ensure qdrant_client is NOT available
        _remove_fake_qdrant()
        sys.modules["qdrant_client"] = None  # type: ignore[assignment]

    def teardown_method(self) -> None:
        sys.modules.pop("qdrant_client", None)
        sys.modules.pop("qdrant_client.models", None)
        for key in list(sys.modules):
            if key.startswith("trelix.store.vector_qdrant"):
                del sys.modules[key]

    def test_helpful_importerror_when_qdrant_client_missing(self) -> None:
        """ImportError raised when qdrant_client is absent must include pip install hint."""
        with pytest.raises((ImportError, ModuleNotFoundError)) as exc_info:
            # Remove any cached import of the module
            for key in list(sys.modules):
                if "vector_qdrant" in key:
                    del sys.modules[key]
            from trelix.store.vector_qdrant import QdrantVectorStore  # noqa: F401

            config = _make_config()
            QdrantVectorStore(config, dimension=4)

        error_text = str(exc_info.value).lower()
        assert "qdrant" in error_text or "pip install" in error_text
