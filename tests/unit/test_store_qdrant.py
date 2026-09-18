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
    mock_client_instance.query_points.return_value = MagicMock(points=[])
    mock_client_instance.delete_collection.return_value = True
    # Existing-collection describe path used by _warn_if_quantization_mismatch().
    # No quantization by default -- tests that need a specific existing state
    # override this return_value directly.
    mock_client_instance.get_collection.return_value = MagicMock(
        config=MagicMock(quantization_config=None)
    )

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

    # -- Quantization models. Field names mirror qdrant-client 1.19.0's real
    # `qdrant_client.models` shapes exactly (verified by direct introspection):
    # ScalarQuantizationConfig(type, quantile, always_ram, memory),
    # ScalarQuantization(scalar=...), BinaryQuantizationConfig(always_ram, memory,
    # encoding, query_encoding), BinaryQuantization(binary=...),
    # QuantizationSearchParams(ignore, rescore, oversampling),
    # SearchParams(hnsw_ef, exact, quantization, ...).
    class ScalarType:
        INT8 = "int8"

    class ScalarQuantizationConfig:
        def __init__(
            self,
            type: str,
            quantile: float | None = None,
            always_ram: bool | None = None,
            memory: Any = None,
        ) -> None:
            self.type = type
            self.quantile = quantile
            self.always_ram = always_ram
            self.memory = memory

    class ScalarQuantization:
        def __init__(self, scalar: Any) -> None:
            self.scalar = scalar

    class BinaryQuantizationConfig:
        def __init__(
            self,
            always_ram: bool | None = None,
            memory: Any = None,
            encoding: Any = None,
            query_encoding: Any = None,
        ) -> None:
            self.always_ram = always_ram
            self.memory = memory
            self.encoding = encoding
            self.query_encoding = query_encoding

    class BinaryQuantization:
        def __init__(self, binary: Any) -> None:
            self.binary = binary

    class QuantizationSearchParams:
        def __init__(
            self,
            ignore: bool | None = False,
            rescore: bool | None = None,
            oversampling: float | None = None,
        ) -> None:
            self.ignore = ignore
            self.rescore = rescore
            self.oversampling = oversampling

    class SearchParams:
        def __init__(self, quantization: Any = None, **kwargs: Any) -> None:
            self.quantization = quantization
            for key, value in kwargs.items():
                setattr(self, key, value)

    # models submodule
    fake_models = types.ModuleType("qdrant_client.models")
    fake_models.PointStruct = PointStruct  # type: ignore[attr-defined]
    fake_models.PointIdsList = PointIdsList  # type: ignore[attr-defined]
    fake_models.Distance = Distance  # type: ignore[attr-defined]
    fake_models.HnswConfigDiff = HnswConfigDiff  # type: ignore[attr-defined]
    fake_models.VectorParams = VectorParams  # type: ignore[attr-defined]
    fake_models.ScalarType = ScalarType  # type: ignore[attr-defined]
    fake_models.ScalarQuantizationConfig = ScalarQuantizationConfig  # type: ignore[attr-defined]
    fake_models.ScalarQuantization = ScalarQuantization  # type: ignore[attr-defined]
    fake_models.BinaryQuantizationConfig = BinaryQuantizationConfig  # type: ignore[attr-defined]
    fake_models.BinaryQuantization = BinaryQuantization  # type: ignore[attr-defined]
    fake_models.QuantizationSearchParams = QuantizationSearchParams  # type: ignore[attr-defined]
    fake_models.SearchParams = SearchParams  # type: ignore[attr-defined]

    # root package
    fake_pkg = types.ModuleType("qdrant_client")

    fake_qdrant_client_cls = MagicMock(return_value=mock_client_instance)

    fake_pkg.QdrantClient = fake_qdrant_client_cls  # type: ignore[attr-defined]
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
    qdrant_quantization: str | None = None,
    qdrant_quantization_rescore: bool = True,
) -> Any:
    """Build a minimal IndexConfig-like object for testing."""
    from trelix.core.config import IndexConfig, StoreConfig

    store = StoreConfig(db_path=".trelix/index.db")  # type: ignore[call-arg]
    # Override via direct attribute assignment to avoid pydantic env-prefix issues
    object.__setattr__(store, "backend", backend)
    object.__setattr__(store, "qdrant_url", qdrant_url)
    object.__setattr__(store, "qdrant_api_key", qdrant_api_key)
    object.__setattr__(store, "qdrant_collection", qdrant_collection)
    object.__setattr__(store, "qdrant_quantization", qdrant_quantization)
    object.__setattr__(store, "qdrant_quantization_rescore", qdrant_quantization_rescore)

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
        self.mock_client.query_points.return_value = MagicMock(points=[hit1, hit2])

        store = self._make_store()
        results = store.search([0.1, 0.2, 0.3, 0.4], k=5)

        assert results == [(42, 0.95), (7, 0.80)]

    def test_search_passes_k_as_limit(self) -> None:
        """search() must forward k as `limit` to client.query_points."""
        store = self._make_store()
        store.search([0.0, 0.0, 1.0, 0.0], k=17)

        _, kwargs = self.mock_client.query_points.call_args
        assert kwargs.get("limit") == 17

    def test_search_passes_collection_name(self) -> None:
        store = self._make_store(collection="my_collection")
        store.search([0.0, 1.0, 0.0, 0.0], k=5)

        _, kwargs = self.mock_client.query_points.call_args
        assert kwargs.get("collection_name") == "my_collection"

    def test_search_empty_result(self) -> None:
        store = self._make_store()
        self.mock_client.query_points.return_value = MagicMock(points=[])

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


class TestQdrantCloudConnectionOptions:
    def test_prefer_grpc_and_timeout_passed_to_client(self, tmp_path: Path) -> None:
        _inject_fake_qdrant()
        try:
            from trelix.core.config import EmbedderConfig, IndexConfig
            from trelix.store.vector_qdrant import QdrantVectorStore

            config = IndexConfig(repo_path=str(tmp_path), embedder=EmbedderConfig())
            config.store.backend = "qdrant"
            config.store.qdrant_prefer_grpc = True
            config.store.qdrant_timeout = 30.0

            fake_module = sys.modules["qdrant_client"]
            QdrantVectorStore(config, dimension=4)

            call_kwargs = fake_module.QdrantClient.call_args.kwargs
            assert call_kwargs["prefer_grpc"] is True
            assert call_kwargs["timeout"] == 30.0
        finally:
            _remove_fake_qdrant()

    def test_defaults_preserve_current_behavior(self, tmp_path: Path) -> None:
        _inject_fake_qdrant()
        try:
            from trelix.core.config import EmbedderConfig, IndexConfig
            from trelix.store.vector_qdrant import QdrantVectorStore

            config = IndexConfig(repo_path=str(tmp_path), embedder=EmbedderConfig())
            config.store.backend = "qdrant"

            fake_module = sys.modules["qdrant_client"]
            QdrantVectorStore(config, dimension=4)

            call_kwargs = fake_module.QdrantClient.call_args.kwargs
            assert call_kwargs["prefer_grpc"] is False
            assert call_kwargs["timeout"] == 10.0
        finally:
            _remove_fake_qdrant()


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


# ---------------------------------------------------------------------------
# QdrantVectorStore — quantization: create_collection wiring
# ---------------------------------------------------------------------------


class TestQdrantQuantizationCreateCollection:
    def setup_method(self) -> None:
        self.mock_client = _inject_fake_qdrant()

    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def _make_store(self, **config_kwargs: Any) -> Any:
        from trelix.store.vector_qdrant import QdrantVectorStore

        return QdrantVectorStore(_make_config(**config_kwargs), dimension=4)

    def test_quantization_none_produces_byte_identical_create_collection_call(self) -> None:
        """Regression: qdrant_quantization=None (default) must not add a
        `quantization_config` kwarg to create_collection at all -- the call
        must be identical to the pre-quantization call."""
        self._make_store(qdrant_quantization=None)

        self.mock_client.create_collection.assert_called_once()
        _, kwargs = self.mock_client.create_collection.call_args
        assert set(kwargs.keys()) == {"collection_name", "vectors_config"}
        assert "quantization_config" not in kwargs

    def test_quantization_int8_passes_scalar_quantization_config(self) -> None:
        """qdrant_quantization='int8' must produce a ScalarQuantization(scalar=...)
        with type=INT8, passed as create_collection's quantization_config kwarg."""
        self._make_store(qdrant_quantization="int8")

        _, kwargs = self.mock_client.create_collection.call_args
        quant = kwargs["quantization_config"]
        assert type(quant).__name__ == "ScalarQuantization"
        assert quant.scalar.type == "int8"

    def test_quantization_binary_passes_binary_quantization_config(self) -> None:
        """qdrant_quantization='binary' must produce a BinaryQuantization(binary=...),
        passed as create_collection's quantization_config kwarg."""
        self._make_store(qdrant_quantization="binary")

        _, kwargs = self.mock_client.create_collection.call_args
        quant = kwargs["quantization_config"]
        assert type(quant).__name__ == "BinaryQuantization"
        assert quant.binary is not None

    def test_quantization_config_does_not_affect_vectors_config(self) -> None:
        """Turning on quantization must not change the HNSW/vector params."""
        self._make_store(qdrant_quantization="int8")

        _, kwargs = self.mock_client.create_collection.call_args
        vparams = kwargs["vectors_config"]
        assert vparams.size == 4
        assert vparams.hnsw_config.m == 16
        assert vparams.hnsw_config.ef_construct == 200


# ---------------------------------------------------------------------------
# QdrantVectorStore — quantization: search() wiring
# ---------------------------------------------------------------------------


class TestQdrantQuantizationSearch:
    def setup_method(self) -> None:
        self.mock_client = _inject_fake_qdrant()

    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def _make_store(self, **config_kwargs: Any) -> Any:
        from trelix.store.vector_qdrant import QdrantVectorStore

        return QdrantVectorStore(_make_config(**config_kwargs), dimension=4)

    def test_search_without_quantization_passes_no_search_params(self) -> None:
        """Regression: search() must not add a `search_params` kwarg at all
        when qdrant_quantization is unset -- byte-identical to the
        pre-quantization call."""
        store = self._make_store(qdrant_quantization=None)
        store.search([0.1, 0.2, 0.3, 0.4], k=5)

        _, kwargs = self.mock_client.query_points.call_args
        assert set(kwargs.keys()) == {"collection_name", "query", "limit"}
        assert "search_params" not in kwargs

    def test_search_with_quantization_passes_search_params_with_default_rescore_true(
        self,
    ) -> None:
        """qdrant_quantization set (rescore left at its True default) must pass
        search_params=SearchParams(quantization=QuantizationSearchParams(rescore=True))."""
        store = self._make_store(qdrant_quantization="int8")
        store.search([0.1, 0.2, 0.3, 0.4], k=5)

        _, kwargs = self.mock_client.query_points.call_args
        search_params = kwargs["search_params"]
        assert search_params.quantization.rescore is True

    def test_search_with_quantization_rescore_false_is_forwarded(self) -> None:
        """qdrant_quantization_rescore=False must be forwarded verbatim -- this is
        a real user-facing tradeoff (recall for max speed), not clamped to True."""
        store = self._make_store(qdrant_quantization="binary", qdrant_quantization_rescore=False)
        store.search([0.1, 0.2, 0.3, 0.4], k=5)

        _, kwargs = self.mock_client.query_points.call_args
        search_params = kwargs["search_params"]
        assert search_params.quantization.rescore is False

    def test_search_still_forwards_collection_query_and_limit_when_quantized(self) -> None:
        """Turning on quantization must not disturb the other search() args."""
        store = self._make_store(qdrant_quantization="int8", qdrant_collection="my_coll")
        store.search([0.0, 1.0, 0.0, 0.0], k=9)

        _, kwargs = self.mock_client.query_points.call_args
        assert kwargs["collection_name"] == "my_coll"
        assert kwargs["query"] == [0.0, 1.0, 0.0, 0.0]
        assert kwargs["limit"] == 9


# ---------------------------------------------------------------------------
# QdrantVectorStore — existing-collection quantization mismatch warning
# ---------------------------------------------------------------------------


class TestQdrantQuantizationMismatchWarning:
    def setup_method(self) -> None:
        self.mock_client = _inject_fake_qdrant()
        # Force the "collection already exists" branch of _ensure_collection.
        self.mock_client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name="trelix")]
        )
        # MagicMock(name=...) sets the mock's repr name, not a `.name` attribute --
        # set it explicitly so `{c.name for c in ...}` sees "trelix".
        self.mock_client.get_collections.return_value.collections[0].name = "trelix"

    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def _make_store(self, **config_kwargs: Any) -> Any:
        from trelix.store.vector_qdrant import QdrantVectorStore

        return QdrantVectorStore(_make_config(**config_kwargs), dimension=4)

    def _set_existing_quantization(self, quantization_config: Any) -> None:
        self.mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(quantization_config=quantization_config)
        )

    def test_no_warning_when_neither_side_has_quantization(self, caplog: Any) -> None:
        self._set_existing_quantization(None)

        with caplog.at_level("WARNING"):
            self._make_store(qdrant_quantization=None)

        assert not any("quantization" in rec.message.lower() for rec in caplog.records)
        self.mock_client.create_collection.assert_not_called()

    def test_warns_when_existing_collection_unquantized_but_int8_configured(
        self, caplog: Any
    ) -> None:
        self._set_existing_quantization(None)

        with caplog.at_level("WARNING"):
            self._make_store(qdrant_quantization="int8")

        messages = [rec.message for rec in caplog.records]
        assert any("quantization" in m.lower() and "trelix" in m for m in messages)

    def test_no_warning_or_get_collection_when_existing_has_int8_but_config_unset(
        self, caplog: Any
    ) -> None:
        """When qdrant_quantization is unset, the mismatch check is skipped
        entirely (no get_collection() call, no warning) even if the existing
        collection happens to have quantization from a prior run -- the check
        only matters when the user has actually configured something. This is
        an intentional trade-off (see `_ensure_collection`): it trades away
        detecting this one reverse-mismatch direction for zero overhead on the
        much more common already-created, no-quantization-configured path."""
        # No import needed: _quantization_kind() only ever checks hasattr(x,
        # "scalar")/hasattr(x, "binary") on the real object (see its own
        # docstring), so a bare MagicMock with a `.scalar` attribute is
        # sufficient to simulate an existing int8-quantized collection.
        self._set_existing_quantization(MagicMock(scalar=MagicMock()))

        with caplog.at_level("WARNING"):
            self._make_store(qdrant_quantization=None)

        assert not any("quantization" in rec.message.lower() for rec in caplog.records)
        self.mock_client.get_collection.assert_not_called()

    def test_no_warning_when_int8_configured_matches_existing_int8(self, caplog: Any) -> None:
        # No import needed -- see the identical note on
        # test_no_warning_or_get_collection_when_existing_has_int8_but_config_unset above.
        self._set_existing_quantization(MagicMock(scalar=MagicMock()))

        with caplog.at_level("WARNING"):
            self._make_store(qdrant_quantization="int8")

        assert not any("quantization" in rec.message.lower() for rec in caplog.records)

    def test_no_create_collection_call_when_collection_already_exists(self) -> None:
        """The collection-exists branch must never call create_collection --
        confirms the mismatch check truly no-ops the write path."""
        self._set_existing_quantization(None)
        self._make_store(qdrant_quantization="int8")

        self.mock_client.create_collection.assert_not_called()

    def test_no_get_collection_call_when_quantization_unset_and_collection_exists(
        self,
    ) -> None:
        """Regression: when qdrant_quantization is unset, the mismatch check is
        meaningless (there is nothing to mismatch), so __init__ must not pay for
        a get_collection() network round-trip on the common (already-created
        collection, no quantization configured) hot path."""
        self._set_existing_quantization(None)

        self._make_store(qdrant_quantization=None)

        self.mock_client.get_collection.assert_not_called()

    def test_get_collection_failure_does_not_propagate_when_quantization_configured(
        self, caplog: Any
    ) -> None:
        """Regression: a get_collection() failure (timeout, transient outage,
        write-only API key, a just-deleted collection) must not turn
        constructing the store into a hard failure -- it should log and
        continue, matching the pre-existing-collection-is-always-a-no-op-success
        behavior."""
        self.mock_client.get_collection.side_effect = RuntimeError("boom")

        with caplog.at_level("WARNING"):
            store = self._make_store(qdrant_quantization="int8")

        assert store is not None
        assert any("quantization" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# QdrantVectorStore — recreate()
# ---------------------------------------------------------------------------


class TestQdrantRecreate:
    def setup_method(self) -> None:
        self.mock_client = _inject_fake_qdrant()

    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def _make_store(self, **config_kwargs: Any) -> Any:
        from trelix.store.vector_qdrant import QdrantVectorStore

        return QdrantVectorStore(_make_config(**config_kwargs), dimension=4)

    def test_recreate_calls_delete_then_create_collection(self) -> None:
        store = self._make_store(qdrant_collection="my_coll")
        # First create_collection call happened during __init__ (collection did
        # not exist yet, per the default fake get_collections()).
        assert self.mock_client.create_collection.call_count == 1

        store.recreate()

        self.mock_client.delete_collection.assert_called_once_with(collection_name="my_coll")
        assert self.mock_client.create_collection.call_count == 2
        _, kwargs = self.mock_client.create_collection.call_args
        assert kwargs["collection_name"] == "my_coll"

    def test_recreate_picks_up_currently_configured_quantization(self) -> None:
        """recreate() must build quantization_config from whatever is
        CURRENTLY configured, not whatever the collection had before."""
        store = self._make_store(qdrant_quantization="binary")

        store.recreate()

        _, kwargs = self.mock_client.create_collection.call_args
        assert type(kwargs["quantization_config"]).__name__ == "BinaryQuantization"

    def test_recreate_returns_none(self) -> None:
        """Matches BaseVectorStore.recreate()'s contract: no return value."""
        store = self._make_store()
        assert store.recreate() is None

    def test_store_usable_for_upsert_and_search_immediately_after_recreate(self) -> None:
        """After recreate(), the store must still be able to upsert and search --
        i.e. it is left pointing at a real, usable collection, not a deleted one."""
        store = self._make_store()
        store.recreate()

        store.upsert_batch([(1, [0.1, 0.2, 0.3, 0.4])])
        self.mock_client.upsert.assert_called_once()

        self.mock_client.query_points.return_value = MagicMock(points=[MagicMock(id=1, score=0.99)])
        results = store.search([0.1, 0.2, 0.3, 0.4], k=5)
        assert results == [(1, 0.99)]
