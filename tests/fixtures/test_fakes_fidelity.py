"""Fake-fidelity gate: proves ``FakeEmbedder``/``FakeVectorStore`` (fakes.py)
actually implement the REAL ``BaseEmbedder``/``BaseVectorStore`` ABCs --
same required methods, same parameter names/order -- and proves the ABC
mechanism itself is what a missing method would trip, not an assumption
about it.

Why this file exists rather than trusting "it subclasses the ABC, done":
subclassing alone only proves every ``@abstractmethod`` got SOME
implementation. It does not prove the implementation's signature still
matches the contract callers depend on -- a fake could override
``def search(self, *args, **kwargs)`` and satisfy ``ABCMeta`` while being
silently incompatible with every real caller. The signature-parity check
below is the part ``ABCMeta`` does NOT do for you.
"""

from __future__ import annotations

import inspect

import pytest

from tests.fixtures.fakes import FakeEmbedder, FakeVectorStore
from trelix.embedder.base import BaseEmbedder
from trelix.store.vector import BaseVectorStore

# Explicit table (rule 2): which base class's abstractmethods a fake must
# mirror. Not derived by iterating something else — each pair is what this
# file exists to pin, both ways (rule 2's set-equality requirement, adapted
# to methods rather than a plain collection).
_FAKE_TO_BASE: dict[type, type] = {
    FakeEmbedder: BaseEmbedder,
    FakeVectorStore: BaseVectorStore,
}


def _param_names(cls: type, method_name: str) -> tuple[str, ...]:
    """Parameter names (excluding `self`) for `cls.method_name`.

    `dimension` is a `@property` on both BaseEmbedder and FakeEmbedder --
    `inspect.signature` cannot introspect a `property` object directly, so
    this unwraps `.fget` for that case rather than special-casing it away
    entirely (a property IS a zero-argument method for this comparison's
    purposes, and asserting that explicitly is the point).
    """
    attr = inspect.getattr_static(cls, method_name)
    target = attr.fget if isinstance(attr, property) else attr
    sig = inspect.signature(target)
    return tuple(p for p in sig.parameters if p != "self")


class TestFakeImplementsEveryAbstractMethod:
    """FALSIFIED BY: a fake missing one of the base class's abstractmethods --
    caught by ABCMeta at class-definition/instantiation time, re-asserted
    explicitly here rather than only inferred from a successful `import`."""

    def test_fake_embedder_has_zero_unimplemented_abstractmethods(self) -> None:
        assert FakeEmbedder.__abstractmethods__ == frozenset()

    def test_fake_vector_store_has_zero_unimplemented_abstractmethods(self) -> None:
        assert FakeVectorStore.__abstractmethods__ == frozenset()

    def test_fake_embedder_can_be_constructed(self) -> None:
        """A non-empty __abstractmethods__ makes instantiation raise TypeError
        BEFORE this line -- so a passing construction is itself evidence,
        not just the frozenset check above."""
        FakeEmbedder(dimension=4)

    def test_fake_vector_store_can_be_constructed(self) -> None:
        FakeVectorStore(dimension=4)


class TestSignatureParity:
    """Same required-method NAMES, same PARAMETER NAMES/ORDER, both ways
    (rule 2) -- the check ABCMeta itself does not perform."""

    def test_every_abstractmethod_the_base_declares_is_on_the_fake_with_matching_params(
        self,
    ) -> None:
        mismatches: list[str] = []
        for fake_cls, base_cls in _FAKE_TO_BASE.items():
            for name in sorted(base_cls.__abstractmethods__):
                base_params = _param_names(base_cls, name)
                fake_params = _param_names(fake_cls, name)
                if base_params != fake_params:
                    mismatches.append(
                        f"{fake_cls.__name__}.{name}{fake_params} != "
                        f"{base_cls.__name__}.{name}{base_params}"
                    )
        assert mismatches == [], "\n".join(mismatches)

    def test_the_fake_does_not_silently_diverge_from_the_current_abstract_set(
        self,
    ) -> None:
        """The inverse direction (rule 2, both ways): every abstractmethod
        name the fakes claim to implement must still BE an abstractmethod on
        the base today -- guards against the base dropping/renaming a method
        while a stale fake keeps an orphaned override nobody calls anymore."""
        base_embedder_names = BaseEmbedder.__abstractmethods__
        base_store_names = BaseVectorStore.__abstractmethods__
        fake_embedder_overrides = {
            "embed",
            "embed_query",
            "dimension",
        }
        fake_store_overrides = {
            "upsert_batch",
            "search",
            "delete_batch",
            "count",
            "upsert_file_summary_embedding",
            "search_file_summaries",
            "upsert_sub_chunk_embedding",
            "search_sub_chunks",
        }
        assert fake_embedder_overrides == base_embedder_names
        assert fake_store_overrides == base_store_names


class TestMissingMethodIsATypeErrorAtConstruction:
    """Proves the CLAIM this whole module rests on: an ABC subclass missing
    one required method fails at __init__, not at the first call site that
    happens to touch it -- a plain class or a Mock would not catch this
    (rule 3's whole point). Built via a LOCAL, throwaway subclass so the
    proof does not require mutating the real fakes.py file on disk to run
    every time this suite runs."""

    def test_embedder_missing_dimension_raises_type_error_not_attribute_error(self) -> None:
        class _IncompleteEmbedder(BaseEmbedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return []

            def embed_query(self, text: str) -> list[float]:
                return []

            # `dimension` deliberately omitted.

        with pytest.raises(TypeError, match="dimension"):
            _IncompleteEmbedder()  # type: ignore[abstract]

    def test_vector_store_missing_search_raises_type_error_not_attribute_error(self) -> None:
        class _IncompleteStore(BaseVectorStore):
            def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
                pass

            def delete_batch(self, chunk_ids: list[int]) -> None:
                pass

            def count(self) -> int:
                return 0

            def upsert_file_summary_embedding(self, file_id: int, embedding: list[float]) -> None:
                pass

            def search_file_summaries(
                self, query_embedding: list[float], k: int
            ) -> list[tuple[int, float]]:
                return []

            def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
                pass

            def search_sub_chunks(
                self, query_embedding: list[float], k: int
            ) -> list[tuple[int, float]]:
                return []

            # `search` deliberately omitted.

        with pytest.raises(TypeError, match="search"):
            _IncompleteStore()  # type: ignore[abstract]
