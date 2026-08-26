"""Close store.vector's 3 diagnosed gaps (round 10 commit 1b02e44, store.vector section):

    * `_is_chunk_id`'s own boundary at chunk_id=0 and chunk_id==_SUB_CHUNK_OFFSET
      -- both untested, 2 mutants.
    * `search`'s default k value (20) -- untested because every existing call
      site in every test passes k explicitly, 1 mutant.

`_is_chunk_id` is `0 < chunk_id < cls._SUB_CHUNK_OFFSET` (vector.py). Both ends
are open (strict `<`), so both boundary values themselves must be False, not
True -- that is exactly the FALSIFYING mutation each test below names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.store.vector import BaseVectorStore, SQLiteVectorStore

_DIM = 4
# Used only to SELECT which input to probe (the sentinel boundary itself), never to
# recompute the expected boolean -- that stays a literal below (rule 1).
_OFFSET = BaseVectorStore._SUB_CHUNK_OFFSET


class TestIsChunkIdBoundary:
    """`BaseVectorStore._is_chunk_id` is a classmethod: no instance needed."""

    def test_zero_is_not_a_real_chunk_id(self) -> None:
        """FALSIFIED BY: `0 < chunk_id` mutated to `0 <= chunk_id`, which would make
        chunk_id=0 register as a real chunk vector instead of the sentinel-adjacent
        boundary it actually is."""
        assert SQLiteVectorStore._is_chunk_id(0) is False

    def test_one_is_a_real_chunk_id(self) -> None:
        """Precondition for the test above: if the left boundary comparison were
        broken in the OTHER direction (e.g. `chunk_id < cls._SUB_CHUNK_OFFSET`
        alone, dropping the lower bound entirely), 0 would already be rejected by
        this method for an unrelated reason and the mutation above would stop
        discriminating. Pin the immediate next integer as True to rule that out."""
        assert SQLiteVectorStore._is_chunk_id(1) is True

    def test_sub_chunk_offset_itself_is_not_a_real_chunk_id(self) -> None:
        """FALSIFIED BY: `chunk_id < cls._SUB_CHUNK_OFFSET` mutated to
        `chunk_id <= cls._SUB_CHUNK_OFFSET`, which would make chunk_id ==
        _SUB_CHUNK_OFFSET (10,000,000 -- the first sub-chunk sentinel, sub_chunk_id=0)
        register as a real chunk vector."""
        assert SQLiteVectorStore._is_chunk_id(_OFFSET) is False

    def test_one_below_sub_chunk_offset_is_a_real_chunk_id(self) -> None:
        """Precondition for the test above: if the upper bound comparison were
        broken in the OTHER direction (e.g. `0 < chunk_id` alone, dropping the
        upper bound entirely), _SUB_CHUNK_OFFSET would already be accepted by this
        method for an unrelated reason and the mutation above would stop
        discriminating. Pin the immediate prior integer as True to rule that out."""
        assert SQLiteVectorStore._is_chunk_id(_OFFSET - 1) is True


class TestSearchDefaultK:
    """`SQLiteVectorStore.search`'s `k` parameter defaults to 20."""

    def test_search_without_an_explicit_k_returns_exactly_twenty(self, tmp_path: Path) -> None:
        """FALSIFIED BY: `def search(self, query_embedding, k: int = 20)` mutated to
        any other default (e.g. `k: int = 21`) -- every other test in this suite
        passes `k` explicitly, so only THIS call, which omits it, exercises the
        literal default value at all.
        """
        store = SQLiteVectorStore(tmp_path / "vectors.db", dimension=_DIM)
        store.upsert_batch([(i, [float(i)] * _DIM) for i in range(1, 26)])

        # Precondition: more real chunks than the k=20 default, and none of them a
        # sentinel -- otherwise a truncation to *any* value at or above 20 would
        # look identical and the assertion below would stop discriminating.
        assert store.count() == 25, (
            "fixture must hold more rows than the k=20 default for the truncation "
            "below to mean anything"
        )

        results = store.search([1.0] * _DIM)  # k intentionally omitted

        assert len(results) == 20, (
            f"expected exactly 20 results from search()'s k=20 default, got {len(results)}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
