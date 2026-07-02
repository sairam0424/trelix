"""
Embedding dimension mismatch guard.

When a user switches embedding providers (e.g. Azure 3072-dim -> local 384-dim),
the existing HNSW index contains vectors of the wrong dimension. trelix would
silently return wrong results or crash with a cryptic sqlite-vec error.

DimensionGuard:
1. Records the dimension at the end of a successful index run (DimensionGuard.record)
2. Checks stored vs current dimension at Retriever/Indexer startup (DimensionGuard.check)
3. Raises DimensionMismatchError with a clear migration hint if they differ
4. Provides reset() to clear stored dimension when the user re-indexes

Migration workflow:
    trelix migrate-vectors --reset   # clear old embeddings
    trelix index ./my-repo           # re-index with new provider
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.store.db import Database

logger = logging.getLogger("trelix.store.dimension_guard")


class DimensionMismatchError(Exception):
    """Raised when the current embedding dimension differs from the stored one."""

    def __init__(self, stored: int, current: int, provider: str) -> None:
        self.stored = stored
        self.current = current
        self.provider = provider
        super().__init__(
            f"Embedding dimension mismatch: index was built with {stored}-dim vectors "
            f"but the current provider '{provider}' produces {current}-dim vectors.\n\n"
            f"Fix: run `trelix migrate-vectors --reset ./your-repo` to clear the old "
            f"embeddings, then re-index: `trelix index ./your-repo`\n\n"
            f"Note: --reset deletes all stored embeddings. You must re-run trelix index."
        )


class DimensionGuard:
    """Static methods for dimension mismatch detection and recovery."""

    @staticmethod
    def check(db: Database, current_dimension: int, provider: str) -> None:
        """
        Check that the current dimension matches the stored one.

        Raises DimensionMismatchError if there is a mismatch.
        No-op when no dimension is stored yet (first index run).
        No-op on unexpected errors (e.g. corrupted index_metadata or closed connection).
        """
        try:
            stored = db.get_embedding_dimension()
        except Exception:
            logger.warning(
                "DimensionGuard.check(): could not read stored dimension — skipping check",
                exc_info=True,
            )
            return
        if not isinstance(stored, int):
            return  # First run or unexpected return type — no check needed
        if stored != current_dimension:
            raise DimensionMismatchError(
                stored=stored,
                current=current_dimension,
                provider=provider,
            )

    @staticmethod
    def record(db: Database, dimension: int, provider: str) -> None:
        """Record the embedding dimension after a successful index run."""
        db.set_embedding_dimension(dimension)
        logger.debug("DimensionGuard: recorded %d-dim (%s)", dimension, provider)

    @staticmethod
    def reset(db: Database) -> None:
        """Clear the stored dimension record for re-indexing."""
        db.delete_embedding_dimension_key()
        logger.info("DimensionGuard: dimension record cleared — ready for fresh index")
