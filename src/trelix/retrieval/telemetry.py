"""
Query telemetry writer — records retrieve() calls to the query_telemetry table.

Off by default (telemetry_enabled=False). When enabled, every retrieve() call
appends one row: query text, intent, elapsed_ms, result count.
Used for: debugging slow queries, tracking improvement over time, computing
nDCG@10 against a golden set via `trelix eval`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.models import RetrievedContext
    from trelix.retrieval.query_expansion import ExpandResult
    from trelix.store.db import Database

logger = logging.getLogger("trelix.retrieval.telemetry")


class TelemetryWriter:
    """Write query telemetry to the query_telemetry table. Never raises."""

    def __init__(self, db: Database, enabled: bool = True) -> None:
        self._db = db
        self._enabled = enabled

    def record(
        self,
        context: RetrievedContext,
        elapsed_ms: float,
        expansion_result: ExpandResult | None = None,
    ) -> None:
        """Record a single retrieve() call. No-op when disabled. Never raises."""
        if not self._enabled:
            return
        try:
            query = str(getattr(context, "query", ""))
            intent = str(getattr(context, "intent", ""))
            results = getattr(context, "results", [])
            result_count = len(results) if hasattr(results, "__len__") else 0

            exp_used: bool | None = None
            exp_variants: int | None = None
            exp_elapsed: float | None = None
            if expansion_result is not None:
                exp_used = expansion_result.llm_used
                exp_variants = len(expansion_result.queries)
                exp_elapsed = expansion_result.elapsed_ms

            self._db.insert_query_telemetry(
                query=query,
                intent=intent,
                elapsed_ms=elapsed_ms,
                result_count=result_count,
                expansion_used=exp_used,
                expansion_variants=exp_variants,
                expansion_elapsed_ms=exp_elapsed,
            )
        except Exception as exc:
            logger.debug("Telemetry record failed (non-fatal): %s", exc)
