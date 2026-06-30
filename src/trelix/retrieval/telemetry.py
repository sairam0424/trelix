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
    from trelix.store.db import Database

logger = logging.getLogger("trelix.retrieval.telemetry")


class TelemetryWriter:
    """Write query telemetry to the query_telemetry table. Never raises."""

    def __init__(self, db: Database, enabled: bool = True) -> None:
        self._db = db
        self._enabled = enabled

    def record(self, context: RetrievedContext, elapsed_ms: float) -> None:
        """Record a single retrieve() call. No-op when disabled."""
        if not self._enabled:
            return
        try:
            query = str(getattr(context, "query", ""))
            intent = str(getattr(context, "intent", ""))
            results = getattr(context, "results", [])
            result_count = len(results) if hasattr(results, "__len__") else 0
            self._db.insert_query_telemetry(
                query=query,
                intent=intent,
                elapsed_ms=elapsed_ms,
                result_count=result_count,
            )
        except Exception as exc:
            logger.debug("Telemetry record failed (non-fatal): %s", exc)
