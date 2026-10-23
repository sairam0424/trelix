"""When the sqlite-vec store has grown past what an exact flat scan can answer quickly.

The `sqlite` backend is an exact flat scan: every search compares the query with every
stored vector, so its cost is linear in rows x width. sqlite-vec has never shipped an HNSW or
any other approximate index, which means there is no setting that changes this and the only
remedy is a different backend (`TRELIX_STORE_BACKEND=lance` or `qdrant`).

The thresholds below are MEASURED, not guessed: the row counts at which the p95 latency of
a k-nearest query passes 100 ms at 384, 768 and 1024 dimensions, read out of the benchmark
report named by `ADVISORY_SOURCE_REPORT` (`scripts/bench_vector_search.py` writes it).
`tests/unit/test_scale_advisory.py` pins the table to that file, so the two cannot drift.
They describe the machine that ran the benchmark, which is why the advisory says "near".

Pure functions, no I/O and no config import: `trelix.cli.main` reads the row count and calls
`flat_scan_advisory`; this module only decides whether, and what to say.
"""

from __future__ import annotations

from collections.abc import Mapping

# Repo-relative path of the committed report the table below was read from.
ADVISORY_SOURCE_REPORT = "docs/reports/vector-search-bench-2026-10-07-apple-m4-shared.json"
ADVISORY_P95_MS = 100.0

# Width (dimensions) -> rows at which measured p95 search latency passes ADVISORY_P95_MS.
FLAT_SCAN_P95_100MS_ROWS: dict[int, int] = {
    384: 32_765,  # measured
    768: 12_979,  # measured
    1024: 28_956,  # measured
}


def flat_scan_threshold_rows(dimension: int, table: Mapping[int, int] | None = None) -> int:
    """Rows past which a flat scan at `dimension` passes the latency target.

    The smallest measured width at or above `dimension` applies. Rounding up is conservative
    only while the table is monotone (a narrower index scans no slower than the width it is
    rounded up to); the committed table is not, because the 768 cell was load-depressed
    (see the Storage section of docs/CONFIGURATION.md). Above the widest measured width the
    widest threshold is scaled down in proportion.
    """
    # bound at call time so a monkeypatch of the module constant reaches the CLI call sites
    table = FLAT_SCAN_P95_100MS_ROWS if table is None else table
    if dimension < 1:
        raise ValueError(f"dimension must be at least 1, got {dimension}")
    if not table:
        raise ValueError("the flat-scan threshold table is empty")
    widths_at_or_above = [width for width in table if width >= dimension]
    if widths_at_or_above:
        return table[min(widths_at_or_above)]
    widest = max(table)
    return table[widest] * widest // dimension


def flat_scan_advisory(
    rows: int, dimension: int, table: Mapping[int, int] | None = None
) -> str | None:
    """The advisory line for an index of `rows` vectors, or None when it is within range."""
    threshold = flat_scan_threshold_rows(dimension, table)
    if rows <= threshold:
        return None
    return (
        f"Flat-scan advisory: this index holds {rows:,} vectors at {dimension} dimensions. "
        "sqlite-vec is an exact flat scan, and measured p95 search latency passes "
        f"{ADVISORY_P95_MS:.0f} ms near {threshold:,} rows at this width "
        f"(see {ADVISORY_SOURCE_REPORT}). "
        "Consider TRELIX_STORE_BACKEND=lance or qdrant; see docs/CONFIGURATION.md#storage."
    )
