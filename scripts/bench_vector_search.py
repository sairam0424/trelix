#!/usr/bin/env python3
"""Benchmark the sqlite-vec vec0 flat scan over a grid of widths and row counts.

Why
---
trelix's sqlite store (``src/trelix/store/vector.py``) answers every search with an exact
flat scan, so search cost is linear in rows x width and a row-count threshold is only
meaningful per dimension, and only when it was measured. The flat-scan advisory planned for
``trelix index`` / ``trelix stats`` (branch ``feat/flat-scan-advisory``) will quote the row
count at which the warm p95 search latency passes 100 ms for each width; this script
measures those crossings and writes the JSON report the advisory pins to.

How to run
----------
    python scripts/bench_vector_search.py --workdir <scratch dir outside the repo> --label apple-m4
    python scripts/bench_vector_search.py --quick --workdir <scratch dir> --label apple-m4

``--label`` names the machine class (a chip, never a hostname); default ``platform.machine()``.
``--out`` defaults to ``docs/reports/vector-search-bench-<UTC date>-<label>.json`` under this
repository, whatever the current directory is.

Cost
----
The default grid is 384/768/1024 dimensions x 10k/100k/1M rows. The 1M x 1024 cell alone writes
about 4.1 GB. The design's quiet M-series one-off inserted roughly 47k rows/s at 384 dimensions,
4.3k at 768 and 3.1k at 1024, so the full grid takes 15-25 minutes on a quiet machine and wants
at least 6 GB free in ``--workdir`` (the largest cell's estimate with a 25 % margin, plus SQLite's
``-journal``). One database file is written per cell and deleted when the cell finishes. The
script refuses to start when ``--workdir`` has less free space than the largest cell's estimate
(5,120,000,000 bytes for the default grid). ``--quick`` drops the 1M cells and times 30 queries
per cell instead of 100.

Not a test
----------
Like ``tests/perf/test_query_latency.py`` ("Manual performance test — NOT run in CI") this
is a manual benchmark: pytest and CI never run it, and its report describes one machine on
one day, not a guarantee.

Exit codes
----------
    0  the grid was measured and the report written
    1  sqlite-vec failed to load; --workdir lacks the space or holds an earlier run's cell file
    2  usage error
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import sqlite_vec

LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
THRESHOLD_MS = 100.0
SCHEMA_VERSION = 1
BATCH_ROWS = 5000
QUERY_SQL = "SELECT chunk_id, distance FROM v WHERE embedding MATCH ? ORDER BY distance LIMIT ?"


@dataclass(frozen=True)
class Settings:
    """Validated run parameters."""

    dims: list[int]
    sizes: list[int]
    k: int
    queries: int
    warmup: int
    seed: int
    workdir: Path | None
    out: Path
    label: str
    note: str


def parse_int_list(text: str) -> list[int]:
    """Parse ``"100000,10000"`` into sorted distinct positive ints; ValueError otherwise."""
    values: set[int] = set()
    for item in text.split(","):
        try:
            value = int(item.strip())
        except ValueError:
            raise ValueError(f"not an integer: {item.strip()!r}") from None
        if value < 1:
            raise ValueError(f"not a positive integer: {value}")
        values.add(value)
    return sorted(values)


def make_vectors(rng: np.random.Generator, n: int, dim: int) -> npt.NDArray[np.float32]:
    """Draw ``n`` seeded unit vectors of width ``dim`` (float32 on the draw, not a cast)."""
    draw = rng.standard_normal((n, dim), dtype=np.float32)
    unit: npt.NDArray[np.float32] = draw / np.linalg.norm(draw, axis=1, keepdims=True)
    return unit


def connect(path: Path | str) -> sqlite3.Connection:
    """Open ``path`` with the sqlite-vec extension loaded, as store/vector.py does."""
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def build_table(
    conn: sqlite3.Connection,
    dim: int,
    rows: int,
    rng: np.random.Generator,
    batch: int = BATCH_ROWS,
) -> float:
    """Create ``v`` and insert ``rows`` seeded unit vectors in batches; return insert seconds."""
    conn.execute(
        f"CREATE VIRTUAL TABLE v USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
    )
    started = time.perf_counter()
    inserted = 0
    while inserted < rows:
        count = min(batch, rows - inserted)
        vectors = make_vectors(rng, count, dim)
        conn.executemany(
            "INSERT INTO v(chunk_id, embedding) VALUES (?, ?)",
            zip(range(inserted, inserted + count), (vec.tobytes() for vec in vectors)),
        )
        conn.commit()
        inserted += count
    return time.perf_counter() - started


def time_queries(
    conn: sqlite3.Connection, queries: npt.NDArray[np.float32], k: int, warmup: int
) -> list[float]:
    """Time every row of ``queries`` as a k-NN search; drop the first ``warmup`` timings."""
    ms: list[float] = []
    for row in queries:
        started = time.perf_counter()
        conn.execute(QUERY_SQL, (row.tobytes(), k)).fetchall()
        ms.append((time.perf_counter() - started) * 1000.0)
    return ms[warmup:]


def percentile(sorted_ms: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    index = max(math.ceil(p / 100 * len(sorted_ms)) - 1, 0)
    return sorted_ms[index]


def rows_at_threshold(points: list[tuple[int, float]], threshold_ms: float) -> tuple[int, bool]:
    """Rows where p95 crosses ``threshold_ms``, and whether the crossing was extrapolated.

    A virtual origin ``(0, 0.0)`` precedes the sorted points; the first bracket
    ``p95[i] < T <= p95[i+1]`` is interpolated and ``extrapolated`` is True exactly when its
    lower point is the origin. Without a bracket, project from the last one or two points.
    """
    measured = sorted(points)
    if any(p95 <= 0.0 for _, p95 in measured):
        raise ValueError("p95 values must be positive")
    rows = [0, *(r for r, _ in measured)]
    p95 = [0.0, *(p for _, p in measured)]
    for i in range(len(measured)):
        if p95[i] < threshold_ms <= p95[i + 1]:
            span = rows[i + 1] - rows[i]
            crossing = rows[i] + int((threshold_ms - p95[i]) * span // (p95[i + 1] - p95[i]))
            return crossing, i == 0
    last_rows, last_p95 = measured[-1]
    if len(measured) == 1:
        return int(threshold_ms * last_rows // last_p95), True
    slope = (last_p95 - measured[-2][1]) / (last_rows - measured[-2][0])
    if slope <= 0:
        return int(threshold_ms * last_rows // last_p95), True
    return last_rows + int((threshold_ms - last_p95) / slope), True


def estimate_cell_bytes(dim: int, rows: int) -> int:
    """Scratch bytes one cell needs: raw float32 bytes plus 25 % (files measure 1.01-1.04x)."""
    return rows * dim * 4 * 5 // 4


def platform_info(conn: sqlite3.Connection) -> dict[str, object]:
    """Platform facts for the report: never a hostname, a user name or a path."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "sqlite_vec": sqlite_vec.__version__,
        "vec_version": conn.execute("SELECT vec_version()").fetchone()[0],
        "cpu_count": os.cpu_count(),
    }


def load_average() -> list[float] | None:
    """The 1/5/15-minute load averages, or None where the platform has none."""
    if not hasattr(os, "getloadavg"):
        return None
    return list(os.getloadavg())


def run_cell(
    dim: int,
    rows: int,
    *,
    k: int,
    queries: int,
    warmup: int,
    seed: int,
    workdir: Path,
) -> tuple[dict[str, object], float]:
    """Measure one (dim, rows) cell; return its result row and the raw (unrounded) p95."""
    rng = np.random.default_rng(seed)
    db_path = workdir / f"bench-{dim}-{rows}.db"
    if db_path.exists():  # an earlier run's cell: never open, measure or delete a file not ours
        raise SystemExit(f"error: {db_path} already exists; remove it or choose another --workdir")
    try:
        conn = connect(db_path)
        try:
            insert_seconds = build_table(conn, dim, rows, rng)
            query_vectors = make_vectors(rng, queries + warmup, dim)
            ms = sorted(time_queries(conn, query_vectors, k, warmup))
        finally:
            conn.close()
        db_bytes = db_path.stat().st_size
    finally:
        for stale in (db_path, Path(f"{db_path}-journal")):
            stale.unlink(missing_ok=True)
    raw_p95 = percentile(ms, 95)
    result: dict[str, object] = {
        "dim": dim,
        "rows": rows,
        "insert_seconds": round(insert_seconds, 3),
        "insert_rows_per_s": int(rows / max(insert_seconds, 1e-9)),
        "db_bytes": db_bytes,
        "warm_p50_ms": round(percentile(ms, 50), 2),
        "warm_p95_ms": round(raw_p95, 2),
        "warm_p99_ms": round(percentile(ms, 99), 2),
    }
    return result, raw_p95


def build_parser() -> argparse.ArgumentParser:
    """The command line; defaults that depend on ``--quick`` are resolved later."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dims", default="384,768,1024", help="comma-separated vector widths")
    parser.add_argument("--sizes", help="row counts (default 10000,100000,1000000)")
    parser.add_argument("--k", type=int, default=20, help="neighbours per query")
    parser.add_argument("--queries", type=int, help="timed queries per cell (default 100)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed queries before the timed")
    parser.add_argument("--seed", type=int, default=0, help="seed for vectors and queries")
    parser.add_argument("--workdir", type=Path, help="existing scratch directory for cell files")
    parser.add_argument("--out", type=Path, help="report path (default docs/reports/, see above)")
    parser.add_argument("--label", default=platform.machine(), help="machine class, not a hostname")
    parser.add_argument("--note", default="", help="free text recorded as notes.comment")
    parser.add_argument("--quick", action="store_true", help="sizes 10000,100000 and 30 queries")
    return parser


def default_out_path(label: str, today: date) -> Path:
    """The report path beside this repository's docs/, never relative to the cwd."""
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "reports"
        / f"vector-search-bench-{today.isoformat()}-{label}.json"
    )


def _resolve_grid(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[list[int], list[int], int]:
    """Parse --dims/--sizes and apply the --quick preset where no explicit flag was given."""
    try:
        dims = parse_int_list(args.dims)
    except ValueError:
        parser.error(
            f"--dims must be a comma-separated list of positive integers, got {args.dims!r}"
        )
    if args.quick:
        sizes, queries = [10_000, 100_000], 30
    else:
        sizes, queries = [10_000, 100_000, 1_000_000], 100
    if args.sizes is not None:
        try:
            sizes = parse_int_list(args.sizes)
        except ValueError:
            parser.error(
                f"--sizes must be a comma-separated list of positive integers, got {args.sizes!r}"
            )
    if args.queries is not None:
        queries = args.queries
    return dims, sizes, queries


def _validate(parser: argparse.ArgumentParser, settings: Settings) -> None:
    """Usage checks in a fixed order; each failure is ``parser.error`` (exit 2)."""
    if settings.k < 1:
        parser.error("--k must be at least 1")
    if settings.queries < 1:
        parser.error("--queries must be at least 1")
    if settings.warmup < 0:
        parser.error("--warmup must not be negative")
    if settings.seed < 0:
        parser.error("--seed must not be negative")
    if settings.k > min(settings.sizes):
        parser.error("--k must not exceed the smallest --sizes value")
    if LABEL_PATTERN.fullmatch(settings.label) is None:
        parser.error("--label must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if not settings.out.parent.is_dir():
        parser.error(f"--out parent directory does not exist: {settings.out.parent}")
    if settings.out.is_dir():
        parser.error(f"--out must be a file path, not a directory: {settings.out}")
    if settings.workdir is not None and not settings.workdir.is_dir():
        parser.error(f"--workdir must be an existing directory: {settings.workdir}")


def resolve_settings(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, today: date
) -> Settings:
    """Turn parsed flags into validated ``Settings``; usage errors exit 2 via the parser."""
    dims, sizes, queries = _resolve_grid(parser, args)
    out = default_out_path(args.label, today) if args.out is None else args.out
    settings = Settings(
        dims=dims,
        sizes=sizes,
        k=args.k,
        queries=queries,
        warmup=args.warmup,
        seed=args.seed,
        workdir=args.workdir,
        out=out,
        label=args.label,
        note=args.note,
    )
    _validate(parser, settings)
    return settings


def check_disk(workdir: Path, dims: list[int], sizes: list[int]) -> tuple[int, str | None]:
    """Free bytes in ``workdir`` and a refusal line when the largest cell would not fit."""
    free = shutil.disk_usage(workdir).free
    dim, rows = max(dims), max(sizes)
    estimate = estimate_cell_bytes(dim, rows)
    if free >= estimate:
        return free, None
    return free, (
        f"error: --workdir {workdir} has {free:,} bytes free; the largest cell "
        f"({dim} x {rows:,} rows) needs about {estimate:,} bytes"
    )


def run_grid(
    settings: Settings, workdir: Path
) -> tuple[list[dict[str, object]], dict[int, list[tuple[int, float]]]]:
    """Run every cell, dims then sizes ascending; print a line per cell; collect p95 points."""
    results: list[dict[str, object]] = []
    points: dict[int, list[tuple[int, float]]] = {}
    for dim in settings.dims:
        for rows in settings.sizes:
            result, raw_p95 = run_cell(
                dim,
                rows,
                k=settings.k,
                queries=settings.queries,
                warmup=settings.warmup,
                seed=settings.seed,
                workdir=workdir,
            )
            results.append(result)
            points.setdefault(dim, []).append((rows, raw_p95))
            print(
                f"dim={dim} rows={rows:,} insert_s={result['insert_seconds']:.2f} "
                f"p50_ms={result['warm_p50_ms']:.2f} p95_ms={result['warm_p95_ms']:.2f}",
                flush=True,
            )
    return results, points


def build_report(
    settings: Settings,
    platform_facts: dict[str, object],
    results: list[dict[str, object]],
    points: dict[int, list[tuple[int, float]]],
    *,
    generated_at: str,
    load_start: list[float] | None,
    load_end: list[float] | None,
    free_bytes: int,
) -> dict[str, object]:
    """Assemble the report: the design's fixed keys plus the additive ``notes`` object."""
    cpus = os.cpu_count() or 1
    inflated = any(sample is not None and sample[0] > cpus for sample in (load_start, load_end))
    crossings = {dim: rows_at_threshold(cell, THRESHOLD_MS) for dim, cell in points.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "platform": platform_facts,
        "params": {
            "k": settings.k,
            "queries": settings.queries,
            "seed": settings.seed,
            "warmup": settings.warmup,
        },
        "notes": {
            "comment": settings.note,
            "inflated": inflated,
            "load_avg_end": load_end,
            "load_avg_start": load_start,
            "numpy": np.__version__,
            "python_full": sys.version,
            "workdir_free_bytes": free_bytes,
        },
        "results": results,
        "advisory": {
            "threshold_ms": THRESHOLD_MS,
            "rows_at_p95_100ms": {str(dim): rows for dim, (rows, _) in crossings.items()},
            "extrapolated": {str(dim): flag for dim, (_, flag) in crossings.items()},
        },
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    """Write the report with sorted keys, two-space indent and a trailing newline."""
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _measure(settings: Settings, workdir: Path, *, generated_at: str) -> int:
    """Check disk, probe the extension, run the grid and write the report; exit code."""
    free, refusal = check_disk(workdir, settings.dims, settings.sizes)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1
    cpus = os.cpu_count() or 1
    load_start = load_average()
    if load_start is not None and load_start[0] > cpus:
        print(
            f"warning: load average {load_start[0]:.1f} exceeds cpu_count {cpus}; "
            "timings will be inflated, label this run -shared",
            file=sys.stderr,
        )
    try:
        probe = connect(":memory:")
    except (sqlite3.Error, AttributeError, OSError) as exc:
        print(f"error: sqlite-vec extension could not be loaded: {exc}", file=sys.stderr)
        return 1
    platform_facts = platform_info(probe)
    probe.close()
    results, points = run_grid(settings, workdir)
    load_end = load_average()
    report = build_report(
        settings,
        platform_facts,
        results,
        points,
        generated_at=generated_at,
        load_start=load_start,
        load_end=load_end,
        free_bytes=free,
    )
    write_report(settings.out, report)
    print(f"wrote {settings.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: one timestamp names the file and stamps ``generated_at``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    started = datetime.now(UTC)
    settings = resolve_settings(parser, args, today=started.date())
    created = settings.workdir is None
    workdir = settings.workdir
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="trelix-bench-"))
    try:
        return _measure(settings, workdir, generated_at=started.strftime("%Y-%m-%dT%H:%M:%SZ"))
    finally:
        if created:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
