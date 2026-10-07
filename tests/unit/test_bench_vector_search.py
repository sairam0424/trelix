"""Pins scripts/bench_vector_search.py, the manual sqlite-vec flat-scan benchmark.

The script is not a test and nothing in CI runs it; what this file pins is the part the
flat-scan advisory depends on: the report's shape, the seeded vectors, the nearest-rank
percentile, the 100 ms crossing rule, the exit-1 paths, and that every committed report
under docs/reports/ was produced under sqlite-vec 0.1.9 (the command line, its defaults
and its usage errors are in tests/unit/test_bench_vector_search_flags.py, which imports
``script`` and ``_argv`` from here so both files exercise one loaded module). Every
expected value is a literal; the script is imported by path (scripts/ is not a package)
and nothing is imported from trelix. Each test's docstring names the mutations that
fail it.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import platform
import re
import sqlite3
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sqlite_vec

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "bench_vector_search.py"


def _load_script() -> ModuleType:
    """Import scripts/bench_vector_search.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_trelix_bench_vector_search", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _argv(tmp_path: Path) -> list[str]:
    """AC-2a's argv: a tiny grid with --workdir and --out both under tmp_path."""
    return [
        *("--dims", "4", "--sizes", "20,40", "--queries", "3", "--warmup", "1", "--k", "2"),
        *("--seed", "0", "--workdir", str(tmp_path), "--out", str(tmp_path / "r.json")),
    ]


def _assert_report_shape(report: dict[str, Any], text: str) -> None:
    """The design's fixed key sets, the additive notes object and the serialisation."""
    expected_top = {"advisory", "generated_at", "notes", "params", "platform", "results"}
    assert set(report) == expected_top | {"schema_version"}
    assert set(report["platform"]) == {
        *("system", "release", "machine", "python"),
        *("sqlite", "sqlite_vec", "vec_version", "cpu_count"),
    }
    notes = report["notes"]
    assert set(notes) == {
        *("comment", "inflated", "load_avg_end", "load_avg_start"),
        *("numpy", "python_full", "workdir_free_bytes"),
    }
    assert isinstance(notes["workdir_free_bytes"], int) and notes["workdir_free_bytes"] > 0
    assert notes["python_full"].startswith(report["platform"]["python"])
    for result in report["results"]:
        assert set(result) == {
            *("dim", "rows", "insert_seconds", "insert_rows_per_s"),
            *("db_bytes", "warm_p50_ms", "warm_p95_ms", "warm_p99_ms"),
        }
    assert report["schema_version"] == 1
    advisory = report["advisory"]
    assert advisory["threshold_ms"] == 100.0
    pairs = [(result["dim"], result["rows"]) for result in report["results"]]
    assert pairs == sorted(pairs)  # dims ascending, then sizes ascending (the grid order)
    dims = {str(result["dim"]) for result in report["results"]}
    assert set(advisory["rows_at_p95_100ms"]) == set(advisory["extrapolated"]) == dims
    assert all(isinstance(v, int) and v > 0 for v in advisory["rows_at_p95_100ms"].values())
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", report["generated_at"])
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    assert "/Users/" not in text


# ---------------------------------------------------------------------------
# AC-2a: the end-to-end report
# ---------------------------------------------------------------------------


def test_a_tiny_grid_produces_the_report_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutations: sizes not sorted; SCHEMA_VERSION 2; a result key renamed; sort_keys or
    the trailing newline dropped; platform.node() added, or written under the `system` key
    (every platform value is pinned to the same stdlib reading, so a hostname cannot hide
    under a correct key); chunk_id restarted per batch; notes.numpy or notes.python_full
    read from the wrong module; DB cleanup removed; db_bytes replaced by a constant below
    one SQLite page, insert_rows_per_s or platform.cpu_count by any constant; the per-cell
    progress line (the only sign of life in a nohup log) dropped."""
    assert script.main(_argv(tmp_path)) == 0
    out = capsys.readouterr().out
    assert out.count("dim=4 rows=") == 2
    assert "dim=4 rows=20 insert_s=" in out
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    report = json.loads(text)
    _assert_report_shape(report, text)
    assert report["params"] == {"k": 2, "queries": 3, "seed": 0, "warmup": 1}
    assert report["platform"]["system"] == platform.system()
    assert report["platform"]["release"] == platform.release()
    assert report["platform"]["machine"] == platform.machine()
    assert report["platform"]["python"] == platform.python_version()
    assert report["platform"]["sqlite"] == sqlite3.sqlite_version
    assert report["platform"]["sqlite_vec"] == sqlite_vec.__version__
    assert report["platform"]["cpu_count"] == os.cpu_count()
    assert report["notes"]["numpy"] == np.__version__
    assert report["notes"]["python_full"] == sys.version
    assert len(report["results"]) == 2
    assert report["results"][0]["rows"] == 20
    assert report["results"][1]["rows"] == 40
    for result in report["results"]:
        assert result["db_bytes"] >= 4096
        assert result["insert_rows_per_s"] > 0
    assert report["advisory"]["rows_at_p95_100ms"]["4"] > 40
    assert report["advisory"]["extrapolated"] == {"4": True}
    assert str(tmp_path) not in text
    assert list(tmp_path.glob("bench-*.db*")) == []


def test_build_report_keeps_an_interpolated_crossing_unflagged() -> None:
    """Mutations: build_report's `extrapolated` value replaced by a constant True (AC-2a's
    two-point grid only ever extrapolates, so it pins the constant False alone); the
    advisory's two maps swapped (a bool passes the shape test's positive-int check).
    """
    settings = script.Settings(
        dims=[4],
        sizes=[20, 40],
        k=2,
        queries=3,
        warmup=1,
        seed=0,
        workdir=None,
        out=Path("r.json"),
        label="x",
        note="",
    )
    report = script.build_report(
        settings,
        {"cpu_count": 1},
        [],
        {4: [(20, 50.0), (40, 150.0)]},
        generated_at="2026-10-07T00:00:00Z",
        load_start=None,
        load_end=None,
        free_bytes=1,
    )
    assert report["advisory"] == {
        "threshold_ms": 100.0,
        "rows_at_p95_100ms": {"4": 30},
        "extrapolated": {"4": False},
    }


# ---------------------------------------------------------------------------
# AC-2b, AC-2c, AC-2d: the arithmetic
# ---------------------------------------------------------------------------


def test_seeded_vectors_are_unit_length_and_reproducible() -> None:
    """Mutations: L2 normalisation removed; the draw ignores the passed rng; dtype=float32
    on the draw replaced by a cast (the literal becomes 0.186517, -0.195973, ...)."""
    first = script.make_vectors(np.random.default_rng(0), 1, 4)
    assert first.dtype == np.float32
    assert [round(float(x), 6) for x in first[0]] == [0.558747, -0.693484, -0.213262, -0.401748]
    block = script.make_vectors(np.random.default_rng(0), 3, 8)
    assert block.shape == (3, 8)
    assert np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=1e-6)
    again = script.make_vectors(np.random.default_rng(0), 3, 8)
    assert np.array_equal(block, again)
    other = script.make_vectors(np.random.default_rng(1), 3, 8)
    assert not np.array_equal(block, other)


def test_percentile_is_nearest_rank() -> None:
    """Mutation: math.ceil -> math.floor (p95 of 1..10 gives 9)."""
    ten = [float(i) for i in range(1, 11)]
    assert script.percentile(ten, 50) == 5
    assert script.percentile(ten, 95) == 10
    assert script.percentile(ten, 99) == 10
    assert script.percentile([7.0], 95) == 7


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(100_000, 45.0), (300_000, 135.0)], (222_222, False)),
        ([(10_000, 16.0), (100_000, 52.0)], (220_000, True)),
        ([(100_000, 188.0)], (53_191, True)),
        ([(10_000, 150.0), (100_000, 900.0)], (6_666, True)),
        ([(100_000, 100.0), (300_000, 135.0)], (100_000, True)),
        ([(10_000, 16.0)], (62_500, True)),
        ([(10_000, 16.0), (100_000, 16.0)], (625_000, True)),
        ([(50_000, 40.0), (100_000, 100.0), (300_000, 135.0)], (100_000, False)),
        ([(300_000, 135.0), (100_000, 45.0)], (222_222, False)),
    ],
)
def test_rows_at_threshold_interpolates_the_crossing(
    points: list[tuple[int, float]], expected: tuple[int, bool]
) -> None:
    """Mutations: numerator/denominator swapped; extrapolated always False; upper bound
    `T < p95[i+1]` instead of `<=` (only the three-point row sees it: p95 equals T at a
    measured point whose lower neighbour is measured); the slope <= 0 branch removed
    (ZeroDivisionError on the flat row); `sorted(points)` dropped (the reversed row)."""
    assert script.rows_at_threshold(points, 100.0) == expected


def test_rows_at_threshold_rejects_non_positive_p95() -> None:
    """Mutation: the positive-p95 check removed."""
    with pytest.raises(ValueError, match="p95 values must be positive"):
        script.rows_at_threshold([(10, 0.0)], 100.0)


# ---------------------------------------------------------------------------
# The measurement plumbing
# ---------------------------------------------------------------------------


def test_time_queries_discards_the_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutations: `ms[warmup:]` -> `ms` (length 4); `ms[warmup:]` -> `ms[1:]` (the warm-up
    hard-coded: warmup 2 and 0 then keep 3, not 2 and 4); the discard taken from the tail,
    `ms[: len(ms) - warmup]` (the clock below times the first, cold query at 1000.0 and the
    three warm ones at 500.0, so the kept list must start at 500.0); `* 1000.0` -> `* 1.0`
    (seconds recorded as milliseconds: the same clock must give 500.0, not 0.5)."""
    conn = script.connect(tmp_path / "t.db")
    try:
        script.build_table(conn, 4, 20, np.random.default_rng(0))
        queries = script.make_vectors(np.random.default_rng(1), 4, 4)
        timings = script.time_queries(conn, queries, 2, 1)
        assert len(timings) == 3
        assert all(isinstance(ms, float) and ms > 0.0 for ms in timings)
        assert len(script.time_queries(conn, queries, 2, 2)) == 2
        assert len(script.time_queries(conn, queries, 2, 0)) == 4
        clock = iter([0.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5])
        monkeypatch.setattr(script, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
        assert script.time_queries(conn, queries, 2, 1) == [500.0, 500.0, 500.0]
    finally:
        conn.close()


def test_build_table_generates_vectors_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutations: one make_vectors(rng, rows, dim) call for the whole cell; chunk_id
    restarted at 0 in every batch (IntegrityError); conn.commit() dropped from the batch loop
    (close() rolls the inserts back, so the reopened file holds 0 rows)."""
    seen: list[int] = []
    real = script.make_vectors

    def recorder(rng: np.random.Generator, n: int, dim: int) -> Any:
        seen.append(n)
        return real(rng, n, dim)

    monkeypatch.setattr(script, "make_vectors", recorder)
    conn = script.connect(tmp_path / "b.db")
    try:
        script.build_table(conn, 4, 12, np.random.default_rng(0), batch=5)
        assert conn.execute("SELECT count(*) FROM v").fetchone()[0] == 12
        assert conn.execute("SELECT min(chunk_id), max(chunk_id) FROM v").fetchone() == (0, 11)
    finally:
        conn.close()
    assert seen == [5, 5, 2]
    again = script.connect(tmp_path / "b.db")
    assert again.execute("SELECT count(*) FROM v").fetchone()[0] == 12
    again.close()


def test_run_cell_seeds_the_generator_and_draws_queries_plus_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutations: run_cell builds default_rng() without the seed; it draws `queries`
    instead of `queries + warmup` vectors (recorded n becomes [20, 2]); the query draw
    restarted from a fresh default_rng(seed) (its first vectors would equal the first
    inserted rows); the DB/-journal cleanup removed."""
    seen: list[tuple[int, Any]] = []
    real = script.make_vectors

    def recorder(rng: np.random.Generator, n: int, dim: int) -> Any:
        array = real(rng, n, dim)
        seen.append((n, array.copy()))
        return array

    monkeypatch.setattr(script, "make_vectors", recorder)
    runs = []
    for seed in (3, 3, 4):
        seen.clear()
        result, raw_p95 = script.run_cell(
            4, 20, k=2, queries=2, warmup=1, seed=seed, workdir=tmp_path
        )
        assert [n for n, _ in seen] == [20, 3]
        assert not np.array_equal(seen[1][1], seen[0][1][:3])
        assert set(result) == {
            *("dim", "rows", "insert_seconds", "insert_rows_per_s"),
            *("db_bytes", "warm_p50_ms", "warm_p95_ms", "warm_p99_ms"),
        }
        assert raw_p95 > 0.0
        runs.append(seen[0][1])
    assert np.array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])
    assert list(tmp_path.glob("bench-*.db*")) == []


def test_run_cell_sorts_timings_before_taking_percentiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One hundred distinct timings in reverse order: nearest rank over the sorted list gives
    p50 = 50 (index 49), p95 = 95 (index 94) and p99 = 99 (index 98), three different values
    that also differ from p100 (index 99); with fewer than 100 timings p99 and p100 share an
    index, and with three they share it with p95 too. Mutations: `sorted(...)` dropped around
    time_queries (the reversed list reads 51 / 6 / 2); `warm_p99_ms` read from
    `percentile(ms, 95)` (95.0) or `percentile(ms, 100)` (100.0); insert_rows_per_s computed
    from rows * seconds, or offset by one (with time_queries patched, build_table's two readings
    are the only clock reads, so a 0.5-s step makes insert_seconds exactly 0.5 and 20 rows
    exactly 40 rows/s)."""
    shuffled = [float(i) for i in range(100, 0, -1)]
    monkeypatch.setattr(script, "time_queries", lambda conn, queries, k, warmup: shuffled)
    clock = itertools.count(0.0, 0.5)
    monkeypatch.setattr(script, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    result, raw_p95 = script.run_cell(4, 20, k=2, queries=3, warmup=0, seed=0, workdir=tmp_path)
    assert (result["warm_p50_ms"], result["warm_p95_ms"], result["warm_p99_ms"]) == (
        50.0,
        95.0,
        99.0,
    )
    assert raw_p95 == 95.0
    assert (result["insert_seconds"], result["insert_rows_per_s"]) == (0.5, 40)


# ---------------------------------------------------------------------------
# Exit 1 paths, the scratch directory and the notes object
# ---------------------------------------------------------------------------


def test_disk_check_refuses_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutations: the check_disk call removed; the estimate formula changed; `free >=
    estimate` -> `>` (exactly enough space is refused); `max(dims)` or `max(sizes)` -> `min`
    (the two-dim grid must name the 8 x 40 cell and 1,600 bytes); disk_usage measured on
    the cwd (or any path but --workdir): the one recorded argument must be tmp_path."""
    seen: list[Path] = []

    def tiny_disk(path: Path) -> SimpleNamespace:
        seen.append(Path(path))
        return SimpleNamespace(total=1, used=0, free=1)

    monkeypatch.setattr(script.shutil, "disk_usage", tiny_disk)
    assert script.main(_argv(tmp_path)) == 1
    assert seen == [tmp_path]
    err = capsys.readouterr().err
    assert "bytes free; the largest cell (4 x 40 rows) needs about 800 bytes" in err
    assert not (tmp_path / "r.json").exists()
    refusal = script.check_disk(tmp_path, [4, 8], [20, 40])[1]
    assert refusal is not None and "(8 x 40 rows) needs about 1,600 bytes" in refusal
    monkeypatch.setattr(
        script.shutil, "disk_usage", lambda p: SimpleNamespace(total=800, used=0, free=800)
    )
    assert script.check_disk(tmp_path, [4], [40]) == (800, None)


def test_extension_load_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutation: the except around connect(":memory:") removed."""

    def raiser(conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("boom")

    monkeypatch.setattr(script.sqlite_vec, "load", raiser)
    assert script.main(_argv(tmp_path)) == 1
    assert "error: sqlite-vec extension could not be loaded: boom" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_default_workdir_is_created_and_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutation: the finally that removes the script-created scratch dir removed.

    Patches the global tempfile module for the test's duration on purpose: the script calls
    tempfile.mkdtemp(prefix=...) through the module attribute, and the *args/**kwargs
    signature keeps any other caller in the process from a TypeError."""
    made = tmp_path / "scratch"
    made.mkdir()
    monkeypatch.setattr(script.tempfile, "mkdtemp", lambda *args, **kwargs: str(made))
    argv = _argv(tmp_path)
    workdir_at = argv.index("--workdir")
    del argv[workdir_at : workdir_at + 2]
    assert script.main(argv) == 0
    assert (tmp_path / "r.json").exists()
    assert not made.exists()


@pytest.mark.parametrize(
    ("samples", "expected_start", "expected_end", "expected_inflated", "expected_warning"),
    [
        ([[99.0, 1.0, 1.0], [99.0, 1.0, 1.0]], [99.0, 1.0, 1.0], [99.0, 1.0, 1.0], True, True),
        ([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False, False),
        ([[0.5, 0.5, 0.5], [99.0, 1.0, 1.0]], [0.5, 0.5, 0.5], [99.0, 1.0, 1.0], True, False),
        ([None, None], None, None, False, False),
        ([[10.0, 1.0, 1.0], [10.0, 1.0, 1.0]], [10.0, 1.0, 1.0], [10.0, 1.0, 1.0], False, False),
    ],
)
def test_notes_record_load_and_inflation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    samples: list[list[float] | None],
    expected_start: list[float] | None,
    expected_end: list[float] | None,
    expected_inflated: bool,
    expected_warning: bool,
) -> None:
    """Mutations: the inflated comparison `>` -> `<`; the end-of-run load sample dropped
    (None, or a copy of the start sample) or taken before the grid (the call order becomes
    load, load, grid); workdir_free_bytes read from .total instead of .free
    (2_000_000_000_000); a third load_average() call (StopIteration); the start-of-run
    stderr warning dropped, or keyed on `inflated` instead of the start sample (the third
    row inflates on the END sample and must see no warning).

    99.0 exceeds os.cpu_count() here and on every CI runner; the patched free space is
    deterministic where the real reading is not. cpu_count is patched to 10 so the last row
    pins the boundary: a load exactly equal to cpu_count is quiet (`>` -> `>=` fails it)."""
    monkeypatch.setattr(script.os, "cpu_count", lambda: 10)
    order: list[str] = []
    it = iter(samples)

    def sampled_load() -> list[float] | None:
        order.append("load")
        return next(it)

    real_run_grid = script.run_grid

    def recorded_run_grid(*args: Any, **kwargs: Any) -> Any:
        order.append("grid")
        return real_run_grid(*args, **kwargs)

    free = SimpleNamespace(total=2_000_000_000_000, used=1_000_000_000_000, free=1_000_000_000_000)
    monkeypatch.setattr(script, "load_average", sampled_load)
    monkeypatch.setattr(script, "run_grid", recorded_run_grid)
    monkeypatch.setattr(script.shutil, "disk_usage", lambda p: free)
    assert script.main([*_argv(tmp_path), "--note", "shared machine"]) == 0
    assert order == ["load", "grid", "load"]
    err = capsys.readouterr().err
    assert ("warning: load average 99.0 exceeds cpu_count" in err) is expected_warning
    notes = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["notes"]
    assert notes["load_avg_start"] == expected_start
    assert notes["load_avg_end"] == expected_end
    assert notes["inflated"] is expected_inflated
    assert notes["comment"] == "shared machine"
    assert notes["workdir_free_bytes"] == 1_000_000_000_000


def test_load_average_reads_three_floats() -> None:
    """Mutation: load_average's body replaced by `return None` (the notes rows patch the
    function itself, so only this test reads the real sample the LABEL RULE rests on)."""
    sample = script.load_average()
    if not hasattr(os, "getloadavg"):
        assert sample is None
        return
    assert isinstance(sample, list) and len(sample) == 3
    assert all(isinstance(value, float) for value in sample)


# ---------------------------------------------------------------------------
# The committed report(s) under docs/reports/
# ---------------------------------------------------------------------------


def test_committed_report_matches_the_schema() -> None:
    """Mutations: a committed report edited by hand (a key dropped, sqlite_vec changed, a
    `-shared` label on a report whose notes.inflated is false, or the reverse); the report
    deleted (the non-vacuity assertion is why this test lands with the report).

    No Python-version assertion: the owner's later report may come from another interpreter."""
    files = sorted((_ROOT / "docs" / "reports").glob("vector-search-bench-*.json"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        report = json.loads(text)
        _assert_report_shape(report, text)
        assert report["platform"]["sqlite_vec"] == "0.1.9"
        assert report["platform"]["vec_version"] == "v0.1.9"
        assert len(report["results"]) in (6, 9)
        assert ("-shared" in path.stem) == report["notes"]["inflated"]
