"""Pins scripts/bench_vector_search.py, the manual sqlite-vec flat-scan benchmark.

The script is not a test and nothing in CI runs it; what this file pins is the part the
flat-scan advisory depends on: the report's shape, the seeded vectors, the nearest-rank
percentile, the 100 ms crossing rule, the usage errors, and that every committed report
under docs/reports/ was produced under sqlite-vec 0.1.9. Every expected value is a literal;
the script is imported by path (scripts/ is not a package) and nothing is imported from
trelix. Each test's docstring names the mutations that fail it.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import re
import sqlite3
import sys
from datetime import date
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


def _settings(argv: list[str], today: date = date(2026, 10, 7)) -> Any:
    parser = script.build_parser()
    return script.resolve_settings(parser, parser.parse_args(argv), today=today)


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
    dims = {str(result["dim"]) for result in report["results"]}
    assert set(advisory["rows_at_p95_100ms"]) == set(advisory["extrapolated"]) == dims
    assert all(isinstance(v, int) and v > 0 for v in advisory["rows_at_p95_100ms"].values())
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", report["generated_at"])
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    assert "/Users/" not in text


# ---------------------------------------------------------------------------
# AC-2a: the end-to-end report
# ---------------------------------------------------------------------------


def test_a_tiny_grid_produces_the_report_shape(tmp_path: Path) -> None:
    """Mutations: sizes not sorted; SCHEMA_VERSION 2; a result key renamed; sort_keys or
    the trailing newline dropped; platform.node() added; chunk_id restarted per batch;
    notes.numpy or notes.python_full read from the wrong module; DB cleanup removed."""
    assert script.main(_argv(tmp_path)) == 0
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    report = json.loads(text)
    _assert_report_shape(report, text)
    assert report["params"] == {"k": 2, "queries": 3, "seed": 0, "warmup": 1}
    assert report["platform"]["sqlite_vec"] == sqlite_vec.__version__
    assert report["notes"]["numpy"] == np.__version__
    assert report["notes"]["python_full"] == sys.version
    assert len(report["results"]) == 2
    assert report["results"][0]["rows"] == 20
    assert report["results"][1]["rows"] == 40
    assert report["advisory"]["rows_at_p95_100ms"]["4"] > 40
    assert report["advisory"]["extrapolated"] == {"4": True}
    assert str(tmp_path) not in text
    assert list(tmp_path.glob("bench-*.db*")) == []


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
    ],
)
def test_rows_at_threshold_interpolates_the_crossing(
    points: list[tuple[int, float]], expected: tuple[int, bool]
) -> None:
    """Mutations: numerator/denominator swapped; extrapolated always False; upper bound
    `T < p95[i+1]` instead of `<=` (only the three-point row sees it: p95 equals T at a
    measured point whose lower neighbour is measured); the slope <= 0 branch removed
    (ZeroDivisionError on the flat row)."""
    assert script.rows_at_threshold(points, 100.0) == expected


def test_rows_at_threshold_rejects_non_positive_p95() -> None:
    """Mutation: the positive-p95 check removed."""
    with pytest.raises(ValueError, match="p95 values must be positive"):
        script.rows_at_threshold([(10, 0.0)], 100.0)


# ---------------------------------------------------------------------------
# The measurement plumbing
# ---------------------------------------------------------------------------


def test_time_queries_discards_the_warmup(tmp_path: Path) -> None:
    """Mutation: `ms[warmup:]` -> `ms` (length 4)."""
    conn = script.connect(tmp_path / "t.db")
    try:
        script.build_table(conn, 4, 20, np.random.default_rng(0))
        queries = script.make_vectors(np.random.default_rng(1), 4, 4)
        timings = script.time_queries(conn, queries, 2, 1)
    finally:
        conn.close()
    assert len(timings) == 3
    assert all(isinstance(ms, float) and ms > 0.0 for ms in timings)


def test_build_table_generates_vectors_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutations: one make_vectors(rng, rows, dim) call for the whole cell; chunk_id
    restarted at 0 in every batch (IntegrityError)."""
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


def test_run_cell_seeds_the_generator_and_draws_queries_plus_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutations: run_cell builds default_rng() without the seed; it draws `queries`
    instead of `queries + warmup` vectors (recorded n becomes [20, 2]); the DB/-journal
    cleanup removed."""
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
    """Mutation: `sorted(...)` dropped around time_queries (the unsorted answer is
    1.0 / 3.0 / 3.0)."""
    monkeypatch.setattr(script, "time_queries", lambda conn, queries, k, warmup: [5.0, 1.0, 3.0])
    result, raw_p95 = script.run_cell(4, 20, k=2, queries=3, warmup=0, seed=0, workdir=tmp_path)
    assert (result["warm_p50_ms"], result["warm_p95_ms"], result["warm_p99_ms"]) == (
        3.0,
        5.0,
        5.0,
    )
    assert raw_p95 == 5.0


# ---------------------------------------------------------------------------
# Flags and settings
# ---------------------------------------------------------------------------


def test_parse_int_list() -> None:
    """Mutations: sorted() dropped; duplicates kept; the positive check removed."""
    assert script.parse_int_list("100000,10000") == [10000, 100000]
    assert script.parse_int_list("5,5") == [5]
    for bad in ("0", "a", ""):
        with pytest.raises(ValueError):
            script.parse_int_list(bad)


def test_settings_defaults_and_quick() -> None:
    """Mutations: a default changed; --quick overriding an explicit --queries or --sizes;
    default_out_path made cwd-relative."""
    defaults = _settings([])
    assert defaults.dims == [384, 768, 1024]
    assert defaults.sizes == [10000, 100000, 1000000]
    assert (defaults.k, defaults.queries, defaults.warmup, defaults.seed) == (20, 100, 5, 0)
    assert defaults.workdir is None
    assert defaults.label == platform.machine()
    expected_name = f"vector-search-bench-2026-10-07-{platform.machine()}.json"
    assert defaults.out == _ROOT / "docs" / "reports" / expected_name
    quick = _settings(["--quick"])
    assert (quick.sizes, quick.queries) == ([10000, 100000], 30)
    explicit = _settings(["--quick", "--queries", "7", "--sizes", "50"])
    assert (explicit.sizes, explicit.queries) == ([50], 7)
    out = script.default_out_path("arm64", date(2026, 10, 7))
    assert out.is_absolute()
    assert str(out).endswith("docs/reports/vector-search-bench-2026-10-07-arm64.json")


@pytest.mark.parametrize(
    ("override", "expected_text"),
    [
        (("--k", "50"), "--k must not exceed the smallest --sizes value"),
        (("--label", "../x"), "--label must match"),
        (("--out", "/nonexistent/dir/r.json"), "--out parent directory does not exist"),
        (("--dims", "0"), "--dims must be a comma-separated list of positive integers"),
        (("--queries", "0"), "--queries must be at least 1"),
        (("--workdir", "{tmp}/missing"), "--workdir must be an existing directory"),
    ],
)
def test_usage_errors_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    override: tuple[str, str],
    expected_text: str,
) -> None:
    """Mutations: the k <= min(sizes) check removed; the --workdir existence check removed
    (FileNotFoundError instead of exit 2); the _validate call dropped from
    resolve_settings (every row but --dims)."""
    flag, value = override
    argv = _argv(tmp_path)
    value = value.replace("{tmp}", str(tmp_path))
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:  # --label is not part of the AC-2a argv (it defaults to platform.machine())
        argv += [flag, value]
    with pytest.raises(SystemExit) as excinfo:
        script.main(argv)
    assert excinfo.value.code == 2
    assert expected_text in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Exit 1 paths, the scratch directory and the notes object
# ---------------------------------------------------------------------------


def test_disk_check_refuses_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutations: the check_disk call removed; the estimate formula changed."""
    monkeypatch.setattr(
        script.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=0, free=1)
    )
    assert script.main(_argv(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "bytes free; the largest cell (4 x 40 rows) needs about 800 bytes" in err
    assert not (tmp_path / "r.json").exists()


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
    ("samples", "expected_start", "expected_end", "expected_inflated"),
    [
        ([[99.0, 1.0, 1.0], [99.0, 1.0, 1.0]], [99.0, 1.0, 1.0], [99.0, 1.0, 1.0], True),
        ([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False),
        ([[0.5, 0.5, 0.5], [99.0, 1.0, 1.0]], [0.5, 0.5, 0.5], [99.0, 1.0, 1.0], True),
        ([None, None], None, None, False),
    ],
)
def test_notes_record_load_and_inflation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    samples: list[list[float] | None],
    expected_start: list[float] | None,
    expected_end: list[float] | None,
    expected_inflated: bool,
) -> None:
    """Mutations: the inflated comparison `>` -> `<`; the end-of-run load sample dropped
    (None, or a copy of the start sample); workdir_free_bytes read from .total instead of
    .free (2_000_000_000_000); a third load_average() call (StopIteration).

    99.0 exceeds os.cpu_count() here and on every CI runner; the patched free space is
    deterministic where the real reading is not."""
    it = iter(samples)
    monkeypatch.setattr(script, "load_average", lambda: next(it))
    monkeypatch.setattr(
        script.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(
            total=2_000_000_000_000, used=1_000_000_000_000, free=1_000_000_000_000
        ),
    )
    assert script.main([*_argv(tmp_path), "--note", "shared machine"]) == 0
    notes = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["notes"]
    assert notes["load_avg_start"] == expected_start
    assert notes["load_avg_end"] == expected_end
    assert notes["inflated"] is expected_inflated
    assert notes["comment"] == "shared machine"
    assert notes["workdir_free_bytes"] == 1_000_000_000_000
