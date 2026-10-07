"""Pins the command line of scripts/bench_vector_search.py: the list flags, the defaults and
the --quick preset, the usage errors (exit 2) and the --k boundary.

Split from tests/unit/test_bench_vector_search.py, which keeps the measurement and the report
tests and loads the script once; ``script`` and the AC-2a argv come from there, so a
monkeypatch in either file touches the same module. Every expected value is a literal and
each test's docstring names the mutations that fail it.
"""

from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.test_bench_vector_search import _ROOT, _argv, script


def _settings(argv: list[str], today: date = date(2026, 10, 7)) -> Any:
    parser = script.build_parser()
    return script.resolve_settings(parser, parser.parse_args(argv), today=today)


def test_parse_int_list() -> None:
    """Mutations: sorted() dropped; duplicates kept; the positive check removed."""
    assert script.parse_int_list("100000,10000") == [10000, 100000]
    assert script.parse_int_list("5,5") == [5]
    for bad in ("0", "a", ""):
        with pytest.raises(ValueError):
            script.parse_int_list(bad)


def test_settings_defaults_and_quick() -> None:
    """Mutations: a default changed; --quick overriding an explicit --queries or --sizes;
    default_out_path made cwd-relative; `--warmup 0` refused (`< 0` -> `< 1`)."""
    defaults = _settings([])
    assert _settings(["--warmup", "0"]).warmup == 0
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
        (("--k", "30"), "--k must not exceed the smallest --sizes value"),
        (("--k", "0"), "--k must be at least 1"),
        (("--label", "../x"), "--label must match"),
        (("--label", "apple m4"), "--label must match"),
        (("--label", ".x"), "--label must match"),
        (("--out", "/nonexistent/dir/r.json"), "--out parent directory does not exist"),
        (("--out", "{tmp}/afile/r.json"), "--out parent directory does not exist"),
        (("--out", "{tmp}"), "--out must be a file path, not a directory"),
        (("--dims", "0"), "--dims must be a comma-separated list of positive integers"),
        (("--queries", "0"), "--queries must be at least 1"),
        (("--warmup", "-1"), "--warmup must not be negative"),
        (("--seed", "-1"), "--seed must not be negative"),
        (("--workdir", "{tmp}/missing"), "--workdir must be an existing directory"),
    ],
)
def test_usage_errors_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    override: tuple[str, str],
    expected_text: str,
) -> None:
    """Mutations: the k <= min(sizes) check removed or weakened to max(sizes) (the --k 30
    row: 30 lies between the sizes 20 and 40); the k >= 1, warmup >= 0, seed >= 0 or
    --out-is-a-directory check removed (seed would die in numpy with exit 1, the last would
    run the grid and fail only in write_report); the label check weakened from fullmatch to
    match (the `apple m4` row: match accepts the `apple` prefix, `../x` fails either way);
    the pattern's leading class widened to `[A-Za-z0-9._-]+` (the `.x` row: a label may not
    start with a dot or a dash, so the file is never hidden or flag-shaped); the
    --workdir existence check removed (FileNotFoundError instead of exit 2); the
    _validate call dropped from resolve_settings (every row but --dims); the --out parent
    check loosened to `.exists()` (the `afile` row: a regular file is not a directory)."""
    flag, value = override
    argv = _argv(tmp_path)
    value = value.replace("{tmp}", str(tmp_path))
    if "afile" in value:  # the parent must be a regular FILE: `.exists()` would let it through
        (tmp_path / "afile").write_bytes(b"")
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:  # --label is not part of the AC-2a argv (it defaults to platform.machine())
        argv += [flag, value]
    with pytest.raises(SystemExit) as excinfo:
        script.main(argv)
    assert excinfo.value.code == 2
    assert expected_text in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir()] == (["afile"] if "afile" in value else [])


def test_k_equal_to_the_smallest_size_is_accepted(tmp_path: Path) -> None:
    """Mutation: `k > min(sizes)` -> `>=` in _validate (exit 2 for a legitimate --k)."""
    argv = _argv(tmp_path)
    argv[argv.index("--k") + 1] = "20"
    assert _settings(argv).k == 20


def test_the_scratch_dir_is_removed_on_a_refusal_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutation: the scratch dir removed only on exit 0 (the finally turned into a
    success-only branch): a disk refusal must leave no trelix-bench-* directory behind."""
    made = tmp_path / "scratch"
    made.mkdir()
    monkeypatch.setattr(script.tempfile, "mkdtemp", lambda *args, **kwargs: str(made))
    monkeypatch.setattr(
        script.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=0, free=1)
    )
    argv = _argv(tmp_path)
    workdir_at = argv.index("--workdir")
    del argv[workdir_at : workdir_at + 2]
    assert script.main(argv) == 1
    assert not made.exists()


def test_a_pre_existing_cell_file_is_refused_and_kept(tmp_path: Path) -> None:
    """Mutation: the existence check before connect() dropped (the stale file is opened,
    `file is not a database` tracebacks, and run_cell's finally deletes a file this run did
    not create)."""
    stale = tmp_path / "bench-4-20.db"
    stale.write_bytes(b"not a database")
    with pytest.raises(SystemExit) as excinfo:
        script.main(_argv(tmp_path))
    assert "already exists" in str(excinfo.value)
    assert stale.read_bytes() == b"not a database"
    assert not (tmp_path / "r.json").exists()


def test_batch_rows_is_the_documented_memory_bound() -> None:
    """Mutation: BATCH_ROWS = 50000 (a tenfold per-batch working set)."""
    assert script.BATCH_ROWS == 5000


def test_results_come_out_dims_then_sizes_ascending(tmp_path: Path) -> None:
    """Mutation: the run_grid loops swapped (sizes outer, dims inner), which the one-dim
    AC-2a grid and the committed report cannot see; two dims make the order observable."""
    argv = _argv(tmp_path)
    argv[argv.index("--dims") + 1] = "8,4"
    assert script.main(argv) == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert [(r["dim"], r["rows"]) for r in report["results"]] == [
        (4, 20),
        (4, 40),
        (8, 20),
        (8, 40),
    ]
