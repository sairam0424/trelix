"""The flat-scan advisory: when it fires, what it says, and where the numbers come from.

The `sqlite` backend is an exact flat scan, so search latency grows with rows x width.
`trelix index` and `trelix stats` print a `Flat-scan advisory` on stderr once the store holds
more vectors than the measured threshold for the index's width. Pinned here:

  * the rule that turns a width into a threshold (nearest measured width at or above, scaled
    above the widest) and the strict `>` that decides whether the line prints;
  * the production table equals the committed benchmark report it cites, and
    `docs/CONFIGURATION.md` quotes the same three numbers;
  * both CLI call sites, the backend guard, the stream, and that an unreadable or missing
    index is silence rather than an error and creates no `.trelix/`.

Every patch of the threshold table targets the SOURCE module (`scale_advisory_mod`), never
`trelix.cli.main`, which imports `flat_scan_advisory` at call time inside the helper.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from trelix.store import scale_advisory as scale_advisory_mod
from trelix.store.scale_advisory import (
    ADVISORY_P95_MS,
    ADVISORY_SOURCE_REPORT,
    FLAT_SCAN_P95_100MS_ROWS,
    flat_scan_advisory,
    flat_scan_threshold_rows,
)

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig

runner = CliRunner()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIM = 4
_VEC = [0.1, 0.2, 0.3, 0.4]
_FIXTURE = {384: 200_000, 768: 50_000, 1024: 40_000}
# Copied from the committed report's `advisory.rows_at_p95_100ms`; a refresh of the report
# changes this table, the module and the CONFIGURATION.md table, and nothing else here.
_REPORT_TABLE = {384: 32_765, 768: 12_979, 1024: 28_956}
_FIRST_LINE = "Flat-scan advisory: this index holds 3 vectors at 4 dimensions"


def _number(output: str, label: str) -> int:
    """First number after `label`. Copied from tests/unit/test_cli_stats_coverage.py."""
    match = re.search(rf"{label}\D*?([\d,]+)", output)
    assert match, f"no {label!r} row in:\n{output}"
    return int(match.group(1).replace(",", ""))


def _seeded_repo(tmp_path: Path, *, chunks: int, embedded: int, record_dim: bool) -> Path:
    """An index with `chunks` chunk rows of which `embedded` have vectors (width 4).

    Copied from tests/unit/test_cli_stats_coverage.py: no model, no paid call.
    """
    from trelix.core.config import IndexConfig
    from trelix.core.models import Chunk, IndexedFile, Language, Symbol, SymbolKind
    from trelix.store.db import Database
    from trelix.store.vector import VectorStore

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def f():\n    return 1\n")
    db_path = IndexConfig(repo_path=str(repo)).db_path_absolute
    db_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_ids: list[int] = []
    with Database(db_path) as db:
        file_id = db.upsert_file(
            IndexedFile(
                path=str(repo / "mod.py"),
                rel_path="mod.py",
                language=Language.PYTHON,
                hash="0" * 64,
                size_bytes=1,
            )
        )
        for n in range(chunks):
            symbol_id = db.insert_symbol(
                Symbol(
                    file_id=file_id,
                    name=f"f{n}",
                    qualified_name=f"f{n}",
                    kind=SymbolKind.FUNCTION,
                    line_start=1,
                    line_end=2,
                    signature=f"def f{n}()",
                    body="return 1",
                )
            )
            chunk_ids.append(
                db.insert_chunk(
                    Chunk(symbol_id=symbol_id, chunk_text=f"def f{n}(): return 1", token_count=7)
                )
            )
        if record_dim:
            db.set_embedding_dimension(_DIM)
        db._conn.commit()

    store = VectorStore(db_path, dimension=_DIM)
    for chunk_id in chunk_ids[:embedded]:
        store.upsert(chunk_id=chunk_id, embedding=_VEC)
    store.close()
    return repo


def _config(repo: Path, backend: str = "sqlite") -> IndexConfig:
    from trelix.core.config import IndexConfig, StoreConfig

    return IndexConfig(repo_path=str(repo), store=StoreConfig(backend=backend))


def _recorders(monkeypatch: pytest.MonkeyPatch) -> tuple[Console, Console]:
    """Replace both module consoles with wide recorders: (stderr, stdout)."""
    from trelix.cli import main as cli_main

    err = Console(record=True, width=400, no_color=True, legacy_windows=False)
    out = Console(record=True, width=400, no_color=True, legacy_windows=False)
    monkeypatch.setattr(cli_main, "err_console", err)
    monkeypatch.setattr(cli_main, "console", out)
    return err, out


def _patch_table(monkeypatch: pytest.MonkeyPatch, table: dict[int, int]) -> None:
    monkeypatch.setattr(scale_advisory_mod, "FLAT_SCAN_P95_100MS_ROWS", table)


@pytest.fixture
def _stub_provenance_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """For the `index`-site cases ONLY (never autouse).

    `index` reads provenance from the fake's MagicMock db and uses the result only in the
    prune guard, so None is right there. `stats` dereferences the record, so this must not
    reach the `stats` cases: they run the real `read_provenance`.
    """
    from trelix.store import provenance as provenance_mod

    monkeypatch.setattr(provenance_mod, "read_provenance", lambda db: None)


def _fake_indexer(monkeypatch: pytest.MonkeyPatch, *, dimension: int | str | None) -> None:
    """A real-class Indexer whose __init__ and index() do nothing, with or without a width."""
    from trelix.indexing.indexer import Indexer

    def _fake_init(self, config, *args, **kwargs):
        self.db = MagicMock()
        if dimension is not None:
            self.embedder = SimpleNamespace(dimension=dimension)

    monkeypatch.setattr(Indexer, "__init__", _fake_init)
    monkeypatch.setattr(
        Indexer,
        "index",
        lambda self: {"files_found": 1, "files_indexed": 1, "chunks_embedded": 3},
    )


class TestTheThresholdRule:
    @pytest.mark.parametrize(
        ("dimension", "expected"),
        [
            (256, 200_000),
            (384, 200_000),
            (500, 50_000),
            (768, 50_000),
            (1024, 40_000),
            (1536, 26_666),
            (3072, 13_333),
        ],
    )
    def test_threshold_is_the_nearest_measured_width_at_or_above_then_scaled(
        self, dimension: int, expected: int
    ) -> None:
        """MUTATIONS: nearest-lower width (500 gives 200_000); scaling dropped (1536 gives
        40_000); `//` turned into `/` (1536 gives a float)."""
        result = flat_scan_threshold_rows(dimension, _FIXTURE)

        assert result == expected
        assert isinstance(result, int)

    def test_a_width_below_one_and_an_empty_table_are_refused(self) -> None:
        """MUTATIONS: the `dimension < 1` check removed (0 would return 200_000); the empty
        table check removed (the error would come from `max()` with another message)."""
        with pytest.raises(ValueError, match="at least 1"):
            flat_scan_threshold_rows(0, _FIXTURE)
        with pytest.raises(ValueError, match="table is empty"):
            flat_scan_threshold_rows(4, {})

    def test_table_defaults_are_bound_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MUTATION: either default restored to `= FLAT_SCAN_P95_100MS_ROWS`, which binds at
        definition time so a monkeypatch of the module constant never reaches the CLI."""
        _patch_table(monkeypatch, {4: 1})

        assert flat_scan_threshold_rows(4) == 1
        assert flat_scan_advisory(2, 4) is not None


class TestTheAdvisoryLine:
    def test_advisory_fires_strictly_above_the_threshold(self) -> None:
        """MUTATIONS: `>` turned into `>=` (200_000 would fire); the `:,` thousands format
        dropped; a backend name or the docs anchor edited."""
        assert flat_scan_advisory(199_999, 384, _FIXTURE) is None
        assert flat_scan_advisory(200_000, 384, _FIXTURE) is None

        message = flat_scan_advisory(200_001, 384, _FIXTURE)

        assert message is not None
        assert message.startswith(
            "Flat-scan advisory: this index holds 200,001 vectors at 384 dimensions."
        )
        assert "passes 100 ms near 200,000 rows at this width" in message
        assert "(see docs/reports/vector-search-bench-" in message
        assert message.endswith(
            "Consider TRELIX_STORE_BACKEND=lance or qdrant; see docs/CONFIGURATION.md#storage."
        )
        assert "\n" not in message

    def test_a_tiny_table_reads_back_its_own_numbers(self) -> None:
        """MUTATION: rows and threshold swapped in the text."""
        message = flat_scan_advisory(3, 4, {4: 1})

        assert message is not None
        assert "3 vectors at 4 dimensions" in message
        assert "near 1 rows" in message


class TestTheProductionTable:
    def test_production_table_matches_the_committed_report(self) -> None:
        """MUTATIONS: any table literal edited; ADVISORY_SOURCE_REPORT pointed at a missing
        file (FileNotFoundError); the threshold constant changed."""
        report = json.loads((_REPO_ROOT / ADVISORY_SOURCE_REPORT).read_text(encoding="utf-8"))

        assert report["schema_version"] == 1
        assert report["advisory"]["threshold_ms"] == ADVISORY_P95_MS == 100.0
        from_report = {int(k): v for k, v in report["advisory"]["rows_at_p95_100ms"].items()}
        assert from_report == dict(FLAT_SCAN_P95_100MS_ROWS)
        assert set(FLAT_SCAN_P95_100MS_ROWS) == {384, 768, 1024}

    def test_the_table_literal_is_what_this_file_expects(self) -> None:
        """MUTATION: any table literal edited. A report refresh changes this literal."""
        assert dict(FLAT_SCAN_P95_100MS_ROWS) == _REPORT_TABLE

    def test_configuration_quotes_the_measured_thresholds(self) -> None:
        """MUTATIONS: a number in docs/CONFIGURATION.md edited; the advisory paragraph or the
        report link dropped."""
        text = (_REPO_ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
        report_name = ADVISORY_SOURCE_REPORT.removeprefix("docs/")

        for dimension, rows in _REPORT_TABLE.items():
            assert f"| {dimension} | {rows:,} |" in text
        assert "Flat-scan advisory" in text
        assert report_name in text


class TestTheCommandSites:
    def test_index_prints_the_advisory_past_the_threshold(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _stub_provenance_read: None,
    ) -> None:
        """MUTATIONS: the `index` call site deleted; `indexer.embedder.dimension` read
        strictly (the no-embedder case and four batch-flag tests exit 1); a non-width passed
        as the dimension (`3 dimensions`)."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        _fake_indexer(monkeypatch, dimension=_DIM)
        _patch_table(monkeypatch, {4: 1})

        result = runner.invoke(app, ["index", str(repo)])

        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert _FIRST_LINE in collapsed, collapsed
        assert "near 1 rows" in collapsed, collapsed

    def test_index_is_silent_at_the_threshold(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _stub_provenance_read: None,
    ) -> None:
        """MUTATION: `if message is None: return` dropped (the word None is printed)."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        _fake_indexer(monkeypatch, dimension=_DIM)
        _patch_table(monkeypatch, {4: 3})

        result = runner.invoke(app, ["index", str(repo)])

        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert "Flat-scan advisory" not in collapsed, collapsed
        assert "None" not in collapsed, collapsed

    @pytest.mark.parametrize("dimension", [None, "384"])
    def test_index_with_a_stub_indexer_that_has_no_usable_width_is_silent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _stub_provenance_read: None,
        dimension: str | None,
    ) -> None:
        """No embedder at all, or one whose width is not an int (whatever a stub passes).

        MUTATIONS: `indexer.embedder.dimension` read strictly (AttributeError, exit 1); the
        `isinstance(dimension, int)` guard weakened to `dimension is None` (the string "384"
        reaches the threshold lookup and raises TypeError, exit 1)."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        _fake_indexer(monkeypatch, dimension=dimension)
        _patch_table(monkeypatch, {4: 1})

        result = runner.invoke(app, ["index", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Flat-scan advisory" not in " ".join(result.output.split())

    def test_stats_prints_the_advisory_past_the_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATIONS: the `stats` call site deleted; the coverage dimension replaced by a
        non-width. Runs the REAL read_provenance (the seeded index records none)."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        _patch_table(monkeypatch, {4: 1})

        result = runner.invoke(app, ["stats", str(repo)])

        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert _FIRST_LINE in collapsed, collapsed
        assert "near 1 rows" in collapsed, collapsed
        assert _number(collapsed, "Chunks with vectors") == 3, collapsed

    def test_stats_is_silent_at_the_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `if message is None: return` dropped (None is printed; meaningful only
        because provenance is real, so nothing else on this path prints None)."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        _patch_table(monkeypatch, {4: 3})

        result = runner.invoke(app, ["stats", str(repo)])

        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert "Flat-scan advisory" not in collapsed, collapsed
        assert "None" not in collapsed, collapsed

    @pytest.mark.parametrize("recorded", [None, "0", "-1"])
    def test_stats_is_silent_when_the_width_is_not_a_usable_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str | None
    ) -> None:
        """No recorded width, or a corrupt one: `stats` is the command meant to survive a
        damaged `index_metadata`, and base `origin/develop` exits 0 on a recorded 0 or -1.

        MUTATIONS: the `isinstance(dimension, int)` guard dropped (a missing width raises
        TypeError in the threshold lookup); the `dimension < 1` guard dropped (a recorded 0 or
        -1 raises ValueError); either way `stats` exits 1."""
        from trelix.cli.main import app
        from trelix.store.db import Database

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=False)
        if recorded is not None:
            with Database(_config(repo).db_path_resolved) as db:
                db.set_index_metadata("embedding_dimension", recorded)
        _patch_table(monkeypatch, {4: 1})

        result = runner.invoke(app, ["stats", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Flat-scan advisory" not in " ".join(result.output.split())


class TestTheHelpers:
    @pytest.mark.parametrize(
        ("width", "said"), [(4, "3 vectors at 4 dimensions"), (1, "3 vectors at 1 dimensions")]
    )
    def test_the_advisory_goes_to_stderr_only_on_the_sqlite_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, said: str
    ) -> None:
        """MUTATIONS: the backend guard removed (qdrant prints); `err_console` swapped for
        `console`; `if message is None: return` dropped (None lands on stderr); the
        `dimension < 1` guard widened to `< 2` (the smallest valid width, 1, goes silent)."""
        from trelix.cli import main as cli_main

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        err, out = _recorders(monkeypatch)
        _patch_table(monkeypatch, {4: 1})

        cli_main._print_flat_scan_advisory(_config(repo, "qdrant"), width)
        assert err.export_text() == ""

        cli_main._print_flat_scan_advisory(_config(repo, "sqlite"), width)
        assert f"Flat-scan advisory: this index holds {said}" in err.export_text()
        assert out.export_text() == ""

    def test_nothing_prints_at_the_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `if message is None: return` dropped (None lands on stderr)."""
        from trelix.cli import main as cli_main

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        err, _ = _recorders(monkeypatch)
        _patch_table(monkeypatch, {4: 3})

        cli_main._print_flat_scan_advisory(_config(repo), 4)

        assert err.export_text() == ""

    def test_an_unreadable_index_is_silent_and_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATIONS: the first `except Exception` in `_sqlite_vector_rows` removed
        (sqlite3.OperationalError escapes); `db_path_resolved` turned into `db_path_absolute`
        (a `.trelix/` directory appears)."""
        from trelix.cli import main as cli_main

        err, out = _recorders(monkeypatch)
        _patch_table(monkeypatch, {4: 0})
        empty = tmp_path / "empty"
        empty.mkdir()

        cli_main._print_flat_scan_advisory(_config(empty), 4)

        assert err.export_text() == ""
        assert out.export_text() == ""
        assert not (empty / ".trelix").exists()

    def test_an_index_with_no_vector_table_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the second `except Exception` in `_sqlite_vector_rows` removed
        (`no such table: chunk_embeddings` escapes)."""
        from trelix.cli import main as cli_main
        from trelix.store.db import Database

        repo = tmp_path / "schema_only"
        repo.mkdir()
        config = _config(repo)
        db_path = config.db_path_absolute
        with Database(db_path):
            pass
        err, _ = _recorders(monkeypatch)
        _patch_table(monkeypatch, {4: 0})

        cli_main._print_flat_scan_advisory(config, 4)

        assert err.export_text() == ""

    def test_the_row_count_includes_sentinels_and_closes_its_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATIONS: sentinels filtered out of the count (3 instead of 5); `conn.close()`
        removed from the `finally`."""
        from trelix.cli import main as cli_main
        from trelix.core.config import IndexConfig
        from trelix.store.vector import VectorStore

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        config = _config(repo)
        store = VectorStore(IndexConfig(repo_path=str(repo)).db_path_resolved, dimension=_DIM)
        store.upsert_file_summary_embedding(1, _VEC)
        store.upsert_sub_chunk_embedding(1, _VEC)
        store.close()
        opened: list[sqlite3.Connection] = []
        real_open = cli_main._readonly_vec_conn

        def _spy(db_path: Path) -> sqlite3.Connection:
            conn = real_open(db_path)
            opened.append(conn)
            return conn

        monkeypatch.setattr(cli_main, "_readonly_vec_conn", _spy)

        assert cli_main._sqlite_vector_rows(config) == 5
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")
