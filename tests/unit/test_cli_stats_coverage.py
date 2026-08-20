"""`trelix stats` must say when chunks have no vector, and when it could not tell.

The reproduced failure was silent in three places at once: the index reported itself
complete, `trelix index` said "Files to embed 0", and `stats` counted 68,880 chunks
without ever asking how many of them were retrievable. This pins the third.

Two things are asserted that a single "coverage" number would break:

  * the two directions are reported SEPARATELY and never netted. Netting is exactly what
    makes `count()` lie (it is sentinel-inclusive), and the reverse direction — vectors
    with no chunk row — is the only signal for LanceDB's swallowed `delete_batch`.
  * "not recorded" is distinct from 0. An index whose `index_metadata` a kill emptied has
    no dimension AND no working `DimensionGuard`; printing 0 there would assert health
    that nothing checked.

Exit code stays 0 in every case. `tests/unit/test_cli_closed_stdout.py` exists because a
non-zero status on this command broke CI twice, and any script that pipes `stats` would
break again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.console import Console
from rich.table import Table
from typer.testing import CliRunner

runner = CliRunner()

_DIM = 4
_VEC = [0.1, 0.2, 0.3, 0.4]


def _number(output: str, label: str) -> int:
    """First number after `label`. `\\D*?` skips the table's box-drawing separator.

    Same helper as `tests/unit/test_dry_run.py` — the rows under test sit in a rich table,
    so label and value are separated by a `│`.
    """
    match = re.search(rf"{label}\D*?([\d,]+)", output)
    assert match, f"no {label!r} row in:\n{output}"
    return int(match.group(1).replace(",", ""))


def _seeded_repo(tmp_path: Path, *, chunks: int, embedded: int, record_dim: bool) -> Path:
    """An index with `chunks` chunk rows of which `embedded` have vectors.

    Built with `Database` + `VectorStore` directly, following
    `test_cli_closed_stdout.py`'s fixture: `trelix index` needs an embedder, and this
    needs neither a model nor a paid call.
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

    store = VectorStore(db_path, dimension=_DIM)
    for chunk_id in chunk_ids[:embedded]:
        store.upsert(chunk_id=chunk_id, embedding=_VEC)
    store.close()
    return repo


def _rendered(coverage: object, monkeypatch: pytest.MonkeyPatch, repo: str = "/tmp/r") -> str:
    """The coverage rows plus any remedy line, as plain text.

    Width pinned wide because rich falls back to 80 columns off a terminal and would
    ellipsize the sentence under the table — a width failure would look exactly like the
    defect under test.
    """
    from trelix.cli import main as cli_main

    recorder = Console(record=True, width=400, no_color=True, legacy_windows=False)
    monkeypatch.setattr(cli_main, "console", recorder)
    table = Table(show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    cli_main._add_coverage_rows(table, coverage)  # type: ignore[arg-type]
    recorder.print(table)
    cli_main._print_coverage_remedy(coverage, repo)  # type: ignore[arg-type]
    return recorder.export_text()


class TestTheRenderer:
    def test_holes_are_reported_in_both_directions_and_never_netted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trelix.cli.main import _VectorCoverage

        output = " ".join(
            _rendered(
                _VectorCoverage(missing=7, orphaned=3, with_vectors=61, dimension=3072), monkeypatch
            ).split()
        )
        # Both directions carry their own number; nothing netted them to 4.
        assert _number(output, "Chunks with vectors") == 61, output
        assert _number(output, "Chunks missing vectors") == 7, output
        assert _number(output, "Vectors with no chunk row") == 3, output
        assert _number(output, "Embedding dimension") == 3072, output

    def test_a_missing_dimension_says_the_guard_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row that makes the reproduced bug loud before any hole is counted."""
        from trelix.cli.main import _VectorCoverage

        output = _rendered(
            _VectorCoverage(unavailable_reason="no embedding dimension recorded"), monkeypatch
        )
        assert "not recorded" in output
        assert "dimension guard disabled" in output

    def test_an_unreadable_store_says_not_checked_rather_than_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trelix.cli.main import _VectorCoverage

        output = _rendered(
            _VectorCoverage(dimension=768, unavailable_reason="OperationalError"), monkeypatch
        )
        assert "not checked" in output
        assert "Chunks missing vectors" not in output

    def test_the_remedy_line_appears_only_when_there_are_holes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trelix.cli.main import _VectorCoverage

        holed = _rendered(
            _VectorCoverage(missing=7, orphaned=0, with_vectors=61, dimension=4), monkeypatch
        )
        assert "can never be retrieved" in holed
        assert "trelix index" in holed
        assert "--dry-run" in holed

        healthy = _rendered(
            _VectorCoverage(missing=0, orphaned=0, with_vectors=68, dimension=4), monkeypatch
        )
        assert "can never be retrieved" not in healthy


class TestTheCommand:
    def test_a_half_embedded_index_reports_its_holes(self, tmp_path: Path) -> None:
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=5, embedded=2, record_dim=True)
        result = runner.invoke(app, ["stats", str(repo)])
        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert _number(collapsed, "Chunks with vectors") == 2, collapsed
        assert _number(collapsed, "Chunks missing vectors") == 3, collapsed
        assert _number(collapsed, "Vectors with no chunk row") == 0, collapsed
        assert "can never be retrieved" in collapsed, collapsed

    def test_a_fully_embedded_index_reports_zero_and_no_remedy(self, tmp_path: Path) -> None:
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=4, embedded=4, record_dim=True)
        result = runner.invoke(app, ["stats", str(repo)])
        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert _number(collapsed, "Chunks with vectors") == 4, collapsed
        assert _number(collapsed, "Chunks missing vectors") == 0, collapsed
        assert "can never be retrieved" not in collapsed, collapsed

    def test_an_index_with_no_recorded_dimension_says_so_and_still_exits_zero(
        self, tmp_path: Path
    ) -> None:
        """The exact state the SIGKILLed index was in: index_metadata empty, so the
        store cannot be opened at a known width and the dimension guard is disarmed."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=4, embedded=1, record_dim=False)
        result = runner.invoke(app, ["stats", str(repo)])
        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert "not recorded" in collapsed, collapsed
        assert "dimension guard disabled" in collapsed, collapsed
