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
  * a missing dimension suppresses NEITHER number. That state is exactly what a kill
    leaves, so gating the coverage block on a recorded dimension made this command blind
    on the only index that ever needed it. The width must not be guessed; a read-only
    projection of `chunk_id` needs no width and creates nothing.
  * id-space exhaustion is its own answer, with its own remedy. A chunk row past the
    sub-chunk offset is unretrievable but is NOT a hole `trelix index` can fill, and
    counting it as one printed a remedy that could never clear while charging for it
    every run.

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
        # Explicit, because `insert_symbol` / `insert_chunk` do not commit and neither does
        # `close()` — only `set_index_metadata` did, so `record_dim=False` used to leave a
        # fixture with ZERO chunk rows on disk. That went unnoticed while the only assertion
        # on that branch was a string match; the coverage numbers below depend on the rows.
        db._conn.commit()

    store = VectorStore(db_path, dimension=_DIM)
    for chunk_id in chunk_ids[:embedded]:
        store.upsert(chunk_id=chunk_id, embedding=_VEC)
    store.close()
    return repo


def _rendered(
    coverage: object,
    monkeypatch: pytest.MonkeyPatch,
    repo: str = "/tmp/r",
    backend: str = "sqlite",
) -> str:
    """The coverage rows plus any remedy line, as plain text.

    Width pinned wide because rich falls back to 80 columns off a terminal and would
    ellipsize the sentence under the table — a width failure would look exactly like the
    defect under test.

    `backend` reaches the remedy through a real `IndexConfig`: the id-space remedy names
    where the vectors live, and on lance/qdrant that is not the index file. Its `repo_path`
    is the temp dir rather than `repo`, because `IndexConfig` validates that the path exists
    and nothing here opens the store — only the rendered sentence is under test.
    """
    import tempfile

    from trelix.cli import main as cli_main
    from trelix.core.config import IndexConfig, StoreConfig

    config = IndexConfig(  # type: ignore[arg-type]
        repo_path=tempfile.gettempdir(), store=StoreConfig(backend=backend)
    )
    recorder = Console(record=True, width=400, no_color=True, legacy_windows=False)
    monkeypatch.setattr(cli_main, "console", recorder)
    table = Table(show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    cli_main._add_coverage_rows(table, coverage)  # type: ignore[arg-type]
    recorder.print(table)
    cli_main._print_coverage_remedy(coverage, repo, config)  # type: ignore[arg-type]
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

    def test_the_remedy_does_not_promise_that_nothing_else_is_re_embedded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain `trelix index` also embeds every file whose content hash moved, which is
        its job. Promising it "will not re-embed anything else" under-quotes the run the
        line is recommending."""
        from trelix.cli.main import _VectorCoverage

        output = " ".join(
            _rendered(
                _VectorCoverage(missing=7, orphaned=0, with_vectors=61, dimension=4), monkeypatch
            ).split()
        )
        assert "will not re-embed anything else" not in output, output
        assert "on top of anything that changed since the last run" in output, output

    def test_the_remedy_says_watch_mode_will_not_repair_the_holes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Watch mode deliberately does not heal — a save event on one file is the wrong
        trigger for a repo-wide store scan — and this is the command the CHANGELOG says
        states it. It was stated only in a source comment, so the one user who runs
        `trelix watch` and waits for the holes to clear waits forever."""
        from trelix.cli.main import _VectorCoverage

        output = " ".join(
            _rendered(
                _VectorCoverage(missing=7, orphaned=0, with_vectors=61, dimension=4), monkeypatch
            ).split()
        )
        assert "trelix watch" in output, output
        assert "will not repair" in output, output

    def test_a_healthy_index_is_not_told_about_watch_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It belongs to the remedy, not to the table: with no holes there is nothing for
        watch mode to fail to repair."""
        from trelix.cli.main import _VectorCoverage

        output = _rendered(
            _VectorCoverage(missing=0, orphaned=0, with_vectors=68, dimension=4), monkeypatch
        )
        assert "trelix watch" not in output, output

    def test_the_id_space_remedy_names_the_vector_store_not_just_the_index_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "Remove the index file" renumbers `chunks` from 1, which is the point — but on
        lance/qdrant the vectors do not live in that file, so a user who follows it clears
        the id-space row and strands every old vector as a `Vectors with no chunk row`
        orphan that nothing reclaims. `_partial_index_error` already names both."""
        from trelix.cli.main import _VectorCoverage

        output = " ".join(
            _rendered(
                _VectorCoverage(
                    missing=0, orphaned=0, with_vectors=61, dimension=4, id_space_exhausted=2
                ),
                monkeypatch,
                backend="lance",
            ).split()
        )
        assert ".trelix/lance" in output, output
        assert "index.db" in output, output

    def test_id_space_exhaustion_gets_its_own_row_and_its_own_remedy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never merged into the hole count, and never sent to `trelix index`: re-embedding
        an id past the sub-chunk offset is money for a vector `search()` filters out."""
        from trelix.cli.main import _VectorCoverage

        output = " ".join(
            _rendered(
                _VectorCoverage(
                    missing=0, orphaned=0, with_vectors=61, dimension=4, id_space_exhausted=2
                ),
                monkeypatch,
            ).split()
        )
        assert _number(output, "Chunks missing vectors") == 0, output
        assert _number(output, "Chunks past the id-space limit") == 2, output
        assert "re-key" in output, output
        assert "can never be retrieved" not in output, output

    def test_a_healthy_index_shows_no_id_space_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A permanent `0` row for a condition no normal index is in invites it to be read
        as a hole count."""
        from trelix.cli.main import _VectorCoverage

        output = _rendered(
            _VectorCoverage(
                missing=0, orphaned=0, with_vectors=68, dimension=4, id_space_exhausted=0
            ),
            monkeypatch,
        )
        assert "id-space" not in output, output


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

    def test_an_index_with_no_recorded_dimension_still_reports_its_coverage(
        self, tmp_path: Path
    ) -> None:
        """The exact state the SIGKILLed index was in — `index_metadata` with 0 rows — and
        therefore the one state this reporting has to work in.

        It previously answered "not checked — no embedding dimension recorded" here, so on
        the very index the coverage feature exists for, `stats` reported neither the holes
        nor the remedy. The gate was right that a width must never be GUESSED (a vec0 table's
        dimension is fixed at creation), but a read-only projection of `chunk_id` needs no
        width and creates nothing — `--dry-run` prices the identical question that way.

        The disarmed-guard row still prints: it is a second, independent finding, not a
        substitute for the coverage numbers.
        """
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=4, embedded=1, record_dim=False)
        result = runner.invoke(app, ["stats", str(repo)])
        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert "not recorded" in collapsed, collapsed
        assert "dimension guard disabled" in collapsed, collapsed
        assert "not checked" not in collapsed, collapsed
        assert _number(collapsed, "Chunks with vectors") == 1, collapsed
        assert _number(collapsed, "Chunks missing vectors") == 3, collapsed
        assert _number(collapsed, "Vectors with no chunk row") == 0, collapsed
        assert "can never be retrieved" in collapsed, collapsed

    def test_the_command_says_watch_mode_will_not_repair(self, tmp_path: Path) -> None:
        """End to end on the fixture shape the real damaged index had — `index_metadata`
        emptied, so no recorded dimension — because that is the index whose owner is most
        likely to leave `trelix watch` running and expect the holes to close."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=4, embedded=1, record_dim=False)
        result = runner.invoke(app, ["stats", str(repo)])
        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert "trelix watch" in collapsed, collapsed
        assert "will not repair" in collapsed, collapsed

    def test_a_healthy_index_with_no_recorded_dimension_reports_zero_holes(
        self, tmp_path: Path
    ) -> None:
        """The other half: the dimension-free path must not report a fully embedded index as
        holed, or the remedy it prints is a bill for nothing."""
        from trelix.cli.main import app

        repo = _seeded_repo(tmp_path, chunks=4, embedded=4, record_dim=False)
        result = runner.invoke(app, ["stats", str(repo)])
        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert _number(collapsed, "Chunks with vectors") == 4, collapsed
        assert _number(collapsed, "Chunks missing vectors") == 0, collapsed
        assert "can never be retrieved" not in collapsed, collapsed

    def test_a_chunk_past_the_id_space_limit_is_not_counted_as_a_hole(self, tmp_path: Path) -> None:
        """End to end: `all_chunk_ids()`'s partition reaching the rendered table.

        Without it, `stats` prints "N chunk(s) can never be retrieved" with a `trelix index`
        remedy that can never clear it — and `trelix index` charges for it every run.
        """
        import sqlite3

        from trelix.cli.main import app
        from trelix.core.config import IndexConfig
        from trelix.store.vector import BaseVectorStore

        repo = _seeded_repo(tmp_path, chunks=3, embedded=3, record_dim=True)
        db_path = IndexConfig(repo_path=str(repo)).db_path_absolute
        conn = sqlite3.connect(str(db_path))
        try:
            symbol_id = int(conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()[0])
            conn.execute(
                "INSERT INTO chunks (id, symbol_id, chunk_text, token_count) VALUES (?, ?, ?, ?)",
                (BaseVectorStore._SUB_CHUNK_OFFSET, symbol_id, "past the offset", 4),
            )
            conn.commit()
        finally:
            conn.close()

        result = runner.invoke(app, ["stats", str(repo)])
        assert result.exit_code == 0, result.output
        collapsed = " ".join(result.output.split())
        assert _number(collapsed, "Chunks missing vectors") == 0, collapsed
        assert _number(collapsed, "Chunks past the id-space limit") == 1, collapsed
        assert "can never be retrieved" not in collapsed, collapsed
        assert "re-key" in collapsed, collapsed
