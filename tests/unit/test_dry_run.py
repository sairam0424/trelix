"""`trelix index --dry-run` must price the run without buying any of it.

`scripts/self-index.sh --dry-run` already answers "which files would be indexed", so the
only reason for a flag on the CLI is the question that script does not ask: what will this
cost. That makes two things load-bearing, and both are pinned here.

1. **Nothing is embedded.** The Indexer is replaced by a class that fails if it is
   constructed at all, so a preview that quietly instantiates an embedder — which for
   `local` downloads a model, and for `openai` bills — fails the test rather than the user.

2. **An unpriceable provider says "unknown".** A cost preview exists to be trusted, so a
   made-up rate is worse than no number: Azure bills by deployment name and tier, and the
   deployment name is chosen by whoever created it, so it cannot be priced from config.
   The table must say so instead of quoting OpenAI's list price.

The token count itself comes from the chunker's own tiktoken encoder, not from a
characters/4 heuristic, so it is the same number Phase 3 reports when the run is real.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trelix.core.config import IndexConfig
from trelix.core.models import IndexedFile
from trelix.indexing.walker import FileWalker
from trelix.store.db import Database

runner = CliRunner()

_SAMPLE = '''
def greet(name: str) -> str:
    """Say hello to name."""
    return f"hello {name}"


class Greeter:
    def shout(self, name: str) -> str:
        """Louder."""
        return greet(name).upper()
'''


@pytest.fixture(autouse=True)
def _indexer_must_not_be_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run that constructs an Indexer has already paid for a model load.

    `Indexer.__init__` is what calls `make_embedder`, so patching the constructor rather
    than the class pins "nothing was embedded" while leaving the rest of the class
    reachable — the preview borrows `_parse_one` from it on purpose.
    """
    from trelix.indexing.indexer import Indexer

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--dry-run constructed an Indexer; it must embed nothing")

    monkeypatch.setattr(Indexer, "__init__", _explode)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "sample.py").write_text(_SAMPLE, encoding="utf-8")
    return tmp_path


def _run(repo: Path, *extra: str) -> str:
    from trelix.cli.main import app

    result = runner.invoke(app, ["index", str(repo), "--dry-run", *extra])
    assert result.exit_code == 0, result.output
    # Rich wraps at 80 columns off a terminal, so a phrase under test can straddle a
    # newline. Collapsing whitespace keeps these assertions about content, not width.
    return " ".join(result.output.split())


def _number(output: str, label: str) -> int:
    """First number after `label`. `\\D*?` skips the table's box-drawing separator."""
    match = re.search(rf"{label}\D*?([\d,]+)", output)
    assert match, f"no {label!r} row in:\n{output}"
    return int(match.group(1).replace(",", ""))


class TestItCountsWhatWouldBeEmbedded:
    def test_tokens_and_chunks_are_reported(self, repo: Path) -> None:
        output = _run(repo)
        assert _number(output, "Chunks to embed") >= 2, output  # greet + Greeter.shout
        assert _number(output, "Embedding tokens") > 0, output

    def test_files_already_indexed_and_unchanged_are_not_charged_for(self, repo: Path) -> None:
        """The real run skips them on hash, so a preview that counts them overstates cost."""
        config = IndexConfig(repo_path=str(repo))
        walked = list(FileWalker(config).walk())
        assert walked, "walker yielded nothing — the fixture is not indexable"
        with Database(config.db_path_absolute) as db:
            for file in walked:
                db.upsert_file(file)

        output = _run(repo)
        assert _number(output, "Files to embed") == 0, output
        assert _number(output, "Embedding tokens") == 0, output

    def test_a_changed_file_is_charged_for_again(self, repo: Path) -> None:
        config = IndexConfig(repo_path=str(repo))
        with Database(config.db_path_absolute) as db:
            for file in FileWalker(config).walk():
                db.upsert_file(
                    IndexedFile(
                        path=file.path,
                        rel_path=file.rel_path,
                        language=file.language,
                        hash="0" * 64,  # stale on purpose
                        size_bytes=file.size_bytes,
                    )
                )
        output = _run(repo)
        assert _number(output, "Files to embed") == 1, output
        assert _number(output, "Embedding tokens") > 0, output


class TestItPricesTheRepairOfAnInterruptedRun:
    """A run killed mid-Phase-3 leaves chunk rows with no vector, and the next
    `trelix index` re-embeds exactly those. The preview must show that work and, crucially,
    include it in the dollar figure — otherwise it under-quotes the very run it is pricing,
    which is the same class of dishonesty as an index reporting itself complete while 10.5%
    of it is unretrievable.

    A repair test can live in this file even though the autouse fixture forbids building an
    Indexer: the dry run builds none, and the prior state is seeded through `Database` and
    `VectorStore` directly, as the tests above already do.
    """

    def _seed(self, repo: Path, *, embed_all: bool) -> None:
        """Index the fixture's chunk rows, giving vectors to all of them or to none."""
        from trelix.indexing.chunker import Chunker
        from trelix.indexing.indexer import Indexer
        from trelix.store.vector import VectorStore

        config = IndexConfig(repo_path=str(repo))
        chunker = Chunker(config.chunker)
        chunk_ids: list[int] = []
        with Database(config.db_path_absolute) as db:
            for file in FileWalker(config).walk():
                file_id = db.upsert_file(file)
                parsed = Indexer._parse_one(None, file)  # type: ignore[arg-type]
                assert parsed.parse_result is not None
                for symbol in parsed.parse_result.symbols:
                    symbol.file_id = file_id
                    symbol.id = db.insert_symbol(symbol)
                for chunk in chunker.build_chunks(
                    symbols=parsed.parse_result.symbols,
                    imports=[],
                    file_rel_path=file.rel_path,
                    language=file.language.value,
                    parent_symbols={s.id: s for s in parsed.parse_result.symbols if s.id},
                ):
                    chunk_ids.append(db.insert_chunk(chunk))
            db.set_embedding_dimension(4)
        assert chunk_ids, "fixture produced no chunks"

        store = VectorStore(config.db_path_absolute, dimension=4)
        if embed_all:
            store.upsert_batch([(cid, [0.1, 0.2, 0.3, 0.4]) for cid in chunk_ids])
        store.close()

    def test_vector_less_chunks_are_counted_and_their_tokens_priced(self, repo: Path) -> None:
        self._seed(repo, embed_all=False)
        output = _run(repo)
        # Every file is unchanged on hash, so the from-scratch rows are zero — and the
        # preview still has work to report. That combination is the bug: it used to print
        # only the zeros.
        assert _number(output, "Files to embed") == 0, output
        assert _number(output, "Chunks to embed") == 0, output
        assert _number(output, "Chunks missing vectors") >= 2, output
        assert _number(output, "Repair tokens") > 0, output

    def test_a_fully_covered_index_reports_no_repair(self, repo: Path) -> None:
        self._seed(repo, embed_all=True)
        output = _run(repo)
        assert _number(output, "Chunks missing vectors") == 0, output
        assert _number(output, "Repair tokens") == 0, output

    def test_the_priced_total_includes_the_repair_tokens(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A preview that prices only the from-scratch tokens quotes $0 for a run that
        will spend money."""
        monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "openai")
        monkeypatch.setenv("TRELIX_EMBEDDER_OPENAI_MODEL", "text-embedding-3-large")
        self._seed(repo, embed_all=False)
        output = _run(repo)

        repair_tokens = _number(output, "Repair tokens")
        assert repair_tokens > 0, output
        assert _number(output, "Embedding tokens") == 0, output

        # The load-bearing assertion: the token count the estimate is BUILT ON, not the
        # rounded dollar string. At fixture scale 124 tokens prices below $0.0001 and
        # formats as $0.0000, so asserting on the dollars would test rich's float format
        # rather than whether the repair reached the estimate at all — which before this
        # change it did not.
        priced = re.search(r"Estimated cost.*?([\d,]+) tokens at", output)
        assert priced, output
        assert int(priced.group(1).replace(",", "")) == repair_tokens, output

        rate = re.search(r"\$([\d.]+) per 1M tokens", output)
        assert rate, output
        cost = re.search(r"Estimated cost\s+\$([\d.]+)", output)
        assert cost, output
        expected = repair_tokens / 1_000_000 * float(rate.group(1))
        assert abs(float(cost.group(1)) - expected) < 0.005, output

    def test_a_chunk_past_the_sentinel_offset_is_not_billed(self, repo: Path) -> None:
        """No real run will ever embed it, so quoting it quotes a spend that cannot happen.

        `Indexer._chunks_missing_vectors` excludes ids at or above the sub-chunk offset from
        the repair — every backend's `stored_chunk_ids()` filters them out as sentinels and
        `search()` drops their vectors regardless — and this preview has to agree with the run
        it is pricing, in both directions.
        """
        import sqlite3

        from trelix.store.vector import BaseVectorStore

        self._seed(repo, embed_all=True)
        db_path = IndexConfig(repo_path=str(repo)).db_path_absolute
        conn = sqlite3.connect(str(db_path))
        try:
            symbol_id = int(conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()[0])
            conn.execute(
                "INSERT INTO chunks (id, symbol_id, chunk_text, token_count) VALUES (?, ?, ?, ?)",
                (BaseVectorStore._SUB_CHUNK_OFFSET, symbol_id, "past the offset", 999),
            )
            conn.commit()
        finally:
            conn.close()

        output = _run(repo)
        assert _number(output, "Chunks missing vectors") == 0, output
        assert _number(output, "Repair tokens") == 0, output

    def test_a_non_sqlite_backend_says_not_checked_rather_than_zero(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opening a Lance or Qdrant handle is a WRITE — `_get_or_create_table` creates on
        open failure and `_ensure_collection` creates a collection — and a cost preview must
        not be able to modify the index it prices. So it declines, visibly."""
        monkeypatch.setenv("TRELIX_STORE_BACKEND", "lance")
        output = _run(repo)
        assert "not checked" in output, output
        assert "lance" in output, output
        # And the dollar figure says which direction it is wrong in. Declining to price the
        # repair does not mean the run will decline to perform it.
        assert "LOWER BOUND" in " ".join(output.split()), output


class TestPricingIsHonest:
    def test_a_known_openai_model_is_priced_from_the_token_count(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "openai")
        monkeypatch.setenv("TRELIX_EMBEDDER_OPENAI_MODEL", "text-embedding-3-large")
        output = _run(repo)

        tokens = _number(output, "Embedding tokens")
        cost = re.search(r"Estimated cost\s+\$([\d.]+)", output)
        assert cost, output
        # Pins the arithmetic, not the rate: the rate is asserted to be the one printed.
        rate = re.search(r"\$([\d.]+) per 1M tokens", output)
        assert rate, output
        expected = tokens / 1_000_000 * float(rate.group(1))
        assert abs(float(cost.group(1)) - expected) < 0.005, output

    def test_azure_is_unknown_rather_than_guessed(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Azure bills per deployment and tier; the deployment name carries no rate."""
        monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "azure")
        output = _run(repo)
        assert "unknown" in output.lower(), output
        assert not re.search(r"Estimated cost\s+\$", output), output

    def test_a_local_provider_reports_no_api_spend(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "local")
        output = _run(repo)
        assert "no API cost" in output, output

    def test_a_non_openai_tokenizer_is_disclosed_as_approximate(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Voyage does not tokenise with cl100k_base, so its bill will differ."""
        monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "voyage")
        output = _run(repo)
        assert "cl100k_base" in output, output


class TestTheCostsItCannotPriceAreNamed:
    def test_contextual_chunking_is_flagged_as_unpriced_llm_calls(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_CHUNKER_CONTEXTUAL", "true")
        output = _run(repo)
        assert "contextual" in output.lower(), output

    def test_file_summaries_are_flagged_as_unpriced_llm_calls(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_FILE_SUMMARIES_ENABLED", "true")
        output = _run(repo)
        assert "summar" in output.lower(), output


class TestFlagsThatContradictEachOther:
    def test_dry_run_with_yes_is_refused(self, repo: Path) -> None:
        from trelix.cli.main import app

        result = runner.invoke(app, ["index", str(repo), "--dry-run", "--yes"])
        assert result.exit_code != 0
        assert "--dry-run" in result.output


class TestTheBorrowedParseStaysBorrowable:
    def test_parse_one_touches_no_instance_state(self) -> None:
        """The preview calls `Indexer._parse_one` unbound, with None for self.

        Borrowing it rather than copying it is deliberate: a private copy in the CLI would
        drift from the parse whose cost it estimates — silently mis-counting the
        line-window fallback, which on this repository is 12 files the extractors cannot
        parse. The price of borrowing is this constraint, so it is pinned here rather than
        left to be discovered as an AttributeError inside a user's dry run.
        """
        import inspect

        from trelix.indexing.indexer import Indexer

        body = inspect.getsource(Indexer._parse_one)
        _signature, _, statements = body.partition(") -> _ParsedFile:")
        assert statements, "the _parse_one signature moved — re-check the unbound call"
        assert "self." not in statements, (
            "_parse_one now reads instance state, so cli/main.py's unbound call in "
            "_print_cost_preview will raise AttributeError on None. Either keep it "
            "state-free or give the preview a real (embedder-free) construction path."
        )


class TestItDoesNotDuplicateTheShellScript:
    def test_the_shell_dry_run_still_owns_file_discovery_reporting(self) -> None:
        """Division of labour: the script answers "which files", the flag answers "what cost".

        Pinned because the cheap way to write this feature is to reimplement the script's
        per-directory breakdown in Python, which leaves two things to keep in step and
        still does not tell anyone what a run will cost.
        """
        script = Path(__file__).resolve().parents[2] / "scripts" / "self-index.sh"
        body = script.read_text(encoding="utf-8")
        assert "--dry-run" in body
        assert "files would be indexed" in body
        assert "Estimated cost" not in body
