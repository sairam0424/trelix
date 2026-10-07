"""`chunks_from_cache` on the index-time paths `test_embedding_cache_indexer.py` does not
reach: the Phase 2.5/2.6 side embeds of the batch pipeline, the streaming pipeline's
crash-recovery repair, and `index_file()` (its summary side embed and its early return for
an unchanged file).

Same real-Indexer harness as that file, imported from it rather than copied (it is at the
500-line limit). Each test names the mutation that must fail it.
"""

from __future__ import annotations

import pathlib
import sqlite3
from typing import Any
from unittest.mock import ANY

import pytest
import sqlite_vec

from tests.fixtures.fakes import CountingEmbedder
from tests.unit.test_embedding_cache_indexer import (
    _DIM,
    _config,
    _enable_cache,
    _fresh_db,
    _indexer,
    _quiet_progress,
    _repo,
    _run,
)


class _ConstantSummarizer:
    """`FileSummarizer.summarize` without an LLM: one fixed line per file, so Phase 2.5
    generates, stores and EMBEDS a summary for every file through the same wrapper."""

    def summarize(self, *, rel_path: str, symbols: object, language: object) -> str:
        return f"Summary of {rel_path}"


def _indexer_with_side_embeds(repo: pathlib.Path, embedder: CountingEmbedder) -> Any:
    """A real Indexer whose Phase 2.5 (file summaries) and Phase 2.6 (multi-granularity
    sub-chunks) both run. Neither needs the network: the summarizer above is constant and
    the sub-chunker is tree-sitter over the symbol body."""
    cfg = _config(repo)
    cfg.chunker.multi_granularity_enabled = True
    indexer = _indexer(cfg, embedder)
    indexer._file_summarizer = _ConstantSummarizer()
    return indexer


def _delete_one_vector(repo: pathlib.Path) -> None:
    """Open one hole: the vector row of the lowest chunk id, which is what a kill between
    "chunk row committed" and "vector written" leaves (test_indexer_vector_repair.py)."""
    conn = sqlite3.connect(str(repo / ".trelix" / "index.db"))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = (SELECT min(id) FROM chunks)")
        conn.commit()
    finally:
        conn.close()


class TestSideEmbedsAndRepairAccounting:
    def test_side_embed_hits_are_not_chunks_from_cache(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `hits_before = self._cache_hits()` taken before `_summarize_files`
        (Phase 2.5 summary hits counted) or before `_insert_and_chunk_all` (Phase 2.6
        sub-chunk hits counted too): on the warm run `chunks_from_cache` overshoots
        `chunks_embedded`.

        Side embeds go through the same wrapper, so they ARE cached (the warm run makes no
        provider call at all), but they are not chunks: `chunks_from_cache` means Phase-3
        chunks, as `chunks_embedded` does. The first-run assertions prove this repo really
        produced both kinds of side embed, otherwise the warm-run equality would be vacuous.
        """
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")

        first_embedder = CountingEmbedder(_DIM)
        first = _run(_indexer_with_side_embeds(repo, first_embedder))
        assert first["chunks_embedded"] > 0
        assert first["file_summaries_embedded"] == 3
        side_embeds = len(first_embedder.texts) - first["chunks_embedded"]
        assert side_embeds > 3  # the three summaries plus at least one sub-chunk
        assert first["chunks_from_cache"] == 0

        _fresh_db(repo)
        again = CountingEmbedder(_DIM)
        second = _run(_indexer_with_side_embeds(repo, again))
        assert again.calls == []
        assert second["file_summaries_embedded"] == 3
        assert second["chunks_embedded"] == first["chunks_embedded"]
        assert second["chunks_from_cache"] == second["chunks_embedded"]

    def test_index_file_does_not_count_the_summary_hit_as_a_chunk(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`index_file()` builds a summary request whenever a file summarizer is configured
        (watch mode and `trelix update-index` with summaries on), and on a warm cache that
        summary vector is a hit like the chunks are. MUTATION: `hits_before` taken before
        `_summarize_files` in `index_file` (the summary hit is reported as a chunk:
        `chunks_from_cache` reads 2 for alpha.py's one chunk and exceeds `chunks_updated`).
        """
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        first = _run(_indexer_with_side_embeds(repo, CountingEmbedder(_DIM)))
        assert first["file_summaries_embedded"] == 3

        _fresh_db(repo)
        again = CountingEmbedder(_DIM)
        with _quiet_progress():
            result = _indexer_with_side_embeds(repo, again).index_file(str(repo / "alpha.py"))

        assert result["status"] == "ok"
        assert again.calls == []  # the summary and the chunk were both hits
        assert result["chunks_updated"] == 1
        assert result["chunks_from_cache"] == 1

    def test_the_streaming_repair_counts_its_cache_hits(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the `results["chunks_from_cache"] += self._cache_hits() - hits_before`
        line after the streaming repair deleted (the repaired chunk is served from the
        cache, lands in `chunks_reconciled` and `chunks_embedded`, and is missing from
        `chunks_from_cache`, so the two pipelines no longer report the same thing).

        Warm cache, one chunk without its vector, every file unchanged: the repair is the
        only embed of the run, and it is a hit.
        """
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        _run(_indexer(_config(repo, streaming=True), CountingEmbedder(_DIM)))
        _delete_one_vector(repo)

        again = CountingEmbedder(_DIM)
        stats = _run(_indexer(_config(repo, streaming=True), again))
        assert again.calls == []
        assert stats["files_skipped"] == 3
        assert stats["chunks_missing_vectors"] == 1
        assert stats["chunks_reconciled"] == 1
        assert stats["chunks_embedded"] == 1
        assert stats["chunks_from_cache"] == 1

    def test_index_file_on_an_unchanged_file_keeps_the_documented_shape(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`trelix update-index <unchanged file>` prints `index_file()`'s dict as JSON, and
        the CHANGELOG says `chunks_from_cache` is on every `ok` result. MUTATION: the key
        dropped from the `skipped: True` early return (`_index_streaming` hides it behind
        `.get(.., 0)`, so only the printed shape changes)."""
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        with _quiet_progress():
            indexer = _indexer(_config(repo), CountingEmbedder(_DIM))
            first = indexer.index_file(str(repo / "alpha.py"))
            unchanged = indexer.index_file(str(repo / "alpha.py"))

        assert first["status"] == "ok" and "skipped" not in first
        assert unchanged == {
            "status": "ok",
            "symbols_updated": 0,
            "chunks_updated": 0,
            "chunks_from_cache": 0,
            "ms": ANY,
            "skipped": True,
        }
