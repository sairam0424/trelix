"""Tests for CoIR-style evaluation harness and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trelix.eval.ndcg import mrr, ndcg_at_k, recall_at_k


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        score = ndcg_at_k(ranked, relevant, k=5)
        assert score == pytest.approx(1.0)

    def test_no_relevant_in_top_k(self) -> None:
        ranked = [10, 11, 12]
        relevant = {99}
        assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        ranked = [1, 10, 2, 11, 12]
        relevant = {1, 2}
        score = ndcg_at_k(ranked, relevant, k=5)
        assert 0.0 < score < 1.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_relevant_found(self) -> None:
        assert recall_at_k([1, 2, 3], {1, 2}, k=3) == pytest.approx(1.0)

    def test_none_found(self) -> None:
        assert recall_at_k([10, 11], {1}, k=2) == pytest.approx(0.0)

    def test_partial(self) -> None:
        assert recall_at_k([1, 10, 11], {1, 2}, k=3) == pytest.approx(0.5)


class TestMRR:
    def test_first_hit_at_rank_1(self) -> None:
        assert mrr([1, 2, 3], {1}) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self) -> None:
        assert mrr([10, 1, 2], {1}) == pytest.approx(0.5)

    def test_no_hit(self) -> None:
        assert mrr([10, 11, 12], {1}) == pytest.approx(0.0)


class TestEvalHarness:
    def test_run_returns_metrics_dict(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.eval.harness import EvalHarness

        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            json.dumps({"query": "how does auth work", "relevant_files": ["auth.py"]}) + "\n"
        )

        mock_ctx = MagicMock()
        mock_result = MagicMock()
        mock_result.file.rel_path = "auth.py"
        mock_ctx.results = [mock_result]

        config = IndexConfig(repo_path=str(tmp_path))
        harness = EvalHarness(config)

        with patch.object(harness, "_retriever") as mock_r:
            mock_r.retrieve.return_value = mock_ctx
            metrics = harness.run(str(golden))

        assert "ndcg@10" in metrics
        assert "recall@10" in metrics
        assert "mrr" in metrics
        assert 0.0 <= metrics["ndcg@10"] <= 1.0


class TestMetricsStayWithinTheirDocumentedRange:
    """nDCG and recall must stay in [0, 1] even when an ID repeats in the ranking.

    Both functions document a `[0, 1]` result, and `docs/CLI_REFERENCE.md` shows
    example output in that range. Neither deduplicated, so a repeated ID scored
    once per occurrence.

    That is not a corner case for trelix: `EvalHarness` maps chunk-level results
    onto file-level IDs (`r.file.rel_path`), and this index averages ~77 chunks per
    file, so a single relevant file routinely supplies several of the top 10 hits.
    Measured before the fix: one relevant file appearing 5 times in the top 10
    produced recall@10 = 5.0 and nDCG@10 = 2.52.
    """

    def test_recall_cannot_exceed_one(self) -> None:
        ranked = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]  # the one relevant file, 5 times
        assert recall_at_k(ranked, {0}, k=10) == 1.0

    def test_recall_cannot_exceed_one_when_every_hit_repeats(self) -> None:
        assert recall_at_k([0] * 10, {0}, k=10) == 1.0

    def test_ndcg_cannot_exceed_one(self) -> None:
        ranked = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]
        assert 0.0 <= ndcg_at_k(ranked, {0}, k=10) <= 1.0

    def test_ndcg_is_one_for_a_perfect_repeated_ranking(self) -> None:
        """A relevant file first, repeated below, is still a perfect ranking."""
        assert ndcg_at_k([0] * 10, {0}, k=10) == pytest.approx(1.0)

    def test_repeats_do_not_beat_a_cleaner_ranking(self) -> None:
        """Ranking A at 1 must not score below ranking B just because A repeats.

        This is the ordering property the inflation destroyed: padding a result
        list with duplicates used to raise the score.
        """
        relevant = {0, 1}
        padded = ndcg_at_k([0, 0, 0, 0, 1], relevant, k=10)
        clean = ndcg_at_k([0, 1], relevant, k=10)
        assert clean >= padded

    def test_partial_relevance_is_still_partial(self) -> None:
        """Deduplicating must not turn a partial hit into a perfect one."""
        # Two relevant files (0, 1); only 0 is retrieved, three times.
        assert recall_at_k([0, 0, 0, 9], {0, 1}, k=10) == pytest.approx(0.5)


class TestHarnessRanksFilesNotChunks:
    """Ground truth is file-level, so the ranking the metrics see must be too."""

    @staticmethod
    def _harness_metrics(tmp_path: Path, ranked_rel_paths: list[str], relevant: list[str]):
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.eval.harness import EvalHarness

        golden = tmp_path / "golden.jsonl"
        golden.write_text(json.dumps({"query": "q", "relevant_files": relevant}) + "\n")

        mock_ctx = MagicMock()
        mock_ctx.results = []
        for path in ranked_rel_paths:
            hit = MagicMock()
            hit.file.rel_path = path
            mock_ctx.results.append(hit)

        harness = EvalHarness(IndexConfig(repo_path=str(tmp_path)))
        with patch.object(harness, "_retriever") as mock_r:
            mock_r.retrieve.return_value = mock_ctx
            return harness.run(str(golden))

    def test_many_chunks_from_one_file_score_once(self, tmp_path: Path) -> None:
        """The realistic shape: every top hit is a different chunk of one file."""
        metrics = self._harness_metrics(
            tmp_path, ["src/walker.py"] * 8 + ["src/other.py"] * 2, ["src/walker.py"]
        )
        assert metrics["recall@10"] == 1.0
        assert 0.0 <= metrics["ndcg@10"] <= 1.0

    def test_k_counts_distinct_files(self, tmp_path: Path) -> None:
        """A relevant file at chunk-rank 11 is within the top 10 *files*.

        Ten chunks of one file are one result as far as file-level ground truth is
        concerned; collapsing them is what makes `@10` mean ten files.
        """
        ranked = ["src/noise.py"] * 10 + ["src/target.py"]
        metrics = self._harness_metrics(tmp_path, ranked, ["src/target.py"])
        assert metrics["recall@10"] == 1.0, (
            "src/target.py is the 2nd distinct file retrieved and must count as a hit"
        )

    def test_a_genuine_miss_is_still_a_miss(self, tmp_path: Path) -> None:
        """Deduplication must not manufacture hits that were never retrieved."""
        ranked = [f"src/noise{i}.py" for i in range(12)]
        metrics = self._harness_metrics(tmp_path, ranked, ["src/target.py"])
        assert metrics["recall@10"] == 0.0
        assert metrics["ndcg@10"] == 0.0
        assert metrics["mrr"] == 0.0


class TestMrrDeduplicates:
    """MRR must agree with nDCG/recall about what a rank is."""

    def test_repeats_before_the_hit_do_not_depress_the_score(self) -> None:
        # A occupies ranks 1-3 as three chunks of one file; B is the 2nd distinct file.
        assert mrr([0, 0, 0, 1], {1}) == pytest.approx(0.5)

    def test_first_rank_still_scores_one(self) -> None:
        assert mrr([1, 1, 0], {1}) == pytest.approx(1.0)

    def test_no_hit_is_still_zero(self) -> None:
        assert mrr([0, 0, 2], {9}) == 0.0


# ---------------------------------------------------------------------------
# Golden-file validation, area/limit filtering
# ---------------------------------------------------------------------------


def _write_golden(tmp_path: Path, entries: list[dict], name: str = "golden.jsonl") -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return str(path)


def _run_golden(tmp_path: Path, golden: str, ranked: dict[str, list[str]], **kwargs: object):
    """Run `golden` with a retriever that returns `ranked[query]` as chunk hits."""
    from unittest.mock import MagicMock, patch

    from trelix.core.config import IndexConfig
    from trelix.eval.harness import EvalHarness

    def retrieve(query: str, *args: object, **kw: object) -> MagicMock:
        ctx = MagicMock()
        ctx.results = []
        for rel_path in ranked[query]:
            hit = MagicMock()
            hit.file.rel_path = rel_path
            ctx.results.append(hit)
        return ctx

    harness = EvalHarness(IndexConfig(repo_path=str(tmp_path)))
    with patch.object(harness, "_retriever") as mock_r:
        mock_r.retrieve.side_effect = retrieve
        return harness.run(golden, **kwargs)  # type: ignore[arg-type]


class TestAnUnusableGoldenEntryIsRefusedNotSkipped:
    """A malformed entry must not be able to raise the reported score.

    `run()` used to `continue` past any entry whose `relevant_files` was empty or
    missing. The denominator is the number of *scored* entries, so a skipped entry
    left behind the mean of the survivors. Measured on the two-entry fixture below
    (one query answered perfectly, one missed): intact it reports 0.5 for all three
    metrics; delete `relevant_files` from the missed query and it reports 1.0 with
    `n_queries` 2 -> 1. The score went up because the fixture broke, and the only
    trace was a count in a table nobody diffs.

    Scoring such an entry 0.0 instead would be the mirror-image lie — nDCG, recall
    and MRR are undefined without ground truth, so a 0.0 there is a retrieval
    failure that never happened. The entry is refused.
    """

    def test_empty_relevant_files_raises(self, tmp_path: Path) -> None:
        golden = _write_golden(
            tmp_path,
            [
                {"query": "q1", "relevant_files": ["a.py"]},
                {"query": "q2", "relevant_files": []},
            ],
        )
        with pytest.raises(ValueError, match="line 2"):
            _run_golden(tmp_path, golden, {"q1": ["a.py"], "q2": ["z.py"]})

    def test_missing_relevant_files_key_raises(self, tmp_path: Path) -> None:
        golden = _write_golden(tmp_path, [{"query": "q1"}])
        with pytest.raises(ValueError, match="relevant_files"):
            _run_golden(tmp_path, golden, {"q1": ["a.py"]})

    def test_empty_query_raises(self, tmp_path: Path) -> None:
        golden = _write_golden(tmp_path, [{"query": "   ", "relevant_files": ["a.py"]}])
        with pytest.raises(ValueError, match="query"):
            _run_golden(tmp_path, golden, {"   ": ["a.py"]})

    def test_an_empty_golden_file_raises_instead_of_reporting_zero(self, tmp_path: Path) -> None:
        """Zeros from an empty file are indistinguishable from total retrieval failure."""
        golden = _write_golden(tmp_path, [])
        with pytest.raises(ValueError, match="no golden entries"):
            _run_golden(tmp_path, golden, {})

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text('{"query": "q1", "relevant_files": ["a.py"]}\n{"query": oops}\n')
        with pytest.raises(ValueError, match="line 2"):
            _run_golden(tmp_path, str(path), {"q1": ["a.py"]})

    def test_every_offending_line_is_reported_at_once(self, tmp_path: Path) -> None:
        """Fixing a 54-line golden set one exception per run is the wrong workflow."""
        golden = _write_golden(
            tmp_path,
            [
                {"query": "q1", "relevant_files": []},
                {"query": "q2", "relevant_files": ["./a.py"]},
                {"query": "q3", "relevant_files": ["/abs/a.py"]},
            ],
        )
        with pytest.raises(ValueError) as exc:
            _run_golden(tmp_path, golden, {})
        message = str(exc.value)
        assert "line 1" in message and "line 2" in message and "line 3" in message

    def test_a_valid_entry_still_scores(self, tmp_path: Path) -> None:
        golden = _write_golden(tmp_path, [{"query": "q1", "relevant_files": ["src/a.py"]}])
        metrics = _run_golden(tmp_path, golden, {"q1": ["src/a.py"]})
        assert metrics == {"ndcg@10": 1.0, "recall@10": 1.0, "mrr": 1.0, "n_queries": 1.0}


class TestNonNormalisedPathsAreRefused:
    """`rel_path` comparison is exact string equality, so near-misses score 0.

    `FileWalker` builds `rel_path` as `str(path.relative_to(repo_root))` — no `./`
    prefix, no absolute paths, POSIX separators on POSIX. Measured before this
    change: a golden entry naming `./a.py` against a retrieved `a.py` scored
    nDCG@10 = recall@10 = MRR = 0.0, byte-identical to a genuine miss. Normalising
    the path silently would hide the fixture bug; the entry is refused instead.
    """

    @pytest.mark.parametrize(
        "bad_path",
        ["./a.py", "/abs/a.py", "src\\a.py", "src/../a.py", "src//a.py", "src/a.py/", "../a.py"],
    )
    def test_refused(self, tmp_path: Path, bad_path: str) -> None:
        golden = _write_golden(tmp_path, [{"query": "q1", "relevant_files": [bad_path]}])
        with pytest.raises(ValueError, match="line 1"):
            _run_golden(tmp_path, golden, {"q1": ["a.py"]})

    def test_a_normalised_relative_posix_path_is_accepted(self, tmp_path: Path) -> None:
        golden = _write_golden(
            tmp_path, [{"query": "q1", "relevant_files": ["src/trelix/eval/harness.py"]}]
        )
        metrics = _run_golden(tmp_path, golden, {"q1": ["src/trelix/eval/harness.py"]})
        assert metrics["recall@10"] == 1.0


class TestAreaAndLimitFilters:
    """Per-area scores are the only way to see a per-area regression.

    `eval/golden.jsonl` carries 54 queries across six areas (10 each except `ops`,
    which has 4); the area labels live in the sibling `golden-metadata.json`. An
    aggregate over all 54 hides a collapse confined to one area, and until now the
    aggregate was the only number obtainable.
    """

    @staticmethod
    def _fixture(tmp_path: Path) -> tuple[str, dict[str, list[str]]]:
        golden = _write_golden(
            tmp_path,
            [
                {"query": "q1", "relevant_files": ["a.py"]},
                {"query": "q2", "relevant_files": ["b.py"]},
                {"query": "q3", "relevant_files": ["c.py"]},
            ],
        )
        (tmp_path / "golden-metadata.json").write_text(
            json.dumps(
                {
                    "queries": [
                        {"query": "q1", "area": "indexing"},
                        {"query": "q2", "area": "storage"},
                        {"query": "q3", "area": "storage"},
                    ]
                }
            )
        )
        # q1 hits, q2/q3 miss — so the two areas must score differently.
        ranked = {"q1": ["a.py"], "q2": ["z.py"], "q3": ["z.py"]}
        return golden, ranked

    def test_area_selects_only_that_areas_queries(self, tmp_path: Path) -> None:
        golden, ranked = self._fixture(tmp_path)
        assert _run_golden(tmp_path, golden, ranked, area="storage") == {
            "ndcg@10": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "n_queries": 2.0,
        }
        assert _run_golden(tmp_path, golden, ranked, area="indexing")["n_queries"] == 1.0

    def test_an_inline_area_key_wins_over_the_sidecar(self, tmp_path: Path) -> None:
        golden = _write_golden(
            tmp_path, [{"query": "q1", "relevant_files": ["a.py"], "area": "ops"}]
        )
        assert _run_golden(tmp_path, golden, {"q1": ["a.py"]}, area="ops")["n_queries"] == 1.0

    def test_an_unknown_area_lists_the_areas_that_exist(self, tmp_path: Path) -> None:
        golden, ranked = self._fixture(tmp_path)
        with pytest.raises(ValueError, match="indexing, storage"):
            _run_golden(tmp_path, golden, ranked, area="retrieval")

    def test_area_without_any_labels_says_so(self, tmp_path: Path) -> None:
        golden = _write_golden(tmp_path, [{"query": "q1", "relevant_files": ["a.py"]}])
        with pytest.raises(ValueError, match="no area labels"):
            _run_golden(tmp_path, golden, {"q1": ["a.py"]}, area="storage")

    def test_partially_labelled_lines_are_named(self, tmp_path: Path) -> None:
        """An unlabelled entry can never be selected by --area, so it must not be silent."""
        golden = _write_golden(
            tmp_path,
            [
                {"query": "q1", "relevant_files": ["a.py"], "area": "ops"},
                {"query": "q2", "relevant_files": ["b.py"]},
            ],
        )
        with pytest.raises(ValueError, match="line 2"):
            _run_golden(tmp_path, golden, {"q1": ["a.py"], "q2": ["b.py"]}, area="ops")

    def test_limit_takes_the_first_n_in_file_order(self, tmp_path: Path) -> None:
        golden, ranked = self._fixture(tmp_path)
        metrics = _run_golden(tmp_path, golden, ranked, limit=1)
        assert metrics["n_queries"] == 1.0
        assert metrics["recall@10"] == 1.0, "q1 is the first line and it hits"

    def test_limit_composes_with_area(self, tmp_path: Path) -> None:
        golden, ranked = self._fixture(tmp_path)
        assert _run_golden(tmp_path, golden, ranked, area="storage", limit=1)["n_queries"] == 1.0

    def test_limit_below_one_is_refused(self, tmp_path: Path) -> None:
        golden, ranked = self._fixture(tmp_path)
        with pytest.raises(ValueError, match="limit"):
            _run_golden(tmp_path, golden, ranked, limit=0)
