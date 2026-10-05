"""Per-query eval records: `EvalHarness.run_detailed`, the JSON file, and `trelix eval`.

The fixture is three queries whose metrics were worked out by hand (DCG = sum of
1 / log2(rank + 1) over the relevant files found; ideal DCG puts every relevant file at
the top):

* ``q one``   — the one relevant file at rank 1:                 nDCG 1, recall 1, MRR 1.
* ``q two``   — two relevant files at ranks 2 and 4:
                nDCG = (1/log2(3) + 1/log2(5)) / (1 + 1/log2(3)) = 0.65092...,
                recall 1, MRR 1/2.
* ``q three`` — the one relevant file at rank 10, the edge of ``@10`` (at @9 it would
                score 0): nDCG = 1/log2(11) = 0.28906..., recall 1, MRR 1/10.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from trelix.cli import main as cli_main
from trelix.eval import harness as harness_module
from trelix.eval.harness import EvalHarness, QueryRecord, aggregate_metrics

_GOLDEN: list[dict[str, Any]] = [
    {"query": "q one", "relevant_files": ["a.py"]},
    {"query": "q two", "relevant_files": ["b.py", "c.py"]},
    {"query": "q three", "relevant_files": ["t.py"]},
]
_RANKED: dict[str, list[str] | Exception] = {
    "q one": ["a.py", "x.py", "y.py"],
    "q two": ["x.py", "b.py", "y.py", "c.py"],
    "q three": [f"n{i}.py" for i in range(1, 10)] + ["t.py"],
}


class _StubRetriever:
    """Returns the registered ranking for a query, or raises the registered exception."""

    def __init__(self, ranked: dict[str, list[str] | Exception]) -> None:
        self._ranked = ranked

    def retrieve(self, query: str) -> Any:
        outcome = self._ranked[query]
        if isinstance(outcome, Exception):
            raise outcome
        hits = [SimpleNamespace(file=SimpleNamespace(rel_path=path)) for path in outcome]
        return SimpleNamespace(results=hits, rerank=None)


def _write_golden(directory: Path, entries: list[dict[str, Any]] = _GOLDEN) -> str:
    path = directory / "golden.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return str(path)


def _harness(repo: Path, ranked: dict[str, list[str] | Exception] = _RANKED) -> EvalHarness:
    """An EvalHarness with no Retriever and no database."""
    harness = EvalHarness.__new__(EvalHarness)
    harness._config = SimpleNamespace(repo_path=str(repo))  # type: ignore[assignment]
    harness._retriever = _StubRetriever(ranked)  # type: ignore[assignment]
    harness._rerank_outcomes = ()
    return harness


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    directory = tmp_path / "example-repo"
    directory.mkdir()
    return directory


class TestRunDetailedScores:
    def test_each_query_has_its_hand_computed_scores(self, tmp_path: Path, repo: Path) -> None:
        records = _harness(repo).run_detailed(_write_golden(tmp_path))

        assert [r.id for r in records] == ["q0001", "q0002", "q0003"]
        one, two, three = records
        assert (one.ndcg, one.recall, one.mrr) == (1.0, 1.0, 1.0)
        assert two.ndcg == pytest.approx(0.6509209298071326, abs=1e-12)
        assert (two.recall, two.mrr) == (1.0, 0.5)
        assert three.ndcg == pytest.approx(0.2890648263178879, abs=1e-12)
        assert (three.recall, three.mrr) == (1.0, 0.1)

    def test_a_miss_scores_zero_without_an_error(self, tmp_path: Path, repo: Path) -> None:
        golden = _write_golden(tmp_path, [{"query": "q one", "relevant_files": ["z.py"]}])
        (record,) = _harness(repo).run_detailed(golden)
        assert (record.ndcg, record.recall, record.mrr) == (0.0, 0.0, 0.0)
        assert record.top10 == ("a.py", "x.py", "y.py")
        assert record.error is None

    def test_repo_is_the_basename_of_the_repository_directory(
        self, tmp_path: Path, repo: Path
    ) -> None:
        records = _harness(repo).run_detailed(_write_golden(tmp_path))
        assert {r.repo for r in records} == {"example-repo"}

    def test_repo_is_resolved_when_the_configured_path_is_relative(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(repo)
        records = _harness(Path(".")).run_detailed(_write_golden(tmp_path))
        assert {r.repo for r in records} == {"example-repo"}

    def test_top10_is_the_first_ten_distinct_files_in_rank_order(
        self, tmp_path: Path, repo: Path
    ) -> None:
        ranked = [
            "f1.py",
            "f1.py",
            "f2.py",
            "f3.py",
            "f3.py",
            *[f"f{i}.py" for i in range(4, 13)],
        ]
        golden = _write_golden(tmp_path, [{"query": "dup", "relevant_files": ["f1.py"]}])
        (record,) = _harness(repo, {"dup": ranked}).run_detailed(golden)
        assert record.top10 == tuple(f"f{i}.py" for i in range(1, 11))


class TestRunAgreesWithRunDetailed:
    def test_run_is_the_mean_of_the_records(self, tmp_path: Path, repo: Path) -> None:
        golden = _write_golden(tmp_path)
        records = _harness(repo).run_detailed(golden)
        aggregate = _harness(repo).run(golden)

        n = len(records)
        assert aggregate == {
            "ndcg@10": sum(r.ndcg for r in records) / n,
            "recall@10": sum(r.recall for r in records) / n,
            "mrr": sum(r.mrr for r in records) / n,
            "n_queries": float(n),
        }

    def test_the_aggregate_has_the_hand_computed_values(self, tmp_path: Path, repo: Path) -> None:
        aggregate = _harness(repo).run(_write_golden(tmp_path))
        assert aggregate["ndcg@10"] == pytest.approx(0.6466619187083401, abs=1e-12)
        assert aggregate["recall@10"] == 1.0
        assert aggregate["mrr"] == pytest.approx(0.5333333333333333, abs=1e-12)
        assert aggregate["n_queries"] == 3.0
        assert list(aggregate) == ["ndcg@10", "recall@10", "mrr", "n_queries"]

    def test_aggregating_no_records_raises_instead_of_dividing_by_zero(self) -> None:
        with pytest.raises(ValueError, match="no records"):
            aggregate_metrics([])


class TestLabels:
    def test_golden_labels_are_carried_onto_the_record(self, tmp_path: Path, repo: Path) -> None:
        golden = _write_golden(
            tmp_path,
            [
                {
                    "query": "q one",
                    "relevant_files": ["a.py"],
                    "id": "walker-01",
                    "kind": "symbol",
                    "lang": "python",
                    "split": "dev",
                }
            ],
        )
        (record,) = _harness(repo).run_detailed(golden)
        assert (record.id, record.kind, record.lang, record.split) == (
            "walker-01",
            "symbol",
            "python",
            "dev",
        )

    def test_a_v1_entry_gets_a_positional_id_and_no_other_labels(
        self, tmp_path: Path, repo: Path
    ) -> None:
        records = _harness(repo).run_detailed(_write_golden(tmp_path))
        assert [(r.id, r.kind, r.lang, r.split) for r in records] == [
            ("q0001", None, None, None),
            ("q0002", None, None, None),
            ("q0003", None, None, None),
        ]

    def test_unlabelled_entries_are_numbered_around_a_labelled_one(
        self, tmp_path: Path, repo: Path
    ) -> None:
        entries = [dict(e) for e in _GOLDEN]
        entries[1]["id"] = "custom"
        records = _harness(repo).run_detailed(_write_golden(tmp_path, entries))
        assert [r.id for r in records] == ["q0001", "custom", "q0003"]

    def test_blank_and_non_string_labels_are_ignored(self, tmp_path: Path, repo: Path) -> None:
        golden = _write_golden(
            tmp_path,
            [
                {
                    "query": "q one",
                    "relevant_files": ["a.py"],
                    "id": "  ",
                    "kind": 7,
                    "lang": "",
                    "split": ["dev"],
                }
            ],
        )
        (record,) = _harness(repo).run_detailed(golden)
        assert (record.id, record.kind, record.lang, record.split) == ("q0001", None, None, None)

    def test_ids_count_golden_entries_not_lines(self, tmp_path: Path, repo: Path) -> None:
        one, two, three = (json.dumps(entry) for entry in _GOLDEN)
        golden = tmp_path / "golden.jsonl"
        golden.write_text(f"\n{one}\n\n{two}\n   \n{three}\n\n", encoding="utf-8")
        records = _harness(repo).run_detailed(str(golden))
        assert [r.id for r in records] == ["q0001", "q0002", "q0003"]

    def test_the_positional_id_survives_an_area_filter_and_a_limit(
        self, tmp_path: Path, repo: Path
    ) -> None:
        entries = [dict(e, area="a" if i == 0 else "b") for i, e in enumerate(_GOLDEN)]
        golden = _write_golden(tmp_path, entries)
        by_area = _harness(repo).run_detailed(golden, area="b")
        assert [r.id for r in by_area] == ["q0002", "q0003"]
        limited = _harness(repo).run_detailed(golden, area="b", limit=1)
        assert [r.id for r in limited] == ["q0002"]


class TestAFailedQuery:
    _FAILING: dict[str, list[str] | Exception] = {
        **_RANKED,
        "q two": RuntimeError("index exploded"),
    }

    def test_it_is_recorded_with_its_message_and_scored_zero(
        self, tmp_path: Path, repo: Path
    ) -> None:
        records = _harness(repo, self._FAILING).run_detailed(_write_golden(tmp_path))

        assert len(records) == 3, "a failing query must not stop the run"
        failed = records[1]
        assert failed.id == "q0002"
        assert failed.error == "index exploded"
        assert (failed.ndcg, failed.recall, failed.mrr) == (0.0, 0.0, 0.0)
        assert failed.top10 == ()
        assert [r.error for r in (records[0], records[2])] == [None, None]
        assert records[2].ndcg == pytest.approx(0.2890648263178879, abs=1e-12)

    def test_it_keeps_the_golden_entrys_repo_and_labels(self, tmp_path: Path, repo: Path) -> None:
        # A failed query must stay in its kind/lang/split subgroup, or a per-subgroup
        # comparison drops exactly the failures that the error field exists to expose.
        entries = [dict(e) for e in _GOLDEN]
        entries[1].update(id="walker-02", kind="symbol", lang="python", split="dev")
        golden = _write_golden(tmp_path, entries)

        (_, failed, _) = _harness(repo, self._FAILING).run_detailed(golden)

        assert failed == QueryRecord(
            id="walker-02",
            repo="example-repo",
            kind="symbol",
            lang="python",
            split="dev",
            ndcg=0.0,
            recall=0.0,
            mrr=0.0,
            top10=(),
            error="index exploded",
        )

    def test_the_mean_counts_it_as_a_zero_never_a_hit(self, tmp_path: Path, repo: Path) -> None:
        aggregate = _harness(repo, self._FAILING).run(_write_golden(tmp_path))
        assert aggregate["ndcg@10"] == pytest.approx(0.42968827543929594, abs=1e-12)
        assert aggregate["recall@10"] == pytest.approx(2 / 3, abs=1e-12)
        assert aggregate["mrr"] == pytest.approx(0.3666666666666667, abs=1e-12)
        assert aggregate["n_queries"] == 3.0

    def test_the_message_is_capped_at_200_characters(self, tmp_path: Path, repo: Path) -> None:
        ranked = {**_RANKED, "q two": RuntimeError("x" * 500)}
        (_, failed, _) = _harness(repo, ranked).run_detailed(_write_golden(tmp_path))
        assert failed.error == "x" * 200

    def test_an_exception_without_a_message_still_leaves_an_error(
        self, tmp_path: Path, repo: Path
    ) -> None:
        ranked = {**_RANKED, "q two": RuntimeError()}
        (_, failed, _) = _harness(repo, ranked).run_detailed(_write_golden(tmp_path))
        assert failed.error == "RuntimeError"

    def test_it_has_no_rerank_verdict_but_keeps_the_lists_aligned(
        self, tmp_path: Path, repo: Path
    ) -> None:
        harness = _harness(repo, self._FAILING)
        harness.run_detailed(_write_golden(tmp_path))
        assert harness.rerank_outcomes == (None, None, None)

    def test_a_frozen_plan_miss_still_propagates(self, tmp_path: Path, repo: Path) -> None:
        from trelix.retrieval.planner.agent import PlanCacheMissError

        ranked = {**_RANKED, "q two": PlanCacheMissError("no frozen plan")}
        with pytest.raises(PlanCacheMissError):
            _harness(repo, ranked).run_detailed(_write_golden(tmp_path))


def _install_stub_harness(
    monkeypatch: pytest.MonkeyPatch, ranked: dict[str, list[str] | Exception]
) -> None:
    """Make `trelix eval` build a harness with the stub retriever and no index."""

    class _Stub(EvalHarness):
        def __init__(self, config: Any) -> None:
            self._config = config
            self._retriever = _StubRetriever(ranked)  # type: ignore[assignment]
            self._rerank_outcomes = ()

    monkeypatch.setattr(harness_module, "EvalHarness", _Stub)


def _invoke(repo: Path, golden: str, *extra: str) -> Any:
    return CliRunner().invoke(cli_main.app, ["eval", str(repo), "--golden", golden, *extra])


class TestEvalCommandPerQueryOut:
    def test_the_flag_writes_the_file_and_the_printed_results_do_not_change(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, _RANKED)
        golden = _write_golden(tmp_path)
        out = tmp_path / "per-query.json"

        plain = _invoke(repo, golden)
        flagged = _invoke(repo, golden, "--per-query-out", str(out))

        assert plain.exit_code == 0, plain.output
        assert flagged.exit_code == 0, flagged.output
        assert flagged.output == plain.output
        for shown in ("nDCG@10", "0.6467", "Recall@10", "1.0000", "MRR", "0.5333"):
            assert shown in plain.output

        document = json.loads(out.read_text(encoding="utf-8"))
        assert set(document) == {"schema_version", "records", "aggregate"}
        assert document["schema_version"] == 1
        assert [r["id"] for r in document["records"]] == ["q0001", "q0002", "q0003"]
        assert {r["repo"] for r in document["records"]} == {"example-repo"}
        assert document["aggregate"]["n_queries"] == 3.0
        assert document["aggregate"]["mrr"] == pytest.approx(0.5333333333333333, abs=1e-12)

    def test_the_file_aggregate_is_the_mean_of_its_own_records(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, _RANKED)
        out = tmp_path / "per-query.json"
        assert _invoke(repo, _write_golden(tmp_path), "--per-query-out", str(out)).exit_code == 0

        document = json.loads(out.read_text(encoding="utf-8"))
        records = document["records"]
        assert document["aggregate"] == {
            "ndcg@10": sum(r["ndcg"] for r in records) / 3,
            "recall@10": sum(r["recall"] for r in records) / 3,
            "mrr": sum(r["mrr"] for r in records) / 3,
            "n_queries": 3.0,
        }

    def test_without_the_flag_no_file_is_written(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, _RANKED)
        golden = _write_golden(tmp_path)
        before = sorted(p.name for p in tmp_path.iterdir())
        assert _invoke(repo, golden).exit_code == 0
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_an_unwritable_path_is_a_one_line_error_not_a_traceback(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, _RANKED)
        missing = tmp_path / "no-such-directory" / "per-query.json"
        result = _invoke(repo, _write_golden(tmp_path), "--per-query-out", str(missing))
        assert result.exit_code == 1
        assert "Could not write per-query results" in result.output
        assert "Traceback" not in result.output

    @pytest.mark.parametrize("path", ["", "."])
    def test_a_path_with_no_file_name_is_a_one_line_error_not_a_traceback(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        _install_stub_harness(monkeypatch, _RANKED)
        result = _invoke(repo, _write_golden(tmp_path), "--per-query-out", path)
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "Could not write per-query results" in result.output


class TestEvalCommandFailsWhenAQueryRaised:
    _FAILING: dict[str, list[str] | Exception] = {
        **_RANKED,
        "q two": RuntimeError("index exploded"),
    }

    def test_exit_code_is_one_and_the_failure_is_named(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, self._FAILING)
        result = _invoke(repo, _write_golden(tmp_path))

        assert result.exit_code == 1
        assert "1 of 3 queries raised" in result.output
        assert "q0002: index exploded" in result.output
        assert "not a valid measurement" in result.output
        # The table is still printed: the operator sees what the run produced.
        assert "0.4297" in result.output
        # The failure list is on stderr, so a script reading stdout gets only the table.
        assert "1 of 3 queries raised" in result.stderr
        assert "q0002: index exploded" in result.stderr
        assert "queries raised" not in result.stdout
        assert "q0002" not in result.stdout

    def test_bracketed_text_in_an_id_or_a_message_prints_literally(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exception text and golden ids are arbitrary strings. Rich reads "[/red]" as a
        # closing tag, so unescaped it raises MarkupError and the listing is lost.
        message = "install 'trelix[local]' and [/red] oops"
        entries = [dict(e) for e in _GOLDEN]
        entries[1]["id"] = "[/bold]walker"
        _install_stub_harness(monkeypatch, {**_RANKED, "q two": RuntimeError(message)})

        result = _invoke(repo, _write_golden(tmp_path, entries))

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert f"[/bold]walker: {message}" in result.output

    def test_the_per_query_file_is_still_written(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, self._FAILING)
        out = tmp_path / "per-query.json"
        result = _invoke(repo, _write_golden(tmp_path), "--per-query-out", str(out))

        assert result.exit_code == 1
        records = json.loads(out.read_text(encoding="utf-8"))["records"]
        assert [r["error"] for r in records] == [None, "index exploded", None]

    def test_an_undecodable_file_name_in_the_message_reaches_the_file(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, {**_RANKED, "q two": RuntimeError("bad\udcff path")})
        out = tmp_path / "per-query.json"
        result = _invoke(repo, _write_golden(tmp_path), "--per-query-out", str(out))

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "1 of 3 queries raised" in result.output
        records = json.loads(out.read_text(encoding="utf-8"))["records"]
        assert [r["error"] for r in records] == [None, "bad\udcff path", None]

    def test_only_the_first_five_failures_are_listed(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = [{"query": f"q{i}", "relevant_files": ["a.py"]} for i in range(1, 8)]
        ranked: dict[str, list[str] | Exception] = {
            f"q{i}": RuntimeError(f"boom {i}") for i in range(1, 8)
        }
        _install_stub_harness(monkeypatch, ranked)
        result = _invoke(repo, _write_golden(tmp_path, entries))

        assert result.exit_code == 1
        assert "7 of 7 queries raised" in result.output
        assert "q0005: boom 5" in result.output
        assert "q0006" not in result.output
        assert "... and 2 more" in result.output
        assert "... and 2 more" in result.stderr
        assert "... and" not in result.stdout

    def test_exactly_five_failures_are_all_listed_and_there_is_no_more_line(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entries = [{"query": f"q{i}", "relevant_files": ["a.py"]} for i in range(1, 6)]
        ranked: dict[str, list[str] | Exception] = {
            f"q{i}": RuntimeError(f"boom {i}") for i in range(1, 6)
        }
        _install_stub_harness(monkeypatch, ranked)
        result = _invoke(repo, _write_golden(tmp_path, entries))

        assert result.exit_code == 1
        assert "5 of 5 queries raised" in result.output
        assert "q0005: boom 5" in result.output
        assert "... and" not in result.output

    def test_a_clean_run_still_exits_zero(
        self, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_stub_harness(monkeypatch, _RANKED)
        result = _invoke(repo, _write_golden(tmp_path))
        assert result.exit_code == 0
        assert "queries raised" not in result.output
