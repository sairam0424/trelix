"""`run_suite`: the results file of one arm, and the configuration a run forces.

The "remote" is the real two-commit repository of `eval_suite_harness.py`, cloned by path.
Nothing is built here (`index_fn` is injected) and the ranking comes from a stub retriever
under the real `EvalHarness.run_detailed` (`eval_suite_harness.stub_run`). The refusals and
the git isolation are in `test_eval_suite_run_refusals.py`; the real `Indexer`, `Retriever`
and `EvalHarness` run in `test_eval_suite_run_indexer.py`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. the committed plans file passed to the planner instead of the run-directory copy, or the
   index placed anywhere but `<run dir>/index.db`
                                                  (test_the_harness_reads_the_run_directory_copy_...)
2. any ONE of the ten forced settings dropped from `run_config` (one parametrized case each)
                                                  (TestForcedSettings)
3. `created_at` moved out of `run`                 (test_two_clocks_differ_only_in_run)
4. `library_version` raising on a missing package, or asking for another package, or the
   embedder dimension not the one the index build reported
                                                  (test_a_missing_embedder_library_is_null,
                                                   test_the_embedder_block_...,
                                                   test_the_embedder_dimension_is_...)
5. a secret-shaped field kept in `pipeline.config` (one case per word), `max_tokens_per_chunk`
   dropped with the other token counts, or a credential in `store.qdrant_url` reaching the file
                                                  (test_a_field_named_like_a_secret_is_left_out,
                                                   test_max_tokens_per_chunk_is_kept_by_name,
                                                   test_a_credential_in_a_store_url_...)
6. the suite identity, the `repo` label, a score or the aggregate changed, or a query that
   raised dropped from the file                    (TestTheResultsFile)
7. the writer drifting from the reader or the judge (test_the_file_satisfies_the_reader_...)
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

import pytest
import yaml

from tests.unit.eval_compare_fixtures import prereg_text
from tests.unit.eval_suite_harness import (
    FIRST_FILES,
    GOLDEN_SHA256,
    RANKED,
    Remote,
    clock,
    clone_local,
    load_local,
    make_remote,
    out_file,
    results_doc,
    stub_harness_factory,
    stub_run,
    write_suite,
)
from tests.unit.eval_suite_harness import remote as remote
from tests.unit.eval_suite_harness import spec as spec
from trelix import __version__
from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.eval.compare import compare
from trelix.eval.prereg import parse_prereg
from trelix.eval.results import load_results
from trelix.eval.suite import SuiteSpec
from trelix.eval.suite_run import IndexOutcome, strip_secret_fields


class TestTheResultsFile:
    def test_the_identity_block_is_the_suite_literals(
        self, tmp_path: Path, remote: Remote, spec: SuiteSpec
    ) -> None:
        plans_sha256 = hashlib.sha256((tmp_path / "suite" / "plans.jsonl").read_bytes()).hexdigest()
        doc = results_doc(stub_run(tmp_path, spec))
        assert doc["suite"] == {
            "name": "demo",
            "golden_version": "v1",
            "repo_url": str(remote.path),
            "repo_sha": remote.first,
            "license": "MIT",
            "golden_sha256": GOLDEN_SHA256,
            "plans_sha256": plans_sha256,
        }
        assert doc["arm"] == "baseline"
        assert doc["schema_version"] == 1
        assert doc["trelix_version"] == __version__
        assert doc["run"] == {"created_at": "2026-10-05T12:00:00+00:00"}
        assert sorted(doc) == [
            "aggregate",
            "arm",
            "embedder",
            "pipeline",
            "records",
            "run",
            "schema_version",
            "suite",
            "trelix_version",
        ]

    def test_every_record_is_labelled_with_the_suite_name_and_scored(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        run = stub_run(tmp_path, spec)
        doc = results_doc(run)
        assert [r["repo"] for r in doc["records"]] == ["demo", "demo"]
        assert [r["id"] for r in doc["records"]] == ["q0001", "q0002"]
        first, second = doc["records"]
        assert (first["ndcg"], first["recall"], first["mrr"]) == (1.0, 1.0, 1.0)
        assert first["top10"] == ["src/app.py", "src/extra.py"]
        assert second["ndcg"] == pytest.approx(0.6309297535714575, abs=1e-12)
        assert (second["recall"], second["mrr"]) == (1.0, 0.5)
        assert doc["aggregate"]["ndcg@10"] == pytest.approx(0.8154648767857288, abs=1e-12)
        assert doc["aggregate"]["recall@10"] == 1.0
        assert doc["aggregate"]["mrr"] == 0.75
        assert doc["aggregate"]["n_queries"] == 2.0
        assert run.aggregate == doc["aggregate"]
        assert [r.error for r in run.records] == [None, None]

    def test_the_embedder_block_names_the_local_model_and_the_dimension(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[str] = []

        def version(name: str) -> str:
            asked.append(name)
            return "9.9.9"

        monkeypatch.setattr("trelix.eval.suite_run.metadata.version", version)
        embedder = results_doc(stub_run(tmp_path, spec))["embedder"]
        assert embedder == {
            "provider": "local",
            "model": EmbedderConfig.model_fields["local_model"].default,
            "dimension": 8,
            "library_version": "9.9.9",
        }
        assert asked == ["sentence-transformers"]

    def test_the_embedder_dimension_is_the_one_the_index_build_reported(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        """Every other stub here reports 8; `eval-compare` reads this field for its
        `note: embedder differs` line."""

        def wide(config: IndexConfig) -> IndexOutcome:
            return IndexOutcome(errors=0, dimension=384)

        run = stub_run(tmp_path, spec, index_fn=wide)
        assert run.dimension == 384
        assert results_doc(run)["embedder"]["dimension"] == 384
        assert load_results(run.out).embedder.dimension == 384

    def test_a_missing_embedder_library_is_null(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that installs no `local` extra still writes its results."""

        def missing(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr("trelix.eval.suite_run.metadata.version", missing)
        run = stub_run(tmp_path, spec)
        assert results_doc(run)["embedder"]["library_version"] is None
        assert load_results(run.out).embedder.library_version is None

    def test_the_pipeline_block_records_the_forced_flags_and_the_configuration(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        pipeline = results_doc(stub_run(tmp_path, spec))["pipeline"]
        assert {key: pipeline[key] for key in pipeline if key != "config"} == {
            "rerank": False,
            "hyde_fallback_enabled": False,
            "multi_query_enabled": False,
            "flare_enabled": False,
            "plans": "replayed",
            "rerank_summary": "disabled",
        }
        config = pipeline["config"]  # the file is written with sorted keys
        assert list(config) == [
            "chunker",
            "indexer",
            "parser",
            "retrieval",
            "sparse",
            "store",
            "walker",
        ]
        assert config["walker"]["follow_symlinks"] is False
        assert config["store"]["backend"] == "sqlite"
        assert config["chunker"]["contextual"] is False
        assert config["retrieval"]["rerank"] is False
        assert config["indexer"]["streaming_enabled"] is False
        assert "db_path" not in config["store"]
        assert "plan_cache_file" not in config["retrieval"]
        assert "qdrant_api_key" not in config["store"]
        assert "qdrant_url" not in config["store"]
        assert "lance_uri" not in config["store"]
        assert "cohere_api_key" not in config["retrieval"]
        assert "cohere_endpoint" not in config["retrieval"]
        assert "otel_exporter_endpoint" not in config["retrieval"]
        assert "top_k_tokens" not in config["sparse"]
        assert isinstance(config["chunker"]["max_tokens_per_chunk"], int)

    @pytest.mark.parametrize(
        "name",
        [
            "api_key",
            "client_secret",
            "auth_token",
            "db_password",
            "service_endpoint",
            "qdrant_url",
            "lance_uri",
        ],
    )
    def test_a_field_named_like_a_secret_is_left_out(self, name: str) -> None:
        """One case per word of the rule, so dropping any one word fails exactly one case."""
        section = {name: "x", "top_k": 10, "Rerank": True}
        assert strip_secret_fields(section) == {"top_k": 10, "Rerank": True}

    def test_max_tokens_per_chunk_is_kept_by_name(self) -> None:
        """The one token count that shapes the index survives the word rule; the others do not."""
        section = {"max_tokens_per_chunk": 512, "contextual_max_tokens": 100, "overlap": 50}
        assert strip_secret_fields(section) == {"max_tokens_per_chunk": 512, "overlap": 50}

    def test_a_credential_in_a_store_url_does_not_reach_the_file_and_the_chunk_size_does(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator's daily `TRELIX_STORE_QDRANT_URL` may embed a password; the results file
        is a pull request artifact. The control shows the plain configuration takes the values."""
        monkeypatch.setenv("TRELIX_STORE_QDRANT_URL", "http://u:canary@h:6333")
        monkeypatch.setenv("TRELIX_STORE_LANCE_URI", "/mnt/canary-lance")
        monkeypatch.setenv("TRELIX_CHUNKER_MAX_TOKENS_PER_CHUNK", "333")
        plain = IndexConfig(repo_path=str(tmp_path))
        assert plain.store.qdrant_url == "http://u:canary@h:6333"
        assert plain.store.lance_uri == "/mnt/canary-lance"
        assert plain.chunker.max_tokens_per_chunk == 333
        run = stub_run(tmp_path, spec)
        text = run.out.read_text(encoding="utf-8")
        assert "canary" not in text
        assert results_doc(run)["pipeline"]["config"]["chunker"]["max_tokens_per_chunk"] == 333

    def test_two_clocks_differ_only_in_run(self, tmp_path: Path, spec: SuiteSpec) -> None:
        first = stub_run(
            tmp_path,
            spec,
            cache_root=tmp_path / "c1",
            out=out_file(tmp_path, "a.json"),
            now=clock("2026-10-05T12:00:00+00:00"),
        )
        second = stub_run(
            tmp_path,
            spec,
            cache_root=tmp_path / "c2",
            out=out_file(tmp_path, "b.json"),
            now=clock("2026-10-06T01:02:03+00:00"),
        )
        doc_a, doc_b = results_doc(first), results_doc(second)
        assert doc_a["run"] == {"created_at": "2026-10-05T12:00:00+00:00"}
        assert doc_b["run"] == {"created_at": "2026-10-06T01:02:03+00:00"}
        assert doc_a != doc_b
        rest_a = {key: value for key, value in doc_a.items() if key != "run"}
        rest_b = {key: value for key, value in doc_b.items() if key != "run"}
        assert json.dumps(rest_a, sort_keys=True) == json.dumps(rest_b, sort_keys=True)
        assert "2026-10-05" not in json.dumps(rest_a)

    def test_a_query_that_raised_is_recorded_and_the_file_is_still_written(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        ranked = {**RANKED, "what is in the readme": RuntimeError("boom")}
        run = stub_run(tmp_path, spec, harness_factory=stub_harness_factory(ranked))
        assert run.out.is_file()
        failed = results_doc(run)["records"][1]
        assert failed["error"] == "boom"
        assert (failed["ndcg"], failed["recall"], failed["mrr"], failed["top10"]) == (
            0.0,
            0.0,
            0.0,
            [],
        )
        assert [r.error for r in run.records] == [None, "boom"]

    def test_the_file_satisfies_the_reader_and_the_judge(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        """The contract with `trelix.eval.results` and `eval-compare`: two written files load and
        are judged (two queries are fewer than `min_queries`, so INCONCLUSIVE, not REFUSED)."""
        base = stub_run(tmp_path, spec, arm="baseline", out=out_file(tmp_path, "base.json"))
        cand = stub_run(tmp_path, spec, arm="flag-on", out=out_file(tmp_path, "cand.json"))
        prereg = parse_prereg(yaml.safe_load(prereg_text()))
        verdict = compare(load_results(base.out), load_results(cand.out), prereg)
        assert verdict.outcome == "INCONCLUSIVE"
        assert len(verdict.reasons) == 1
        assert "fewer than min_queries" in verdict.reasons[0]
        assert load_results(base.out).arm == "baseline"
        assert load_results(cand.out).pipeline.config["store"]["backend"] == "sqlite"


# ---------------------------------------------------------------------------
# the configuration of a run under a hostile environment
# ---------------------------------------------------------------------------

_HOSTILE = {
    "TRELIX_RETRIEVAL_RERANK": "true",
    "TRELIX_RETRIEVAL_HYDE_FALLBACK": "true",
    "TRELIX_RETRIEVAL_MULTI_QUERY": "true",
    "TRELIX_RETRIEVAL_FLARE": "true",
    "TRELIX_EMBEDDER_PROVIDER": "openai",
    "TRELIX_RETRIEVAL_PLAN_CACHE_FILE": "/nonexistent",
    "TRELIX_STORE_BACKEND": "lance",
    "TRELIX_CHUNKER_CONTEXTUAL": "true",
    "TRELIX_FILE_SUMMARIES_ENABLED": "true",
    "TRELIX_WALKER_FOLLOW_SYMLINKS": "true",
    "TRELIX_USE_BATCH_API": "true",
}
_FORCED: list[tuple[str, object]] = [
    ("retrieval.rerank", False),
    ("retrieval.hyde_fallback_enabled", False),
    ("retrieval.multi_query_enabled", False),
    ("retrieval.flare_enabled", False),
    ("embedder.provider", "local"),
    ("store.backend", "sqlite"),
    ("chunker.contextual", False),
    ("file_summaries_enabled", False),
    ("walker.follow_symlinks", False),
    ("use_batch_api", False),
]
# The control: what the plain configuration takes from the same environment.
_TAKEN: list[tuple[str, object]] = [
    ("retrieval.rerank", True),
    ("retrieval.hyde_fallback_enabled", True),
    ("retrieval.multi_query_enabled", True),
    ("retrieval.flare_enabled", True),
    ("embedder.provider", "openai"),
    ("retrieval.plan_cache_file", Path("/nonexistent")),
    ("store.backend", "lance"),
    ("chunker.contextual", True),
    ("file_summaries_enabled", True),
    ("walker.follow_symlinks", True),
    ("use_batch_api", True),
]


def _field(config: IndexConfig, dotted: str) -> object:
    value: object = config
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


class TestForcedSettings:
    @pytest.fixture
    def hostile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in _HOSTILE.items():
            monkeypatch.setenv(name, value)

    @pytest.mark.usefixtures("hostile")
    @pytest.mark.parametrize(("field", "expected"), _TAKEN)
    def test_control_the_plain_configuration_obeys_the_hostile_environment(
        self, tmp_path: Path, field: str, expected: object
    ) -> None:
        assert _field(IndexConfig(repo_path=str(tmp_path)), field) == expected

    @pytest.mark.usefixtures("hostile")
    @pytest.mark.parametrize(("field", "expected"), _FORCED)
    def test_the_run_forces_the_setting(
        self, tmp_path: Path, spec: SuiteSpec, field: str, expected: object
    ) -> None:
        factory = stub_harness_factory()
        stub_run(tmp_path, spec, harness_factory=factory)
        [config] = factory.configs
        assert _field(config, field) == expected

    @pytest.mark.usefixtures("hostile")
    def test_the_harness_reads_the_run_directory_copy_and_the_committed_plans_file_is_untouched(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        committed = spec.plans_file.read_bytes()
        factory = stub_harness_factory()
        run = stub_run(tmp_path, spec, harness_factory=factory)
        [config] = factory.configs
        assert run.run_dir == tmp_path / "cache" / "arms" / spec.repo_sha / "demo" / "baseline"
        assert config.retrieval.plan_cache_file == run.run_dir / "plans.jsonl"
        assert config.retrieval.plan_cache_file != spec.plans_file
        assert config.store.db_path == str(run.run_dir / "index.db")
        assert (run.run_dir / "plans.jsonl").read_bytes() == committed
        assert (run.run_dir / "golden.jsonl").read_bytes() == spec.golden_file.read_bytes()
        assert spec.plans_file.read_bytes() == committed
        assert hashlib.sha256(committed).hexdigest() == spec.plans_sha256

    def test_a_planted_dotenv_in_the_current_directory_does_not_configure_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        planted = make_remote(
            tmp_path,
            {**FIRST_FILES, ".env": "TRELIX_EMBEDDER_PROVIDER=openai\n"},
            {"src/extra.py": "X = 1\n"},
            name="dotenv",
        )
        spec = load_local(write_suite(tmp_path / "suite", str(planted.path), planted.first))
        clone = clone_local(spec, tmp_path / "cache")
        assert (clone / ".env").read_text() == "TRELIX_EMBEDDER_PROVIDER=openai\n"
        monkeypatch.chdir(clone)
        factory = stub_harness_factory()
        stub_run(tmp_path, spec, harness_factory=factory)
        assert factory.configs[0].embedder.provider == "local"
