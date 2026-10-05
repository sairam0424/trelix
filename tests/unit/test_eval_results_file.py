"""results.json schema 1: the reader's rules for the header blocks and for the documented example.

The two-query document is `tests.unit.eval_results_demo.DEMO`, written out by hand; every
expected value is a literal.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. the exact-key check removed (an extra or a missing key accepted)   (TestKeysAndTypes)
2. `pipeline.config` not required                                     (test_the_pipeline_config_...)
3. the schema_version check removed                                   (test_another_schema_...)
4. `0 <= value <= 1` -> `0 < value <= 1`, or `< 1`                     (the 0.0 and 1.0 scores)
5. `top10` length bound 10 -> 11 or dropped                           (the ten/eleven test)
6. the embedder fingerprint accepted again  (test_the_embedder_fingerprint_...)
7. a bool accepted as the embedder dimension  (test_a_bad_embedder_dimension_...)
8. a blank `library_version` accepted (`_is_text` -> `isinstance`)   (test_a_blank_library_...)
9. the early return after an unsupported schema_version removed  (..._reports_only_the_version)
10. `Path(path)` dropped from load_results (a str path crashes)   (test_the_two_query_document_...)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.unit.eval_compare_fixtures import results_doc, write_json
from tests.unit.eval_results_demo import (
    DEMO,
    mutated,
    refusals_of,
)
from trelix.eval.results import load_results


class TestTheReader:
    def test_the_two_query_document_loads_to_the_written_values(self, tmp_path: Path) -> None:
        results = load_results(write_json(tmp_path, "r.json", DEMO))  # a str path, as 3b has it

        assert results.arm == "baseline"
        assert results.trelix_version == "3.4.3"
        assert results.created_at == "2026-10-05T12:00:00+00:00"
        assert results.suite.name == "demo"
        assert results.suite.repo_sha == "a" * 40
        assert results.suite.golden_sha256 == "b" * 64
        assert results.embedder.provider == "local"
        assert results.embedder.dimension == 384
        assert results.embedder.library_version == "5.1.0"
        assert results.pipeline.rerank is False
        assert results.pipeline.plans == "replayed"
        assert results.pipeline.config == {
            "retrieval": {"declaration_boost_enabled": False, "top_k": 10}
        }
        assert [r.id for r in results.records] == ["q0001", "q0002"]
        assert results.records[1].ndcg == 0.6309297535714575
        assert results.records[1].top10 == ("src/app.py", "README.md")
        assert results.records[1].split == "test"

    def test_the_example_in_the_readme_loads(self, tmp_path: Path) -> None:
        readme = Path(__file__).resolve().parents[2] / "eval" / "README.md"
        blocks = re.findall(r"```json\n(.*?)```", readme.read_text(encoding="utf-8"), re.DOTALL)
        example = next(block for block in blocks if '"suite"' in block)
        text = example.replace("<64 hex>", "b" * 64).replace("<40 hex>", "a" * 40)

        results = load_results(Path(write_json(tmp_path, "r.json", json.loads(text))))

        assert results.suite.name == "demo"
        assert results.records[0].top10 == ("src/app.py",)
        assert results.pipeline.config == {"retrieval": {"top_k": 10}}

    def test_a_null_library_version_is_not_an_error(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["embedder"].update(library_version=None))

        assert (
            load_results(Path(write_json(tmp_path, "r.json", doc))).embedder.library_version is None
        )

    def test_pipeline_and_run_ignore_keys_they_do_not_know(self, tmp_path: Path) -> None:
        doc = mutated(
            lambda d: (d["pipeline"].update(overrides={"x": 1}), d["run"].update(host="h"))
        )

        results = load_results(Path(write_json(tmp_path, "r.json", doc)))

        assert results.arm == "baseline"

    def test_the_loader_does_not_judge_the_frozen_flags(self, tmp_path: Path) -> None:
        # Refusing a run that was not rerank-off is eval-compare's call, not the reader's.
        doc = mutated(lambda d: d["pipeline"].update(rerank=True, plans="live"))

        results = load_results(Path(write_json(tmp_path, "r.json", doc)))

        assert results.pipeline.rerank is True
        assert results.pipeline.plans == "live"

    def test_scores_of_exactly_zero_and_one_are_valid(self, tmp_path: Path) -> None:
        doc = results_doc("baseline", [0.0, 1.0], [1.0, 0.0], ids=["q1", "q2"])

        assert len(load_results(Path(write_json(tmp_path, "r.json", doc))).records) == 2

    def test_ten_files_in_top10_are_valid_and_eleven_are_not(self, tmp_path: Path) -> None:
        ten = mutated(lambda d: d["records"][0].update(top10=[f"f{i}.py" for i in range(10)]))
        assert len(load_results(Path(write_json(tmp_path, "ten.json", ten))).records) == 2

        eleven = mutated(lambda d: d["records"][0].update(top10=[f"f{i}.py" for i in range(11)]))
        assert "record q0001: 'top10' must be a list of at most 10 strings" in refusals_of(
            tmp_path, eleven
        )


class TestKeysAndTypes:
    @pytest.mark.parametrize(
        "key", ["aggregate", "arm", "embedder", "pipeline", "records", "run", "trelix_version"]
    )
    def test_a_missing_top_level_key_is_refused(self, tmp_path: Path, key: str) -> None:
        doc = mutated(lambda d: d.pop(key))

        assert f"results: missing key {key!r}" in refusals_of(tmp_path, doc)

    def test_an_unsupported_top_level_key_is_refused(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d.update(extra=1))

        assert refusals_of(tmp_path, doc) == ("results: unsupported key 'extra'",)

    @pytest.mark.parametrize("version", [2, "1", True, 1.0, None])
    def test_another_schema_version_is_unsupported(self, tmp_path: Path, version: object) -> None:
        doc = mutated(lambda d: d.update(schema_version=version))

        problems = refusals_of(tmp_path, doc)

        assert len(problems) == 1
        assert problems[0].startswith("unsupported schema_version ")
        assert problems[0].endswith("this reader understands 1")

    def test_another_schema_version_reports_only_the_version(self, tmp_path: Path) -> None:
        """A file of another schema is not judged by this one's rules: an invalid record too."""
        doc = mutated(
            lambda d: d.update(
                schema_version=2, records=[{**d["records"][0], "ndcg": 7}, *d["records"][1:]]
            )
        )

        problems = refusals_of(tmp_path, doc)

        assert len(problems) == 1
        assert problems[0].startswith("unsupported schema_version 2")

    @pytest.mark.parametrize("version", ["", " ", 5, None])
    def test_a_blank_trelix_version_is_refused(self, tmp_path: Path, version: object) -> None:
        doc = mutated(lambda d: d.update(trelix_version=version))

        assert refusals_of(tmp_path, doc) == ("trelix_version: must be a non-empty string",)

    def test_a_suite_that_is_not_an_object_is_refused(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d.update(suite=["demo"]))

        assert refusals_of(tmp_path, doc) == ("suite: must be an object",)

    @pytest.mark.parametrize("arm", ["Base", "", 5, "a" * 64, "-x", "a b"])
    def test_a_bad_arm_is_refused(self, tmp_path: Path, arm: object) -> None:
        doc = mutated(lambda d: d.update(arm=arm))

        problems = refusals_of(tmp_path, doc)

        assert len(problems) == 1
        assert problems[0].startswith("arm: must match [a-z0-9][a-z0-9_-]{0,62} (got ")

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("golden_sha256", "B" * 64),
            ("golden_sha256", "b" * 63),
            ("plans_sha256", "c" * 65),
            ("repo_sha", "a" * 39),
            ("repo_sha", "5fa2a032"),
            ("name", "Demo"),
            ("name", "../x"),
            ("golden_version", "-x"),
            ("golden_version", ""),
        ],
    )
    def test_a_bad_suite_value_is_refused(self, tmp_path: Path, key: str, value: str) -> None:
        doc = mutated(lambda d: d["suite"].update({key: value}))

        problems = refusals_of(tmp_path, doc)

        assert any(p.startswith(f"suite: {key!r} is not a valid value (got ") for p in problems)

    @pytest.mark.parametrize("key", ["repo_url", "license"])
    def test_a_blank_suite_text_is_refused(self, tmp_path: Path, key: str) -> None:
        doc = mutated(lambda d: d["suite"].update({key: " "}))

        assert refusals_of(tmp_path, doc) == (f"suite: {key!r} must be a non-empty string",)

    def test_a_suite_key_is_missing_or_unsupported(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: (d["suite"].pop("license"), d["suite"].update(origin="x")))

        assert refusals_of(tmp_path, doc) == (
            "suite: unsupported key 'origin'",
            "suite: missing key 'license'",
        )

    @pytest.mark.parametrize("dimension", [0, -1, True, 384.0, "384", None])
    def test_a_bad_embedder_dimension_is_refused(self, tmp_path: Path, dimension: object) -> None:
        doc = mutated(lambda d: d["embedder"].update(dimension=dimension))

        problems = refusals_of(tmp_path, doc)

        assert len(problems) == 1
        assert problems[0].startswith(
            "embedder: 'dimension' must be an integer of at least 1 (got "
        )

    def test_the_embedder_fingerprint_is_gone(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["embedder"].update(fingerprint="0" * 64))

        assert refusals_of(tmp_path, doc) == ("embedder: unsupported key 'fingerprint'",)

    def test_an_embedder_key_is_missing(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["embedder"].pop("library_version"))

        assert refusals_of(tmp_path, doc) == ("embedder: missing key 'library_version'",)

    def test_an_embedder_text_is_blank_or_a_library_version_is_not_text(
        self, tmp_path: Path
    ) -> None:
        doc = mutated(lambda d: d["embedder"].update(provider="", model=5, library_version=5))

        assert refusals_of(tmp_path, doc) == (
            "embedder: 'provider' must be a non-empty string",
            "embedder: 'model' must be a non-empty string",
            "embedder: 'library_version' must be a non-empty string or null",
        )

    @pytest.mark.parametrize("version", ["", "  "])
    def test_a_blank_library_version_is_refused(self, tmp_path: Path, version: str) -> None:
        doc = mutated(lambda d: d["embedder"].update(library_version=version))

        assert refusals_of(tmp_path, doc) == (
            "embedder: 'library_version' must be a non-empty string or null",
        )

    def test_the_pipeline_config_is_required(self, tmp_path: Path) -> None:
        doc = mutated(lambda d: d["pipeline"].pop("config"))

        assert refusals_of(tmp_path, doc) == ("pipeline: missing key 'config'",)

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("config", [], "pipeline: 'config' must be an object"),
            ("rerank", "no", "pipeline: 'rerank' must be true or false"),
            ("flare_enabled", 0, "pipeline: 'flare_enabled' must be true or false"),
            ("plans", 5, "pipeline: 'plans' must be a string"),
            ("rerank_summary", None, "pipeline: 'rerank_summary' must be a string"),
        ],
    )
    def test_a_mistyped_pipeline_value_is_refused(
        self, tmp_path: Path, key: str, value: object, message: str
    ) -> None:
        doc = mutated(lambda d: d["pipeline"].update({key: value}))

        assert refusals_of(tmp_path, doc) == (message,)

    @pytest.mark.parametrize("run", [None, [], {}, {"created_at": 5}])
    def test_run_needs_a_created_at_string(self, tmp_path: Path, run: object) -> None:
        doc = mutated(lambda d: d.update(run=run))

        problems = refusals_of(tmp_path, doc)

        assert len(problems) == 1
        assert problems[0] in ("run: must be an object", "run: 'created_at' must be a string")
