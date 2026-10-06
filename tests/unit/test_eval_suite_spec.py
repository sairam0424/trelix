"""`suite.json`: the schema, the hashes of the files it names, and the checks before any clone.

The clone is in `test_eval_suite_clone.py`, the git hardening in
`test_eval_suite_git_isolation.py`, and the gold paths in `test_eval_suite_gold.py`; the
helpers and the git fixture are in `eval_suite_harness.py`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. a schema rule dropped: the `[0-9a-f]{40}` sha, the name, the file-name or the licence
   pattern, the exact-keys check, the `schema_version` test (`True` or `1.0` accepted)
                                                         (TestSchemaRefusals)
2. the duplicate-key hook or the NaN hook removed        (TestJsonRefusals)
3. the local-path url accepted without `allow_local`, or refused with it     (TestLoad)
   (the host allow-list and the shape of `repo.url` are in `test_eval_suite_repo_url.py`)
4. a hash compared on decoded text, or not compared       (TestFiles)
5. the symlink/parent check of `_verified_file` removed   (test_a_symlink_to_a_file_elsewhere_...)
6. the plans pre-flight reads records itself instead of through `_FrozenPlanCache`
                                                   (test_a_query_differing_in_case_...)
7. the empty-plans sentinel removed                       (test_an_empty_plans_file_...)
8. the uncovered list not capped at five, or not clipped  (TestPlans)
9. the pre-flight moved after the clone, or the hash check after it
                                          (test_a_refused_suite_leaves_the_cache_untouched)
10. a non-object `repo`, `golden` or `plans` indexed instead of refused   (TestSchemaRefusals)
11. the loader's refusal of a sidecar it cannot read turned into a traceback    (TestGolden)
12. the suite directory not resolved: a relative, `..` or symlinked path to suite.json
                                                 (test_a_relative_or_indirect_path_to_suite_json_...)
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.unit.eval_suite_harness import (
    GOLDEN,
    GOLDEN_SHA256,
    Remote,
    plan_record,
    write_suite,
)
from tests.unit.eval_suite_harness import remote as remote
from trelix.eval.golden_validate import validate_golden
from trelix.eval.suite import SuiteError, input_problems, load_suite
from trelix.eval.suite_prepare import prepare_suite

_SHA = "5fa2a032147f64d698077e8849e603872d76b5d0"
_URL = "https://github.com/example/demo.git"
_Mutate = Callable[[dict[str, Any]], None]


def _suite(tmp_path: Path, mutate: _Mutate | None = None, **kwargs: Any) -> Path:
    return write_suite(tmp_path / "suite", _URL, _SHA, mutate=mutate, **kwargs)


def _problems(path: Path, *, allow_local: bool = False) -> str:
    with pytest.raises(SuiteError) as caught:
        load_suite(path, allow_local=allow_local)
    return " | ".join(caught.value.problems)


class TestLoad:
    def test_a_valid_suite_loads_with_its_literal_fields(self, tmp_path: Path) -> None:
        spec = load_suite(_suite(tmp_path))
        assert spec.name == "demo"
        assert spec.golden_version == "v1"
        assert spec.repo_url == "https://github.com/example/demo.git"
        assert spec.repo_sha == "5fa2a032147f64d698077e8849e603872d76b5d0"
        assert spec.license == "MIT"
        assert spec.golden_sha256 == GOLDEN_SHA256
        assert spec.golden_file == (tmp_path / "suite" / "golden.jsonl").resolve()
        assert spec.plans_file == (tmp_path / "suite" / "plans.jsonl").resolve()

    @pytest.mark.parametrize(
        "spelling", ["suite/suite.json", "suite/../suite/suite.json", "link/suite.json"]
    )
    def test_a_relative_or_indirect_path_to_suite_json_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        """`eval/suites/NAME/suite.json` is how the README spells it: relative to the cwd."""
        _suite(tmp_path)
        (tmp_path / "link").symlink_to(tmp_path / "suite")
        monkeypatch.chdir(tmp_path)
        spec = load_suite(spelling)
        assert spec.golden_file == (tmp_path / "suite" / "golden.jsonl").resolve()
        assert spec.plans_file == (tmp_path / "suite" / "plans.jsonl").resolve()

    def test_a_local_path_is_refused_unless_the_tests_allow_it(self, tmp_path: Path) -> None:
        """A pull request's suite.json must not name a repository on the runner."""
        path = write_suite(tmp_path / "suite", "/srv/demo.git", _SHA)
        assert "repo.url must be https://<host>/<owner>/<repo>[.git] with <host> one of" in (
            _problems(path)
        )

    @pytest.mark.parametrize("url", ["/srv/demo.git", "file:///srv/demo.git"])
    def test_a_local_path_and_a_file_url_load_when_allowed(self, tmp_path: Path, url: str) -> None:
        path = write_suite(tmp_path / "suite", url, _SHA)
        assert load_suite(path, allow_local=True).repo_url == url

    @pytest.mark.parametrize(
        "url", ["repo", "file://host/srv/x", "/srv/x\nnew", "-oProxyCommand=x"]
    )
    def test_a_relative_or_malformed_local_url_is_refused_even_when_allowed(
        self, tmp_path: Path, url: str
    ) -> None:
        path = write_suite(tmp_path / "suite", url, _SHA)
        assert "repo.url" in _problems(path, allow_local=True)


_BAD_FIELDS: list[tuple[str, _Mutate, str]] = [
    ("short sha", lambda d: d["repo"].update(sha="5fa2a032"), "repo.sha"),
    ("upper-case sha", lambda d: d["repo"].update(sha=_SHA.upper()), "repo.sha"),
    ("a tag instead of a sha", lambda d: d["repo"].update(sha="v3.4.3"), "repo.sha"),
    ("an option as the url", lambda d: d["repo"].update(url="-oProxyCommand=x"), "repo.url"),
    ("ext transport", lambda d: d["repo"].update(url="ext::sh -c id"), "repo.url"),
    ("git transport", lambda d: d["repo"].update(url="git://h/x"), "repo.url"),
    ("http", lambda d: d["repo"].update(url="http://h/x"), "repo.url"),
    ("ssh", lambda d: d["repo"].update(url="ssh://h/x"), "repo.url"),
    ("credentials", lambda d: d["repo"].update(url="https://u:p@h/x"), "repo.url"),
    ("a query string", lambda d: d["repo"].update(url="https://h/x?y=1"), "repo.url"),
    ("a bare word url", lambda d: d["repo"].update(url="repo"), "repo.url"),
    ("a null url", lambda d: d["repo"].update(url=None), "repo.url"),
    ("a list as the url", lambda d: d["repo"].update(url=["https://github.com/o/r"]), "repo.url"),
    ("licence with spaces", lambda d: d["repo"].update(license="MIT AND X"), "repo.license"),
    ("name with a dot", lambda d: d.update(name="a.b"), "name must match"),
    ("name that climbs", lambda d: d.update(name="../x"), "name must match"),
    ("upper-case name", lambda d: d.update(name="Demo"), "name must match"),
    ("label with a slash", lambda d: d.update(golden_version="a/b"), "golden_version"),
    (
        "golden path that climbs",
        lambda d: d["golden"].update(path="../golden.jsonl"),
        "golden.path",
    ),
    (
        "golden path in a subdirectory",
        lambda d: d["golden"].update(path="sub/g.jsonl"),
        "golden.path",
    ),
    ("plans path absolute", lambda d: d["plans"].update(path="/etc/hosts"), "plans.path"),
    ("short golden hash", lambda d: d["golden"].update(sha256="abc"), "golden.sha256"),
    ("upper-case plans hash", lambda d: d["plans"].update(sha256="A" * 64), "plans.sha256"),
    ("unknown top-level key", lambda d: d.update(extra=1), "unsupported key 'extra'"),
    ("unknown repo key", lambda d: d["repo"].update(branch="main"), "unsupported key 'branch'"),
    ("missing plans", lambda d: d.pop("plans"), "missing key 'plans'"),
    ("missing licence", lambda d: d["repo"].pop("license"), "missing key 'license'"),
    ("schema 2", lambda d: d.update(schema_version=2), "unsupported schema_version 2"),
    ("schema true", lambda d: d.update(schema_version=True), "unsupported schema_version True"),
    ("schema 1.0", lambda d: d.update(schema_version=1.0), "unsupported schema_version 1.0"),
    # `<name>.partial` is the clone in progress, so no suite may own a name with a dot in it.
    ("name that is a partial", lambda d: d.update(name="demo.partial"), "name must match"),
    ("name that is only a suffix", lambda d: d.update(name=".partial"), "name must match"),
    # Each names its own keys as text: were it indexed, `"url" in "url ..."` would then crash.
    ("repo not an object", lambda d: d.update(repo="url sha license"), "repo must be an object"),
    ("golden not an object", lambda d: d.update(golden="path sha256"), "golden must be an object"),
    (
        "plans not an object",
        lambda d: d.update(plans=["path", "sha256"]),
        "plans must be an object",
    ),
]


class TestSchemaRefusals:
    @pytest.mark.parametrize(
        ("label", "mutate", "expected"), _BAD_FIELDS, ids=[b[0] for b in _BAD_FIELDS]
    )
    def test_each_bad_field_is_refused_with_its_name(
        self, tmp_path: Path, label: str, mutate: _Mutate, expected: str
    ) -> None:
        assert expected in _problems(_suite(tmp_path, mutate))

    def test_every_problem_is_reported_in_one_go(self, tmp_path: Path) -> None:
        def mutate(doc: dict[str, Any]) -> None:
            doc["repo"]["sha"] = "abc"
            doc["name"] = "A"
            doc["extra"] = 1

        with pytest.raises(SuiteError) as caught:
            load_suite(_suite(tmp_path, mutate))
        assert len(caught.value.problems) == 3


class TestJsonRefusals:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "suite.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_repeated_key_is_refused(self, tmp_path: Path) -> None:
        text = json.dumps(json.loads(_suite(tmp_path).read_text()), indent=2)
        doubled = text.replace('"name": "demo",', '"name": "demo", "name": "other",', 1)
        assert "duplicate key 'name'" in _problems(self._write(tmp_path, doubled))

    def test_a_repeated_key_in_a_nested_object_is_refused(self, tmp_path: Path) -> None:
        text = json.dumps(json.loads(_suite(tmp_path).read_text()), indent=2)
        doubled = text.replace('"license": "MIT"', '"license": "MIT", "license": "GPL"', 1)
        assert "duplicate key 'license'" in _problems(self._write(tmp_path, doubled))

    def test_a_nan_is_refused(self, tmp_path: Path) -> None:
        assert "the constant NaN is not allowed" in _problems(
            self._write(tmp_path, '{"schema_version": NaN}')
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("{", "not valid JSON"), ("[]", "must be an object"), ("", "not valid JSON")],
    )
    def test_text_that_is_not_an_object_is_refused(
        self, tmp_path: Path, text: str, expected: str
    ) -> None:
        assert expected in _problems(self._write(tmp_path, text))

    def test_a_file_over_the_cap_is_refused_unread(self, tmp_path: Path) -> None:
        assert "cannot read suite.json: is larger than 65536 bytes" in _problems(
            self._write(tmp_path, " " * 65537)
        )

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        assert "cannot read suite.json" in _problems(tmp_path / "absent.json")


class TestFiles:
    def test_a_changed_golden_file_is_refused_with_both_digests(self, tmp_path: Path) -> None:
        path = _suite(tmp_path)
        changed = b'{"query": "x", "relevant_files": ["README.md"]}\n'
        (tmp_path / "suite" / "golden.jsonl").write_bytes(changed)
        expected = (
            f"golden: sha256 of golden.jsonl is {hashlib.sha256(changed).hexdigest()} "
            f"but suite.json says {GOLDEN_SHA256}"
        )
        assert expected in _problems(path)

    def test_both_files_are_checked_and_both_reported(self, tmp_path: Path) -> None:
        path = _suite(tmp_path)
        (tmp_path / "suite" / "golden.jsonl").write_bytes(b"x\n")
        (tmp_path / "suite" / "plans.jsonl").write_bytes(b"y\n")
        text = _problems(path)
        assert "golden: sha256 of golden.jsonl" in text
        assert "plans: sha256 of plans.jsonl" in text

    def test_the_hash_is_over_raw_bytes_so_a_line_ending_rewrite_is_refused(
        self, tmp_path: Path
    ) -> None:
        path = _suite(tmp_path)
        golden = tmp_path / "suite" / "golden.jsonl"
        golden.write_bytes(golden.read_bytes().replace(b"\n", b"\r\n"))
        assert "golden: sha256 of golden.jsonl is" in _problems(path)

    def test_a_missing_plans_file_is_refused(self, tmp_path: Path) -> None:
        path = _suite(tmp_path)
        (tmp_path / "suite" / "plans.jsonl").unlink()
        assert "plans: cannot find 'plans.jsonl' beside suite.json" in _problems(path)

    def test_a_directory_in_place_of_the_golden_file_is_refused(self, tmp_path: Path) -> None:
        path = _suite(tmp_path)
        (tmp_path / "suite" / "golden.jsonl").unlink()
        (tmp_path / "suite" / "golden.jsonl").mkdir()
        assert "golden: 'golden.jsonl' must be a regular file inside the suite directory" in (
            _problems(path)
        )

    def test_a_data_file_over_the_cap_is_refused_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _suite(tmp_path)
        monkeypatch.setattr("trelix.eval.suite.MAX_DATA_FILE_BYTES", 10)
        text = _problems(path)
        assert "golden: cannot read 'golden.jsonl': is larger than 10 bytes" in text
        assert "plans: cannot read 'plans.jsonl': is larger than 10 bytes" in text

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file with no permissions")
    def test_an_unreadable_data_file_is_refused(self, tmp_path: Path) -> None:
        path = _suite(tmp_path)
        golden = tmp_path / "suite" / "golden.jsonl"
        golden.chmod(0)
        try:
            text = _problems(path)
        finally:
            golden.chmod(0o600)
        assert "golden: cannot read 'golden.jsonl': " in text

    def test_a_symlink_loop_in_place_of_a_data_file_is_refused(self, tmp_path: Path) -> None:
        path = _suite(tmp_path)
        golden = tmp_path / "suite" / "golden.jsonl"
        golden.unlink()
        golden.symlink_to("golden.jsonl")
        assert "golden: cannot find 'golden.jsonl' beside suite.json" in _problems(path)

    def test_a_symlink_to_a_file_elsewhere_is_refused_even_with_the_right_hash(
        self, tmp_path: Path
    ) -> None:
        """The link's target has the same bytes, so only the directory check can refuse it."""
        path = _suite(tmp_path)
        golden = tmp_path / "suite" / "golden.jsonl"
        outside = tmp_path / "elsewhere.jsonl"
        outside.write_bytes(golden.read_bytes())
        golden.unlink()
        golden.symlink_to(outside)
        assert "golden: 'golden.jsonl' must be a regular file inside the suite directory" in (
            _problems(path)
        )


def _golden(*queries: str) -> list[dict[str, Any]]:
    return [{"query": q, "relevant_files": ["README.md"]} for q in queries]


class TestPlans:
    def test_the_demo_suite_has_no_problems(self, tmp_path: Path) -> None:
        assert input_problems(load_suite(_suite(tmp_path))) == []

    def test_a_golden_query_without_a_plan_is_named(self, tmp_path: Path) -> None:
        spec = load_suite(_suite(tmp_path, plans=["how does login work"]))
        assert input_problems(spec) == [
            "plans: 1 of 2 golden queries have no recorded plan in plans.jsonl: "
            "'what is in the readme'"
        ]

    def test_at_most_five_uncovered_queries_are_listed_and_the_rest_counted(
        self, tmp_path: Path
    ) -> None:
        queries = [f"question {n}" for n in range(1, 9)]
        spec = load_suite(_suite(tmp_path, golden=_golden(*queries), plans=queries[:1]))
        assert input_problems(spec) == [
            "plans: 7 of 8 golden queries have no recorded plan in plans.jsonl: "
            "'question 2', 'question 3', 'question 4', 'question 5', 'question 6', and 2 more"
        ]

    def test_a_long_query_is_clipped_in_the_message(self, tmp_path: Path) -> None:
        long_query = "a" * 200
        spec = load_suite(_suite(tmp_path, golden=_golden(long_query), plans=["other"]))
        [problem] = input_problems(spec)
        assert "'" + "a" * 76 + "..." in problem
        assert "a" * 80 not in problem

    @pytest.mark.parametrize("content", [b"", b"\n\n", b"  \n"])
    def test_an_empty_plans_file_is_refused_not_treated_as_record_mode(
        self, tmp_path: Path, content: bytes
    ) -> None:
        """A plans file with no records makes the planner call the LLM and record."""
        path = _suite(tmp_path)
        plans = tmp_path / "suite" / "plans.jsonl"
        plans.write_bytes(content)
        doc = json.loads(path.read_text())
        doc["plans"]["sha256"] = hashlib.sha256(content).hexdigest()
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert input_problems(load_suite(path)) == [
            "plans: plans.jsonl holds no plans. eval-suite never records plans: "
            "an empty plans file would make the planner call the LLM and record"
        ]
        assert plans.read_bytes() == content

    def test_a_query_differing_in_case_and_padding_is_covered_as_the_planner_covers_it(
        self, tmp_path: Path
    ) -> None:
        """The pre-flight asks the planner's own cache class, so its key rule cannot differ."""
        spec = load_suite(
            _suite(tmp_path, golden=_golden("How Does Login Work "), plans=["how does login work"])
        )
        assert input_problems(spec) == []

    def test_the_real_planner_replays_what_the_pre_flight_called_covered(
        self, tmp_path: Path
    ) -> None:
        from trelix.core.config import EmbedderConfig, RetrievalConfig
        from trelix.retrieval.planner.agent import PlanCacheMissError, QueryPlanner

        spec = load_suite(
            _suite(tmp_path, golden=_golden("How Does Login Work "), plans=["how does login work"])
        )
        retrieval = RetrievalConfig(
            plan_cache_size=0, plan_cache_file=spec.plans_file, _env_file=None
        )
        planner = QueryPlanner(
            EmbedderConfig(provider="local", _env_file=None), retrieval_config=retrieval
        )  # type: ignore[call-arg]
        assert planner.plan("How Does Login Work ").raw_query == "how does login work"
        with pytest.raises(PlanCacheMissError):
            planner.plan("what is in the readme")

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("{not json", "plans: plans.jsonl is not usable: ValueError"),
            ('{"query": "q"}', "plans: plans.jsonl is not usable: ValueError"),
            (
                json.dumps(
                    {**plan_record("q"), "plan": {"intent": "feature_flow", "sub_queries": [{}]}}
                ),
                "plans: plans.jsonl is not usable: KeyError",
            ),
        ],
    )
    def test_an_unusable_plans_file_is_refused(
        self, tmp_path: Path, line: str, expected: str
    ) -> None:
        path = _suite(tmp_path)
        content = (line + "\n").encode()
        (tmp_path / "suite" / "plans.jsonl").write_bytes(content)
        doc = json.loads(path.read_text())
        doc["plans"]["sha256"] = hashlib.sha256(content).hexdigest()
        path.write_text(json.dumps(doc), encoding="utf-8")
        [problem] = input_problems(load_suite(path))
        assert problem.startswith(expected)


class TestGolden:
    def test_a_golden_violation_is_reported_with_its_line_and_the_plans_wait(
        self, tmp_path: Path
    ) -> None:
        golden = [*GOLDEN, {"query": "HOW does login work ", "relevant_files": ["README.md"]}]
        spec = load_suite(_suite(tmp_path, golden=golden))
        assert input_problems(spec) == [
            "golden: line 3: duplicate query (same as line 1 once stripped and case-folded)"
        ]

    def test_a_path_that_is_not_normalised_is_refused(self, tmp_path: Path) -> None:
        golden = [{"query": "q", "relevant_files": ["./README.md"]}]
        [problem] = input_problems(load_suite(_suite(tmp_path, golden=golden)))
        assert problem.startswith('golden: line 1: "relevant_files" ')
        assert "is not normalised" in problem

    def test_the_v2_strata_thresholds_of_eval_validate_do_not_apply_to_a_suite(
        self, tmp_path: Path
    ) -> None:
        """One `nl` entry and no reviewed status fails `eval-validate`'s defaults (20 per kind,
        95 percent reviewed): they decide whether a golden file may be committed, not whether a
        suite may run."""
        golden = [{"query": "q", "relevant_files": ["README.md"], "kind": "nl"}]
        spec = load_suite(_suite(tmp_path, golden=golden))
        control = validate_golden(
            spec.golden_file, tree=None, min_per_stratum=20, min_validated=0.95
        )
        assert len(control.violations) == 2
        assert input_problems(spec) == []

    def test_a_golden_file_with_no_entries_is_refused(self, tmp_path: Path) -> None:
        spec = load_suite(_suite(tmp_path, golden=[], plans=[]))
        assert input_problems(spec) == ["golden: file: no golden entries"]

    def test_a_sidecar_the_loader_cannot_parse_is_refused_and_not_raised(
        self, tmp_path: Path
    ) -> None:
        """`golden-metadata.json` beside the golden file is read by the loader (for `area`) and
        not by `validate_golden`, so only the loader's own `ValueError` can report it."""
        spec = load_suite(_suite(tmp_path))
        (tmp_path / "suite" / "golden-metadata.json").write_text("{", encoding="utf-8")
        [problem] = input_problems(spec)
        assert problem.startswith("golden: ")
        assert "golden-metadata.json" in problem

    def test_a_sidecar_that_is_a_directory_is_refused_and_not_raised(self, tmp_path: Path) -> None:
        """The loader opens the sidecar whenever it exists; a directory raises an OSError, which
        is not a ValueError. MUTATION THAT MUST FAIL THIS TEST: `except (OSError, ValueError)`
        narrowed back to `except ValueError` around `_parse_golden`."""
        spec = load_suite(_suite(tmp_path))
        (tmp_path / "suite" / "golden-metadata.json").mkdir()
        [problem] = input_problems(spec)
        assert problem.startswith("golden: ")
        assert "golden-metadata.json" in problem


class TestNothingIsClonedUntilTheInputsAreProven:
    def _cache(self, tmp_path: Path) -> Path:
        return tmp_path / "cache"

    def test_a_refused_suite_leaves_the_cache_untouched(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        """One real, reachable remote: were the clone first, the cache directory would exist."""
        url = str(remote.path)
        wrong_hash = write_suite(tmp_path / "a", url, remote.first)
        (tmp_path / "a" / "golden.jsonl").write_bytes(b"x\n")
        uncovered = write_suite(tmp_path / "b", url, remote.first, plans=["how does login work"])
        empty = write_suite(tmp_path / "c", url, remote.first)
        (tmp_path / "c" / "plans.jsonl").write_bytes(b"")
        doc = json.loads(empty.read_text())
        doc["plans"]["sha256"] = hashlib.sha256(b"").hexdigest()
        empty.write_text(json.dumps(doc), encoding="utf-8")

        for suite_json in (wrong_hash, uncovered, empty):
            with pytest.raises(SuiteError):
                prepare_suite(suite_json, self._cache(tmp_path), allow_local=True)
            assert not self._cache(tmp_path).exists()
