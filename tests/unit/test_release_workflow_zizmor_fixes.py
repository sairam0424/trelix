"""Pins for the zizmor fixes in release.yml and the files its guards depend on.

SCANS (nothing more): the parsed YAML of .github/workflows/release.yml, docker-publish.yml and
ci.yml (step `name:`, `uses:`, `with:`, `env:`, `run:` and the `on:` triggers), every file under
.github/ as plain text for `# zizmor: ignore[...]` comments, and whether a zizmor config file
exists. It does not run zizmor or GitHub Actions and does not evaluate expressions.
verify-release.yml is in test_verify_release_workflow_guards.py.

WHY. release.yml publishes to PyPI, so a step there that can SAVE a cache lets an earlier step (a
pip-installed dependency) poison what a later run restores (zizmor: cache-poisoning). Its two
Hugging Face cache steps are restore-only and ci.yml is the only writer, so the restore must read
exactly the path and key ci.yml writes (or it never hits) and must not stop or fail on a miss
(`lookup-only`, `fail-on-cache-miss`). A miss costs time, not correctness, only because a LATER
step in the same job prefetches the model with the offline switches off: the smoke job runs with
HF_HUB_OFFLINE=1, so without it a miss ends in LocalEntryNotFoundError. Two other findings are
accepted, each by a `# zizmor: ignore[...]` with its reason on the same line. zizmor also reads
`rules.<rule>.ignore` from a config file (all four paths below honoured by 1.30.1), which would
silence a finding with no comment and no reason, so no config file may exist.
Each checker takes a parsed document and returns problems; the fixtures below break each rule.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit.workflow_yaml_helpers import GITHUB, ROOT, action, load_workflow, steps, triggers

_HF_WITH = {
    "path": "~/.cache/huggingface",
    "key": "hf-hub-minilm-l6-v2-${{ hashFiles('pyproject.toml') }}",
}
_RESTORE = "actions/cache/restore"
_PREFETCH_NAME = "Prefetch the local embedder model"
_MODEL_LOAD = "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
_TAGS_ONLY = {"tags": ["v*"]}
_IGNORE = re.compile(r"#\s*zizmor:\s*ignore\[([^\]]*)\]\s*(.*)$")
_RULES = {"dangerous-triggers", "superfluous-actions"}
_RELEASE, _VERIFY = ".github/workflows/release.yml", ".github/workflows/verify-release.yml"
_ACCEPTED = [(_RELEASE, "superfluous-actions"), (_VERIFY, "dangerous-triggers")]
_ZIZMOR_CONFIGS = ("zizmor.yml", "zizmor.yaml", ".github/zizmor.yml", ".github/zizmor.yaml")


def _is_prefetch(step: dict[str, Any]) -> bool:
    env = step.get("env") or {}
    return (
        step.get("name") == _PREFETCH_NAME
        and env.get("HF_HUB_OFFLINE") == "0"
        and env.get("TRANSFORMERS_OFFLINE") == "0"
        and _MODEL_LOAD in str(step.get("run"))
    )


def release_cache_problems(workflow: dict[str, Any]) -> list[str]:
    """Steps that can save a cache, and restore steps that read anything but the HF cache or
    have no prefetch step (model download, offline switches off) after them in their job."""
    problems = []
    for job in workflow["jobs"].values():
        job_steps = job.get("steps", [])
        for index, step in enumerate(job_steps):
            name = step.get("name")
            if action(step) in ("actions/cache", "actions/cache/save"):
                problems.append(f"{name}: {action(step)} can save a cache")
            elif action(step) == _RESTORE:
                if step.get("with") != _HF_WITH:
                    problems.append(f"{name}: restore has {step.get('with')}")
                if not any(_is_prefetch(later) for later in job_steps[index + 1 :]):
                    problems.append(f"{name}: no prefetch step after it in the same job")
    return problems


def ci_cache_problems(workflow: dict[str, Any]) -> list[str]:
    """ci.yml writes the cache release.yml reads: same literal path and key."""
    hf = [
        s
        for s in steps(workflow)
        if action(s).startswith("actions/cache") and re.search(r"hugging|hf-hub", str(s), re.I)
    ]
    if not hf:
        return ["ci.yml has no Hugging Face cache step"]
    return [f"{s.get('name')}: with is {s.get('with')}" for s in hf if s.get("with") != _HF_WITH]


def trigger_problems(release: dict[str, Any], docker: dict[str, Any]) -> list[str]:
    """verify-release.yml's guard needs the push trigger of both publish workflows to be tags
    only (Docker Publish may also have a `workflow_dispatch`; that fails the guard's push test)."""
    problems = []
    if triggers(release) != {"push": _TAGS_ONLY}:
        problems.append(f"release.yml triggers are {triggers(release)}")
    if triggers(docker).get("push") != _TAGS_ONLY:
        problems.append(f"docker-publish.yml push trigger is {triggers(docker).get('push')}")
    return problems


def ignore_problems(files: dict[str, str]) -> list[str]:
    """`files` maps a repo path to its text: a known rule and a reason per ignore, and the
    ignores found must be exactly the accepted ones."""
    found, problems = [], []
    for path, text in sorted(files.items()):
        for number, line in enumerate(text.splitlines(), start=1):
            if (match := _IGNORE.search(line)) is None:
                continue
            for rule in (r.strip() for r in match.group(1).split(",")):
                found.append((path, rule))
                if rule not in _RULES:
                    problems.append(f"{path}:{number}: unknown rule '{rule}'")
            if not match.group(2).strip():
                problems.append(f"{path}:{number}: no reason")
    if sorted(found) != _ACCEPTED:
        problems.append(f"ignores found: {sorted(found)}")
    return problems


def config_problems(root: Path) -> list[str]:
    """A zizmor config file can ignore any rule for any file, with no comment and no reason."""
    return [f"{name} exists" for name in _ZIZMOR_CONFIGS if (root / name).exists()]


def test_release_only_restores_the_hf_cache_in_both_jobs() -> None:
    release = load_workflow("release.yml")
    assert release_cache_problems(release) == []
    for job in ("test", "smoke-test-built-artifacts"):
        assert [s for s in release["jobs"][job]["steps"] if action(s) == _RESTORE], job


def test_ci_writes_the_cache_release_reads() -> None:
    assert ci_cache_problems(load_workflow("ci.yml")) == []


def test_publish_workflows_push_on_tags_only() -> None:
    assert trigger_problems(load_workflow("release.yml"), load_workflow("docker-publish.yml")) == []


def test_zizmor_ignores_are_the_two_accepted_ones_with_reasons() -> None:
    texts = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8", errors="ignore")
        for path in GITHUB.rglob("*")
        if path.is_file()
    }
    assert ignore_problems(texts) == []


def test_there_is_no_zizmor_config_file() -> None:
    assert config_problems(ROOT) == []


@pytest.mark.parametrize(
    "name", ["zizmor.yml", "zizmor.yaml", ".github/zizmor.yml", ".github/zizmor.yaml"]
)
def test_a_zizmor_config_file_is_flagged(name: str, tmp_path: Path) -> None:
    (tmp_path / name).parent.mkdir(exist_ok=True)
    (tmp_path / name).write_text("rules: {}\n", encoding="utf-8")
    assert config_problems(tmp_path) == [f"{name} exists"]


_OFF = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
_PREFETCH: dict[str, Any] = {
    "name": "Prefetch the local embedder model",
    "env": _OFF,
    "run": 'python -c "from sentence_transformers import SentenceTransformer; '
    "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\"",
}


def _workflow(
    uses: str,
    with_: dict[str, Any] | None,
    *,
    before: tuple[dict[str, Any], ...] = (),
    after: tuple[dict[str, Any], ...] = (_PREFETCH,),
) -> dict[str, Any]:
    """One job: the cache step, with correct prefetch step(s) after it unless told otherwise."""
    cache = {"name": "cache", "uses": uses, "with": with_}
    return {"jobs": {"test": {"steps": [*before, cache, *after]}}}


_KEY_V3 = "hf-hub-minilm-l6-v3-${{ hashFiles('pyproject.toml') }}"
_MUTATIONS = {
    "K01-key-changed": {**_HF_WITH, "key": _KEY_V3},
    "K02-path-changed": {**_HF_WITH, "path": "~/.cache/huggingface/hub"},
    "K03-static-key": {**_HF_WITH, "key": "hf-hub-minilm-l6-v2"},
    "K04-fail-on-cache-miss": {**_HF_WITH, "fail-on-cache-miss": True},
    "K05-lookup-only": {**_HF_WITH, "lookup-only": True},
}


def test_cache_fixtures_that_read_or_write_the_right_thing_are_clean() -> None:
    assert release_cache_problems(_workflow(f"{_RESTORE}@v4", _HF_WITH)) == []
    other_step_between = _workflow(f"{_RESTORE}@v4", _HF_WITH, after=({"run": "x"}, _PREFETCH))
    assert release_cache_problems(other_step_between) == []
    assert ci_cache_problems(_workflow("actions/cache@v4", _HF_WITH)) == []


@pytest.mark.parametrize("with_", _MUTATIONS.values(), ids=_MUTATIONS.keys())
def test_a_cache_step_that_reads_or_writes_something_else_is_flagged(with_: dict[str, Any]) -> None:
    assert release_cache_problems(_workflow(f"{_RESTORE}@v4", with_)) != []
    assert ci_cache_problems(_workflow("actions/cache@v4", with_)) != []


@pytest.mark.parametrize("uses", ["actions/cache@v4", "Actions/Cache@v4", "actions/cache/save@v4"])
def test_a_step_that_can_save_a_cache_is_flagged(uses: str) -> None:
    assert release_cache_problems(_workflow(uses, _HF_WITH)) != []


def test_a_restore_without_with_and_a_ci_without_a_cache_step_are_flagged() -> None:
    assert release_cache_problems(_workflow(f"{_RESTORE}@v4", None)) != []
    assert ci_cache_problems(_workflow("actions/cache@v4", {"path": "p", "key": "k"})) != []


_PREFETCH_BITES = {
    "PF1-prefetch-removed": {"after": ()},
    "PF2-prefetch-before-the-restore": {"before": (_PREFETCH,), "after": ()},
    "PF3-hub-offline-still-on": {"after": ({**_PREFETCH, "env": {**_OFF, "HF_HUB_OFFLINE": "1"}},)},
    "PF3-transformers-offline-still-on": {
        "after": ({**_PREFETCH, "env": {**_OFF, "TRANSFORMERS_OFFLINE": "1"}},)
    },
    "PF4-no-model-download": {"after": ({**_PREFETCH, "run": "print(1)"},)},
    "PF4-other-model": {"after": ({**_PREFETCH, "run": _PREFETCH["run"].replace("L6", "L12")},)},
    "PF5-other-step-name": {"after": ({**_PREFETCH, "name": "Warm the cache"},)},
}


@pytest.mark.parametrize("steps_around", _PREFETCH_BITES.values(), ids=_PREFETCH_BITES.keys())
def test_a_restore_without_a_correct_prefetch_after_it_is_flagged(steps_around: Any) -> None:
    assert release_cache_problems(_workflow(f"{_RESTORE}@v4", _HF_WITH, **steps_around)) != []


def test_a_prefetch_in_another_job_does_not_count() -> None:
    restore = {"name": "cache", "uses": f"{_RESTORE}@v4", "with": _HF_WITH}
    jobs = {"a": {"steps": [restore]}, "b": {"steps": [_PREFETCH]}}
    assert release_cache_problems({"jobs": jobs}) != []


_TAGS = "{push: {tags: ['v*']}}"
_TRIGGER_FIXTURES = {
    "release-branch-push": ("{push: {branches: [main]}}", _TAGS),
    "release-tags-and-branches": ("{push: {tags: ['v*'], branches: [main]}}", _TAGS),
    "release-extra-trigger": ("{push: {tags: ['v*']}, pull_request: {}}", _TAGS),
    "release-other-tags": ("{push: {tags: ['*']}}", _TAGS),
    "docker-branch-push": (_TAGS, "{push: {branches: [main]}}"),
    "docker-tags-and-branches": (_TAGS, "{push: {tags: ['v*'], branches: [main]}}"),
    "docker-no-push": (_TAGS, "{workflow_dispatch: {}}"),
}


def _on(text: str) -> dict[str, Any]:
    return {"on": yaml.safe_load(text)}


def test_tag_push_triggers_are_clean_and_other_triggers_are_flagged() -> None:
    assert trigger_problems(_on(_TAGS), _on("{push: {tags: ['v*']}, workflow_dispatch: {}}")) == []
    for release, docker in _TRIGGER_FIXTURES.values():
        assert trigger_problems(_on(release), _on(docker)) != [], (release, docker)


def _ignore(rule: str, reason: str = "accepted") -> str:
    return f"x  # zizmor: ignore[{rule}] {reason}\n"


_GOOD = {_RELEASE: _ignore("superfluous-actions"), _VERIFY: _ignore("dangerous-triggers")}
_IGNORE_FIXTURES = {
    "no-reason": {**_GOOD, _RELEASE: _ignore("superfluous-actions", "")},
    "misspelt-rule": {**_GOOD, _VERIFY: _ignore("dangerous-trigger")},
    "second-rule": {**_GOOD, _VERIFY: _ignore("dangerous-triggers,artipacked")},
    "unreviewed-file": {**_GOOD, ".github/workflows/ci.yml": _ignore("dangerous-triggers")},
    "missing": {**_GOOD, _VERIFY: "no ignore here\n"},
    "duplicate": {**_GOOD, _RELEASE: _GOOD[_RELEASE] * 2},
}


def test_ignore_fixtures_the_two_accepted_are_clean_and_others_are_flagged() -> None:
    assert ignore_problems(_GOOD) == []
    assert ignore_problems({**_GOOD, "a.md": "# zizmor: cache-poisoning, ACCEPTED\n"}) == []


@pytest.mark.parametrize("files", _IGNORE_FIXTURES.values(), ids=_IGNORE_FIXTURES.keys())
def test_a_bad_or_unreviewed_ignore_is_flagged(files: dict[str, str]) -> None:
    assert ignore_problems(files) != []
