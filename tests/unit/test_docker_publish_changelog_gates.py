"""docker-publish.yml must refuse a tag that release.yml would refuse, before it pushes anything.

Both workflows start on the same `v*` tag push. release.yml refuses a tag whose CHANGELOG has
no `## [X.Y.Z]` section, or whose section is dated to any day but the tag's UTC creation day;
docker-publish.yml only compared the tag with pyproject.toml, so the same tag stopped PyPI but
still pushed four GHCR tags, which are hard to take back. A real tag release cannot be run
first, so this pins that the two gate scripts are release.yml's text, that the gates sit before
the login and the push on tag pushes only, and (with bash in a scratch git repository) where the
scripts pass and fail, including either side of UTC midnight and for a tagger east of UTC.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_SECTION_GATE = "CHANGELOG has a section for this version"
_DATE_GATE = "CHANGELOG section is dated to the tag's UTC day"
_REF = "v9.9.9"
# Dated far from every tag below, so a gate that read the commit date instead would fail.
_COMMIT_INSTANT = "2000-01-01T00:00:00+0000"


def _steps(workflow: str) -> list[dict[str, Any]]:
    job = "verify-version" if workflow == "release.yml" else "publish"
    data = yaml.safe_load((_WORKFLOWS / workflow).read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = data["jobs"][job]["steps"]
    return steps


def _step(workflow: str, name: str) -> dict[str, Any]:
    found = [step for step in _steps(workflow) if step.get("name") == name]
    assert len(found) == 1, f"{workflow}: expected one step named {name!r}, found {len(found)}"
    return found[0]


def _position(name: str) -> int:
    names = [step.get("name") for step in _steps("docker-publish.yml")]
    assert name in names, f"docker-publish.yml has no step named {name!r}"
    return names.index(name)


def _shape(workflow: str, name: str) -> dict[str, Any]:
    """The step minus its `if` (only docker-publish.yml has one), with `run` right-trimmed, so
    an `env:` or `shell:` added to one copy is drift too."""
    step = {key: value for key, value in _step(workflow, name).items() if key != "if"}
    step["run"] = "\n".join(line.rstrip() for line in step["run"].rstrip().splitlines())
    return step


@pytest.mark.parametrize("name", [_SECTION_GATE, _DATE_GATE])
class TestTheGates:
    def test_the_step_is_release_ymls_step(self, name: str) -> None:
        assert _shape("docker-publish.yml", name) == _shape("release.yml", name)

    def test_it_precedes_the_login_and_the_build(self, name: str) -> None:
        assert _position(name) < _position("Log in to GHCR") < _position("Build and push")

    def test_it_is_a_hard_gate_on_tag_pushes_only(self, name: str) -> None:
        """A workflow_dispatch backfill builds a tree that need not describe its version."""
        step = _step("docker-publish.yml", name)
        assert step.get("if") == "github.event_name == 'push'"
        assert "continue-on-error" not in step


def test_the_checkout_has_the_tag_on_a_push_and_is_the_default_on_a_dispatch() -> None:
    """Depth 0 and tags on a push (fetch-tags alone aborts on a shallow fetch of a tag ref);
    depth 1 and no tags, the defaults, on a dispatch. The tempting `event_name == 'push' && 0 || 1`
    is wrong: 0 is falsy in a GitHub expression, so `||` would take its right side and give 1."""
    checkout = _steps("docker-publish.yml")[0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == "${{ github.event_name != 'push' && 1 || 0 }}"
    assert checkout["with"]["fetch-tags"] == "${{ github.event_name == 'push' }}"
    assert checkout["with"]["persist-credentials"] is False


@pytest.mark.parametrize("name", ["Log in to GHCR", "Build and push"])
def test_a_failed_gate_leaves_no_way_to_log_in_or_push(name: str) -> None:
    """Actions skips a step after a failed one unless it has an `if` (`always()`, `failure()`)."""
    assert "if" not in _step("docker-publish.yml", name)


def _env(repo: Path, **extra: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(repo.parent),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        **extra,
    }


def _scratch_repo(tmp_path: Path, heading: str, tag_instant: str | None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n{heading}\n\n- a change\n", encoding="utf-8"
    )

    def git(*args: str, instant: str = _COMMIT_INSTANT) -> None:
        env = _env(repo, GIT_AUTHOR_DATE=instant, GIT_COMMITTER_DATE=instant)
        who = ["-c", "user.name=canary", "-c", "user.email=canary@example.test"]
        subprocess.run(["git", *who, *args], cwd=repo, env=env, check=True, capture_output=True)

    git("init", "-q")
    git("add", "CHANGELOG.md")
    git("commit", "-q", "-m", "release")
    git("tag", "-a", "v0.0.1", "-m", "older")  # the real repo has many tags; the gate reads one
    if tag_instant is not None:
        git("tag", "-a", _REF, "-m", "release", instant=tag_instant)
    return repo


def _run_gate(name: str, repo: Path) -> tuple[int, str]:
    """Run docker-publish.yml's copy of a gate the way Actions does (`bash -e`), with the
    runner's TZ east of UTC so that only the script's own `TZ=UTC` can give the UTC day."""
    script = repo.parent / "gate.sh"
    script.write_text(_step("docker-publish.yml", name)["run"], encoding="utf-8")
    env = _env(repo, GITHUB_REF_NAME=_REF, TZ="IST-5:30")
    argv = ["bash", "--noprofile", "--norc", "-e", str(script)]
    done = subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True, check=False)
    return done.returncode, done.stdout + done.stderr


_NOON = "2026-10-01T12:00:00+0000"
_LAST_SECOND = "2026-10-01T23:59:59+0000"
_MIDNIGHT = "2026-10-02T00:00:00+0000"
_EAST = "2026-10-02T01:30:00+0530"  # 2026-10-01T20:00:00Z: already the 2nd in the tagger's zone

# (tag instant, day in the heading, the tag's UTC day); the gate passes when the last two agree.
_DAY_CASES = [
    pytest.param(_NOON, "2026-10-01", "2026-10-01", id="midday"),
    pytest.param(_LAST_SECOND, "2026-10-01", "2026-10-01", id="23:59:59Z"),
    pytest.param(_MIDNIGHT, "2026-10-02", "2026-10-02", id="00:00:00Z"),
    pytest.param(_EAST, "2026-10-01", "2026-10-01", id="tagger-east-of-utc"),
    pytest.param(_NOON, "2026-09-30", "2026-10-01", id="day-before"),
    pytest.param(_NOON, "2026-10-02", "2026-10-01", id="day-after"),
    pytest.param(_LAST_SECOND, "2026-10-02", "2026-10-01", id="23:59:59Z-next-day"),
    pytest.param(_MIDNIGHT, "2026-10-01", "2026-10-02", id="00:00:00Z-previous-day"),
    pytest.param(_EAST, "2026-10-02", "2026-10-01", id="taggers-local-day"),
]
# (heading, tag instant or None for no tag, text the refusal must contain)
_REFUSALS = [
    pytest.param("## [9.9.9]", _NOON, "has no 'YYYY-MM-DD' date", id="no-date"),
    pytest.param("## [9.9.9] — 2026-10", _NOON, "has no 'YYYY-MM-DD' date", id="partial-date"),
    pytest.param("## [9.9.9] — 2026-10-01", None, "could not read the creation date", id="no-tag"),
]


@pytest.mark.parametrize(
    ("heading", "found"),
    [
        pytest.param("## [9.9.9] — 2026-10-01", True, id="section-present"),
        pytest.param("## [9.9.90] — 2026-10-01", False, id="only-a-longer-version"),
        pytest.param("## [8.8.8] — 2026-10-01", False, id="only-another-version"),
        pytest.param("### [9.9.9] — 2026-10-01", False, id="wrong-heading-level"),
    ],
)
def test_the_section_gate_passes_only_when_the_tags_section_exists(
    tmp_path: Path, heading: str, found: bool
) -> None:
    code, output = _run_gate(_SECTION_GATE, _scratch_repo(tmp_path, heading, _NOON))

    assert code == (0 if found else 1), output
    assert ("no '## [9.9.9]' section" in output) is not found


@pytest.mark.parametrize(("tag_instant", "heading_day", "tag_day"), _DAY_CASES)
def test_the_date_gate_follows_the_tags_utc_day(
    tmp_path: Path, tag_instant: str, heading_day: str, tag_day: str
) -> None:
    repo = _scratch_repo(tmp_path, f"## [9.9.9] — {heading_day}", tag_instant)

    code, output = _run_gate(_DATE_GATE, repo)

    if heading_day == tag_day:
        assert code == 0, output
        assert f"dated {heading_day} matches tag creation {tag_day} (UTC)" in output
    else:
        assert code == 1, output
        assert f"dated {heading_day} but tag {_REF} was created {tag_day} (UTC)" in output


@pytest.mark.parametrize(("heading", "tag_instant", "fragment"), _REFUSALS)
def test_the_date_gate_refuses_what_it_cannot_check(
    tmp_path: Path, heading: str, tag_instant: str | None, fragment: str
) -> None:
    """No date, or no tag object to compare it with: the gate checked nothing, so it fails."""
    code, output = _run_gate(_DATE_GATE, _scratch_repo(tmp_path, heading, tag_instant))

    assert code == 1, output
    assert fragment in output
    assert "was created" not in output, "the refusal must come before any comparison"
