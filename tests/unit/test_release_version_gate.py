"""The release workflow must refuse a tag that disagrees with the tree.

WHY THIS EXISTS. `release.yml` fires on `push: tags: v*` and derives nothing from the
tree, while all four PyPI publish steps set `skip-existing: true`. So tagging `v3.2.0` on a
tree still stamped `3.1.2` built `trelix-3.1.2`, PyPI skipped it as already present, and
the run went **green** — leaving a GitHub Release full of binaries and zero new packages,
with nothing in the logs saying "published nothing". `skip-existing` is deliberately kept
(it is what makes a re-run after a partial failure safe), so the guard has to be a
separate gate.

`ci.yml` triggers on push/PR to `[main, develop]` only, never on tags, so the entire
pre-publish gate was `python -m build` + `twine check` + a `--help` smoke test.

These tests read the workflow YAML rather than mocking Actions: the failure being
prevented is a *missing* job, and only the file can show whether it is there.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_RELEASE = _ROOT / ".github" / "workflows" / "release.yml"
_DOCKER = _ROOT / ".github" / "workflows" / "docker-publish.yml"


def _workflow(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _dunder_version(*parts: str) -> str:
    """Read a `__version__ = "..."` stamp the same way the gate's inline python does.

    Four of the eleven readers below are this exact shape, so going through one helper keeps
    the test honest: a stamp that stops matching fails here loudly instead of raising
    `AttributeError` on `None.group` and reading as an unrelated crash.
    """
    text = _ROOT.joinpath(*parts).read_text(encoding="utf-8")
    match = re.search(r'__version__ = "([^"]+)"', text)
    assert match is not None, f"no __version__ stamp in {'/'.join(parts)}"
    return match.group(1)


def _pyproject_version(*parts: str) -> str:
    with _ROOT.joinpath(*parts).open("rb") as handle:
        version: str = tomllib.load(handle)["project"]["version"]
    return version


class TestReleaseIsGatedOnVersionAgreement:
    def test_a_verify_version_job_exists(self) -> None:
        assert "verify-version" in _workflow(_RELEASE)["jobs"]

    def test_every_build_and_publish_job_waits_for_the_gate(self) -> None:
        """A job that skips the gate can still burn a build, or worse, publish."""
        jobs = _workflow(_RELEASE)["jobs"]

        for name, spec in jobs.items():
            if name == "verify-version":
                continue
            needs = spec.get("needs") or []
            needs = [needs] if isinstance(needs, str) else needs
            # Either directly gated, or gated through something that is.
            transitively = any(
                "verify-version" in ((jobs[n].get("needs") or []) if n in jobs else [])
                for n in needs
            )
            assert "verify-version" in needs or transitively, (
                f"job '{name}' does not depend on verify-version, directly or otherwise"
            )

    def test_the_suite_runs_before_publishing(self) -> None:
        """Tags never triggered ci.yml, so the release path had no test gate at all."""
        jobs = _workflow(_RELEASE)["jobs"]
        assert "test" in jobs, "no test job in the release workflow"
        steps = " ".join(str(s.get("run", "")) for s in jobs["test"]["steps"])
        assert "pytest tests/unit/" in steps
        assert "pytest packages/trelix-mcp/tests/" in steps
        # The publish job uploads four distributions; the release path must test all four.
        # These two suites ran only in ci.yml, which never fires on a tag.
        assert "pytest packages/trelix-langchain/tests/" in steps
        assert "pytest packages/trelix-llama-index/tests/" in steps

    def test_skip_existing_is_still_set(self) -> None:
        """The gate replaces the need to remove it; removing it breaks safe re-runs."""
        assert "skip-existing: true" in _RELEASE.read_text(encoding="utf-8")


class TestEveryVersionSiteIsChecked:
    """The gate must cover the sites the release checklist misses.

    `CONTRIBUTING.md` enumerated five, and its procedure greps for the PREVIOUS version —
    which by construction cannot find a site already stranded on an older line. Every site
    that had actually drifted was of that class: helm `values.yaml`'s `image.tag` (the chart
    advertised `appVersion: 3.1.2` while deploying `2.12.0`), `packages/trelix-mcp/server.json`
    (two entries, both `2.12.0`), and both adapters, frozen at `2.4.0`.

    The adapters are the worst case, because the gate's own purpose failed on them: the
    `publish` job builds and uploads all four distributions, every step sets
    `skip-existing: true`, and nothing compared the adapter stamps to anything — so a core
    tag published two packages, skipped two, and reported success.
    """

    # path as it appears in the workflow -> callable reading the real value off disk
    SITES: dict[str, Callable[[], str]] = {
        "pyproject.toml": lambda: _pyproject_version("pyproject.toml"),
        "src/trelix/__init__.py": lambda: _dunder_version("src", "trelix", "__init__.py"),
        "helm/trelix/Chart.yaml": lambda: str(
            yaml.safe_load((_ROOT / "helm" / "trelix" / "Chart.yaml").read_text())["appVersion"]
        ),
        "helm/trelix/values.yaml": lambda: str(
            yaml.safe_load((_ROOT / "helm" / "trelix" / "values.yaml").read_text())["image"]["tag"]
        ),
        "packages/trelix-mcp/pyproject.toml": lambda: _pyproject_version(
            "packages", "trelix-mcp", "pyproject.toml"
        ),
        "packages/trelix-mcp/src/trelix_mcp/__init__.py": lambda: _dunder_version(
            "packages", "trelix-mcp", "src", "trelix_mcp", "__init__.py"
        ),
        "packages/trelix-mcp/server.json": lambda: str(
            json.loads((_ROOT / "packages" / "trelix-mcp" / "server.json").read_text())["version"]
        ),
        "packages/trelix-langchain/pyproject.toml": lambda: _pyproject_version(
            "packages", "trelix-langchain", "pyproject.toml"
        ),
        "packages/trelix-langchain/src/trelix_langchain/__init__.py": lambda: _dunder_version(
            "packages", "trelix-langchain", "src", "trelix_langchain", "__init__.py"
        ),
        "packages/trelix-llama-index/pyproject.toml": lambda: _pyproject_version(
            "packages", "trelix-llama-index", "pyproject.toml"
        ),
        "packages/trelix-llama-index/src/trelix_llama_index/__init__.py": lambda: _dunder_version(
            "packages", "trelix-llama-index", "src", "trelix_llama_index", "__init__.py"
        ),
    }

    @pytest.mark.parametrize("site", sorted(SITES))
    def test_the_gate_mentions_this_site(self, site: str) -> None:
        assert site in _RELEASE.read_text(encoding="utf-8"), (
            f"{site} carries a version the gate does not check, so a release can ship "
            "with it stale — which is how helm values.yaml sat on 2.12.0 across five "
            "releases while Chart.yaml was bumped by every one"
        )

    @pytest.mark.parametrize("site", sorted(SITES))
    def test_the_site_is_still_readable_the_way_the_gate_reads_it(self, site: str) -> None:
        """A restructured file would make the gate crash or silently pass."""
        value = self.SITES[site]()
        assert isinstance(value, str) and value, f"{site} yielded {value!r}"
        assert re.fullmatch(r"\d+\.\d+\.\d+.*", value), f"{site} is not a version: {value!r}"

    @pytest.mark.parametrize("site", sorted(set(SITES) - {"pyproject.toml"}))
    def test_the_site_agrees_with_the_canonical_stamp(self, site: str) -> None:
        """Every site must already agree BEFORE a tag exists.

        `verify-version` compares each site against the tag, so it can only fail once
        someone has cut one — by which point the wrong thing may already be public. Sites
        disagreeing with each other is the same defect, is detectable now, and is what
        actually happened: both adapters sat on 2.4.0 across the seventeen releases that
        shipped after it, while core moved to 3.1.2, and no test in the tree said otherwise.

        Root `pyproject.toml` is the authority rather than a mutual-equality check so that a
        failure names which file is wrong instead of only reporting that two disagree.
        """
        canonical = self.SITES["pyproject.toml"]()
        assert self.SITES[site]() == canonical, (
            f"{site} is stamped {self.SITES[site]()!r} but pyproject.toml says "
            f"{canonical!r}. Every distribution this repo publishes is released by one "
            "core `v*` tag (see docs/BACKWARDS_COMPATIBILITY.md, 'Why lockstep'), so all "
            "stamps move together. If you intend to break lockstep, change that policy "
            "first — this test encodes it deliberately."
        )


class TestDockerPublishIsGatedToo:
    def test_it_verifies_the_tag_against_the_tree(self) -> None:
        """It derived the image tag from the git ref with nothing checking the tree, so a
        v3.2.0 tag on a 3.1.2 tree published an image labelled 3.2.0 containing 3.1.2."""
        text = _DOCKER.read_text(encoding="utf-8")
        assert "Verify the tag matches the tree" in text

    def test_the_check_is_skipped_for_manual_backfills(self) -> None:
        """workflow_dispatch exists to backfill an older release, where the tree
        legitimately does not match — gating that path would break the feature."""
        assert "if: github.event_name == 'push'" in _DOCKER.read_text(encoding="utf-8")

    def test_no_dispatch_input_is_interpolated_into_a_script_body(self) -> None:
        """`${{ github.event.inputs.* }}` inside `run:` is a shell-injection sink."""
        for line in _DOCKER.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or ":" in stripped.split("${{")[0][:6]:
                continue
            if "github.event.inputs" in line and ("echo " in line or "if [" in line):
                raise AssertionError(f"dispatch input reaches a script body: {stripped}")


class TestEveryReleasedChangelogHeadingIsDated:
    """The date on a released heading is a shipped artifact, not decoration.

    `release.yml` gates a tag on the heading existing AND on its date matching the
    tag's UTC day. These tests pin the same rule where it is cheap to fix — at PR
    time, before a tag exists — because by the time the workflow refuses, the
    correction costs a re-tag.

    Why the field earns its own tests: on 3.1.3 it was wrong twice in one day. First
    it kept the day the release was *prepared*; then it was "corrected" to the next
    day by reading a local clock after midnight, which shipped in the sdist. PyPI is
    immutable, so that date is permanent. Every release before it used the UTC
    publish day — and v3.1.0 proves the rule rather than merely agreeing with it,
    having published at 23:00:08Z, the next day in IST, with a heading that follows
    UTC.
    """

    #: `## [1.2.3] — 2026-08-19`. `[Unreleased]` is deliberately excluded: it has no
    #: date because it has no publish day yet, which is the whole point of the section.
    _HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\](.*)$", re.MULTILINE)
    _DATED = re.compile(r"^ — (\d{4}-\d{2}-\d{2})$")

    def _headings(self) -> list[tuple[str, str]]:
        text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        found = self._HEADING.findall(text)
        assert found, "no released version headings found — has the format changed?"
        return found

    def test_every_released_version_carries_an_iso_date(self) -> None:
        undated = [
            f"## [{version}]{tail}"
            for version, tail in self._headings()
            if not self._DATED.match(tail)
        ]
        assert undated == [], (
            "every released heading needs ' — YYYY-MM-DD', the tag's UTC day, because "
            f"release.yml refuses to publish without it: {undated}"
        )

    def test_the_dates_never_go_backwards_reading_down_the_file(self) -> None:
        """Newest first. A heading dated later than the one above it is a transposition.

        Catches the shape a hand-edited date makes — 3.1.3's wrong value was one day
        AHEAD of its predecessor's, so an ordering check sees it even without a tag.
        """
        dates = [
            match.group(1)
            for _, tail in self._headings()
            if (match := self._DATED.match(tail)) is not None
        ]
        inversions = [(above, below) for above, below in zip(dates, dates[1:]) if above < below]
        assert inversions == [], (
            "CHANGELOG runs newest-first, so each date must be >= the one below it; "
            f"these pairs are inverted (above, below): {inversions}"
        )
