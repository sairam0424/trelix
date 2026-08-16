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
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_RELEASE = _ROOT / ".github" / "workflows" / "release.yml"
_DOCKER = _ROOT / ".github" / "workflows" / "docker-publish.yml"


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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

    def test_skip_existing_is_still_set(self) -> None:
        """The gate replaces the need to remove it; removing it breaks safe re-runs."""
        assert "skip-existing: true" in _RELEASE.read_text(encoding="utf-8")


class TestEveryVersionSiteIsChecked:
    """The gate must cover the sites the release checklist misses.

    `CONTRIBUTING.md` enumerates five, and its procedure greps for the PREVIOUS version —
    which cannot find a site still stamped `2.x`. Two were exactly that: helm
    `values.yaml`'s `image.tag` (the chart advertised `appVersion: 3.1.2` while deploying
    `2.12.0`) and `packages/trelix-mcp/server.json` (two entries, both `2.12.0`).
    """

    # (label fragment in the workflow, callable reading the real value)
    SITES = {
        "pyproject.toml": lambda: tomllib.load((_ROOT / "pyproject.toml").open("rb"))["project"][
            "version"
        ],
        "src/trelix/__init__.py": lambda: re.search(
            r'__version__ = "([^"]+)"', (_ROOT / "src" / "trelix" / "__init__.py").read_text()
        ).group(1),
        "helm/trelix/Chart.yaml": lambda: yaml.safe_load(
            (_ROOT / "helm" / "trelix" / "Chart.yaml").read_text()
        )["appVersion"],
        "helm/trelix/values.yaml": lambda: yaml.safe_load(
            (_ROOT / "helm" / "trelix" / "values.yaml").read_text()
        )["image"]["tag"],
        "packages/trelix-mcp/pyproject.toml": lambda: tomllib.load(
            (_ROOT / "packages" / "trelix-mcp" / "pyproject.toml").open("rb")
        )["project"]["version"],
        "packages/trelix-mcp/server.json": lambda: json.loads(
            (_ROOT / "packages" / "trelix-mcp" / "server.json").read_text()
        )["version"],
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
