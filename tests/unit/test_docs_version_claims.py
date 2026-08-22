"""The places a reader ACTS on a version number must name the version that ships.

WHY THIS EXISTS. `release.yml`'s verify-version job checks twelve stamps, all of them code
or chart metadata. It checks no prose. So every release since v3.1.2 left a set of documents
still calling 3.1.2 "latest" — four releases of drift, discovered by an audit of the 3.1.5
candidate rather than by any gate.

Most of those hits are fine and MUST NOT be touched: "before v3.1.2", "through v3.1.1",
"added in v2.8.1" are history, and a test that rewrote them would destroy the record. Doc
*header* stamps ("Version: 3.1.2" atop a guide) drifted the same way and are covered by the
last two tests here, which scan only stamp POSITIONS — H1, masthead, signature — so that no
regex ever has to guess whether a triple in body prose is a claim or a record.

What this file pins is the narrow set a reader turns into a command or a decision:

  * `.github/SECURITY.md`'s Supported Versions table — tells a user which line receives
    security fixes. Naming a superseded version there says patches land somewhere they do not.
  * `docs/FAQ.md`'s four-way pin block — copy-pasteable into requirements.txt. Stale here
    means a user pins four packages to a version several releases old and believes it current.
  * `.github/ISSUE_TEMPLATE/bug_report.yml` — the version placeholder and the "I am using the
    latest release (X)" checkbox. A reporter ticks that box against the wrong number.
  * `helm/trelix/README.md`'s `image.tag` row, which asserts of itself that it "always equals
    Chart.yaml's appVersion" — a claim that was false for four releases.

Precedent for testing published prose this way: tests/unit/test_readme_install_commands.py,
which exists because a stale `==` pin shipped inside a PyPI long description.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _shipping_version() -> str:
    """The version this tree would publish — read from the site the gate treats as canonical."""
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        version = tomllib.load(fh)["project"]["version"]
    assert isinstance(version, str) and version, "pyproject.toml has no project.version"
    return version


def test_security_md_names_the_shipping_version_as_latest() -> None:
    """A security policy that names a superseded version tells users patches land elsewhere."""
    version = _shipping_version()
    text = (_ROOT / ".github" / "SECURITY.md").read_text(encoding="utf-8")

    row = re.search(r"^\|\s*([0-9]+\.[0-9]+\.[0-9]+)\s*\(latest\)\s*\|", text, re.M)
    assert row is not None, "SECURITY.md no longer has a '<version> (latest)' row to check"
    assert row.group(1) == version, (
        f"SECURITY.md advertises {row.group(1)} as the supported line while this tree ships "
        f"{version}; a reader concludes fixes are not shipping for what they installed"
    )


def test_the_faq_pin_block_matches_the_shipping_version() -> None:
    """This block is copy-pasted into requirements.txt, so a stale number becomes a real pin."""
    version = _shipping_version()
    text = (_ROOT / "docs" / "FAQ.md").read_text(encoding="utf-8")

    pins = dict(re.findall(r"^(trelix(?:-[a-z-]+)?)==([0-9]+\.[0-9]+\.[0-9]+)$", text, re.M))
    assert pins, "the four-way pin block in docs/FAQ.md is gone or reshaped"
    expected = {"trelix", "trelix-mcp", "trelix-langchain", "trelix-llama-index"}
    assert set(pins) == expected, f"the pin block no longer covers all four packages: {pins}"
    wrong = {name: got for name, got in pins.items() if got != version}
    assert not wrong, (
        f"docs/FAQ.md pins packages to a superseded version (ships {version}): {wrong}"
    )


@pytest.mark.parametrize("shape", ["placeholder", "latest-checkbox"])
def test_the_bug_report_template_matches_the_shipping_version(shape: str) -> None:
    """Both version-bearing fields a reporter reads before filing."""
    version = _shipping_version()
    text = (_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(encoding="utf-8")

    if shape == "placeholder":
        found = re.search(r'placeholder:\s*"([0-9]+\.[0-9]+\.[0-9]+)"', text)
        label = "the version placeholder"
    else:
        found = re.search(
            r"label: I am using the latest release \(([0-9]+\.[0-9]+\.[0-9]+)\)", text
        )
        label = 'the "latest release" checkbox'

    assert found is not None, f"{label} is gone from bug_report.yml"
    assert found.group(1) == version, (
        f"{label} says {found.group(1)} while this tree ships {version}"
    )


def test_the_helm_readme_image_tag_matches_the_charts_appversion() -> None:
    """That row asserts of itself that it always equals appVersion. Hold it to that.

    Checked against Chart.yaml rather than pyproject, because that is the claim the row makes —
    testing it against the wrong source would pass while the stated invariant was broken.
    """
    chart = (_ROOT / "helm" / "trelix" / "Chart.yaml").read_text(encoding="utf-8")
    app_version = re.search(r'^appVersion:\s*"?([^"\s]+)"?', chart, re.M)
    assert app_version is not None, "Chart.yaml has no appVersion"

    readme = (_ROOT / "helm" / "trelix" / "README.md").read_text(encoding="utf-8")
    row = re.search(r"^\|\s*`image\.tag`\s*\|\s*`([^`]+)`\s*\|", readme, re.M)
    assert row is not None, "helm/trelix/README.md no longer documents an image.tag default"
    assert row.group(1) == app_version.group(1), (
        f"helm/trelix/README.md documents image.tag as {row.group(1)} while Chart.yaml's "
        f"appVersion is {app_version.group(1)} — the row claims these are always equal"
    )


def test_the_charts_own_version_moves_when_its_contents_do() -> None:
    """Helm requires `version` to change whenever the chart changes; appVersion is not enough.

    Not a version-equality check — the chart's own SemVer is independent of the app's. This
    only pins that it is present and parseable, so the field cannot silently vanish; the
    reason it must be *bumped* is recorded in Chart.yaml's own comment, which had been
    ignored for three consecutive releases before the 3.1.5 audit.
    """
    chart = (_ROOT / "helm" / "trelix" / "Chart.yaml").read_text(encoding="utf-8")

    own = re.search(r"^version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$", chart, re.M)
    assert own is not None, "helm/trelix/Chart.yaml has no parseable chart `version`"
    assert own.group(1) != "0.0.0", "the chart version is a placeholder"


# ── docs/ header, masthead and signature version stamps ────────────────────────
#
# Scope is POSITIONAL on purpose. A version triple in body prose is almost always
# history ("before v3.1.2", "New in v3.0.0") and must never be rewritten, so this
# looks only at the three places a stamp announces what the document is ABOUT: an H1
# title, a labelled `Version:` line in the first few lines, and a `trelix vX.Y.Z`
# signature at the foot. Measured on the tree at 3.1.5, those three shapes matched 16
# sites across 12 pages and produced zero false positives — docs/AUDIT.md and
# docs/SSO.md ("New in v3.0.0") and docs/BACKWARDS_COMPATIBILITY.md ("SemVer 2.0.0")
# all pass untouched.
#
# Body-text and sample-output claims (`# trelix 3.1.2` under `trelix --version`,
# "trelix-mcp v3.1.2 exposes 15 tools") are NOT covered: nothing about their position
# distinguishes them from history, so a regex broad enough to catch them would rewrite
# the record. They stay a manual release-checklist item.

_VERSION_TRIPLE = re.compile(r"(?<![\w.])v?(\d+\.\d+\.\d+)(?![\w.])")
_LABELLED_STAMP = re.compile(r"\bVersion:\**\s*v?(\d+\.\d+\.\d+)")
_SIGNATURE_STAMP = re.compile(r"\btrelix v(\d+\.\d+\.\d+)")
_MASTHEAD_LINES = 5
_SIGNATURE_LINES = 3


def _doc_pages() -> list[Path]:
    """Top-level docs/ pages, minus any whose FILENAME names a release.

    `docs/v2.4.0-world-release-report.md` stamps 2.4.0 in its H1 and its masthead and
    always should: that release is its subject. Same for the nested dated artifacts
    (`docs/reports/`, `docs/migration/`, `docs/superpowers/`), which `glob` skips.
    """
    return [
        page
        for page in sorted((_ROOT / "docs").glob("*.md"))
        if not re.search(r"\d+\.\d+\.\d+", page.name)
    ]


def _version_stamps(page: Path) -> list[tuple[int, str, str]]:
    """Every (line number, version, shape) stamp on one page."""
    lines = page.read_text(encoding="utf-8").splitlines()
    stamps: list[tuple[int, str, str]] = []

    for lineno, line in enumerate(lines[:_MASTHEAD_LINES], start=1):
        if line.startswith("# "):
            title = _VERSION_TRIPLE.search(line)
            if title is not None:
                stamps.append((lineno, title.group(1), "H1 title"))
        labelled = _LABELLED_STAMP.search(line)
        if labelled is not None:
            stamps.append((lineno, labelled.group(1), "masthead Version: stamp"))

    tail_start = max(len(lines) - _SIGNATURE_LINES, 0)
    for lineno, line in enumerate(lines[tail_start:], start=tail_start + 1):
        signature = _SIGNATURE_STAMP.search(line)
        if signature is not None:
            stamps.append((lineno, signature.group(1), "signature line"))

    return stamps


def test_the_stamp_scanner_still_matches_something() -> None:
    """Guards the regexes, not the docs. Silence here would make the next test vacuous.

    This is the failure the four-release drift was hiding behind: a check nobody
    noticed had stopped looking. If a docs reshuffle drops the stamps below this floor,
    fail loudly rather than pass on an empty scan.
    """
    stamped = {page.name: _version_stamps(page) for page in _doc_pages()}
    pages_with_stamps = {name for name, found in stamped.items() if found}
    total = sum(len(found) for found in stamped.values())
    assert len(pages_with_stamps) >= 10, (
        f"only {len(pages_with_stamps)} docs/ pages carry a version stamp — the shapes "
        f"this test recognises have probably changed: {sorted(pages_with_stamps)}"
    )
    assert total >= 14, f"only {total} stamps matched across {len(pages_with_stamps)} pages"


@pytest.mark.parametrize("page", _doc_pages(), ids=lambda page: page.name)
def test_doc_page_version_stamps_name_the_shipping_version(page: Path) -> None:
    """A guide whose own header names a superseded release reads as unmaintained."""
    version = _shipping_version()
    stale = [(n, got, shape) for n, got, shape in _version_stamps(page) if got != version]
    assert not stale, "\n".join(
        f"docs/{page.name}:{n} ({shape}) stamps {got} while this tree ships {version}"
        for n, got, shape in stale
    )
