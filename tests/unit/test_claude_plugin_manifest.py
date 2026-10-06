"""The Claude Code marketplace and plugin manifests are exactly what the docs describe.

WHY. `.claude-plugin/marketplace.json` and `plugins/trelix/` are read by Claude Code, not by
this package, so nothing else in the test suite would notice when they drift: a renamed entry
makes `claude plugin install trelix@trelix` fail with "not found in marketplace"; a `.mcp.json`
that adds `env`, drops the `==` pin, or points at a version PyPI does not have breaks the
server start on every machine that installs the plugin; and an edit under `plugins/trelix/`
that does not bump `plugin.json`'s `version` reaches nobody, because Claude Code keeps every
user on the cached copy until that string changes (plugins/loading: "a manifest that pins
version keeps every user on the cached copy until its author changes the string").

Every expected value is a literal. The pin's companions (the trelix-mcp version stamp, the
CHANGELOG section) are read from files the plugin does not own, as independent observations.
The contents hash (`_PLUGIN_TREE_SHA256`) makes every edit under plugins/trelix/ fail a test
until its literal is updated; its failure message asks for the `version` bump, which no test
can see. `version` is also compared with the `.mcp.json` pin itself, so a pin bump that forgets
`version` fails mechanically. The other pins exist for their readable failure messages.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE = _ROOT / ".claude-plugin" / "marketplace.json"
PLUGIN = _ROOT / "plugins" / "trelix"
PLUGIN_MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
MCP_JSON = PLUGIN / ".mcp.json"
PLUGIN_README = PLUGIN / "README.md"
MCP_PYPROJECT = _ROOT / "packages" / "trelix-mcp" / "pyproject.toml"
CHANGELOG = _ROOT / "CHANGELOG.md"

# The whole server declaration. No `env` (keys come from the user's environment, never the
# plugin), no `type` (an entry without one is stdio), no `--tools` (the pinned release has
# no such flag; the pin-bump PR adds `"--tools", "core"` once 3.4.4 is published).
EXPECTED_MCP_JSON = {
    "mcpServers": {
        "trelix": {
            "command": "uvx",
            "args": ["--from", "trelix-mcp==3.4.3", "trelix-mcp"],
        }
    }
}
_PIN_RE = re.compile(r"^trelix-mcp==(\d+\.\d+\.\d+)$")

# Every file under plugins/trelix/, in sorted order. The README documents each of them and
# every command they run; a file that is not here is one the README does not mention. A
# `CLAUDE.md` here is not loaded and `claude plugin validate` warns on it; a `bin/` directory
# makes claude.ai refuse the plugin.
EXPECTED_PLUGIN_FILES = [
    ".claude-plugin/plugin.json",
    ".mcp.json",
    "README.md",
    "skills/use-trelix-index/SKILL.md",
]
# sha256 over `rel_path\0bytes\0` for every file above, in that order. Any edit under
# plugins/trelix/ changes it and must come with a bump of plugin.json's `version`.
_PLUGIN_TREE_SHA256 = "fdc85bfcb8ad430e893ec08bdc1a79229eab573c0ca33ad76d81d1d14c6a91e4"
# The row of the README's "Every command the plugin executes" table that is the server launch.
_README_LAUNCH_ROW = (
    "| Session start, by Claude Code | `uvx --from trelix-mcp==3.4.3 trelix-mcp` "
    "| The MCP server over stdio |"
)
# Every `trelix-mcp`, `trelix[local]` or bare `trelix` version specifier in the README, with its
# operator: a `>=` warm-up would run a release the plugin does not pin.
_README_PIN_RE = re.compile(r"trelix(?:-mcp|\[local\])?(==|>=|<=|~=|!=|>|<)(\d+\.\d+\.\d+)")
# Bare prose mentions of a release ("trelix-mcp 3.4.3") in the plugin's README and skill: outside
# every specifier check above, so a pin bump that forgets them would leave the skill naming the
# wrong release.
_BARE_RELEASE_RE = re.compile(r"trelix-mcp (\d+\.\d+\.\d+)")


def _load(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.relative_to(_ROOT)} is not a JSON object"
    return data


def _pin() -> str:
    """The `X` in `trelix-mcp==X` from .mcp.json, after the whole-object check has passed."""
    args = _load(MCP_JSON)["mcpServers"]["trelix"]["args"]  # type: ignore[index]
    match = _PIN_RE.match(args[1])
    assert match is not None, f"args[1] is {args[1]!r}, not an exact `trelix-mcp==X.Y.Z` pin"
    return match.group(1)


def test_marketplace_lists_exactly_the_trelix_plugin() -> None:
    """One marketplace, one plugin, install id `trelix@trelix`, source inside the repository.

    MUTATIONS that must make this fail: rename the entry to `trelix-mcp`; point `source` at
    `./plugins`.
    """
    data = _load(MARKETPLACE)
    assert data["name"] == "trelix"
    owner = data["owner"]
    assert isinstance(owner, dict) and owner["name"] == "sairam0424"

    plugins = data["plugins"]
    assert isinstance(plugins, list) and len(plugins) == 1, f"expected one plugin: {plugins!r}"
    entry = plugins[0]
    assert entry["name"] == "trelix", "entry name must equal plugin.json's name (the install id)"
    assert entry["source"] == "./plugins/trelix"
    assert ".." not in entry["source"], "`claude plugin validate` rejects a `..` in source"
    assert isinstance(entry["description"], str) and entry["description"]
    assert (_ROOT / entry["source"] / ".claude-plugin" / "plugin.json").is_file()


def test_plugin_manifest_uses_default_component_locations() -> None:
    """Metadata only: no `mcpServers`, `hooks` or `skills` key, so `.mcp.json` and `skills/`
    load from their default locations; `version` is the trelix-mcp pin.

    MUTATIONS that must make this fail: add `"mcpServers": {}`; set `version` to `"1.0.0"`;
    move the .mcp.json pin to `3.4.4` without touching `version` (the pin-bump PR that forgets
    `version` leaves every user on the cached copy with the old pin).
    """
    data = _load(PLUGIN_MANIFEST)
    assert set(data) == {
        "name",
        "version",
        "description",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
    }
    assert data["name"] == "trelix"
    assert data["license"] == "MIT"
    author = data["author"]
    assert isinstance(author, dict) and isinstance(author["name"], str) and author["name"]
    assert isinstance(data["description"], str) and data["description"]

    version = data["version"]
    pin = _pin()
    assert version == "3.4.3", (
        f"plugin version is {version!r}: move this literal with every plugin bump (the "
        f"trelix-mcp pin is {pin!r}; `{pin}.N` is the form for a plugin-only change)"
    )
    assert version == pin or version.startswith(pin + "."), (
        f"plugin version {version!r} must be the trelix-mcp pin {pin!r} or `{pin}.N`"
    )


def test_mcp_json_is_exactly_the_pinned_uvx_launch() -> None:
    """Whole-object equality: command, the exact `==` pin, the console script, nothing else.

    MUTATIONS that must make this fail: `==` to `>=`; add `"env": {...}`; add `--tools core`.
    """
    assert _load(MCP_JSON) == EXPECTED_MCP_JSON

    with MCP_PYPROJECT.open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]
    args = EXPECTED_MCP_JSON["mcpServers"]["trelix"]["args"]
    assert args[2] in scripts, f"{args[2]!r} is not a console script of packages/trelix-mcp"


def test_pin_trails_the_core_stamp_and_names_a_released_section() -> None:
    """The pin is a published release: it is at most the repository's own trelix-mcp stamp
    and CHANGELOG.md has its `## [X.Y.Z]` section. (This is the offline guard; PyPI itself
    is checked by hand in the pin-bump PR body.)

    MUTATION that must make this fail: pin `3.4.4` while CHANGELOG has no `## [3.4.4]`.
    """
    pin = _pin()
    with MCP_PYPROJECT.open("rb") as handle:
        stamp = tomllib.load(handle)["project"]["version"]
    as_tuple = [tuple(int(part) for part in v.split(".")) for v in (pin, stamp)]
    assert as_tuple[0] <= as_tuple[1], f"pin {pin} is newer than the trelix-mcp stamp {stamp}"

    heading = f"## [{pin}]"
    released = [
        line
        for line in CHANGELOG.read_text(encoding="utf-8").splitlines()
        if line.startswith(heading)
    ]
    assert released, f"CHANGELOG.md has no `{heading}` section: {pin} is not a released version"


def test_plugin_readme_shows_the_exact_launch_command() -> None:
    """The README's command table shows the executed command, pin included, and every
    `trelix-mcp==X`, `trelix[local]==X` or bare `trelix==X` in the README is that pin (the intro,
    the warm-up, the table row and the opt-in local-embeddings command all repeat it).

    MUTATIONS that must make this fail: drop the pin from the table row (the intro and warm-up
    lines still carry it); change one occurrence to `3.4.2`; add a bare `trelix==3.4.2`; loosen
    the warm-up to `trelix-mcp>=3.4.3`.
    """
    readme = PLUGIN_README.read_text(encoding="utf-8")
    assert _README_LAUNCH_ROW in readme, "the command table must show the pinned launch command"
    specifiers = _README_PIN_RE.findall(readme)
    assert specifiers, "the README names no pinned version"
    operators = {operator for operator, _ in specifiers}
    assert operators == {"=="}, f"every README version specifier must be `==`: {sorted(operators)}"
    pins = {pin for _, pin in specifiers}
    assert pins == {"3.4.3"}, f"README pins disagree: {sorted(pins)}"


def test_bare_release_mentions_in_the_plugin_docs_name_the_pin() -> None:
    """Every "trelix-mcp X.Y.Z" in README.md and SKILL.md is the `.mcp.json` pin, read from
    that file here (an independent observation, not the test's own literal).

    MUTATION that must make this fail: change SKILL.md's "trelix-mcp 3.4.3" to 3.4.4.
    """
    args = _load(MCP_JSON)["mcpServers"]["trelix"]["args"]
    pin = next(a for a in args if a.startswith("trelix-mcp==")).split("==", 1)[1]
    skill = PLUGIN / "skills" / "use-trelix-index" / "SKILL.md"
    mentions = {
        (path.name, found)
        for path in (PLUGIN_README, skill)
        for found in _BARE_RELEASE_RE.findall(path.read_text(encoding="utf-8"))
    }
    assert mentions, "the plugin docs name no release"
    assert {found for _, found in mentions} == {pin}, (
        f"bare release mentions off the pin: {sorted(mentions)}"
    )


def _plugin_files() -> list[str]:
    return sorted(p.relative_to(PLUGIN).as_posix() for p in PLUGIN.rglob("*") if p.is_file())


def test_plugin_tree_is_exactly_the_documented_files() -> None:
    """MUTATION that must make this fail: add `plugins/trelix/CLAUDE.md`."""
    assert _plugin_files() == EXPECTED_PLUGIN_FILES


def test_plugin_contents_are_the_ones_the_version_was_bumped_for() -> None:
    """Any byte under plugins/trelix/ moves this hash, so every edit there is loud. The test
    cannot see whether `version` moved with it: the failure message carries that rule.

    MUTATION that must make this fail: change one byte in any file under plugins/trelix/.
    """
    digest = hashlib.sha256()
    for rel in _plugin_files():
        digest.update(rel.encode() + b"\0" + (PLUGIN / rel).read_bytes() + b"\0")
    assert digest.hexdigest() == _PLUGIN_TREE_SHA256, (
        "plugin files changed: update this literal AND bump plugin.json's version "
        "(Claude Code keeps users on the cached copy until it changes; no test checks the bump)"
    )
