"""Every install command a README prints must actually work.

WHY THIS EXISTS. Each distribution's `pyproject.toml` sets `readme = "README.md"`, so these
files are not internal notes — they are the **PyPI long description**, i.e. the project page
a new user copies commands from. Two failure modes shipped that way and neither was
detectable by any existing check:

1. `packages/trelix-langchain/README.md` advertised `pip install "trelix-langchain[bedrock]"`
   and `"trelix-langchain[code-embeddings]"`. That `pyproject.toml` has no
   `[project.optional-dependencies]` table at all, so PyPI reports `provides_extra: None`
   and pip installs the base package **silently** — an unknown extra is a warning at most,
   never an error. `code-embeddings` does not exist on core either.
2. `packages/trelix-mcp/README.md` hard-pinned `trelix-mcp==2.12.0` in seven places while the
   package shipped 3.1.2 — five releases stale, on the live project page, including the
   primary command under its own `## Install` heading.

CONTRIBUTING.md's doc-stamp grep could not have caught (2): it is scoped `docs/*.md *.md`,
which never descends into `packages/*/README.md`. That scope hole is why a stale pin sat in a
published long description. A grep with a blind spot reads as coverage; this test reads the
files the grep cannot reach.

These assertions are deliberately about the *published* surface, not about style. An extra
that does not exist and a version that no longer ships are both commands that fail for a
reader while passing every other gate in the repo.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# dist name -> the pyproject.toml that declares its extras
_OWNERS = {
    "trelix": _ROOT / "pyproject.toml",
    "trelix-mcp": _ROOT / "packages" / "trelix-mcp" / "pyproject.toml",
    "trelix-langchain": _ROOT / "packages" / "trelix-langchain" / "pyproject.toml",
    "trelix-llama-index": _ROOT / "packages" / "trelix-llama-index" / "pyproject.toml",
}

# `pkg[a,b]` and `pkg==1.2.3`, in any of the quoting styles the docs use.
_EXTRAS_RE = re.compile(r"(trelix(?:-[a-z][a-z-]*)?)\[([a-z0-9][a-z0-9,_-]*)\]")
_PIN_RE = re.compile(r"(trelix(?:-[a-z][a-z-]*)?)==([0-9][^\"'\s]*)")


def _readmes() -> list[Path]:
    """Root README plus every package README, excluding vendored trees."""
    found = [_ROOT / "README.md"]
    found += sorted(p for p in _ROOT.glob("packages/*/README.md") if "node_modules" not in p.parts)
    return [p for p in found if p.is_file()]


def _install_lines(path: Path) -> list[tuple[int, str]]:
    return [
        (n, line)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "pip install" in line
    ]


def _declared_extras(pyproject: Path) -> set[str]:
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return set(project.get("optional-dependencies", {}))


def _rel(path: Path) -> str:
    return str(path.relative_to(_ROOT))


def _advertised_extras() -> list[tuple[Path, int, str, str]]:
    """(readme, line, dist, extra) for every extra any README tells a reader to install."""
    out = []
    for readme in _readmes():
        for lineno, line in _install_lines(readme):
            for dist, extras in _EXTRAS_RE.findall(line):
                if dist not in _OWNERS:
                    continue
                for extra in extras.split(","):
                    out.append((readme, lineno, dist, extra.strip()))
    return out


def _advertised_pins() -> list[tuple[Path, int, str, str]]:
    """(readme, line, dist, version) for every hard version pin in an install command."""
    out = []
    for readme in _readmes():
        for lineno, line in _install_lines(readme):
            for dist, version in _PIN_RE.findall(line):
                if dist in _OWNERS:
                    out.append((readme, lineno, dist, version))
    return out


def test_at_least_one_readme_and_one_install_command_were_found() -> None:
    """Guard against the checks below passing because the discovery silently found nothing."""
    assert _readmes(), "no READMEs discovered — the glob is wrong"
    assert _advertised_extras(), "no extras-style install commands found — the regex is wrong"


_ADVERTISED_EXTRAS = _advertised_extras()


@pytest.mark.parametrize(
    ("readme", "lineno", "dist", "extra"),
    _ADVERTISED_EXTRAS,
    # The id must carry the README's path and line number, not just the extra name.
    # `ids=lambda v: v if isinstance(v, str) else ""` produced `--trelix-local` three
    # separate times (and `--trelix-knowledge-graph` five), because several READMEs
    # advertise the same extra. `strict_parametrization_ids` (via `strict = true` in
    # pyproject) turns that into a collection error, and rightly: with duplicate ids `-k`
    # cannot address a single case and a failure report cannot tell you WHICH README line
    # is broken — the only thing this test exists to say.
    ids=[
        f"{_rel(readme)}:{lineno}:{dist}[{extra}]"
        for readme, lineno, dist, extra in _ADVERTISED_EXTRAS
    ],
)
def test_every_advertised_extra_actually_exists(
    readme: Path, lineno: int, dist: str, extra: str
) -> None:
    declared = _declared_extras(_OWNERS[dist])
    assert extra in declared, (
        f'{_rel(readme)}:{lineno} tells a reader to run `pip install "{dist}[{extra}]"`, but '
        f"{_rel(_OWNERS[dist])} declares "
        f"{'no extras at all' if not declared else f'only {sorted(declared)}'}. pip treats an "
        "unknown extra as a warning, not an error, so the reader silently gets the base "
        "package. This README is that distribution's PyPI long description, so the broken "
        "command is on the live project page."
    )


def test_no_readme_hard_pins_a_trelix_version() -> None:
    """A version hardcoded into a doc's own install command rots at the next release.

    `packages/trelix-mcp/README.md` sat on `==2.12.0` for five releases, in seven places
    including the primary command under its own `## Install` heading. All four distributions
    ship on one core tag, so the newest adapter is always the one built alongside the newest
    core — an unpinned install resolves a working pair by construction. Pinning belongs in a
    *reader's* `requirements.txt`, where they control when it moves;
    docs/LANGCHAIN_LLAMAINDEX_GUIDE.md says exactly that and links the FAQ block for it.

    Deliberately ONE assertion over the whole set rather than one parametrized case per pin.
    The healthy state here is an empty list, and `parametrize` over an empty list reports as
    a single SKIPPED test — which reads as "disabled" to anyone scanning output, and would
    have to be justified against this repo's no-skipped-tests rule. A plain assertion is
    green when clean and names every offender at once when it is not.
    """
    pins = _advertised_pins()
    assert not pins, "READMEs hard-pin a trelix version:\n" + "\n".join(
        f"  {_rel(readme)}:{lineno} pins `{dist}=={version}` — this file is {dist}'s PyPI "
        f"long description, so that pin is stale the moment the next release ships"
        for readme, lineno, dist, version in pins
    )


def test_the_nomic_code_extra_declares_einops() -> None:
    """An extra that omits a REQUIRED transitive import is the same class of bug as one
    that does not exist: pip succeeds and the provider fails at use.

    `nomic-ai/CodeRankEmbed`'s `config.json` declares
    `auto_map.AutoModel = modeling_hf_nomic_bert.NomicBertModel`, and that published module
    imports `einops` at its MODULE TOP LEVEL — so the import runs during the
    `trust_remote_code` load, before any trelix code touches the model. Until v3.2.0 `einops`
    was declared by no extra and no installed package provided it transitively, so
    `nomic-code` raised `ModuleNotFoundError` from inside remote model code every time it was
    selected: the provider had never constructed for any user, in any release.

    This assertion is the offline half of the guard. The other half is the ci.yml unit job
    installing `[nomic-code]`, which makes pip resolution itself the positive control — an
    unresolvable or deleted requirement turns that job red. Neither half can prove the
    provider CONSTRUCTS; that needs the model weights, which CI does not download. So the
    claim these two support is exactly "the extra resolves and einops is importable".

    MUTATION: delete the einops line from the nomic-code extra and this fails.
    """
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]

    assert "nomic-code" in extras, "the nomic-code extra is gone; the provider is unreachable"
    requirements = extras["nomic-code"]
    names = {re.split(r"[<>=!~\[ ]", req, maxsplit=1)[0].lower() for req in requirements}

    # Precondition: without this the assertion below could pass on an empty parse.
    assert "sentence-transformers" in names, (
        f"the requirement parse produced {names} — it is not reading the extra correctly, so "
        "the einops assertion below would be vacuous"
    )
    assert "einops" in names, (
        "the nomic-code extra no longer declares einops. CodeRankEmbed's published "
        "modeling_hf_nomic_bert.py imports it at module top level during the "
        "trust_remote_code load, so the provider cannot construct without it — which is the "
        "state trelix shipped in from this provider's introduction through v3.1.7."
    )
