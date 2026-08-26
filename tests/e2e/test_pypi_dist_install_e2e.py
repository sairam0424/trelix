"""Build the real wheel for all 4 packages, install into a fresh venv, verify.

Never editable (`pip install -e .`), never a live PyPI pull — a live pull can't
gate the release that creates the version it would check (it would only ever
confirm a version already published), and an editable install can't catch a
packaging mistake (a file the wheel build forgot, a broken entry point in the
actual built dist). Building the wheel and installing THAT is the one path
that is both a real fresh-install boundary and something that can run before
a version is ever published — see D2 in the implementation plan, which reuses
this exact same logic to hard-gate `release.yml`'s `publish` job.

Ports the exact steps both manual PyPI-install verification passes ran by
hand for the 3.2.1 and 3.2.2 releases, now permanent and parametrized.

MUTATION: delete a package's `__init__.py` from its `[tool.hatch.build.targets.wheel]`
packages list and the corresponding case here fails with a real ModuleNotFoundError
inside the fresh venv — a defect an editable install could never surface, because
editable installs read straight from src/, never from what the wheel build actually
includes.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# (distribution source dir, import name, expected __version__ this process already has)
_PACKAGES = [
    ("trelix", _ROOT, "trelix"),
    ("trelix-mcp", _ROOT / "packages" / "trelix-mcp", "trelix_mcp"),
    ("trelix-langchain", _ROOT / "packages" / "trelix-langchain", "trelix_langchain"),
    ("trelix-llama-index", _ROOT / "packages" / "trelix-llama-index", "trelix_llama_index"),
]


def _installed_version(import_name: str) -> str:
    import importlib

    return str(importlib.import_module(import_name).__version__)


@pytest.fixture(scope="module")
def wheelhouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build every package's wheel once, into one shared directory.

    Shared so that installing trelix-mcp's wheel (which depends on
    ``trelix>=X``) resolves against the wheel THIS run just built via
    ``--find-links``, never a live PyPI pull of an older published trelix.
    """
    out = tmp_path_factory.mktemp("wheelhouse")
    for _dist_name, source_dir, _import_name in _PACKAGES:
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(source_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"building the wheel for {source_dir} failed:\n{result.stdout}\n{result.stderr}"
        )
    built = sorted(p.name for p in out.glob("*.whl"))
    assert len(built) == len(_PACKAGES), f"expected {len(_PACKAGES)} wheels, got {built}"
    return out


@pytest.mark.parametrize(
    ("dist_name", "_source_dir", "import_name"), _PACKAGES, ids=[p[0] for p in _PACKAGES]
)
def test_fresh_venv_install_reports_the_real_version(
    dist_name: str, _source_dir: Path, import_name: str, wheelhouse: Path, tmp_path: Path
) -> None:
    venv_dir = tmp_path / f"venv-{dist_name}"
    # symlinks=True, not the venv.create() default of False: a copied interpreter
    # binary from a framework-style Python build (e.g. uv's python-build-standalone)
    # can fail to resolve its own @rpath/libpythonX.Y.dylib when copied into a new
    # venv directory, crashing ensurepip with SIGABRT. Symlinking keeps the binary
    # at its original path, where the dylib resolves correctly — also what the
    # `python -m venv` CLI does by default on POSIX, unlike this programmatic API.
    venv.create(venv_dir, with_pip=True, symlinks=True)
    venv_python = venv_dir / "bin" / "python"
    assert venv_python.exists(), f"venv creation did not produce {venv_python}"

    # Install the package under test by its EXACT wheel path, not by name: if the
    # working tree's version happens to equal an already-published PyPI version
    # (true right after a release, which is exactly when this matters most), pip
    # given only --find-links (no --no-index) could silently pick the published
    # wheel instead of the one this run just built, making the whole test vacuous.
    # --find-links (without --no-index) still lets pip resolve transitive deps
    # (mcp, fastmcp, langchain-core, trelix itself) from real PyPI as a fallback.
    normalized = dist_name.replace("-", "_")
    matches = list(wheelhouse.glob(f"{normalized}-*.whl"))
    assert len(matches) == 1, (
        f"expected exactly one {normalized} wheel in {wheelhouse}, got {matches}"
    )

    install = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--find-links",
            str(wheelhouse),
            str(matches[0]),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert install.returncode == 0, install.stderr

    version_check = subprocess.run(
        [str(venv_python), "-c", f"import {import_name}; print({import_name}.__version__)"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert version_check.returncode == 0, version_check.stderr
    got_version = version_check.stdout.strip()
    expected_version = _installed_version(import_name)
    assert got_version == expected_version, (
        f"{dist_name}'s built wheel reports version {got_version!r}, but this "
        f"working tree's own {import_name}.__version__ is {expected_version!r} — "
        f"the wheel build and the source tree have drifted apart"
    )
