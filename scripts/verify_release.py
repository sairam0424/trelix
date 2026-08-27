#!/usr/bin/env python3
"""Post-publish production verification. A REPORT, run once per release, by hand no more.

WHAT THIS IS
------------
This codifies the manual production-verification pass this project has now run BY
HAND, TWICE — once for the 3.2.1 release, once for 3.2.2. Both passes found a real
defect that had already shipped past every automated check that existed at the time:

  * trelix-mcp was never copied into the published Docker image (see the Dockerfile's
    own comment on this). `docker run --entrypoint trelix-mcp <image> --help` failed
    with exit code 127 — the console script did not exist inside the image at all.
  * trelix-mcp's console script silently ignored `--help`, `--version`, and every
    other flag: it always started the MCP stdio server regardless of argv, so there
    was no way to ask a published `trelix-mcp` what version it was without starting a
    server and speaking the MCP protocol at it.

Neither defect could have been caught by a unit test, a wheel-build test, or anything
that runs before a tag exists — the first needs the REAL published image, the second
needs the REAL published console script, invoked the way a real user invokes it (a
bare CLI call, not an in-process import). Both were only found because a human ran a
bespoke, re-typed-from-scratch verification pass after the fact. Re-typing that prompt
a third time is the failure mode this script exists to remove: it is the same five
checks, permanent, version-controlled, and runnable with one command instead of a
freshly-remembered ritual.

THE FIVE CHECKS
----------------
  1. check_pypi_installs      — fresh venv, `pip install <pkg>==<version>`, run each
                                 package's real smoke command, assert the version.
  2. check_docker_images       — pull both published tags, run the real entrypoints
                                 inside each one (the exact command that returned
                                 exit 127 on the unpatched 3.2.1 image).
  3. check_helm_chart          — `helm template` the chart AT THE RELEASE TAG, on all
                                 three store backends, and check the rendered image
                                 tag against Chart.yaml's appVersion.
  4. check_github_release_binaries — download every GitHub Release asset, confirm each
                                 one's file format/arch, and actually RUN the one that
                                 matches this host.
  5. check_security_audit      — `pip-audit` every published package, plus a per-wheel
                                 scan for anything .env/.git/credential/secret-shaped
                                 that should never have shipped inside a distribution.

Every check function returns a `list[str]` of human-readable failure descriptions and
NEVER raises — one failing sub-check must never hide the others' results. An empty
list means every sub-check in that category passed. `main()` runs every requested
category unconditionally, then prints one PASS/FAIL block per category and exits 1 if
any category reported a failure, 0 otherwise.

USAGE
-----
    python scripts/verify_release.py --version 3.2.2
    python scripts/verify_release.py --version 3.2.2 --skip-docker --skip-binaries

Run this ONCE per release, after BOTH of these are true:
  * the `v<version>` tag has been pushed to `origin`
  * `.github/workflows/release.yml`'s "Release" workflow AND
    `.github/workflows/docker-publish.yml`'s "Docker Publish" workflow have BOTH
    finished green for that tag.

Running it earlier just reproduces "not found yet" for every category — there is
nothing to verify until the artifacts this script downloads and installs actually
exist on PyPI, GHCR, and the GitHub Release.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import tempfile
import venv
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
GHCR_IMAGE = "ghcr.io/sairam0424/trelix"
GH_REPO = "sairam0424/trelix"

# (PyPI distribution name, top-level import name). trelix's smoke test uses its real
# console script; the other three are pure libraries with no console script, so their
# smoke test is the same "import it, print its __version__" the manual passes used.
_PACKAGES: tuple[tuple[str, str], ...] = (
    ("trelix", "trelix"),
    ("trelix-mcp", "trelix_mcp"),
    ("trelix-langchain", "trelix_langchain"),
    ("trelix-llama-index", "trelix_llama_index"),
)

# helm-lint.yml's own matrix — the exact three backends CI templates on every push.
_HELM_BACKENDS: tuple[str, ...] = ("sqlite", "qdrant", "lance")

# release.yml's "Rename binaries to unique release asset names" step — the exact four
# final asset names attached to the GitHub Release, and how to recognise each one's
# `file(1)` output and which (system, machine) pair from `platform` it belongs to.
_RELEASE_BINARIES: tuple[dict[str, Any], ...] = (
    {
        "asset": "trelix-macos-arm64",
        "file_markers": ("Mach-O", "arm64"),
        "host_match": lambda system, machine: system == "Darwin" and machine == "arm64",
        "host_desc": "macOS arm64",
    },
    {
        "asset": "trelix-linux-x64",
        "file_markers": ("ELF",),
        "file_markers_any": ("x86-64", "x86_64"),
        "host_match": lambda system, machine: system == "Linux" and machine in ("x86_64", "x64"),
        "host_desc": "Linux x86_64",
    },
    {
        "asset": "trelix-linux-arm64",
        "file_markers": ("ELF",),
        "file_markers_any": ("aarch64", "ARM aarch64", "arm64"),
        "host_match": lambda system, machine: system == "Linux" and machine in ("aarch64", "arm64"),
        "host_desc": "Linux aarch64",
    },
    {
        "asset": "trelix-windows-x64.exe",
        "file_markers": ("PE32",),
        "host_match": lambda system, machine: (
            system == "Windows" and machine in ("AMD64", "x86_64")
        ),
        "host_desc": "Windows x86_64",
    },
)

# Filenames that must never appear inside a published wheel. Matched by exact path
# COMPONENT (".git", never ".gitignore" or ".github") or by substring on the basename
# (never on the full path, so a source file that happens to live under a directory
# named e.g. "secrets_test/" is not itself flagged).
_ENV_BASENAME_RE = re.compile(r"^\.env(\..+)?$")
_SECRET_SUBSTRINGS = ("credential", "secret")


def _run(
    cmd: list[str], *, cwd: Path | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """subprocess.run with this script's uniform defaults. Callers still need to catch
    subprocess.TimeoutExpired and OSError themselves -- this helper does not swallow
    either, only fixes the capture/decode/check defaults so every call site agrees."""
    return subprocess.run(
        cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True, check=False
    )


def _truncate(text: str, limit: int = 2000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:] + f"\n...(truncated to the last {limit} chars)"


# ---------------------------------------------------------------------------
# 1. PyPI installs
# ---------------------------------------------------------------------------
def check_pypi_installs(version: str) -> list[str]:
    """Fresh venv, `pip install <pkg>==<version>`, real smoke command, assert version.

    ALWAYS RUN, matching how both manual production-verification passes always
    covered installs regardless of what else they had time for.
    """
    failures: list[str] = []
    for dist_name, import_name in _PACKAGES:
        tmp_root: Path | None = None
        try:
            tmp_root = Path(tempfile.mkdtemp(prefix=f"trelix-verify-pypi-{dist_name}-"))
            venv_dir = tmp_root / "venv"
            # symlinks=True, NOT venv.create()'s default of False -- see
            # tests/e2e/test_pypi_dist_install_e2e.py's own comment on this exact call:
            # a copied interpreter binary from a framework-style Python build can fail
            # to resolve its own @rpath/libpythonX.Y.dylib when copied rather than
            # symlinked into a new venv, crashing ensurepip with SIGABRT.
            venv.create(venv_dir, with_pip=True, symlinks=True)
            venv_python = venv_dir / "bin" / "python"
            if not venv_python.exists():
                failures.append(f"{dist_name}: venv creation did not produce {venv_python}")
                continue

            print(f"  installing {dist_name}=={version} into a fresh venv ...")
            install = _run(
                [str(venv_python), "-m", "pip", "install", "--quiet", f"{dist_name}=={version}"],
                timeout=300,
            )
            if install.returncode != 0:
                failures.append(
                    f"{dist_name}: `pip install {dist_name}=={version}` failed "
                    f"(exit {install.returncode}): {_truncate(install.stderr)}"
                )
                continue

            if dist_name == "trelix":
                # The real console script, invoked the way a user invokes it -- not
                # an in-process import, which could never have caught trelix-mcp's
                # own console-script defect if it had been tested that way instead.
                smoke = _run([str(venv_dir / "bin" / "trelix"), "--version"], timeout=30)
                expected = f"trelix {version}"
                match = expected in smoke.stdout.strip()
            else:
                smoke_code = f"import {import_name}; print({import_name}.__version__)"
                smoke = _run([str(venv_python), "-c", smoke_code], timeout=30)
                expected = version
                match = smoke.stdout.strip() == expected

            if smoke.returncode != 0:
                failures.append(
                    f"{dist_name}: smoke command failed (exit {smoke.returncode}): "
                    f"{_truncate(smoke.stderr)}"
                )
                continue
            if not match:
                failures.append(
                    f"{dist_name}: smoke command printed {smoke.stdout.strip()!r}, "
                    f"expected {expected!r}"
                )
        except subprocess.TimeoutExpired as exc:
            failures.append(f"{dist_name}: timed out running {exc.cmd!r}")
        except OSError as exc:
            failures.append(f"{dist_name}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # accumulate -- one package's bug must not abort the rest
            failures.append(f"{dist_name}: unexpected error: {type(exc).__name__}: {exc}")
        finally:
            if tmp_root is not None:
                shutil.rmtree(tmp_root, ignore_errors=True)
    return failures


# ---------------------------------------------------------------------------
# 2. Docker images
# ---------------------------------------------------------------------------
def check_docker_images(version: str) -> list[str]:
    """Pull both published tags, run the real entrypoints inside each.

    `docker run --rm --entrypoint trelix-mcp <img> --version` is the EXACT command
    that returned exit 127 on the published 3.2.1 image -- trelix-mcp was not copied
    into the image at all. This re-runs it, and the trelix one beside it, on every tag.
    """
    failures: list[str] = []
    for suffix in ("", "-local"):
        image = f"{GHCR_IMAGE}:{version}{suffix}"
        try:
            print(f"  docker pull {image} ...")
            pull = _run(["docker", "pull", image], timeout=600)
            if pull.returncode != 0:
                failures.append(
                    f"{image}: `docker pull` failed (exit {pull.returncode}): "
                    f"{_truncate(pull.stderr)}"
                )
                continue

            smoke_cases = (("trelix", f"trelix {version}"), ("trelix-mcp", f"trelix-mcp {version}"))
            for binary, expected in smoke_cases:
                cmd = ["docker", "run", "--rm", "--entrypoint", binary, image, "--version"]
                run = _run(cmd, timeout=60)
                if run.returncode != 0:
                    failures.append(
                        f"{image}: `docker run --rm --entrypoint {binary} {image} --version` "
                        f"failed (exit {run.returncode}) -- this exact command returned exit "
                        f"127 on the published 3.2.1 image before trelix-mcp was added to the "
                        f"Dockerfile. stderr: {_truncate(run.stderr, 500)}"
                    )
                    continue
                got = run.stdout.strip()
                if expected not in got:
                    failures.append(
                        f"{image}: `--entrypoint {binary} --version` printed {got!r}, "
                        f"expected it to contain {expected!r}"
                    )
        except subprocess.TimeoutExpired as exc:
            failures.append(f"{image}: timed out running {exc.cmd!r}")
        except OSError as exc:
            failures.append(f"{image}: could not run docker ({type(exc).__name__}: {exc})")
        except Exception as exc:  # accumulate -- one tag's bug must not hide the other tag's
            failures.append(f"{image}: unexpected error: {type(exc).__name__}: {exc}")
    return failures


# ---------------------------------------------------------------------------
# 3. Helm chart
# ---------------------------------------------------------------------------
def _read_chart_app_version(chart_yaml: Path) -> str | None:
    match = re.search(r'^appVersion:\s*"?([^"\s]+)"?', chart_yaml.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def _read_rendered_image_tag(rendered: str) -> str | None:
    match = re.search(rf'image:\s*"?{re.escape(GHCR_IMAGE)}:([^"\s]+)"?', rendered)
    return match.group(1) if match else None


def check_helm_chart(version: str) -> list[str]:
    """`helm lint` + `helm template` the chart AT THE RELEASE TAG, on every backend.

    Checks out `v<version>` into a throwaway `git worktree` (never the caller's own
    working tree) so this always verifies what was actually tagged, not whatever is
    currently checked out. Reuses helm-lint.yml's own `--set` keys (confirmed against
    helm/trelix/values.yaml rather than guessed) and its own three-backend matrix, and
    adds the appVersion-vs-rendered-tag assertion helm-lint.yml also gates on.
    """
    failures: list[str] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="trelix-verify-helm-"))
    worktree = tmp_root / "worktree"
    tag = f"v{version}"
    worktree_added = False
    try:
        added = _run(["git", "worktree", "add", str(worktree), tag], cwd=REPO_ROOT, timeout=60)
        if added.returncode != 0:
            # The tag may not be fetched locally yet -- try once more after fetching it.
            refspec = f"refs/tags/{tag}:refs/tags/{tag}"
            _run(["git", "fetch", "origin", refspec], cwd=REPO_ROOT, timeout=60)
            added = _run(["git", "worktree", "add", str(worktree), tag], cwd=REPO_ROOT, timeout=60)
        if added.returncode != 0:
            failures.append(
                f"helm: `git worktree add` for tag {tag!r} failed even after fetching it "
                f"(exit {added.returncode}): {_truncate(added.stderr)}"
            )
            return failures
        worktree_added = True

        chart_dir = worktree / "helm" / "trelix"
        chart_yaml = chart_dir / "Chart.yaml"
        if not chart_yaml.exists():
            failures.append(f"helm: {chart_yaml} does not exist at tag {tag!r}")
            return failures

        app_version = _read_chart_app_version(chart_yaml)
        if app_version is None:
            failures.append(f"helm: could not find an `appVersion:` line in {chart_yaml}")
            return failures
        if app_version != version:
            failures.append(
                f"helm: Chart.yaml's appVersion is {app_version!r} at tag {tag!r}, "
                f"expected {version!r}"
            )

        print(f"  helm lint {chart_dir} ...")
        lint = _run(["helm", "lint", str(chart_dir)], timeout=60)
        if lint.returncode != 0:
            failures.append(
                f"helm lint: failed (exit {lint.returncode}): "
                f"{_truncate(lint.stdout + lint.stderr)}"
            )

        for backend in _HELM_BACKENDS:
            print(f"  helm template (backend={backend}) ...")
            template = _run(
                [
                    "helm",
                    "template",
                    "test",
                    str(chart_dir),
                    "--set",
                    f"store.backend={backend}",
                    "--set",
                    "store.qdrant.apiKey=verify-release-placeholder",
                ],
                timeout=60,
            )
            if template.returncode != 0:
                failures.append(
                    f"helm template (backend={backend}): failed (exit {template.returncode}): "
                    f"{_truncate(template.stdout + template.stderr)}"
                )
                continue
            rendered_tag = _read_rendered_image_tag(template.stdout)
            if rendered_tag is None:
                failures.append(
                    f"helm template (backend={backend}): no `image: {GHCR_IMAGE}:...` line "
                    "found in the rendered manifest"
                )
                continue
            if rendered_tag != app_version:
                failures.append(
                    f"helm template (backend={backend}): rendered image tag {rendered_tag!r} "
                    f"does not match Chart.yaml's appVersion {app_version!r}"
                )
    except subprocess.TimeoutExpired as exc:
        failures.append(f"helm: timed out running {exc.cmd!r}")
    except OSError as exc:
        failures.append(f"helm: {type(exc).__name__}: {exc}")
    except Exception as exc:  # never let an unexpected bug hide the other checks' results
        failures.append(f"helm: unexpected error: {type(exc).__name__}: {exc}")
    finally:
        # Clean up even when an assertion above failed -- a failed check must never
        # leak the worktree.
        if worktree_added:
            _run(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO_ROOT, timeout=30)
            _run(["git", "worktree", "prune"], cwd=REPO_ROOT, timeout=30)
        shutil.rmtree(tmp_root, ignore_errors=True)
    return failures


# ---------------------------------------------------------------------------
# 4. GitHub release binaries
# ---------------------------------------------------------------------------
def check_github_release_binaries(version: str) -> list[str]:
    """Download every release binary, confirm format/arch, and RUN the one host match.

    The other three binaries can only ever have their `file(1)` output checked on this
    host -- actually running an ARM64 Linux binary on a macOS arm64 host, for example,
    is not possible. That limitation is reported explicitly (via `print`, as evidence),
    never silently skipped.
    """
    failures: list[str] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="trelix-verify-binaries-"))
    try:
        tag = f"v{version}"
        print(f"  gh release download {tag} --repo {GH_REPO} ...")
        gh_cmd = [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            GH_REPO,
            "--dir",
            str(tmp_root),
            "--clobber",
        ]
        download = _run(gh_cmd, timeout=300)
        if download.returncode != 0:
            failures.append(
                f"binaries: `gh release download {tag}` failed (exit {download.returncode}): "
                f"{_truncate(download.stderr)}"
            )
            return failures

        system, machine = platform.system(), platform.machine()
        host_binary_checked = False
        for spec in _RELEASE_BINARIES:
            asset_path = tmp_root / spec["asset"]
            if not asset_path.exists():
                failures.append(f"binaries: expected asset {spec['asset']!r} was not downloaded")
                continue

            file_result = _run(["file", str(asset_path)], timeout=30)
            if file_result.returncode != 0:
                failures.append(
                    f"binaries: `file {spec['asset']}` failed (exit {file_result.returncode}): "
                    f"{_truncate(file_result.stderr)}"
                )
                continue
            description = file_result.stdout.strip()
            missing_markers = [m for m in spec["file_markers"] if m not in description]
            any_markers = spec.get("file_markers_any")
            if any_markers and not any(m in description for m in any_markers):
                missing_markers.append(f"one of {any_markers!r}")
            if missing_markers:
                failures.append(
                    f"binaries: {spec['asset']} does not look like {spec['host_desc']} -- "
                    f"`file` said {description!r}, missing {missing_markers!r}"
                )
                continue

            if spec["host_match"](system, machine):
                host_binary_checked = True
                asset_path.chmod(0o755)
                try:
                    version_run = _run([str(asset_path), "--version"], timeout=30)
                except OSError as exc:
                    failures.append(
                        f"binaries: could not execute {spec['asset']} on this host "
                        f"({system}/{machine}): {type(exc).__name__}: {exc}"
                    )
                    continue
                if version_run.returncode != 0:
                    failures.append(
                        f"binaries: {spec['asset']} --version failed (exit "
                        f"{version_run.returncode}): {_truncate(version_run.stderr)}"
                    )
                    continue
                if version not in version_run.stdout:
                    failures.append(
                        f"binaries: {spec['asset']} --version printed "
                        f"{version_run.stdout.strip()!r}, expected it to contain {version!r}"
                    )
            else:
                print(
                    f"  note: {spec['asset']} passed its file-format check but was NOT "
                    f"executed -- this host is {system}/{machine}, that asset targets "
                    f"{spec['host_desc']}"
                )
        if not host_binary_checked:
            failures.append(
                f"binaries: this host ({system}/{machine}) matched none of the four release "
                "binaries, so no binary was actually executed anywhere in this run"
            )
    except subprocess.TimeoutExpired as exc:
        failures.append(f"binaries: timed out running {exc.cmd!r}")
    except OSError as exc:
        failures.append(f"binaries: {type(exc).__name__}: {exc}")
    except Exception as exc:  # never let an unexpected bug hide the other checks' results
        failures.append(f"binaries: unexpected error: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return failures


# ---------------------------------------------------------------------------
# 5. Security audit
# ---------------------------------------------------------------------------
def _is_secret_shaped(archive_member: str) -> bool:
    """True for a wheel-member path that should never ship: .env*, a `.git` path
    component (never `.gitignore`/`.github`, which are ordinary and harmless), or a
    basename containing "credential" or "secret"."""
    normalized = archive_member.rstrip("/")
    basename = normalized.rsplit("/", 1)[-1]
    parts = normalized.split("/")
    if _ENV_BASENAME_RE.match(basename):
        return True
    if ".git" in parts:
        return True
    lower_basename = basename.lower()
    return any(needle in lower_basename for needle in _SECRET_SUBSTRINGS)


def _audit_wheel_contents(
    venv_python: Path, dist_name: str, version: str, out_dir: Path
) -> list[str]:
    failures: list[str] = []
    download = _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--dest",
            str(out_dir),
            f"{dist_name}=={version}",
        ],
        timeout=300,
    )
    if download.returncode != 0:
        failures.append(
            f"{dist_name}: `pip download --no-deps` failed (exit {download.returncode}): "
            f"{_truncate(download.stderr)}"
        )
        return failures

    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        failures.append(f"{dist_name}: `pip download --no-deps` produced no .whl (only sdist?)")
        return failures
    if len(wheels) > 1:
        failures.append(f"{dist_name}: expected exactly one wheel, got {[w.name for w in wheels]}")

    for wheel in wheels:
        try:
            with zipfile.ZipFile(wheel) as archive:
                offenders = [name for name in archive.namelist() if _is_secret_shaped(name)]
        except zipfile.BadZipFile as exc:
            failures.append(f"{dist_name}: {wheel.name} is not a valid zip: {exc}")
            continue
        if offenders:
            failures.append(
                f"{dist_name}: {wheel.name} contains .env/.git/credential/secret-shaped "
                f"path(s): {offenders}"
            )
    return failures


def check_security_audit(version: str) -> list[str]:
    """`pip-audit` every published package, plus a per-wheel secret-shaped-file scan.

    Every finding is reported as its own failure string, unfiltered by severity --
    this script does not have the context to judge severity, a human reading the
    output does.
    """
    failures: list[str] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="trelix-verify-security-"))
    try:
        venv_dir = tmp_root / "venv"
        venv.create(venv_dir, with_pip=True, symlinks=True)
        venv_python = venv_dir / "bin" / "python"
        if not venv_python.exists():
            failures.append(f"security: venv creation did not produce {venv_python}")
            return failures

        # ensurepip bundles whatever pip shipped with this Python build, which lags
        # behind PyPI's latest release -- and pip itself gets CVEs. Without this, every
        # run of this script reports the SAME finding regardless of trelix's own
        # dependency tree ("pip <bundled version>: <pip's own latest CVE>"), which is
        # noise this check exists to cut through, not add to. Upgrading first keeps the
        # audit scoped to what this release actually ships.
        pip_upgrade = _run(
            [str(venv_python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], timeout=60
        )
        if pip_upgrade.returncode != 0:
            failures.append(
                f"security: `pip install --upgrade pip` failed (exit {pip_upgrade.returncode}): "
                f"{_truncate(pip_upgrade.stderr)}"
            )

        for dist_name, _import_name in _PACKAGES:
            print(f"  installing {dist_name}=={version} for the security audit ...")
            install = _run(
                [str(venv_python), "-m", "pip", "install", "--quiet", f"{dist_name}=={version}"],
                timeout=300,
            )
            if install.returncode != 0:
                failures.append(
                    f"security: `pip install {dist_name}=={version}` failed "
                    f"(exit {install.returncode}): {_truncate(install.stderr)}"
                )

        print("  installing pip-audit ...")
        audit_install = _run(
            [str(venv_python), "-m", "pip", "install", "--quiet", "pip-audit"], timeout=180
        )
        if audit_install.returncode != 0:
            failures.append(
                f"security: `pip install pip-audit` failed (exit {audit_install.returncode}): "
                f"{_truncate(audit_install.stderr)}"
            )
        else:
            pip_audit_bin = venv_dir / "bin" / "pip-audit"
            print("  running pip-audit ...")
            audit = _run([str(pip_audit_bin), "--format", "json"], timeout=300)
            try:
                report = json.loads(audit.stdout)
            except json.JSONDecodeError:
                failures.append(
                    f"security: pip-audit did not produce valid JSON (exit {audit.returncode}): "
                    f"{_truncate(audit.stdout + audit.stderr)}"
                )
            else:
                for dependency in report.get("dependencies", []):
                    name = dependency.get("name", "<unknown>")
                    dep_version = dependency.get("version", "<unknown>")
                    for vuln in dependency.get("vulns", []):
                        failures.append(
                            f"security: {name} {dep_version}: {vuln.get('id', '<no id>')} "
                            f"(fix versions: {vuln.get('fix_versions', [])}) -- "
                            f"{_truncate(vuln.get('description', ''), 300)}"
                        )

        for dist_name, _import_name in _PACKAGES:
            print(f"  scanning {dist_name}'s wheel contents for secret-shaped files ...")
            wheel_dir = tmp_root / f"wheel-{dist_name}"
            wheel_dir.mkdir(exist_ok=True)
            failures.extend(_audit_wheel_contents(venv_python, dist_name, version, wheel_dir))
    except subprocess.TimeoutExpired as exc:
        failures.append(f"security: timed out running {exc.cmd!r}")
    except OSError as exc:
        failures.append(f"security: {type(exc).__name__}: {exc}")
    except Exception as exc:  # never let an unexpected bug hide the other checks' results
        failures.append(f"security: unexpected error: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", required=True, help="the released version to verify, e.g. 3.2.2"
    )
    parser.add_argument("--skip-docker", action="store_true", help="skip the Docker image checks")
    parser.add_argument("--skip-helm", action="store_true", help="skip the Helm chart checks")
    parser.add_argument(
        "--skip-binaries", action="store_true", help="skip the GitHub release binary checks"
    )
    parser.add_argument(
        "--skip-security", action="store_true", help="skip the security audit checks"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    version = args.version

    print(f"=== trelix release verification: v{version} ===")

    # PyPI installs always run, matching how both manual production-verification
    # passes always covered installs regardless of what else they had time for.
    categories: list[tuple[str, bool, list[str]]] = []

    print("\n--- 1. PyPI installs ---")
    categories.append(("PyPI installs", True, check_pypi_installs(version)))

    print("\n--- 2. Docker images ---")
    if args.skip_docker:
        print("  skipped (--skip-docker)")
        categories.append(("Docker images", False, []))
    else:
        categories.append(("Docker images", True, check_docker_images(version)))

    print("\n--- 3. Helm chart ---")
    if args.skip_helm:
        print("  skipped (--skip-helm)")
        categories.append(("Helm chart", False, []))
    else:
        categories.append(("Helm chart", True, check_helm_chart(version)))

    print("\n--- 4. GitHub release binaries ---")
    if args.skip_binaries:
        print("  skipped (--skip-binaries)")
        categories.append(("GitHub release binaries", False, []))
    else:
        categories.append(("GitHub release binaries", True, check_github_release_binaries(version)))

    print("\n--- 5. Security audit ---")
    if args.skip_security:
        print("  skipped (--skip-security)")
        categories.append(("Security audit", False, []))
    else:
        categories.append(("Security audit", True, check_security_audit(version)))

    print(f"\n=== SUMMARY: v{version} ===")
    any_failed = False
    for name, ran, failures in categories:
        if not ran:
            print(f"[SKIP] {name}")
            continue
        if failures:
            any_failed = True
            print(f"[FAIL] {name} ({len(failures)} issue(s))")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print(f"[PASS] {name}")

    if any_failed:
        print("\nFAIL: at least one category reported a failure. See details above.")
        return 1
    print("\nPASS: every check that ran passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
