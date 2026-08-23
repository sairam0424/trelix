"""
Regression guard for the console-script resolution in ``test_cli.py``.

THE BUG, reproduced before it was fixed
---------------------------------------
``_resolve_trelix_bin()`` runs at MODULE scope (``TRELIX_BIN =
_resolve_trelix_bin()``), so when it raises, ``tests/integration/test_cli.py``
cannot be COLLECTED -- not merely cannot pass. Measured on this tree with the
venv's ``bin`` removed from ``PATH``:

    $ PATH=/usr/bin:/bin:/usr/sbin:/sbin .venv/bin/python -m pytest \\
          tests/integration --collect-only -q
    ERROR collecting tests/integration/test_cli.py
    E   RuntimeError: `trelix` console script not found at
        <checkout>/.venv/bin/trelix or on PATH.
    !!!! Interrupted: 1 error during collection !!!!
    74 tests collected, 1 error

    ... and after the fix, same command:
    89 tests collected in 0.31s

Both of the original probes were AMBIENT and failed together: ``_VENV`` is
``__file__``-relative (so it points at a ``git worktree`` that has no ``.venv``)
and ``shutil.which`` reads ``PATH`` (so it needs the venv activated, which
``.venv/bin/python -m pytest`` does not require). The fix adds a probe derived
from the RUNNING PROCESS -- ``Path(sys.executable).parent`` -- and keeps both
ambient probes as fallbacks.

WHY THIS TEST IS NOT CIRCULAR. It does not ask "does the resolver find the
script here" -- importing ``test_cli`` at all already proves that, which is
precisely the kind of proxy that would pass whether or not the fix is present
(on this machine the venv IS on PATH under a normal run, so the OLD resolver
also succeeds). It reconstructs the FAILING CONDITION instead: it strips ``PATH``
and points the repo-relative probe at a directory that does not exist, so BOTH
ambient probes are guaranteed to miss, and then asserts the resolver still
returns a real executable. Under the old code that scenario raises.

SCOPE: tests-only. ``src/`` is untouched; the console script was always installed
and correct, and the defect was this file's sibling being unable to find it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration import test_cli as cli_module


def test_resolver_survives_no_path_and_no_repo_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolver finds the script with BOTH ambient probes made to fail.

    Reconstructs the reported condition rather than trusting the happy path:

    * ``PATH`` is set to ``""`` so ``shutil.which("trelix")`` cannot succeed.
    * ``cli_module._VENV`` is repointed at a guaranteed-absent directory so the
      ``<repo>/.venv/bin/trelix`` probe cannot succeed. This is the ``git
      worktree`` case in miniature.

    What remains is the probe this fix added, and it must carry the whole load.

    PRECONDITION, naming the fixtures that could make this vacuous
    (``monkeypatch.setenv`` and ``monkeypatch.setattr``): both sabotages are
    verified to have TAKEN before the assertion runs. Without that, a
    monkeypatch that silently failed would leave the ambient probes working and
    this test would pass while proving nothing about the new one. It skips
    LOUDLY rather than passing if the interpreter's bin dir genuinely has no
    console script -- a layout where the package is importable but its script
    lives only on ``PATH`` (pipx, a wrapper shim) is legitimate, and in that
    layout this particular probe cannot be the one that works.

    MUTATION that must make this fail: delete the
    ``interpreter_bin / "trelix"`` loop from ``_resolve_trelix_bin`` in
    ``tests/integration/test_cli.py`` -- i.e. restore the pre-fix resolver.
    """
    interpreter_bin = Path(sys.executable).parent
    if (
        not (interpreter_bin / "trelix").is_file()
        and not (interpreter_bin / "trelix.exe").is_file()
    ):
        pytest.skip(
            f"no trelix console script beside the running interpreter "
            f"({interpreter_bin}); the probe this test exists to pin cannot be "
            f"exercised in this layout, and passing here would certify nothing"
        )

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(cli_module, "_VENV", Path("/nonexistent-venv-for-this-test"))

    # Preconditions: both sabotages actually took effect.
    assert shutil.which("trelix") is None, "PATH sabotage did not take; which() still resolves"
    assert not (cli_module._VENV / "bin" / "trelix").exists(), (
        "the _VENV sabotage did not take; the repo-relative probe can still succeed"
    )

    resolved = cli_module._resolve_trelix_bin()

    assert Path(resolved).is_file(), (
        f"resolver returned {resolved!r}, which is not a file. With PATH empty "
        f"and the repo-relative venv absent, only the sys.executable-relative "
        f"probe can succeed -- so this is that probe failing."
    )
    assert Path(resolved).parent == interpreter_bin, (
        f"resolver returned {resolved!r}; expected something inside the running "
        f"interpreter's bin dir ({interpreter_bin})"
    )


def test_resolver_still_raises_when_the_script_is_genuinely_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fix must not turn a real miss into a silent pass.

    The original docstring is explicit that raising beats skipping -- "a silent
    module-level skip would let this file report all green while executing
    nothing". Widening the search must not erode that, so this pins the opposite
    direction: with every probe pointed somewhere empty, a ``RuntimeError`` is
    still what comes out.

    ``sys.executable`` is repointed at a real-but-script-free directory rather
    than a nonexistent one, so the failure is "the script is not there" and not
    "the path is malformed" -- the former is the case operators actually hit.

    MUTATION that must make this fail: replace the ``raise RuntimeError`` at the
    end of ``_resolve_trelix_bin`` with ``return "trelix"``, or with
    ``pytest.skip(...)``.
    """
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    fake_python = empty_bin / "python"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(sys, "executable", str(fake_python))
    monkeypatch.setattr(cli_module, "_VENV", Path("/nonexistent-venv-for-this-test"))

    # Precondition: the directory really is script-free, so the expected raise is
    # caused by absence rather than by one of the sabotages misfiring.
    assert not (empty_bin / "trelix").exists()
    assert shutil.which("trelix") is None

    with pytest.raises(RuntimeError, match="console script not found"):
        cli_module._resolve_trelix_bin()


def test_integration_suite_collects_without_the_venv_on_path() -> None:
    """END-TO-END: the reported symptom, asserted as the symptom.

    The two tests above pin the resolver in-process. This one pins the thing the
    report was actually about -- ``pytest tests/integration --collect-only``
    succeeding in a shell where the venv's ``bin`` is not on ``PATH``. A resolver
    that is correct in isolation but is defeated by something else at import time
    (a second ambient lookup added later, say) passes those and fails here.

    Uses ``sys.executable`` so the child is this same interpreter, and a sanitized
    ``PATH`` that deliberately excludes the venv's ``bin``. ``-p no:cacheprovider``
    keeps the child from writing a ``.pytest_cache`` into the checkout.

    Asserts on the RETURN CODE and on the absence of the collection error, not on
    a test count: a count would have to be updated every time a test is added to
    this directory, which turns an unrelated addition into a failure here.

    MUTATION that must make this fail: restore the pre-fix ``_resolve_trelix_bin``
    (drop the ``sys.executable``-relative probe).
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(Path(__file__).parent.parent.parent),
        capture_output=True,
        text=True,
        timeout=300,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            # Keep the child hermetic the same way tests/unit/conftest.py does,
            # so this probe cannot become the one thing in the suite that reaches
            # the Hub or republishes the operator's dotenv.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LITELLM_MODE": "PRODUCTION",
        },
    )
    combined = proc.stdout + proc.stderr
    assert "console script not found" not in combined, (
        "collection still fails on console-script resolution:\n" + combined[-4000:]
    )
    assert "error during collection" not in combined, "collection errored:\n" + combined[-4000:]
    assert proc.returncode == 0, (
        f"`pytest tests/integration --collect-only` exited {proc.returncode} with "
        f"the venv bin off PATH:\n{combined[-4000:]}"
    )
