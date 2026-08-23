"""``sys.modules["pyvis"]`` must not stay mocked after the test that mocked it.

WHAT THIS CLOSES. tests/unit/test_graph_visualizer.py injected MagicMocks with
``sys.modules.setdefault("pyvis", ...)`` / ``setdefault("pyvis.network", ...)``
and never removed them. ``sys.modules`` is process-global, so every later test in
that process reaching ``from pyvis.network import Network`` -- which
src/trelix/graph/visualizer.py does lazily inside ``export_html`` -- got the mock.

WHY THE SUITE CANNOT SEE IT. tests/unit/test_graph_visualizer_escaping.py calls
``pytest.importorskip("pyvis.network")`` at MODULE level, i.e. during COLLECTION,
before any test runs. In any run that collects that module the real package is
already in sys.modules and ``setdefault`` is a no-op, so nothing leaks. Measured:
the leak is invisible to the full serial run (3969 passed) and to ``-n 4 --dist
load`` (3969 passed), since xdist workers each collect the whole directory. It is
armed only by a run that does NOT collect the escaping module -- a single-file
run, a ``-k`` selection, an explicit node-id command line, or a CI leg that shards
tests/unit by file. Hence the child pytest below: in the PARENT the assertion
would be true by construction.

Round-4 recorded "pyvis is NOT installed here". It is: pyvis 0.3.2 is in the
shared venv, and the leak reproduces.

WHY A NEGATIVE CONTROL. The detector passes trivially in any process where nobody
mocked pyvis, so alone it is green-when-vacuous. The control leaks a mock in a
child on purpose and requires the detector to FAIL there, so a detector that stops
discriminating turns that control red instead of silently becoming a no-op.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Both keys are watched: `import pyvis` alone does not bind the `network`
# submodule, so both were installed and both must be restored.
_WATCHED = ("pyvis", "pyvis.network")

# Arms the negative control. Without it the injector SKIPS, so the ordinary suite
# never gets a deliberately poisoned sys.modules. NOT under the TRELIX_ prefix:
# tests/_env_isolation.py scrubs that whole namespace out of the child, which is
# how the first version of this control silently failed to arm.
_LEAK_CONTROL_ENV = "PYTEST_PYVIS_LEAK_CONTROL"

# Literals, not derived from the modules involved: a derived id would keep
# agreeing with the code through a rename, which is when this probe must go red.
_DETECTOR = (
    "tests/unit/test_sys_modules_pyvis_leak.py"
    "::test_no_mock_object_is_installed_under_pyvis_in_sys_modules"
)
_LEAKER = (
    "tests/unit/test_sys_modules_pyvis_leak.py::test_negative_control_leaks_a_pyvis_mock_on_purpose"
)
_VISUALIZER = (
    "tests/unit/test_graph_visualizer.py::TestGraphVisualizer::test_export_html_creates_file"
)
_API_GRAPH = (
    "tests/unit/test_api_graph.py::TestGraphVisualizeContainment"
    "::test_default_output_path_is_accepted"
)


def _mocked_watched_names() -> list[str]:
    """Which of ``_WATCHED`` hold a ``unittest.mock`` object.

    Type-based, not identity-based: the injected mocks live in another process.
    """
    return [n for n in _WATCHED if type(sys.modules.get(n)).__module__ == "unittest.mock"]


def test_no_mock_object_is_installed_under_pyvis_in_sys_modules() -> None:
    """The detector. Not a standalone claim -- see the module docstring.

    Made to fail by reverting ``_install_pyvis_stub`` in
    tests/unit/test_graph_visualizer.py to the unrestored
    ``sys.modules.setdefault(...)`` form and running it in this process without
    collecting tests/unit/test_graph_visualizer_escaping.py.
    """
    leaked = _mocked_watched_names()
    assert leaked == [], (
        f"a unittest.mock object is installed in sys.modules under {leaked}. "
        "Whichever test put it there did not restore it, so every later test in "
        "this process that imports pyvis gets the mock."
    )


def test_negative_control_leaks_a_pyvis_mock_on_purpose() -> None:
    """Arms the detector so the assertion above is provably not vacuous.

    Skips loudly unless the driver set the env flag: an unrestored mock is exactly
    what this file forbids, so it must never happen in the ordinary suite.
    """
    if os.environ.get(_LEAK_CONTROL_ENV) != "1":
        pytest.skip(
            "negative control for test_no_mock_object_is_installed_under_pyvis_in_sys_modules; "
            f"runs only in the child pytest that sets {_LEAK_CONTROL_ENV}=1"
        )
    # Deliberately raw and deliberately unrestored -- this IS the defect shape.
    sys.modules["pyvis"] = MagicMock()
    sys.modules["pyvis.network"] = MagicMock()


def _nested_pytest(
    node_ids: list[str], *, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run *node_ids*, in this order, in a fresh pytest from the repo root.

    ``-p no:randomly`` because the parent may have pytest-randomly active, which
    auto-loads in the child and would shuffle the very order under test.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *node_ids,
            "-q",
            "--no-header",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})},
        timeout=300,
        check=False,
    )


class TestThePyvisStubDoesNotOutliveItsTest:
    def test_negative_control_the_detector_fails_when_a_mock_is_left_behind(self) -> None:
        """PRECONDITION for the two tests below, not a claim of its own.

        "2 passed" here means the detector stopped discriminating (watched keys
        renamed, mock type check broken) and the assertions below are green for
        no reason.
        """
        proc = _nested_pytest([_LEAKER, _DETECTOR], extra_env={_LEAK_CONTROL_ENV: "1"})
        assert proc.returncode != 0, (
            "the detector PASSED in a child where a MagicMock was deliberately left "
            f"in sys.modules['pyvis'] -- it no longer discriminates:\n{proc.stdout}"
        )
        assert "1 failed, 1 passed" in proc.stdout, (
            f"expected exactly the detector to fail:\n{proc.stdout}\n{proc.stderr}"
        )
        assert _DETECTOR in proc.stdout, (
            f"the failure was not the detector:\n{proc.stdout}\n{proc.stderr}"
        )

    def test_the_graph_visualizer_test_leaves_no_pyvis_mock_behind(self) -> None:
        """Reverting ``_install_pyvis_stub`` to ``sys.modules.setdefault(...)``
        without restoration is the mutation that must make this fail. Measured with
        that revert: "1 failed, 1 passed"."""
        proc = _nested_pytest([_VISUALIZER, _DETECTOR])
        assert proc.returncode == 0, (
            "tests/unit/test_graph_visualizer.py left a pyvis mock in sys.modules "
            f"for the rest of the process:\n{proc.stdout}\n{proc.stderr}"
        )
        assert "2 passed" in proc.stdout, f"expected both ids to pass:\n{proc.stdout}"

    def test_the_api_graph_test_leaves_no_pyvis_mock_behind(self) -> None:
        """Same for the second caller of the helper, as a SEPARATE node id.

        tests/unit/test_api_graph.py imports the helper from
        tests/unit/test_graph_visualizer.py, so it is an independent escape route;
        measuring it separately keeps one caller's kill from being credited to the
        other.
        """
        proc = _nested_pytest([_API_GRAPH, _DETECTOR])
        assert proc.returncode == 0, (
            "tests/unit/test_api_graph.py left a pyvis mock in sys.modules "
            f"for the rest of the process:\n{proc.stdout}\n{proc.stderr}"
        )
        assert "2 passed" in proc.stdout, f"expected both ids to pass:\n{proc.stdout}"
