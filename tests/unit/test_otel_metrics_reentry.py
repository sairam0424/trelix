"""Regression test for the ONE order-dependent failure in tests/unit.

WHAT BROKE
----------
``tests/unit/test_otel_metrics.py`` gets its counters from an
``InMemoryMetricReader`` wired into an SDK ``MeterProvider`` that a fixture
installs with ``opentelemetry.metrics.set_meter_provider()``. That global is a
ONE-SHOT process resource: the second call anywhere in the process is refused
and only logs ``"Overriding of current MeterProvider is not allowed"``.

The fixture was ``scope="module"``. A module-scoped fixture is finalized when
pytest LEAVES the module and RE-CREATED when pytest COMES BACK, so any ordering
that splits that file's tests apart produced a second ``set_meter_provider()``
whose brand-new reader was attached to nothing: ``get_metrics_data()`` returned
``None`` and three tests failed with ``assert None == 1`` / ``assert None == 2``.

WHY NO EXISTING RUN CAUGHT IT
-----------------------------
pytest-randomly's shuffle is hierarchical — it permutes module blocks but keeps
each module contiguous (measured: contiguous-module-blocks == distinct-modules
== 183 at three seeds), so under any seed the fixture is still created exactly
once. Only a splitting order reaches it. Splitting orders are real: an explicit
multi-id command line does it, and so does ``pytest-xdist``, which hands
individual tests to workers.

The sibling ``tests/unit/test_otel_tracing.py::_test_tracer_provider`` was
already hardened against this class of bug — it ATTACHES a fresh SpanProcessor
to whichever provider is incumbent instead of installing its own. The metrics
API has no ``add_metric_reader()``, so the same shape was unavailable and the
metrics fixture never got the equivalent treatment. That asymmetry between two
otherwise-parallel modules is the whole defect.

HOW THIS TEST WORKS
-------------------
It shells out to a nested pytest with a three-id command line that forces the
leave-and-return, and requires exit status 0. It asserts the OUTCOME (the module
survives re-entry), not a proxy such as reading the fixture's ``scope`` string —
a scope string can be right while the fixture is broken for other reasons, and
could be spelled differently while still being session-wide.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Node ids written out as literals, deliberately NOT imported or derived from
# test_otel_metrics.py: deriving them would make this test agree with that file
# by construction even after someone renames a test out from under it.
_OTEL_FIRST = (
    "tests/unit/test_otel_metrics.py::TestEnabledRecordsCounters"
    "::test_openai_embed_counts_one_request_per_api_call"
)
_OTEL_SECOND = (
    "tests/unit/test_otel_metrics.py::TestEnabledRecordsCounters"
    "::test_no_token_series_when_provider_reports_no_usage"
)
# Any test from a DIFFERENT module. This one is a pure table assertion over the
# extension map: it touches no OTel state at all, so its only role is to make
# pytest leave test_otel_metrics.py and come back.
_OTHER_MODULE = (
    "tests/unit/test_detect_language_contract.py::TestExtensionMapContract"
    "::test_expected_table_is_exactly_the_extension_map"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _nested_pytest(node_ids: list[str]) -> subprocess.CompletedProcess[str]:
    """Run *node_ids* in a fresh pytest, in the given order, from the repo root.

    ``-p no:randomly`` matters: if the PARENT run has pytest-randomly active the
    plugin auto-loads in the child too and would shuffle these ids, which is
    exactly the order this test needs to control.
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=300,
        check=False,
    )


class TestLeavingAndReenteringTheOtelMetricsModule:
    """Reverting ``_test_meter_provider`` in tests/unit/test_otel_metrics.py from
    ``scope="session"`` back to ``scope="module"`` is the mutation that must make
    these tests fail."""

    def test_the_two_otel_ids_pass_when_nothing_separates_them(self) -> None:
        """PRECONDITION for the test below, not a claim of its own.

        If the two ids in ``_OTEL_FIRST``/``_OTEL_SECOND`` were skipped (no
        ``opentelemetry-sdk`` installed), renamed, or broken for some unrelated
        reason, the re-entry test would pass or fail for reasons that have
        nothing to do with ordering. This pins that the contiguous case is
        genuinely green AND genuinely ran, so the next test discriminates.
        """
        pytest.importorskip("opentelemetry.sdk.metrics", reason="requires pip install trelix[otel]")
        proc = _nested_pytest([_OTEL_FIRST, _OTEL_SECOND])
        assert proc.returncode == 0, f"contiguous run failed:\n{proc.stdout}\n{proc.stderr}"
        assert "2 passed" in proc.stdout, (
            "the two OTel ids did not both PASS in the contiguous order, so the "
            f"re-entry assertion below would not discriminate:\n{proc.stdout}"
        )

    def test_a_test_from_another_module_between_them_does_not_break_the_reader(self) -> None:
        """The three-id order that finalizes and re-creates the provider fixture.

        Fails with ``assert None == 1`` inside
        ``test_no_token_series_when_provider_reports_no_usage`` — plus
        ``"Overriding of current MeterProvider is not allowed"`` in that test's
        captured setup log — whenever the fixture is per-module rather than
        per-session.
        """
        pytest.importorskip("opentelemetry.sdk.metrics", reason="requires pip install trelix[otel]")
        proc = _nested_pytest([_OTEL_FIRST, _OTHER_MODULE, _OTEL_SECOND])
        assert proc.returncode == 0, (
            "leaving tests/unit/test_otel_metrics.py and coming back broke its "
            "InMemoryMetricReader — the global MeterProvider was claimed once and "
            "the second install was refused:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
        assert "3 passed" in proc.stdout, f"expected all three ids to pass:\n{proc.stdout}"

    def test_the_overriding_warning_is_absent_from_the_reentrant_run(self) -> None:
        """Names the mechanism, not just the symptom.

        The counters could conceivably be made to pass while the second
        ``set_meter_provider()`` still happens (e.g. by two readers coincidentally
        agreeing). OTel logs that refusal, so its ABSENCE is the direct evidence
        that the provider is installed exactly once across the re-entry.
        """
        pytest.importorskip("opentelemetry.sdk.metrics", reason="requires pip install trelix[otel]")
        proc = _nested_pytest([_OTEL_FIRST, _OTHER_MODULE, _OTEL_SECOND])
        combined = proc.stdout + proc.stderr
        assert "Overriding of current MeterProvider" not in combined, (
            "the global MeterProvider was set more than once across the "
            f"leave-and-return:\n{combined}"
        )
