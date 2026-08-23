"""The bedrock counter assertions must not require a FRESH counter.

WHAT THIS CLOSES. Round 4 moved test_otel_metrics.py's ``_test_meter_provider``
from module to session scope, because OTel's global MeterProvider can be installed
once per process. That fixed the crash and left a quieter problem: the
``InMemoryMetricReader`` now lives for the whole SESSION and its counters are
CUMULATIVE, so

    assert _counter_total(reader, "trelix.embedder.requests", "bedrock-titan") == 2

is not "this embed made two requests" -- it is that AND "nothing else in this
session ever recorded under bedrock-titan". The second half is a claim about
collection order smuggled into a test about counting. The same shape already bit
the two ``"openai"`` tests: tests/unit in reverse order failed one with
``assert 3 == 2``, which is why those were converted to deltas earlier.

NOT HYPOTHETICAL, NOT CONTRIVED. It was latent only because the bedrock labels had
no second recorder: tests/unit/test_embedder.py drives the same
``BedrockTitanEmbedder.embed`` with TRELIX_OTEL_ENABLED=false (pinned by
tests/_env_isolation.py), so it records nothing. Once test_otel_metrics installs
the global provider it stays installed for the rest of the process, so any
flag-on recorder collides. ``TestASecondRecorderUnderTheSameProviderLabel`` in
test_otel_metrics.py is that recorder, added so the collision is REAL.

WHY A CHILD PYTEST. In-process, the order of collider vs target is whatever the
runner chose, so a test depending on it would pass or fail by seed -- confirmed:
under ``--interleave --interleave-seed=522 --randomly-seed=522`` the in-process
order happened to put the target FIRST and the absolute form survived in-process
while this driver still killed it. The child runs pin the order with node ids and
``-p no:randomly``.

The two targets are driven in SEPARATE child runs: one run containing both would
fail if either regressed and credit one target's kill to the other.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Literals, not derived: an id computed from test_otel_metrics.py would keep
# agreeing with it through a rename, which is when this probe must go red.
_COLLIDER = (
    "tests/unit/test_otel_metrics.py::TestASecondRecorderUnderTheSameProviderLabel"
    "::test_recording_under_both_bedrock_labels_moves_the_shared_counters"
)
_TITAN = (
    "tests/unit/test_otel_metrics.py::TestEnabledRecordsCounters"
    "::test_bedrock_titan_records_provider_reported_tokens"
)
_COHERE = (
    "tests/unit/test_otel_metrics.py::TestEnabledRecordsCounters"
    "::test_no_token_series_when_provider_reports_no_usage"
)


def _nested_pytest(node_ids: list[str]) -> subprocess.CompletedProcess[str]:
    """Run *node_ids* in a fresh pytest, in the given order, from the repo root."""
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


@pytest.fixture(autouse=True)
def _require_otel_sdk() -> None:
    """Skip LOUDLY without opentelemetry-sdk rather than passing vacuously.

    Without the SDK the fixture behind every id below calls ``importorskip`` and
    the child reports "3 skipped" with return code 0 -- so a ``returncode == 0``
    assertion would be green having measured nothing. CI's default
    ``pip install -e ".[local,dev]"`` omits the ``otel`` extra, so this is a real
    environment, not a theoretical one.
    """
    pytest.importorskip("opentelemetry.sdk.metrics", reason="requires pip install trelix[otel]")


class TestTheBedrockTotalsSurviveAnEarlierRecorder:
    def test_the_collider_records_on_its_own(self) -> None:
        """PRECONDITION for the two tests below, not a claim of its own.

        If the collider stops recording (skipped, renamed, assertions relaxed), the
        runs below are green because NOTHING poisoned the counters. "1 passed", not
        just returncode 0: a skip also exits 0.
        """
        proc = _nested_pytest([_COLLIDER])
        assert "1 passed" in proc.stdout, (
            "the second-recorder test did not run and pass on its own, so the two "
            f"assertions below would not discriminate:\n{proc.stdout}\n{proc.stderr}"
        )

    def test_titan_totals_survive_a_previous_bedrock_titan_recorder(self) -> None:
        """Restoring
        ``assert _counter_total(metric_reader, "trelix.embedder.requests", "bedrock-titan") == 2``
        (and ``== 8`` for tokens) in place of the deltas is the mutation that must
        make this fail. Measured with that revert: ``assert 4 == 2`` in the child,
        "1 failed, 1 passed".
        """
        proc = _nested_pytest([_COLLIDER, _TITAN])
        assert proc.returncode == 0, (
            "test_bedrock_titan_records_provider_reported_tokens depends on being the "
            "first bedrock-titan recorder in the process; the session-lived reader's "
            f"counters are cumulative:\n{proc.stdout}\n{proc.stderr}"
        )
        assert "2 passed" in proc.stdout, f"expected both ids to pass:\n{proc.stdout}"

    def test_cohere_totals_survive_a_previous_bedrock_cohere_recorder(self) -> None:
        """The other target, measured alone so the kill is attributable.

        Restoring ``== 1`` / ``== 5`` there is the mutation; measured with that
        revert: ``assert 2 == 1``, "1 failed, 1 passed".
        """
        proc = _nested_pytest([_COLLIDER, _COHERE])
        assert proc.returncode == 0, (
            "test_no_token_series_when_provider_reports_no_usage depends on being the "
            f"first bedrock-cohere recorder in the process:\n{proc.stdout}\n{proc.stderr}"
        )
        assert "2 passed" in proc.stdout, f"expected both ids to pass:\n{proc.stdout}"
