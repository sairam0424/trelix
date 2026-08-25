"""The two "openai" tests in test_otel_metrics.py are order-INDEPENDENT of each other.

THIS FILE IS A MEASUREMENT, NOT A FIX, and that is deliberate.

Rounds 4 and 5 flagged that the two ``"openai"`` tests in
``test_otel_metrics.py::TestEnabledRecordsCounters`` share one cumulative counter series,
and converted ``test_openai_embed_counts_one_request_per_api_call`` to DELTAS after
measuring ``assert 3 == 2`` under reverse collection order. Round 6 was asked whether the
neighbour, ``test_counter_carries_provider_and_model_attributes``, is broken the same way.

MEASURED ANSWER: NO. It asserts on a SET of label VALUES, not on a cumulative TOTAL, and a
set is idempotent under repeated recording -- recording the same (provider, model) pair a
second time adds no member. Both openai tests take their model from the single
``_stub_openai_embedder`` default, so the set has exactly one member however many times
either test runs, in whatever order. All three orderings were run against the shipped
assertion:

    <attributes test> <counting test>  -> 2 passed
    <counting test> <attributes test>  -> 2 passed
    <attributes test> alone            -> 1 passed

The two failures rounds 4/5 measured were about TOTALS. ``_counter_total`` sums point
values; ``models == {...}`` unions label strings. Different quantity, different fault.
``tests/unit/test_otel_metrics_cumulative_totals.py`` owns the totals; this file owns the
labels, and the honest finding about the labels is that there is nothing to convert TODAY.

The three tests below make that negative result PERMANENT rather than leaving it in a round
report, because the next reader who notices the shared series will otherwise re-derive the
wrong conclusion -- inheriting a description instead of measuring is the mistake this
project has already corrected eight times in a single round.

WHAT IS NEVERTHELESS LATENT, MEASURED, AND LEFT UNLANDED
-------------------------------------------------------
``test_counter_carries_provider_and_model_attributes`` reads the label set of the WHOLE
session's ``"openai"`` series, because ``_test_meter_provider`` must be session-scoped
(OTel's global MeterProvider is one-shot per process). Its one-element set-equality
therefore ALSO asserts "no other openai recorder in this session ever used a different
model". Nothing enforces that. Measured, by adding a third openai recorder ahead of it
inside the same module so it shared the session reader, with model
``text-embedding-3-SMALL-probe``:

    assert {'text-embedding-3-SMALL-probe', 'text-embedding-3-large'}
           == {'text-embedding-3-large'}
    E   Extra items in the left set: 'text-embedding-3-SMALL-probe'

That is a true assertion reporting a false cause: the message says trelix labels the wrong
model, when the actual event is "a second model was used somewhere in this process". The
delta-converted sibling survived the same probe unchanged, confirming the delta form is
robust to a differing model and not only to a differing count.

WHY IT IS NOT FIXED HERE. The fix is a real one, not a speculative one, but it lands
INSIDE ``test_otel_metrics.py``: the assertion has to become label-set-intersection
(``after & {"text-embedding-3-large", None} == {"text-embedding-3-large"}`` both ways, plus
``after - before <= {...}``), and it is only load-bearing if a PERMANENT second-model
recorder exists in that file to remove the accident -- the design
``TestASecondRecorderUnderTheSameProviderLabel`` already uses for the bedrock labels.
Both edits belong in that file, in one change, with the collider added in the same commit
as the conversion. That change is written up in this round's findings_left with the exact
measured failure above; landing it half-done (converted but with no collider, or a
collider with no conversion) would be strictly worse than the accident it replaces.

MECHANICS
* Node ids are LITERALS. An id computed from test_otel_metrics.py would keep agreeing with
  it through a rename, which is exactly when this probe must go red.
* Child pytest processes, because in-process the collection order is whatever the runner
  chose, so an in-process version of this would pass or fail by seed.
* Each ordering is its own child run. One run containing both orderings could not tell
  which ordering failed.
* 2.7s measured for the whole file, so it stays out of ``SLOW_FILES`` in
  ``tests/conftest.py`` (that table's documented criterion is ">= 4.0s") and no taxonomy
  entry is added. Reported while looking: its round-5 sibling
  ``unit/test_otel_metrics_cumulative_totals.py`` measures 7.53s by the same command and
  IS absent from that table, which is a real omission against the stated rule -- left for
  whoever owns that file rather than silently amended from here.
* NON-DISCRIMINATING COMPANIONS, stated plainly: no mutation of ``otel_tracing.py`` is
  claimed to make these two fail while leaving the rest of the suite green. Their job is
  to keep a measurement true, not to kill a mutant. Naming that is the rule -- inventing a
  mutation for them would be worse than admitting they have none.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_LABELS = (
    "tests/unit/test_otel_metrics.py::TestEnabledRecordsCounters"
    "::test_counter_carries_provider_and_model_attributes"
)
_COUNTS = (
    "tests/unit/test_otel_metrics.py::TestEnabledRecordsCounters"
    "::test_openai_embed_counts_one_request_per_api_call"
)


def _nested_pytest(node_ids: list[str]) -> subprocess.CompletedProcess[str]:
    """Run *node_ids* in a fresh pytest, IN THE GIVEN ORDER, from the repo root."""
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

    Without the SDK the fixture behind both ids calls ``importorskip`` and the child
    reports "2 skipped" with return code 0 -- so a ``returncode == 0`` assertion would be
    green having measured nothing, which is why "2 passed" is what gets asserted below.
    CI's default ``pip install -e ".[local,dev]"`` omits the ``otel`` extra, so this is a
    real environment and not a theoretical one. Stated as the environment rule requires:
    this file claims nothing stronger than "IF opentelemetry-sdk is importable, both
    orderings hold", and it names the missing package when it is not.

    A per-test ``importorskip`` rather than a module-scope one, deliberately: a
    module-scope ``pytest.importorskip`` must be declared in ``tests/conftest.py``'s
    ``REQUIRES_EXTRA_FILES``, and omitting that declaration was round 5's landing blocker.
    """
    pytest.importorskip("opentelemetry.sdk.metrics", reason="requires pip install trelix[otel]")


class TestTheTwoOpenaiTestsAreOrderIndependentOfEachOther:
    """Both orderings, pinned. See the module docstring for why this is the deliverable."""

    def test_the_label_test_first_then_the_counting_test(self) -> None:
        proc = _nested_pytest([_LABELS, _COUNTS])
        assert "2 passed" in proc.stdout, (
            "the label assertion and the counting deltas do not both hold when the "
            f"label test runs FIRST:\n{proc.stdout}\n{proc.stderr}"
        )

    def test_the_counting_test_first_then_the_label_test(self) -> None:
        proc = _nested_pytest([_COUNTS, _LABELS])
        assert "2 passed" in proc.stdout, (
            "the label assertion and the counting deltas do not both hold when the "
            f"counting test runs FIRST -- which is the ordering round 4 measured "
            f"`assert 3 == 2` in, before the deltas landed:\n{proc.stdout}\n{proc.stderr}"
        )

    def test_the_label_test_alone_still_passes(self) -> None:
        """The third measurement, kept because it is the control for the other two.

        If the label test only passes when SOMETHING else recorded first, then the pair
        above is green for a reason that has nothing to do with ordering robustness.
        """
        proc = _nested_pytest([_LABELS])
        assert "1 passed" in proc.stdout, (
            f"the label assertion does not hold in isolation:\n{proc.stdout}\n{proc.stderr}"
        )
