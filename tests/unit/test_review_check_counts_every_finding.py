"""The review check's title and summary count every finding, not only the annotated ones.

GitHub takes at most 50 annotations per check. The publishing script of
``.github/workflows/trelix-review.yml`` judged the verdict by every finding but reported the
number of annotations it sent, so a review with 60 findings read "trelix found 50 issue(s)" and
nothing said that 10 had no annotation. The GitHub App (``infra/github-app/src/check-posting.ts``)
already counted every finding and said how many went without one. These tests run the script under
node (``tests/unit/review_workflow_harness.py``) and look at the one check it creates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.review_workflow_harness import NODE, findings_json, load_cases, run_publish_script

_needs_node = pytest.mark.skipif(NODE is None, reason="node is required to run the script")

_NOTE = " {} of them could not be shown as inline annotations (GitHub allows 50 per check)."


@_needs_node
@pytest.mark.parametrize(
    ("count", "annotated", "note"),
    [
        (0, 0, ""),
        (1, 1, ""),
        (50, 50, ""),
        (51, 50, _NOTE.format(1)),
        (60, 50, _NOTE.format(10)),
    ],
)
def test_a_clean_exit_counts_every_finding(
    tmp_path: Path, count: int, annotated: int, note: str
) -> None:
    published = run_publish_script(tmp_path, "0", findings_json(["INFO"] * count))

    output = published.call["output"]
    assert output["title"] == f"trelix found {count} issue(s)"
    assert output["summary"] == f"trelix reviewed PR #7 and found {count} issue(s).{note}"
    assert len(output["annotations"]) == annotated


@_needs_node
def test_an_incomplete_review_says_how_many_findings_went_without_an_annotation(
    tmp_path: Path,
) -> None:
    record = json.dumps(load_cases()["valid_outcome"])

    published = run_publish_script(tmp_path, "4", findings_json(["INFO"] * 60), outcome=record)

    output = published.call["output"]
    assert len(output["annotations"]) == 50
    assert (
        "\n60 issue(s) found in the hunks that were reviewed." + _NOTE.format(10) + "\n"
    ) in output["summary"]
