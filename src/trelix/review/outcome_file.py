"""The machine-readable record of what a review did and did not cover.

`trelix review --json` keeps its bare array on stdout, because the workflow, the GitHub App
and third-party scripts parse it. That array cannot say "and three hunks were never reviewed",
so when `TRELIX_REVIEW_OUTCOME_FILE` names a path the command also writes this small JSON
document there. It carries only the fixed vocabulary of `HunkResult`: never refusal text or
model prose, because a caller may publish it in a GitHub Check.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trelix.review.reviewer import ReviewOutcome

SCHEMA_VERSION = 1
# A review of a huge PR can leave hundreds of hunks unreviewed; the first ones are enough to
# say where to look, and a Check summary has a size limit of its own.
MAX_LISTED_HUNKS = 100


def build_outcome_payload(outcome: ReviewOutcome, *, exit_code: int) -> dict[str, Any]:
    """The document for `outcome`: counts, the exit code, and the unreviewed hunks (first 100)."""
    unreviewed = [r for r in outcome.hunk_results if not r.reviewed]
    return {
        "schema_version": SCHEMA_VERSION,
        "hunks_total": outcome.hunks_total,
        "hunks_reviewed": outcome.hunks_reviewed,
        "hunks_unreviewed": outcome.hunks_failed,
        "exit_code": exit_code,
        "hunks": [r.as_dict() for r in unreviewed[:MAX_LISTED_HUNKS]],
        "hunks_omitted": max(len(unreviewed) - MAX_LISTED_HUNKS, 0),
    }


def write_outcome_file(path: str, payload: dict[str, Any]) -> str | None:
    """Write `payload` to `path` atomically; return an error message, or None on success.

    The file is created with mode 0600 and moved into place with `os.replace`, so a reader
    never sees a half-written document and a symlink at `path` is replaced, not followed. The
    parent directory must already exist, and `path` must name a file. A failure is reported,
    never raised: not being able to write this record must not change how the review ends.
    """
    try:
        target = Path(path)
        # Random, so a name planted in a shared directory cannot be guessed; O_EXCL still
        # refuses one that already exists. A path with no file name (".", "/") raises ValueError.
        temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None
