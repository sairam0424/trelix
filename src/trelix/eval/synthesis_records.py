"""One record per golden query for `trelix eval-synthesis`, and their aggregate.

A `SynthesisRecord` is what `SynthesisEvalHarness.run_detailed` keeps for one golden line: the
four GroUSE-style scores, whether the line is answerable, and the error when the query could not
be scored. An errored record's four scores are the placeholders the harness has always reported
(hallucination 1.0, the rest 0.0), not measurements; `error` is the only sign of the difference.

Two exclusion rules shape `aggregate_synthesis_metrics`: the four means run over the ANSWERABLE
records only (an unanswerable query has no answer to score), errored ones included with their
placeholders, so a v1 file (every line answerable) returns exactly the numbers the harness
returned before records existed; and an errored record counts in `n_queries` and `unscoreable`
only, never in `n_unanswerable`, whatever its line said.

The abstention and citation facts (whether the model abstained and why, how many of its `[C#]`
markers verified, which expected citations it hit) arrive with the next change in this series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from trelix.review.outcome_file import write_outcome_file

_PER_QUERY_SCHEMA_VERSION = 1
# The same cut as `trelix eval`'s records: long enough to read, short enough for a table.
_ERROR_CHARS = 200


@dataclass(frozen=True)
class SynthesisRecord:
    """One golden query as scored; a record with `error` set holds placeholder scores."""

    id: str
    query: str
    answerable: bool
    hallucination: float
    completeness: float
    faithfulness: float
    overall: float
    error: str | None


def is_answerable(entry: Mapping[str, object]) -> bool:
    """The line's `answerable`; absent means True (every v1 line is answerable)."""
    return bool(entry.get("answerable", True))


def record_id(entry: Mapping[str, object], position: int) -> str:
    """The line's `id` when it is a non-blank string, else `q0001`-style from `position`,
    which is 1-based over the LOADED entries (a skipped line consumes none), as `trelix eval`."""
    raw = entry.get("id")
    return raw if isinstance(raw, str) and raw.strip() else f"q{position:04d}"


def scored_record(
    entry: Mapping[str, object], position: int, scores: Mapping[str, float]
) -> SynthesisRecord:
    """The record of a query that was retrieved, answered and scored."""
    return SynthesisRecord(
        id=record_id(entry, position),
        query=str(entry.get("query", "")),
        answerable=is_answerable(entry),
        hallucination=scores["hallucination"],
        completeness=scores["completeness"],
        faithfulness=scores["faithfulness"],
        overall=scores["overall"],
        error=None,
    )


def failed_record(
    entry: Mapping[str, object], position: int, exc: BaseException
) -> SynthesisRecord:
    """The record of a query that raised: placeholder scores and the error's first 200 chars
    (`or type(exc).__name__`: a message-less exception must not read as "no error")."""
    return SynthesisRecord(
        id=record_id(entry, position),
        query=str(entry.get("query", "")),
        answerable=is_answerable(entry),
        hallucination=1.0,
        completeness=0.0,
        faithfulness=0.0,
        overall=0.0,
        error=(str(exc) or type(exc).__name__)[:_ERROR_CHARS],
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_synthesis_metrics(records: Sequence[SynthesisRecord]) -> dict[str, float]:
    """The seven aggregate values, every one a float; all 0.0 for no records (never raises)."""
    answerable = [r for r in records if r.answerable]
    return {
        "hallucination_rate": _mean([r.hallucination for r in answerable]),
        "completeness": _mean([r.completeness for r in answerable]),
        "faithfulness": _mean([r.faithfulness for r in answerable]),
        "overall": _mean([r.overall for r in answerable]),
        "n_queries": float(len(records)),
        "unscoreable": float(sum(1 for r in records if r.error is not None)),
        "n_unanswerable": float(sum(1 for r in records if not r.answerable and r.error is None)),
    }


def write_synthesis_per_query_file(
    path: str, records: Sequence[SynthesisRecord], aggregate: Mapping[str, float]
) -> str | None:
    """Write `records` and `aggregate` to `path` as JSON; return an error message, or None.

    The write is `write_outcome_file`'s (a 0600 temporary file moved over `path` with
    `os.replace`, sorted keys, ASCII-escaped), as `trelix eval --per-query-out`'s is. `harness`
    says which harness wrote the file, since both documents carry `schema_version` 1.
    """
    payload = {
        "schema_version": _PER_QUERY_SCHEMA_VERSION,
        "harness": "synthesis",
        "records": [asdict(record) for record in records],
        "aggregate": dict(aggregate),
    }
    return write_outcome_file(path, payload)
