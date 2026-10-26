"""One record per golden query for `trelix eval-synthesis`, and their aggregate.

A `SynthesisRecord` is what `SynthesisEvalHarness.run_detailed` keeps for one golden line, fifteen
fields in this order: `id`, `query`, `answerable`, `abstained`, `abstain_reason`, the four
GroUSE-style scores (`hallucination`, `completeness`, `faithfulness`, `overall`), the citation facts
(`citations_total`, `citations_valid`, `citation_statuses`, `expected_citations_hit`,
`expected_citations_total`) and `error`. An errored record's scores are the placeholders the
harness has always reported (hallucination 1.0, the rest 0.0), not measurements, and its abstention
and citation facts are `False`, `None` and zeros; `error` is the only sign of the difference.

Abstention is derived from the retrieval and the answer (`abstain_reason_for`), never read from the
Synthesizer: an empty retrieval is `no_results`, checked first because the Synthesizer answers it
with a notice that is not an abstention text; an answer opening with `INSUFFICIENT_EVIDENCE:` is
`insufficient_evidence`; anything else is an answer. `abstained` is `abstain_reason is not None`.

Citations are verified only when the retrieval tagged its blocks (`context.citation_sources`
non-empty): with `TRELIX_RETRIEVAL_CITATIONS` off the prompt never asked for tags, so a `[C1]` in
an answer is noise, not an `unknown` citation, and every count is 0. `expected_citations` is read
as a set (a repeated path counts once); a hit is an expected path that a `valid` marker refers to.

`aggregate_synthesis_metrics`: the four means run over the ANSWERABLE records, errored ones included
with their placeholders, so a v1 file returns exactly the numbers the harness returned before
records existed; an errored record counts in `n_queries` and `unscoreable` only; every ratio is
`0.0` on a zero denominator, with its `n_*` key beside it saying how many records it ran over.

Records are compared with `==` and serialised with `asdict`, never hashed or put in a set: the
dataclass is frozen for immutability, and its `dict` field makes `hash()` raise.

The recorded-answers file (`--record`, `--replay`) is `{"schema_version": 1, "answers": {query:
answer}}`: one answer per golden query, keyed by the query text, and nothing else, so a replay
derives every field of the record again from the answer, the retrieval and the golden line.
`load_replay` refuses a file it cannot use with every problem found, as the other eval loaders do.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from trelix.eval._loading import ProblemsError, clip, read_text_capped, show
from trelix.retrieval.citations import AbstainReason, Citation, is_abstention
from trelix.review.outcome_file import write_outcome_file

_PER_QUERY_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
# Answers quote retrieved code and a long run holds many, but 20 MB is thousands of them.
MAX_REPLAY_BYTES = 20 * 1024 * 1024
_REPLAY_KEYS = frozenset({"schema_version", "answers"})
# How many missing queries the refusal names before it counts the rest.
_MISSING_LISTED = 5
# The same cut as `trelix eval`'s records: long enough to read, short enough for a table.
_ERROR_CHARS = 200


@dataclass(frozen=True)
class SynthesisRecord:
    """One golden query as scored; a record with `error` set holds placeholder scores and no
    abstention or citation facts. Compare with `==` and serialise with `asdict`; never hash one
    (the `citation_statuses` dict makes `hash()` raise, which `frozen` does not prevent)."""

    id: str
    query: str
    answerable: bool
    abstained: bool
    abstain_reason: AbstainReason | None
    hallucination: float
    completeness: float
    faithfulness: float
    overall: float
    citations_total: int
    citations_valid: int
    citation_statuses: dict[str, int]
    expected_citations_hit: int
    expected_citations_total: int
    error: str | None


def is_answerable(entry: Mapping[str, object]) -> bool:
    """The line's `answerable`; absent means True (every v1 line is answerable)."""
    return bool(entry.get("answerable", True))


def record_id(entry: Mapping[str, object], position: int) -> str:
    """The line's `id` when it is a non-blank string, else `q0001`-style from `position`,
    which is 1-based over the LOADED entries (a skipped line consumes none), as `trelix eval`."""
    raw = entry.get("id")
    return raw if isinstance(raw, str) and raw.strip() else f"q{position:04d}"


def abstain_reason_for(has_results: bool, answer: str) -> AbstainReason | None:
    """Why `answer` is an abstention, from the retrieval and the answer alone; None when it is not.

    `no_results` when the retrieval found nothing, checked first: the Synthesizer answers an empty
    retrieval with a notice that is not an abstention text. Else `insufficient_evidence` when the
    answer opens with `INSUFFICIENT_EVIDENCE:` (`is_abstention`).
    """
    if not has_results:
        return "no_results"
    if is_abstention(answer):
        return "insufficient_evidence"
    return None


def expected_citation_paths(entry: Mapping[str, object]) -> frozenset[str]:
    """The line's `expected_citations` as a set of paths (a repeated path counts once); empty when
    the field is absent. The loader already refused anything but a list of non-empty strings; the
    `isinstance` checks keep this total under mypy --strict without a cast."""
    raw = entry.get("expected_citations")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(path for path in raw if isinstance(path, str))


def scored_record(
    entry: Mapping[str, object],
    position: int,
    scores: Mapping[str, float],
    *,
    abstain_reason: AbstainReason | None = None,
    citations: Sequence[Citation] = (),
) -> SynthesisRecord:
    """The record of a query that was retrieved, answered and scored.

    `citations` is `verify_citations`' result for the answer, empty when the retrieval carried no
    tags; the keyword defaults mean "no abstention, nothing cited", so a caller that knows only
    the scores still gets a complete record.
    """
    valid = [c for c in citations if c.status == "valid"]
    expected = expected_citation_paths(entry)
    return SynthesisRecord(
        id=record_id(entry, position),
        query=str(entry.get("query", "")),
        answerable=is_answerable(entry),
        abstained=abstain_reason is not None,
        abstain_reason=abstain_reason,
        hallucination=scores["hallucination"],
        completeness=scores["completeness"],
        faithfulness=scores["faithfulness"],
        overall=scores["overall"],
        citations_total=len(citations),
        citations_valid=len(valid),
        citation_statuses=dict(Counter[str](c.status for c in citations)),
        expected_citations_hit=len(expected & {c.path for c in valid if c.path is not None}),
        expected_citations_total=len(expected),
        error=None,
    )


def failed_record(
    entry: Mapping[str, object], position: int, exc: BaseException
) -> SynthesisRecord:
    """The record of a query that raised: placeholder scores, the error's first 200 chars
    (`or type(exc).__name__`: a message-less exception must not read as "no error"), and no
    abstention or citation facts, since nothing was answered."""
    return SynthesisRecord(
        id=record_id(entry, position),
        query=str(entry.get("query", "")),
        answerable=is_answerable(entry),
        abstained=False,
        abstain_reason=None,
        hallucination=1.0,
        completeness=0.0,
        faithfulness=0.0,
        overall=0.0,
        citations_total=0,
        citations_valid=0,
        citation_statuses={},
        expected_citations_hit=0,
        expected_citations_total=0,
        error=(str(exc) or type(exc).__name__)[:_ERROR_CHARS],
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _grouse_means(records: Sequence[SynthesisRecord]) -> dict[str, float]:
    """The seven values the harness reported before the trust metrics: the four means over the
    answerable records (errored placeholders included) and the three counts."""
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


def _trust_metrics(scored: Sequence[SynthesisRecord]) -> dict[str, float]:
    """The nine abstention and citation values over `scored`, the records whose query did not
    raise; each ratio is 0.0 on a zero denominator and its `n_*` key says how many it ran over."""
    abstained = [r for r in scored if r.abstained]
    true_abstentions = sum(1 for r in abstained if not r.answerable)
    unanswerable = sum(1 for r in scored if not r.answerable)
    cited = [r for r in scored if r.citations_total > 0]
    expected = [r for r in scored if r.expected_citations_total > 0]
    return {
        "n_abstained": float(len(abstained)),
        "abstention_precision": _ratio(true_abstentions, len(abstained)),
        "abstention_recall": _ratio(true_abstentions, unanswerable),
        "citation_validity": _mean([r.citations_valid / r.citations_total for r in cited]),
        "n_cited_entries": float(len(cited)),
        "citation_recall": _mean(
            [r.expected_citations_hit / r.expected_citations_total for r in expected]
        ),
        "n_expected_citation_entries": float(len(expected)),
        "citations_total": float(sum(r.citations_total for r in scored)),
        "citations_valid": float(sum(r.citations_valid for r in scored)),
    }


def aggregate_synthesis_metrics(records: Sequence[SynthesisRecord]) -> dict[str, float]:
    """The sixteen aggregate values, every one a float; all 0.0 for no records (never raises).

    An errored record is in `_grouse_means` (its placeholders inside the four means, itself in
    `n_queries` and `unscoreable`) and out of `_trust_metrics`, which runs over `scored` only.
    """
    scored = [r for r in records if r.error is None]
    return _grouse_means(records) | _trust_metrics(scored)


def write_synthesis_per_query_file(
    path: str, records: Sequence[SynthesisRecord], aggregate: Mapping[str, float]
) -> str | None:
    """Write `records` and `aggregate` to `path` as JSON; return an error message, or None.

    The write is `write_outcome_file`'s (a 0600 temporary file moved over `path` with
    `os.replace`, sorted keys, ASCII-escaped), as `trelix eval --per-query-out`'s is. `harness`
    says which harness wrote the file, since both documents carry `schema_version` 1; each
    record's `citation_statuses` is a nested object, sorted like every other key.
    """
    payload = {
        "schema_version": _PER_QUERY_SCHEMA_VERSION,
        "harness": "synthesis",
        "records": [asdict(record) for record in records],
        "aggregate": dict(aggregate),
    }
    return write_outcome_file(path, payload)


class ReplayError(ProblemsError):
    """A recorded-answers file that cannot be used, with every problem found (`problems`)."""


def _answers_problems(doc: Mapping[str, object]) -> list[str]:
    if "answers" not in doc:
        return ['replay file: "answers" is missing']
    answers = doc["answers"]
    if not isinstance(answers, dict):
        return ['replay file: "answers" must be an object mapping each golden query to its answer']
    return [
        f'replay file: "answers"[{show(query)}] must be a string (got {show(answer)})'
        for query, answer in answers.items()
        if not isinstance(answer, str)
    ]


def _replay_problems(doc: object) -> list[str]:
    """Everything wrong with a parsed recorded-answers document, in a fixed order: the
    `schema_version`, then `answers`, then any other key; `[]` when it is usable."""
    if not isinstance(doc, dict):
        return ["replay file: the top level is not an object"]
    problems: list[str] = []
    if "schema_version" not in doc:
        problems.append('replay file: "schema_version" is missing')
    else:
        version = doc["schema_version"]
        # `True == 1` in Python, and a JSON `true` is not a version.
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            problems.append(f'replay file: "schema_version" must be 1 (got {show(version)})')
    problems.extend(_answers_problems(doc))
    problems.extend(
        f"replay file: unsupported key {show(key)}" for key in sorted(doc.keys() - _REPLAY_KEYS)
    )
    return problems


def load_replay(path: Path) -> dict[str, str]:
    """The answers `path` records, keyed by golden query. Raises `ReplayError`, naming every
    problem, for a file that cannot be read, is not JSON or does not have the document's shape.

    An empty `answers` object is fine, and so is an empty answer. The file is read at most
    `MAX_REPLAY_BYTES` large.
    """
    try:
        text = read_text_capped(path, MAX_REPLAY_BYTES)
    except (OSError, ValueError) as exc:
        raise ReplayError([f"replay file: cannot read the file: {exc}"]) from exc
    try:
        doc = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ReplayError([f"replay file: not valid JSON ({clip(str(exc))})"]) from exc
    problems = _replay_problems(doc)
    if problems:
        raise ReplayError(problems)
    return dict(doc["answers"])


def replay_payload(answers: Mapping[str, str]) -> dict[str, object]:
    """The document `--record` writes for `answers`, which `load_replay` reads back."""
    return {"schema_version": REPLAY_SCHEMA_VERSION, "answers": dict(answers)}


def write_replay_file(path: str, answers: Mapping[str, str]) -> str | None:
    """Write `answers` to `path` as a recorded-answers document; return an error message, or None.

    Through `write_outcome_file`, like `--per-query-out`: mode 0600, atomic, an existing file
    replaced. The answers quote retrieved code, so the file is as sensitive as the index.
    """
    return write_outcome_file(path, replay_payload(answers))


def missing_replay_answers(queries: Sequence[str], answers: Mapping[str, str]) -> list[str]:
    """The golden `queries` that `answers` has no entry for, in golden order, each once."""
    return list(dict.fromkeys(query for query in queries if query not in answers))


def missing_replay_message(missing: Sequence[str]) -> str:
    """`replay file has no answer for N queries: 'a', 'b', ...`, naming the first five."""
    count = len(missing)
    listed = ", ".join(show(query) for query in missing[:_MISSING_LISTED])
    if count > _MISSING_LISTED:
        listed += f", ... and {count - _MISSING_LISTED} more"
    return f"replay file has no answer for {count} {'query' if count == 1 else 'queries'}: {listed}"
