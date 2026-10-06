"""`results.json`, schema_version 1: the record of one suite run that `trelix eval-compare` judges.

The writer side is `build_results` and `write_results`, the reader side is `load_results`;
both live here so one module owns the schema and a round trip test can hold them together.
It is a different document from the `trelix eval --per-query-out` file, which also says
`schema_version: 1` but names no repository, commit or golden file, so it cannot be judged
and is refused with a pointed message.

Top level, exactly these keys (anything else is refused):

    schema_version   1
    arm              label the pre-registration's `comparison_id` binds to
    suite            name, golden_version, repo_url, repo_sha, license, golden_sha256, plans_sha256
    trelix_version   the trelix that ran it
    embedder         provider, model, dimension (int >= 1), library_version (string or null)
    pipeline         the four frozen flags (bool), plans, rerank_summary, config (object)
    aggregate        mrr, n_queries, ndcg@10, recall@10: the mean of `records`, exactly
    records          the ten `QueryRecord` fields of each scored query
    run              created_at (string): the one non-deterministic datum, never read by a verdict

`pipeline` ignores keys it does not know, and so does `run`, so a later additive field needs
no schema bump. Everything else is exact. `aggregate` is compared with `==` against
`aggregate_metrics(records)` recomputed from the records as they stand in the file, so a
hand-edited file cannot keep a stale aggregate.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trelix.eval._loading import ProblemsError, clip, read_text_capped, show
from trelix.eval.harness import QueryRecord, aggregate_metrics
from trelix.review.outcome_file import write_outcome_file

RESULTS_SCHEMA_VERSION = 1
MAX_RESULTS_BYTES = 32 * 1024 * 1024

TOP_LEVEL_KEYS = (
    "aggregate",
    "arm",
    "embedder",
    "pipeline",
    "records",
    "run",
    "schema_version",
    "suite",
    "trelix_version",
)
SUITE_KEYS = (
    "name",
    "golden_version",
    "repo_url",
    "repo_sha",
    "license",
    "golden_sha256",
    "plans_sha256",
)
EMBEDDER_KEYS = ("provider", "model", "dimension", "library_version")
AGGREGATE_KEYS = ("mrr", "n_queries", "ndcg@10", "recall@10")
RECORD_KEYS = (
    "id",
    "repo",
    "kind",
    "lang",
    "split",
    "ndcg",
    "recall",
    "mrr",
    "top10",
    "error",
)
# Switched off by the suite runner so a run is repeatable; `eval-compare` refuses a file where
# any of them is true. (The fifth condition, `plans == "replayed"`, is checked there as well.)
FROZEN_FLAGS = ("rerank", "hyde_fallback_enabled", "multi_query_enabled", "flare_enabled")
PIPELINE_KEYS = (*FROZEN_FLAGS, "plans", "rerank_summary", "config")

# Also the shape of a suite name and of an arm: both become path components.
ARM_PATTERN = r"[a-z0-9][a-z0-9_-]{0,62}"
_ARM_RE = re.compile(ARM_PATTERN)
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# The shape of each identity field, shared with `trelix.eval.suite` so that a suite.json the
# loader accepts cannot describe a run that this reader would then refuse.
SUITE_PATTERNS = {
    "name": _ARM_RE,
    "golden_version": _LABEL_RE,
    "repo_sha": _SHA1_RE,
    "golden_sha256": _SHA256_RE,
    "plans_sha256": _SHA256_RE,
}
_SPLITS = (None, "dev", "test")
_TOP10_MAX = 10
_NOT_RESULTS = "not a results.json"


class ResultsError(ProblemsError):
    """A results file that cannot be used; `problems` lists every reason found."""


@dataclass(frozen=True)
class SuiteIdentity:
    """Which repository, commit and golden and plans files a run measured."""

    name: str
    golden_version: str
    repo_url: str
    repo_sha: str
    license: str
    golden_sha256: str
    plans_sha256: str


@dataclass(frozen=True)
class EmbedderInfo:
    provider: str
    model: str
    dimension: int
    library_version: str | None


@dataclass(frozen=True)
class PipelineInfo:
    rerank: bool
    hyde_fallback_enabled: bool
    multi_query_enabled: bool
    flare_enabled: bool
    plans: str
    rerank_summary: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class Results:
    """A validated results file. `records` are in file order; `created_at` is display only."""

    arm: str
    suite: SuiteIdentity
    trelix_version: str
    embedder: EmbedderInfo
    pipeline: PipelineInfo
    records: tuple[QueryRecord, ...]
    created_at: str


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_results(
    *,
    arm: str,
    suite: Mapping[str, str],
    trelix_version: str,
    embedder: Mapping[str, object],
    pipeline: Mapping[str, object],
    records: Sequence[QueryRecord],
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """The schema 1 document for a run: `records` as scored, their `aggregate`, and the labels.

    `now` is injected so a test can freeze the clock. The timestamp lives under `run` and
    nowhere else, so two runs of the same code agree on everything but `run`. `suite`,
    `embedder` and `pipeline` are copied one level deep: a nested value such as
    `pipeline["config"]` is shared with the caller, so write the document before changing it.
    """
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "arm": arm,
        "suite": dict(suite),
        "trelix_version": trelix_version,
        "embedder": dict(embedder),
        "pipeline": dict(pipeline),
        "aggregate": aggregate_metrics(records),
        "records": [{**asdict(record), "top10": list(record.top10)} for record in records],
        "run": {"created_at": now().isoformat(timespec="seconds")},
    }


def write_results(path: str, doc: Mapping[str, Any]) -> str | None:
    """Write `doc` to `path` as JSON; return an I/O error message, or None on success.

    The write is `write_outcome_file`'s (private temp file, atomic rename, sorted keys, ASCII),
    so equal documents are byte-equal. The document is validated first and a ResultsError is
    raised for one the reader would refuse: the writer cannot emit what the reader rejects.
    That includes a NaN, an infinity or a value that is not JSON, in `pipeline.config` or
    any other key `_validate` does not look into.
    """
    problems = _validate(doc) or _json_problems(doc)
    if problems:
        raise ResultsError(problems)
    return write_outcome_file(path, dict(doc))


def _json_problems(doc: Mapping[str, Any]) -> list[str]:
    """Why `doc` cannot be written as JSON the reader accepts, or `[]`."""
    try:
        json.dumps(dict(doc), allow_nan=False, sort_keys=True)  # as write_outcome_file dumps
    except (TypeError, ValueError) as exc:
        return [f"the document cannot be written as JSON: {clip(str(exc))}"]
    return []


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


def _reject_constant(name: str) -> float:
    raise ValueError(f"the constant {name} is not allowed")


def load_results(path: str | Path) -> Results:
    """Read and validate a results file. Raises ResultsError, listing every problem found."""
    try:
        text = read_text_capped(Path(path), MAX_RESULTS_BYTES)
    except (OSError, ValueError) as exc:
        raise ResultsError([f"{_NOT_RESULTS}: cannot read the file: {exc}"]) from exc
    try:
        doc = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ResultsError([f"{_NOT_RESULTS}: not valid JSON ({clip(str(exc))})"]) from exc
    return parse_results(doc)


def parse_results(doc: object) -> Results:
    """Validate an already-parsed document and return it as `Results`."""
    if not isinstance(doc, dict):
        raise ResultsError([f"{_NOT_RESULTS}: the top level is not an object"])
    if "suite" not in doc:
        raise ResultsError(
            [
                f'{_NOT_RESULTS}: it has no "suite" key. A file from '
                "`trelix eval --per-query-out` looks like this: it does not say which "
                "repository, commit or golden file it measured, so it cannot be compared"
            ]
        )
    problems = _version_problems(doc)
    if problems:
        # Nothing else in a file of another schema can be judged by this one's rules.
        raise ResultsError(problems)
    problems = _validate(doc)
    if problems:
        raise ResultsError(problems)
    return _build(doc)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_score(value: object) -> bool:
    return _is_number(value) and 0 <= value <= 1  # type: ignore[operator]


def _exact_keys(where: str, block: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    unsupported = [f"{where}: unsupported key {show(key)}" for key in block if key not in keys]
    missing = [f"{where}: missing key {key!r}" for key in keys if key not in block]
    return unsupported + missing


def _version_problems(doc: Mapping[str, Any]) -> list[str]:
    version = doc.get("schema_version")
    if type(version) is int and version == RESULTS_SCHEMA_VERSION:
        return []
    return [
        f"unsupported schema_version {show(version)}: "
        f"this reader understands {RESULTS_SCHEMA_VERSION}"
    ]


def _validate(doc: Mapping[str, Any]) -> list[str]:
    """Every way `doc` departs from schema 1; `[]` when it is a valid results document."""
    problems = _version_problems(doc) + _exact_keys("results", doc, TOP_LEVEL_KEYS)
    if "arm" in doc and not (isinstance(doc["arm"], str) and _ARM_RE.fullmatch(doc["arm"])):
        problems.append(f"arm: must match {ARM_PATTERN} (got {show(doc['arm'])})")
    if "trelix_version" in doc and not _is_text(doc["trelix_version"]):
        problems.append("trelix_version: must be a non-empty string")
    suite = doc.get("suite")
    problems += _suite_problems(suite)
    problems += _embedder_problems(doc.get("embedder"))
    problems += _pipeline_problems(doc.get("pipeline"))
    problems += _run_problems(doc.get("run"))
    suite_name = suite.get("name") if isinstance(suite, dict) else None
    problems += _records_problems(doc.get("records"), doc.get("aggregate"), suite_name)
    return problems


def _suite_problems(suite: object) -> list[str]:
    if not isinstance(suite, dict):
        return ["suite: must be an object"]
    problems = _exact_keys("suite", suite, SUITE_KEYS)
    for key, pattern in SUITE_PATTERNS.items():
        value = suite.get(key)
        if key in suite and not (isinstance(value, str) and pattern.fullmatch(value)):
            problems.append(f"suite: {key!r} is not a valid value (got {show(value)})")
    for key in ("repo_url", "license"):
        if key in suite and not _is_text(suite[key]):
            problems.append(f"suite: {key!r} must be a non-empty string")
    return problems


def _embedder_problems(embedder: object) -> list[str]:
    if not isinstance(embedder, dict):
        return ["embedder: must be an object"]
    problems = _exact_keys("embedder", embedder, EMBEDDER_KEYS)
    for key in ("provider", "model"):
        if key in embedder and not _is_text(embedder[key]):
            problems.append(f"embedder: {key!r} must be a non-empty string")
    dimension = embedder.get("dimension")
    if "dimension" in embedder and not (type(dimension) is int and dimension >= 1):
        problems.append(
            f"embedder: 'dimension' must be an integer of at least 1 (got {show(dimension)})"
        )
    library_version = embedder.get("library_version")
    if "library_version" in embedder and library_version is not None:
        if not _is_text(library_version):
            problems.append("embedder: 'library_version' must be a non-empty string or null")
    return problems


def _pipeline_problems(pipeline: object) -> list[str]:
    """Unknown keys are accepted and ignored: later additions need no schema bump."""
    if not isinstance(pipeline, dict):
        return ["pipeline: must be an object"]
    problems = [f"pipeline: missing key {key!r}" for key in PIPELINE_KEYS if key not in pipeline]
    for key in FROZEN_FLAGS:
        if key in pipeline and not isinstance(pipeline[key], bool):
            problems.append(f"pipeline: {key!r} must be true or false")
    for key in ("plans", "rerank_summary"):
        if key in pipeline and not isinstance(pipeline[key], str):
            problems.append(f"pipeline: {key!r} must be a string")
    if "config" in pipeline and not isinstance(pipeline["config"], dict):
        problems.append("pipeline: 'config' must be an object")
    return problems


def _run_problems(run: object) -> list[str]:
    if not isinstance(run, dict):
        return ["run: must be an object"]
    if not isinstance(run.get("created_at"), str):
        return ["run: 'created_at' must be a string"]
    return []


def _records_problems(records: object, aggregate: object, suite_name: object) -> list[str]:
    if not isinstance(records, list) or not records:
        return ["records: must be a non-empty list"]
    problems: list[str] = []
    for index, item in enumerate(records):
        problems += _record_problems(item, index, suite_name)
    ids = Counter(
        item["id"] for item in records if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    problems += [
        f"duplicate query id {show(key)} ({count} records)"
        for key, count in ids.items()
        if count > 1
    ]
    if problems:
        return problems
    return _aggregate_problems(aggregate, [_record_from(item) for item in records])


def _record_problems(item: object, index: int, suite_name: object) -> list[str]:
    if not isinstance(item, dict):
        return [f"record #{index}: must be an object"]
    label = clip(item["id"]) if _is_text(item.get("id")) else f"#{index}"
    where = f"record {label}"
    problems = _exact_keys(where, item, RECORD_KEYS)
    problems += [f"{where}: {text}" for text in _record_field_problems(item, suite_name)]
    return problems


def _record_field_problems(item: Mapping[str, Any], suite_name: object) -> list[str]:
    problems: list[str] = []
    for key in ("id", "repo"):
        if key in item and not _is_text(item[key]):
            problems.append(f"{key!r} must be a non-empty string")
    for key in ("kind", "lang"):
        if item.get(key) is not None and not _is_text(item[key]):
            problems.append(f"{key!r} must be a non-empty string or null")
    if "split" in item and item["split"] not in _SPLITS:
        problems.append(f"'split' must be null, 'dev' or 'test' (got {show(item['split'])})")
    for key in ("ndcg", "recall", "mrr"):
        if key in item and not _is_score(item[key]):
            problems.append(f"{key!r} must be a number from 0 to 1 (got {show(item[key])})")
    problems += _top10_and_error_problems(item)
    repo = item.get("repo")
    if isinstance(repo, str) and isinstance(suite_name, str) and repo != suite_name:
        problems.append(f"'repo' is {show(repo)} but the suite is named {show(suite_name)}")
    return problems


def _top10_and_error_problems(item: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    top10 = item.get("top10")
    if "top10" in item and not (
        isinstance(top10, list)
        and len(top10) <= _TOP10_MAX
        and all(isinstance(path, str) for path in top10)
    ):
        problems.append(f"'top10' must be a list of at most {_TOP10_MAX} strings")
    error = item.get("error")
    # Any non-empty text, not `_is_text`: `RuntimeError("\n")` is a failed query whose message is
    # only whitespace, and `_failed_record` already replaces an empty message with the type name.
    if error is not None and not (isinstance(error, str) and error != ""):
        problems.append("'error' must be null or a non-empty string")
    return problems


def _record_from(item: Mapping[str, Any]) -> QueryRecord:
    """A `QueryRecord` from an item `_record_problems` found no fault with."""
    return QueryRecord(
        id=item["id"],
        repo=item["repo"],
        kind=item["kind"],
        lang=item["lang"],
        split=item["split"],
        ndcg=float(item["ndcg"]),
        recall=float(item["recall"]),
        mrr=float(item["mrr"]),
        top10=tuple(item["top10"]),
        error=item["error"],
    )


def _aggregate_problems(aggregate: object, records: Sequence[QueryRecord]) -> list[str]:
    if not isinstance(aggregate, dict):
        return ["aggregate: must be an object"]
    problems = _exact_keys("aggregate", aggregate, AGGREGATE_KEYS)
    if problems:
        return problems
    expected = aggregate_metrics(records)
    wrong = [
        f"{key} is {show(aggregate[key])} in the file but {expected[key]!r} from its records"
        for key in AGGREGATE_KEYS
        if not _is_number(aggregate[key]) or aggregate[key] != expected[key]
    ]
    return [f"aggregate does not match records: {'; '.join(wrong)}"] if wrong else []


def _build(doc: Mapping[str, Any]) -> Results:
    """`Results` from a document `_validate` found no fault with."""
    pipeline = doc["pipeline"]
    return Results(
        arm=doc["arm"],
        suite=SuiteIdentity(**{key: doc["suite"][key] for key in SUITE_KEYS}),
        trelix_version=doc["trelix_version"],
        embedder=EmbedderInfo(**{key: doc["embedder"][key] for key in EMBEDDER_KEYS}),
        pipeline=PipelineInfo(
            **{key: pipeline[key] for key in PIPELINE_KEYS},
        ),
        records=tuple(_record_from(item) for item in doc["records"]),
        created_at=doc["run"]["created_at"],
    )
