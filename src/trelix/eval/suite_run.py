"""Run one arm of a suite: index the verified clone once, replay the frozen plans, write results.

`trelix eval-suite SUITE.json --arm NAME --out results.json` is `prepare_suite` (the files, the
hashes, the pinned clone, the gold paths) followed by, in this order:

0. `--out` is checked before anything is cloned: its directory exists and is writable, and it
   is not a directory itself. `write_results` would otherwise report a missing directory only
   after the index was built.
1. The run directory `<cache>/arms/<repo sha>/<name>/<arm>/` is created with `exist_ok=False`
   and is the claim on the arm. One that exists is refused: an index built by other trelix code
   (a candidate branch with the same version string, a changed chunker) would otherwise be
   measured as if it were current. There is no reuse option and eval-suite deletes nothing;
   one index per arm is the only shape until sharing is designed with its cache key.
2. The verified golden and plans bytes are copied into the run directory, and THOSE copies are
   what the harness reads: the committed files are never written to, and the bytes replayed
   are the bytes hashed. The copy of the golden file has no `<stem>-metadata.json` sidecar
   beside it, so no `area` label from an unhashed file reaches a record.
3. The configuration is `IndexConfig(repo_path=<clone>)` with the settings that change the
   index or the ranking forced to literal values (`run_config`), whatever the operator's
   environment or `~/.config/trelix/env` says: no file summaries, no batch API, the local
   embedder, a walk confined to the clone, no contextual chunking, the sqlite store in the run
   directory, no rerank, HyDE, multi-query or FLARE, and the plans copy as the plan cache. The
   rest of the effective configuration is recorded in `pipeline.config` of the results file,
   so `eval-compare` can show what two arms really differed in.
4. The index is built and the queries run under `isolated_git`, so the Indexer's own git child
   in `trelix.store.provenance` sees the same switched-off operator configuration the clone did.
   An index with any error is refused and no results are written. A golden query the frozen
   plans do not cover at run time (`PlanCacheMissError`) stops the run the same way; it is never
   scored as a miss. So does an index that cannot be built or opened at all (an `OSError`, or the
   `ImportError` of a missing `local` extra).
5. `results.json` (`trelix.eval.results`, schema 1) is written even when queries raised, so the
   file names every failed query; the command then exits 1.

Every refusal after step 1 ends with one more line saying that the claimed run directory is left
as it is and must be deleted before the arm is run again: eval-suite deletes nothing, and the
next run of the same `--arm` would otherwise be refused with no earlier hint.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from trelix import __version__
from trelix.core.config import IndexConfig
from trelix.eval._loading import clip, read_bytes_capped
from trelix.eval.harness import EvalHarness, QueryRecord
from trelix.eval.results import (
    ARM_PATTERN,
    FROZEN_FLAGS,
    ResultsError,
    _utc_now,
    build_results,
    write_results,
)
from trelix.eval.suite import MAX_DATA_FILE_BYTES, SuiteError, SuiteSpec, load_suite
from trelix.eval.suite_git import isolated_git
from trelix.eval.suite_prepare import PreparedSuite, cache_root, prepare_spec
from trelix.retrieval.planner.agent import PlanCacheMissError

_ARM_RE = re.compile(ARM_PATTERN)

# The sections of the configuration recorded in `pipeline.config`, in this order. `embedder` is
# recorded as `embedder` (provider, model, dimension, library version) and `llm`, `git_linker`
# and `image` play no part in indexing or retrieval.
CONFIG_SECTIONS = ("walker", "parser", "chunker", "store", "retrieval", "indexer", "sparse")
# A field whose name contains one of these is left out of the record: a results file is a pull
# request artifact, and a URL or URI (`store.qdrant_url`, `store.lance_uri`; neither backend is
# used by a run) may carry credentials. The rule also drops token counts such as `top_k_tokens`.
SECRET_WORDS = ("key", "secret", "token", "password", "endpoint", "url", "uri")
# Exact names the word rule would drop that are kept: `max_tokens_per_chunk` sets the chunk size,
# which changes the index, and the record exists to show what two arms differed in.
KEPT_FIELDS = frozenset({"max_tokens_per_chunk"})
# Fields that differ between arms by construction and say nothing about the pipeline.
_PER_ARM_FIELDS: Mapping[str, tuple[str, ...]] = {
    "store": ("db_path",),
    "retrieval": ("plan_cache_file",),
}
_EMBEDDER_LIBRARY = "sentence-transformers"


@dataclass(frozen=True)
class IndexOutcome:
    """What a run needs to know about the index it built."""

    errors: int
    dimension: int


@dataclass(frozen=True)
class SuiteRun:
    """One finished arm: where it ran, what it wrote, and the records as scored."""

    prepared: PreparedSuite
    arm: str
    run_dir: Path
    out: Path
    dimension: int
    records: tuple[QueryRecord, ...]
    aggregate: Mapping[str, float]
    rerank_summary: str


# ---------------------------------------------------------------------------
# the checks that run before the clone and the claim on the arm
# ---------------------------------------------------------------------------


def out_problems(out: Path) -> list[str]:
    """Why `--out` cannot be written, found before anything is cloned or indexed; `[]` if it can."""
    if out.is_dir():
        return [f"--out {out} is a directory: name the results file itself"]
    parent = out.parent
    if not parent.is_dir():
        return [f"--out: the directory {parent} does not exist; create it before the run"]
    if not os.access(parent, os.W_OK):
        return [f"--out: cannot write in {parent}"]
    return []


def _arm_problems(arm: str) -> list[str]:
    if _ARM_RE.fullmatch(arm):
        return []
    return [f"--arm must match {ARM_PATTERN} (got {clip(repr(arm))})"]


def run_dir_path(root: Path, spec: SuiteSpec, arm: str) -> Path:
    """Where the index and the input copies of `arm` live under the cache root `root`."""
    return root / "arms" / spec.repo_sha / spec.name / arm


def _claim_run_dir(run_dir: Path, arm: str) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SuiteError(
            [
                f"arm {arm!r} already has a run directory at {run_dir}: choose another --arm "
                "or delete it"
            ]
        ) from exc
    except OSError as exc:
        raise SuiteError([f"cannot create the run directory {run_dir}: {exc}"]) from exc


def _verified_inputs(spec: SuiteSpec) -> dict[str, bytes]:
    """The golden and plans bytes, each re-checked against its hash as it is read now."""
    inputs: dict[str, bytes] = {}
    for what, source, expected in (
        ("golden", spec.golden_file, spec.golden_sha256),
        ("plans", spec.plans_file, spec.plans_sha256),
    ):
        try:
            data = read_bytes_capped(source, MAX_DATA_FILE_BYTES)
        except (OSError, ValueError) as exc:
            raise SuiteError([f"{what}: cannot read {source}: {clip(str(exc))}"]) from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise SuiteError(
                [f"{what}: sha256 of {source.name} is now {actual}, not {expected}: it changed"]
            )
        inputs[what] = data
    return inputs


# ---------------------------------------------------------------------------
# the configuration of a run
# ---------------------------------------------------------------------------


def run_config(clone: Path, run_dir: Path) -> IndexConfig:
    """The configuration every arm indexes and retrieves with: the ambient one, forced where it
    would change the index or the ranking. `model_copy(update=)` does not validate, so every
    value here is a literal of the right type."""
    try:
        base = IndexConfig(repo_path=str(clone))
    except ValueError as exc:  # pydantic's ValidationError is one
        raise SuiteError([f"cannot build the index configuration: {clip(str(exc), 200)}"]) from exc
    return base.model_copy(
        update={
            "file_summaries_enabled": False,
            "use_batch_api": False,
            "embedder": base.embedder.model_copy(update={"provider": "local"}),
            "walker": base.walker.model_copy(update={"follow_symlinks": False}),
            "chunker": base.chunker.model_copy(update={"contextual": False}),
            "store": base.store.model_copy(
                update={"backend": "sqlite", "db_path": str(run_dir / "index.db")}
            ),
            "retrieval": base.retrieval.model_copy(
                update={
                    "rerank": False,
                    "hyde_fallback_enabled": False,
                    "multi_query_enabled": False,
                    "flare_enabled": False,
                    "plan_cache_file": run_dir / "plans.jsonl",
                }
            ),
        }
    )


def strip_secret_fields(section: Mapping[str, Any]) -> dict[str, Any]:
    """`section` without every field whose name contains a word of `SECRET_WORDS`, except the
    exact names in `KEPT_FIELDS`. `lower()` is defensive: every pydantic field name here is
    lowercase, so a case-sensitive match would behave the same today."""
    return {
        name: value
        for name, value in section.items()
        if name in KEPT_FIELDS or not any(word in name.lower() for word in SECRET_WORDS)
    }


def pipeline_config(config: IndexConfig) -> dict[str, dict[str, Any]]:
    """The effective configuration of the run, section by section, for `pipeline.config`."""
    recorded: dict[str, dict[str, Any]] = {}
    for name in CONFIG_SECTIONS:
        section = strip_secret_fields(getattr(config, name).model_dump(mode="json"))
        for field in _PER_ARM_FIELDS.get(name, ()):
            section.pop(field, None)
        recorded[name] = section
    return recorded


def library_version() -> str | None:
    """The installed version of the embedder library, or None when it is not installed."""
    try:
        return metadata.version(_EMBEDDER_LIBRARY)
    except metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def index_repo(config: IndexConfig) -> IndexOutcome:
    """Build the index of `config` with the real `Indexer`, quietly: the default `index_fn`."""
    from trelix.indexing.indexer import Indexer

    indexer = Indexer(config, quiet=True)
    stats = indexer.index()
    return IndexOutcome(errors=int(stats["errors"]), dimension=indexer.embedder.dimension)


def _refuse_index_errors(outcome: IndexOutcome, clone: Path) -> None:
    if outcome.errors == 0:
        return
    raise SuiteError(
        [
            f"indexing {clone} reported {outcome.errors} error(s): a run with a parse or write "
            "error is not a measurement, so no results were written"
        ]
    )


def _left_behind(run_dir: Path) -> str:
    """The last line of every refusal after the claim on the arm."""
    return f"the run directory {run_dir} is left as it is; delete it before running this arm again"


def _copy_inputs(run_dir: Path, inputs: Mapping[str, bytes]) -> None:
    """Write the verified golden and plans bytes into the run directory, as `<what>.jsonl`."""
    for what, data in inputs.items():
        target = run_dir / f"{what}.jsonl"
        try:
            target.write_bytes(data)
        except OSError as exc:
            raise SuiteError([f"cannot write {target}: {exc}"]) from exc


def results_document(
    spec: SuiteSpec,
    *,
    arm: str,
    config: IndexConfig,
    outcome: IndexOutcome,
    rerank_summary: str,
    records: Sequence[QueryRecord],
    now: Callable[[], datetime],
) -> dict[str, Any]:
    """The schema 1 document of one arm (`trelix.eval.results.build_results`)."""
    return build_results(
        arm=arm,
        suite={
            "name": spec.name,
            "repo_url": spec.repo_url,
            "repo_sha": spec.repo_sha,
            "license": spec.license,
            "golden_version": spec.golden_version,
            "golden_sha256": spec.golden_sha256,
            "plans_sha256": spec.plans_sha256,
        },
        trelix_version=__version__,
        embedder={
            "provider": config.embedder.provider,
            "model": config.embedder.local_model,
            "dimension": outcome.dimension,
            "library_version": library_version(),
        },
        pipeline={
            **{flag: getattr(config.retrieval, flag) for flag in FROZEN_FLAGS},
            "plans": "replayed",
            "rerank_summary": rerank_summary,
            "config": pipeline_config(config),
        },
        records=records,
        now=now,
    )


def run_suite(
    spec: SuiteSpec,
    *,
    arm: str,
    cache_root: Path,
    out: Path,
    allow_local: bool = False,
    index_fn: Callable[[IndexConfig], IndexOutcome] = index_repo,
    harness_factory: Callable[[IndexConfig], EvalHarness] = EvalHarness,
    now: Callable[[], datetime] = _utc_now,
) -> SuiteRun:
    """Run `arm` of `spec` under `cache_root` and write its results to `out`.

    Raises SuiteError for every refusal (nothing written); a `PlanCacheMissError` from the
    harness, and an `OSError` or `ImportError` while the index is built or opened, are refusals
    too. Once the run directory is claimed, every refusal also names it as left behind.
    `index_fn`, `harness_factory` and `now` are seams for the tests; `allow_local` lets
    `repo.url` be a local repository, for the tests only.
    """
    problems = _arm_problems(arm) + out_problems(out)
    if problems:
        raise SuiteError(problems)
    prepared = prepare_spec(spec, cache_root, allow_local=allow_local)
    inputs = _verified_inputs(spec)
    run_dir = run_dir_path(cache_root, spec, arm)
    _claim_run_dir(run_dir, arm)
    try:
        return _run_claimed(
            spec,
            prepared,
            inputs,
            run_dir,
            arm=arm,
            out=out,
            index_fn=index_fn,
            harness_factory=harness_factory,
            now=now,
        )
    except SuiteError as exc:
        raise SuiteError([*exc.problems, _left_behind(run_dir)]) from exc
    except PlanCacheMissError as exc:
        raise SuiteError(
            [
                "a golden query had no frozen plan at run time, so no results.json was "
                f"written: {exc}",
                _left_behind(run_dir),
            ]
        ) from exc
    except (OSError, ImportError) as exc:
        raise SuiteError(
            [f"cannot build or open the index: {clip(str(exc), 200)}", _left_behind(run_dir)]
        ) from exc


def _run_claimed(
    spec: SuiteSpec,
    prepared: PreparedSuite,
    inputs: Mapping[str, bytes],
    run_dir: Path,
    *,
    arm: str,
    out: Path,
    index_fn: Callable[[IndexConfig], IndexOutcome],
    harness_factory: Callable[[IndexConfig], EvalHarness],
    now: Callable[[], datetime],
) -> SuiteRun:
    """Steps 2 to 5 of the run, inside the claimed `run_dir`."""
    _copy_inputs(run_dir, inputs)
    config = run_config(prepared.clone, run_dir)
    with isolated_git():
        outcome = index_fn(config)
        _refuse_index_errors(outcome, prepared.clone)
        harness = harness_factory(config)
        records = harness.run_detailed(str(run_dir / "golden.jsonl"))
    doc = results_document(
        spec,
        arm=arm,
        config=config,
        outcome=outcome,
        rerank_summary=harness.rerank_summary(),
        records=records,
        now=now,
    )
    _write(out, doc)
    return SuiteRun(
        prepared=prepared,
        arm=arm,
        run_dir=run_dir,
        out=out,
        dimension=outcome.dimension,
        records=tuple(records),
        aggregate=doc["aggregate"],
        rerank_summary=doc["pipeline"]["rerank_summary"],
    )


def _write(out: Path, doc: Mapping[str, Any]) -> None:
    try:
        error = write_results(str(out), doc)
    except ResultsError as exc:
        raise SuiteError([f"the results document is invalid: {p}" for p in exc.problems]) from exc
    if error is not None:
        raise SuiteError([f"cannot write {out}: {error}"])


def run_suite_file(
    suite_file: Path | str,
    cache_dir: Path | str | None,
    *,
    arm: str,
    out: Path | str,
    allow_local: bool = False,
    index_fn: Callable[[IndexConfig], IndexOutcome] = index_repo,
    harness_factory: Callable[[IndexConfig], EvalHarness] = EvalHarness,
    now: Callable[[], datetime] = _utc_now,
) -> SuiteRun:
    """`run_suite` for the command line: load `suite_file`, resolve the cache root, run."""
    spec = load_suite(suite_file, allow_local=allow_local)
    return run_suite(
        spec,
        arm=arm,
        cache_root=cache_root(cache_dir),
        out=Path(out),
        allow_local=allow_local,
        index_fn=index_fn,
        harness_factory=harness_factory,
        now=now,
    )
