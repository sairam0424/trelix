"""
EvalHarness — run a golden JSONL file through trelix retrieval and report metrics.

Golden file format (one JSON object per line):
    {"query": "how does JWT auth work", "relevant_files": ["src/auth.py"]}

`relevant_files` are repo-relative POSIX paths, compared by exact string equality
against each result's `rel_path` — which `FileWalker` builds as
`str(path.relative_to(repo_root))`, so no `./` prefix and no absolute paths. An entry
that cannot be scored against that, or has no ground truth at all, is refused before
any query runs rather than skipped or scored 0; see `_parse_golden`.

An optional `"area"` key per line (or a sibling `<stem>-metadata.json` carrying
`queries: [{query, area}]`, which is how `eval/golden.jsonl` labels its 54 queries)
makes `run(..., area=...)` able to score one area at a time.

Usage:
    harness = EvalHarness(config)
    metrics = harness.run("golden.jsonl")
    # -> {"ndcg@10": 0.74, "recall@10": 0.81, "mrr": 0.66, "n_queries": 12}
    metrics = harness.run("eval/golden.jsonl", area="storage", limit=5)
"""

from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from trelix.core.config import IndexConfig
from trelix.core.models import RerankOutcome
from trelix.eval.ndcg import mrr, ndcg_at_k, recall_at_k

logger = logging.getLogger("trelix.eval")

_METADATA_SUFFIX = "-metadata.json"


@dataclass(frozen=True)
class _GoldenEntry:
    """One validated golden line. `line_no` is 1-based, to match an editor."""

    line_no: int
    query: str
    relevant_files: frozenset[str]
    area: str | None


def _path_problem(rel_path: object) -> str | None:
    """Describe why `rel_path` can never match a `rel_path` from retrieval, or None.

    Matching is exact string equality, so `./a.py` and `a.py` are different files as
    far as the harness is concerned: the query scores 0 and looks exactly like a
    genuine retrieval miss. Normalising here would paper over the fixture bug (and
    could not touch the sibling failure — a normalised path that is simply wrong), so
    a non-normalised path is reported instead.
    """
    if not isinstance(rel_path, str) or not rel_path.strip():
        return "must be a non-empty string"
    if "\\" in rel_path:
        return f"{rel_path!r} must use POSIX '/' separators"
    if rel_path.startswith("/"):
        return f"{rel_path!r} must be repo-relative, not absolute"
    if ".." in PurePosixPath(rel_path).parts:
        return f"{rel_path!r} must not contain '..'"
    normalised = posixpath.normpath(rel_path)
    if normalised != rel_path:
        return f"{rel_path!r} is not normalised (did you mean {normalised!r}?)"
    return None


def _sidecar_areas(path: Path) -> dict[str, str]:
    """Map query -> area from `<stem>-metadata.json` beside the golden file, if any."""
    sidecar = path.with_name(f"{path.stem}{_METADATA_SUFFIX}")
    if not sidecar.exists():
        return {}
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sidecar}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    queries = meta.get("queries", []) if isinstance(meta, dict) else []
    return {
        entry["query"]: entry["area"]
        for entry in queries
        if isinstance(entry, dict)
        and isinstance(entry.get("query"), str)
        and isinstance(entry.get("area"), str)
    }


def _parse_golden(path: Path) -> list[_GoldenEntry]:
    """Parse and validate every line, raising once with every problem found.

    Refusing rather than skipping is the point. The reported metrics are means over
    the entries that were scored, so an entry dropped for being unusable raises the
    score of a broken golden file: on a two-entry set where one query is answered
    perfectly and one is missed, emptying `relevant_files` on the missed one moved all
    three metrics from 0.5 to 1.0 and `n_queries` from 2 to 1. Scoring it 0.0 instead
    would be the mirror-image lie — nDCG, recall and MRR are undefined without ground
    truth, so a 0.0 would report a retrieval failure that never happened.
    """
    problems: list[str] = []
    entries: list[_GoldenEntry] = []
    areas = _sidecar_areas(path)

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_no}: not valid JSON ({exc.msg} at column {exc.colno})")
            continue
        if not isinstance(item, dict):
            problems.append(f"line {line_no}: expected a JSON object, got {type(item).__name__}")
            continue

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            problems.append(f'line {line_no}: "query" must be a non-empty string')
            continue

        relevant = item.get("relevant_files")
        if not isinstance(relevant, list) or not relevant:
            problems.append(
                f'line {line_no}: "relevant_files" must be a non-empty list of '
                "repo-relative POSIX paths — an entry with no ground truth cannot be "
                "scored, and skipping it would raise the reported mean"
            )
            continue
        path_problems = [p for p in (_path_problem(f) for f in relevant) if p is not None]
        if path_problems:
            problems.extend(f'line {line_no}: "relevant_files" {p}' for p in path_problems)
            continue

        area = item.get("area")
        entries.append(
            _GoldenEntry(
                line_no=line_no,
                query=query,
                relevant_files=frozenset(relevant),
                area=area if isinstance(area, str) and area.strip() else areas.get(query),
            )
        )

    if problems:
        raise ValueError(
            f"{path}: {len(problems)} unusable golden entr"
            f"{'y' if len(problems) == 1 else 'ies'} — refused rather than skipped, "
            "because skipping shrinks the denominator and so raises the score:\n  "
            + "\n  ".join(problems)
        )
    if not entries:
        raise ValueError(
            f"{path}: no golden entries — reporting 0.0 for an empty set is "
            "indistinguishable from total retrieval failure"
        )
    return entries


def _select_area(entries: list[_GoldenEntry], area: str, path: Path) -> list[_GoldenEntry]:
    present = sorted({e.area for e in entries if e.area})
    if not present:
        raise ValueError(
            f"area={area!r} requested but {path} carries no area labels — add an "
            f'"area" key per line, or a sibling {path.stem}{_METADATA_SUFFIX} with '
            '"queries": [{"query": ..., "area": ...}]'
        )
    unlabelled = [e.line_no for e in entries if not e.area]
    if unlabelled:
        raise ValueError(
            f"{path}: {len(unlabelled)} entries carry no area and could never be "
            f"selected by area={area!r}, which would silently shrink every per-area "
            f"run: line{'' if len(unlabelled) == 1 else 's'} "
            + ", ".join(str(n) for n in unlabelled)
        )
    selected = [e for e in entries if e.area == area]
    if not selected:
        raise ValueError(f"no golden entries in area {area!r}; areas present: {', '.join(present)}")
    return selected


class EvalHarness:
    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        from trelix.retrieval.retriever import Retriever

        self._retriever = Retriever(config)
        # One entry per golden entry processed by the most recent run(): what the rerank
        # stage actually did. Set once at the end of run() rather than appended to a
        # shared field, so a caller reading it can never observe a half-filled run.
        self._rerank_outcomes: tuple[RerankOutcome | None, ...] = ()

    @property
    def rerank_outcomes(self) -> tuple[RerankOutcome | None, ...]:
        """Per-query rerank outcomes from the last run(). `None` = not recorded.

        `None` means either the rerank stage was never entered (disabled by config, or
        the planner's strategy skipped it) or the query itself failed before retrieval
        returned. Both are genuinely "no verdict", which is why they share a value.
        """
        return self._rerank_outcomes

    def rerank_summary(self) -> str:
        """One line describing the rerank pipeline the last run() actually used.

        Deliberately does NOT collapse disagreement. If three queries reranked and seven
        fell back on a transient API failure, the reported score is a blend of two
        pipelines, and a summary that said "cohere" would hide exactly the thing that
        makes such a number untrustworthy.
        """
        recorded = [o for o in self._rerank_outcomes if o is not None]
        if not recorded:
            return "disabled" if self._rerank_outcomes else "not run"

        described = {o.describe() for o in recorded}
        missing = len(self._rerank_outcomes) - len(recorded)
        parts = [next(iter(described))] if len(described) == 1 else []
        if not parts:
            applied = sum(1 for o in recorded if o.applied)
            detail = ", ".join(sorted(described))
            parts = [f"MIXED across queries — {applied}/{len(recorded)} applied: {detail}"]
        if missing:
            parts.append(f"+{missing} query(s) with no verdict")
        return "; ".join(parts)

    def run(
        self,
        golden_path: str,
        *,
        area: str | None = None,
        limit: int | None = None,
    ) -> dict[str, float]:
        """
        Run the golden file's queries and return aggregate metrics.

        Args:
            golden_path: JSONL golden set. Every line is validated first; one
                unusable line refuses the whole run (see `_parse_golden`).
            area: score only this area. Areas come from a per-line `"area"` key or
                the sibling `<stem>-metadata.json`. Unknown or unlabelled -> raises.
            limit: score only the first N entries after the area filter, in file
                order.

        Returns dict with keys: ndcg@10, recall@10, mrr, n_queries. `area`/`limit`
        shrink the denominator deliberately, and `n_queries` reports what was
        actually scored — always check it against the run you meant to make.

        Two runs of this method are NOT two samples, for two different reasons that
        used to be conflated:

        * In one process, `query_cache_size` (256) and `plan_cache_size` (128) memoise
          `embed_query` and the `QueryPlan` for the life of the Retriever, so a second
          call replays the first's plan and embedding. That makes an in-process repeat
          report a stability it never sampled — but it cannot move a score, because
          both caches are keyed on the exact text they computed from.
        * Across processes, the LLM planner re-plans every query: 0 of 54 golden plans
          reproduce byte-for-byte at temperature=0.0, which put nDCG@10 at sd
          0.022-0.029 between identical configurations. Zeroing the caches does not
          touch this, and on a golden set whose queries are all distinct under
          `query.strip().lower()` it cannot even produce a cache hit to zero.

        Set `RetrievalConfig.plan_cache_file` (`TRELIX_RETRIEVAL_PLAN_CACHE_FILE`, or
        `trelix eval --plan-cache-file`) to record the plans once and replay them: the
        same pipeline then reproduced nDCG@10 at sd exactly 0.000000 over six runs.
        See `eval/README.md`.
        """
        from trelix.retrieval.planner.agent import PlanCacheMissError

        path = Path(golden_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden file not found: {golden_path}")

        entries = _parse_golden(path)
        if area is not None:
            entries = _select_area(entries, area, path)
        if limit is not None:
            if limit < 1:
                raise ValueError(f"limit must be >= 1, got {limit}")
            entries = entries[:limit]

        ndcg_scores: list[float] = []
        recall_scores: list[float] = []
        mrr_scores: list[float] = []
        rerank_outcomes: list[RerankOutcome | None] = []

        for entry in entries:
            try:
                ctx = self._retriever.retrieve(entry.query)
            except PlanCacheMissError:
                # Never scored as a miss. A frozen-plan run whose cache does not cover
                # the golden set must fail loudly: scoring 0.0 here would report a
                # retrieval failure that never happened AND hand back a mean computed
                # over a mixture of replayed and unplanned queries, which is precisely
                # the "looks reproducible, is not" number the freeze exists to prevent.
                raise
            except Exception as exc:
                logger.warning("Query %r failed: %s", entry.query[:60], exc)
                ndcg_scores.append(0.0)
                recall_scores.append(0.0)
                mrr_scores.append(0.0)
                # No retrieval, so no verdict — recorded to keep this list the same
                # length as the score lists, so "how many queries had no verdict"
                # stays answerable.
                rerank_outcomes.append(None)
                continue

            rerank_outcomes.append(ctx.rerank)

            # Use file rel_path as the ID for matching. Retrieval ranks CHUNKS, so the
            # same file legitimately appears several times here. Ground truth is
            # file-level, so each file must score once: the metric functions collapse
            # repeats to their best rank (see `_dedupe` in eval/ndcg.py), which is what
            # makes `@10` mean ten distinct files rather than ten chunks.
            ranked_files = [r.file.rel_path for r in ctx.results]
            # Convert to integer IDs for metric functions (hash-based)
            file_to_id = {f: i for i, f in enumerate(set(ranked_files) | set(entry.relevant_files))}
            ranked_ids = [file_to_id[f] for f in ranked_files]
            relevant_ids = {file_to_id[f] for f in entry.relevant_files if f in file_to_id}

            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant_ids, k=10))
            recall_scores.append(recall_at_k(ranked_ids, relevant_ids, k=10))
            mrr_scores.append(mrr(ranked_ids, relevant_ids))

        n = len(ndcg_scores)
        # Single assignment at the end: a caller reading rerank_outcomes mid-run would
        # otherwise see a partial picture and could describe the wrong pipeline.
        self._rerank_outcomes = tuple(rerank_outcomes)
        return {
            "ndcg@10": sum(ndcg_scores) / n,
            "recall@10": sum(recall_scores) / n,
            "mrr": sum(mrr_scores) / n,
            "n_queries": float(n),
        }
