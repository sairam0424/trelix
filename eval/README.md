# Evaluation fixtures

Golden sets consumed by `trelix eval` and `trelix eval-synthesis`.

## `golden.jsonl` — retrieval quality

One JSON object per line:

```json
{"query": "how are files excluded from indexing", "relevant_files": ["src/trelix/indexing/walker.py"]}
```

| Field | Meaning |
|---|---|
| `query` | Natural-language question, phrased the way a developer would ask it |
| `relevant_files` | Repo-relative POSIX paths whose contents answer the query |
| `area` | Optional. Usually supplied by `golden-metadata.json` instead — see below |

Run it with:

```bash
trelix eval . --golden eval/golden.jsonl
```

Reports nDCG@10, Recall@10 and MRR over the top-10 **distinct files**.

### Freeze the plans, or the score is not repeatable

**The LLM query planner is the variance source, and freezing it is the only fix.**
Measured on this golden set: nDCG@10 run-to-run **sd 0.02202** (live planner, both
caches off, n=5) and **sd 0.02872** (shipped CLI, n=3), against **sd exactly
0.000000** for the same pipeline replayed from frozen plans — 0.6332934265749293 on
all six runs. The cause is direct: **0 of 54 plans reproduce byte-for-byte** at
`temperature=0.0`, with `bm25_tokens` differing on 53-54 of 54 queries and
`semantic_query` on 39-50 of 54. Different plans mean different `bm25_tokens`,
different embedded text and different grep hints, so 22-40 of 54 per-query scores
move between two runs of an identical configuration.

Record the plans once, then replay them:

```bash
# First run draws every plan and writes one JSONL record per distinct query.
trelix eval . --golden eval/golden.jsonl --plan-cache-file /tmp/plans.jsonl
wc -l /tmp/plans.jsonl        # expect 54 — one line per query in this golden set

# Every later run replays them: no planner LLM call, byte-identical plans.
trelix eval . --golden eval/golden.jsonl --plan-cache-file /tmp/plans.jsonl

# Same mechanism without the flag, for any caller that builds a RetrievalConfig:
TRELIX_RETRIEVAL_PLAN_CACHE_FILE=/tmp/plans.jsonl trelix eval . --golden eval/golden.jsonl
```

A query the file does not contain **raises** rather than drawing a fresh plan. That is
deliberate: a cache that silently re-draws on a miss leaves the run half frozen and
half re-planned while still looking frozen. Delete the file to re-record it, and
re-record whenever the golden set changes.

`TRELIX_RETRIEVAL_PLAN_SEED` forwards a provider sampling seed where the backend
supports one. It narrows the drift; it does not remove it, and `temperature=0.0`
already demonstrates that a sampling control is not a reproducibility guarantee. Use
the plan cache for any number you intend to compare.

**The two in-memory caches are not the control, and never were.** `plan_cache_size`
(default 128) keys the `QueryPlan` on `query.strip().lower()`, and all 54 queries in
`golden.jsonl` are distinct under exactly that key — so a pass has **zero** possible
plan-cache hits and `TRELIX_RETRIEVAL_PLAN_CACHE_SIZE=0` is a strict no-op here.
`query_cache_size` (default 256) keys `embed_query` on the text handed to it, which is
the plan's `semantic_query` or `hyde_snippet`. Both caches return the value computed
for that exact key, so a hit is indistinguishable from a recompute: a cache cannot
make a score move. What they do change is **repeating a query inside one process** —
`EvalHarness` builds one `Retriever` and reuses it, so a repeat loop replays the first
draw and reports a stability it never sampled. Three identical in-process ranks were
one sample echoed twice, while the same query in three separate processes put
`scripts/self-index.sh` at rank 2, 8 and 13 (`docs/reports/self-index-v3.1.2.md`).
Repeat across processes — or freeze the plans and stop needing to.

Earlier revisions of this file named HyDE's LLM rewrite as the underlying source.
Disabling HyDE moves nDCG@10 by **−0.000033**, i.e. wrong by a factor of at least 660,
and pointed every reader at a flag instead of at the planner.

### What the harness refuses outright

Both of the traps this file used to warn about are now errors, raised before any query
runs and listing every offending line at once:

- **An entry with an empty or missing `relevant_files`.** It used to be skipped, and the
  metrics are means over the entries that were *scored* — so breaking the fixture raised
  the score. Measured on a two-entry set where one query is answered perfectly and one
  is missed: intact it reports 0.5 on all three metrics, and emptying `relevant_files`
  on the missed query reports 1.0 with `Queries evaluated` 2 → 1. Scoring such an entry
  0.0 would be the mirror-image lie: nDCG, recall and MRR are undefined without ground
  truth, so a 0.0 reports a retrieval failure that never happened.
- **A path that is not normalised, repo-relative and POSIX.** Matching is exact string
  equality against each result's `rel_path`, which `FileWalker` builds as
  `str(path.relative_to(repo_root))`. A `./` prefix, a leading `/`, a `\` separator, a
  `..` component or a `//` scored 0, indistinguishable from a genuine miss; the harness
  now names the line and suggests the normalised form. It is *not* normalised silently —
  that would hide the fixture bug and could not fix the sibling failure below.
- **An empty golden file, and a line that is not a JSON object.**

`Queries evaluated` therefore equals this file's non-blank line count on any unfiltered
run; it is smaller only when you asked for a subset (next section).

**Still not checked: whether the paths exist.** A path that is well-formed but stale
(renamed or deleted file) scores 0 and looks like a retrieval miss. Verify before adding
an entry:

```bash
python - <<'PY'
import json, pathlib
for n, line in enumerate(open("eval/golden.jsonl"), 1):
    for p in json.loads(line)["relevant_files"]:
        if not pathlib.Path(p).exists():
            print(f"line {n}: missing {p}")
PY
```

### Scoring one area

`golden-metadata.json` labels every query with an `area` — `indexing`, `retrieval`,
`storage`, `llm-embed`, `cli-config-graph` (10 each) and `ops` (4). An aggregate over
all 54 hides a collapse confined to one of them, so `EvalHarness.run()` takes `area=`
and `limit=`:

```python
from trelix.core.config import IndexConfig
from trelix.eval.harness import EvalHarness

harness = EvalHarness(IndexConfig(repo_path="."))
harness.run("eval/golden.jsonl", area="storage")   # 10 queries
harness.run("eval/golden.jsonl", limit=5)          # first 5 lines, for a smoke run
```

Areas come from the sibling `<stem>-metadata.json` matched on query text, or from a
per-line `"area"` key which wins over the sidecar. An unknown area, or a file where only
some lines carry one, is an error rather than a quietly smaller run. `trelix eval` does
not expose these yet — from the CLI you get all 54.

Do not tune against `ops`: 4 queries make Recall@10 move in 0.25 steps, and it measured
0.00/0.50/0.75 across identical configs (`golden-metadata.json`, `notes`). Grow it to
~20 first.

### Metrics count files, not chunks

Retrieval ranks *chunks*, and one file supplies many of them — this repository averages
roughly 77 chunks per file. Ground truth here is file-level, so the metric functions
collapse repeats to each file's best rank before scoring. `@10` therefore means ten
distinct files.

Before v3.1.2 they did not: a relevant file appearing five times in the top ten scored
`recall@10 = 5.0` and `nDCG@10 = 2.52`, against a documented range of `[0, 1]`. **Scores
produced before v3.1.2 are not comparable with scores produced after it.**

### What a difference has to be before it means anything

With a **live planner** — i.e. no `--plan-cache-file` and no
`TRELIX_RETRIEVAL_PLAN_CACHE_FILE` — this is the noise floor of the instrument on the
54-query set:

| Quantity | Value |
|---|---|
| run-to-run sd on nDCG@10 (`sd_d`) | 0.022 (planner live, caches off, n=5) — 0.029 (shipped CLI, n=3) |
| one-run 95% detection band | ±0.061 |
| MDD at 80% power, one run per arm | 0.087 – 0.114 |
| passes per arm to resolve 0.01 | ~105 – 113 |

So a single-run delta below ~0.087 is not evidence. Two readings of the same
configuration are not a range and do not establish a level: two values 0.0028 apart
differ by 4.6% of the detection band. Quote **N**, a confidence interval, and the MDD
at that N, or quote nothing. And "the interval includes 0" is not an equivalence
result — an equivalence margin has to be chosen before the run, not read off it.

Two consecutive live-planner runs over an identical index measured mean precision@10 of
94.0% and 95.0%; that 1-point spread is this floor, not a change in retrieval.

With the plan cache in place the floor is **sd 0.000000** over six runs, so the honest
sequence is: freeze the plans first, then measure. `TRELIX_RETRIEVAL_HYDE_FALLBACK` and
`TRELIX_RETRIEVAL_FLARE` add LLM calls at retrieval time and are worth disabling for
speed, but disabling HyDE moves nDCG@10 by −0.000033 and is not what makes a run
repeatable.

## `golden_synthesis_sample.jsonl` — synthesis quality

Input to `trelix eval-synthesis`, which scores answer faithfulness and completeness
GroUSE-style rather than measuring retrieval.

## Related tooling

`scripts/measure_index_hygiene.py` answers a different question from `trelix eval`: not
"did we rank the right files" but "how much of what we ranked was never source code to
begin with". It needs no golden set, so it works on any index:

```bash
python scripts/measure_index_hygiene.py . --json > docs/reports/index-hygiene-after.json
```
