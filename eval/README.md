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

### The remaining way to get a silently wrong score

**Repeating a query in one process measures it once.** `query_cache_size` (default 256)
memoises `embed_query`, and `plan_cache_size` (default 128) memoises the `QueryPlan` —
both keyed on `query.strip().lower()`, both living as long as the `Retriever`, and
`EvalHarness` builds one `Retriever` and reuses it for every query. A repeat loop
therefore replays the first run's plan (HyDE snippet included) and the first run's
embedding, and reports a stability it never measured: three identical ranks in-process
were one sample echoed twice, while the same query in three separate processes put
`scripts/self-index.sh` at rank 2, 8 and 13 (`docs/reports/self-index-v3.1.2.md`).
Repeat across processes, or set both `TRELIX_RETRIEVAL_QUERY_CACHE_SIZE=0` and
`TRELIX_RETRIEVAL_PLAN_CACHE_SIZE=0`. HyDE's LLM rewrite is the underlying source of
that variance, and has to be disabled rather than averaged over — see "Scores move
between runs".

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

### Scores move between runs

Retrieval is not deterministic when the LLM-dependent legs are on. With
`TRELIX_RETRIEVAL_HYDE_FALLBACK` and `TRELIX_RETRIEVAL_FLARE` enabled, two consecutive
runs over an identical index measured mean precision@10 of 94.0% and 95.0%. Treat small
differences as noise, and disable those flags when you need a repeatable number.

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
