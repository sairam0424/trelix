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
| `id`, `kind`, `lang`, `split` | Optional labels for the query in the per-query output, with the allowed values in "Golden v2 and `trelix eval-validate`" below — see "Per-query results". Entries without an `id` are numbered by position |

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

**Still not checked by `trelix eval`: whether the paths exist.** A path that is well-formed
but stale (renamed or deleted file) scores 0 and looks like a retrieval miss. Verify before
adding an entry, and whenever files move:

```bash
trelix eval-validate eval/golden.jsonl --repo .
```

### Golden v2 and `trelix eval-validate`

Golden v2 adds optional keys to a v1 line; `query` and `relevant_files` stay exactly as
above. A file that uses none of them is a v1 file (`golden.jsonl` is one) and loads and
scores as it always did, and keys trelix does not know are still ignored.

| Field | Allowed values | Meaning |
|---|---|---|
| `id` | non-empty string, unique within the file | Label for the query |
| `lang` | non-empty string | Language of the code the query is about |
| `kind` | `nl`, `keyword`, `commit`, `issue` | How the query was phrased or found |
| `source` | non-empty string | Where the query came from |
| `gold_status` | `validated`, `pooled`, `unreviewed` | The result of checking the entry's `relevant_files` |
| `split` | `dev`, `test` | Which part of the file the query belongs to |

A field that is present with another type or value (a `kind` of `"code"`, an `id` of `""`,
any of them `null`) is refused like every other unusable entry: `trelix eval` names the line
and runs no query.

```bash
trelix eval-validate GOLDEN [--repo PATH] [--rev REV] [--min-per-stratum N] [--min-validated FRACTION]
```

runs no query and checks:

1. every line is schema-valid: the rules above, and the table;
2. no two queries are equal once stripped and case-folded, and no two `id`s are equal
   (an `id` is compared exactly: `A` and `a` are different ids, and an empty one is a
   violation of item 1, not a duplicate);
3. with `--repo`, every `relevant_files` path exists in that git repository at `--rev`
   (default `HEAD`), read with `git ls-tree`. Paths are relative to `--repo`, so a
   subdirectory of a checkout is checked against its own subtree. Without `--repo` this
   check is skipped and a note says so;
4. in a file where some entry has a `kind` or a `gold_status` (a v2 file), every kind that
   occurs has at least `--min-per-stratum` queries (default 20). An entry without a `kind`
   is in no stratum;
5. in a v2 file, at least `--min-validated` (default 0.95) of all entries have a
   `gold_status` of `validated` or `pooled`; an entry with no `gold_status` counts as
   `unreviewed`.

A file where no entry has a `kind` or a `gold_status` is checked as v1 (1 to 3 only), and
the command says so. `id`, `lang`, `source` and `split` alone do not make a v2 file.
Each violation is one line on stdout, `line N: ...` (in line order) or `file: ...` (after
those), followed by any notes and a summary line. The exit code is `0` with no violation and
`1` with any. A golden file that is missing or cannot be opened, a `--repo` that is not a git
repository and an unknown `--rev` print an error on stderr and also exit `1`; a threshold out
of range (`--min-per-stratum` below 0, `--min-validated` outside 0 to 1 or `nan`) is a usage
error, exit `2`. `eval-validate` reads the golden file only: a malformed
`<stem>-metadata.json` beside it, which `trelix eval` refuses, is not reported.

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

### Per-query results

`trelix eval` prints one mean per metric. To compare two runs query by query, add
`--per-query-out`:

```bash
trelix eval . --golden eval/golden.jsonl --plan-cache-file /tmp/plans.jsonl \
    --per-query-out /tmp/run-a.json
```

The table prints exactly as it does without the flag. The file is ASCII-escaped JSON (any
text round-trips) with sorted keys, two-space indentation, a trailing newline and mode 0600,
written to a temporary file beside it and renamed into place, so an interrupted run never
leaves a truncated one. The directory must exist and the path must name a file. With one
record shown:

```json
{
  "aggregate": {
    "mrr": 0.53,
    "n_queries": 54.0,
    "ndcg@10": 0.63,
    "recall@10": 0.81
  },
  "records": [
    {
      "error": null,
      "id": "q0001",
      "kind": null,
      "lang": null,
      "mrr": 1.0,
      "ndcg": 1.0,
      "recall": 1.0,
      "repo": "trelix",
      "split": null,
      "top10": [
        "src/trelix/indexing/walker.py"
      ]
    }
  ],
  "schema_version": 1
}
```

`aggregate` is what `EvalHarness.run()` returns, and `EvalHarness.run_detailed()` returns
the same records as `QueryRecord` objects. One record per scored query:

| Field | Meaning |
|---|---|
| `id` | The golden entry's `id` when it has one, else `q0001`, `q0002`, ... — its 1-based position among the golden entries, counted before `area`/`limit` narrow the run, so a query keeps its id |
| `repo` | Basename of the repository directory that was evaluated |
| `kind`, `lang`, `split` | The golden entry's string of that name, else `null` |
| `ndcg`, `recall`, `mrr` | The same three functions `run()` averages |
| `top10` | The first ten **distinct** files retrieved, in rank order — the files the metrics scored, since one file can supply several chunks |
| `error` | `null`, or the exception message (at most 200 characters) if retrieval raised |

`trelix eval` does not check ids for uniqueness; `trelix eval-validate` reports an explicit
`id` that two lines share. It does not see an explicit `q0001` on a later line repeating the
number given to an entry without an `id`, so keep explicit ids not shaped like `q0001`. The
check in the snippet below ("an id is used twice") catches that one too.

A query that raises is scored 0.0 on all three metrics (never as a hit), recorded with its
`error`, and does not stop the run. `trelix eval` then prints which queries failed and
**exits 1**: the means it printed blend failures into misses. `EvalHarness.run()` keeps
returning the mean.

### Comparing two runs

`trelix.eval.stats` (numpy only) takes per-query scores paired **by position**, so pair the
two files by `id` first. The snippet is for exploring two `--per-query-out` files; a decision
goes through `trelix eval-compare` (next section), which is the only thing that produces a PASS.

```python
import json
from trelix.eval.stats import holm, mde, paired_bootstrap

def scores(path):
    records = json.load(open(path))["records"]
    assert not any(r["error"] for r in records), f"{path}: a query raised, scored 0.0"
    by_id = {r["id"]: r["ndcg"] for r in records}
    assert len(by_id) == len(records), f"{path}: an id is used twice"
    return by_id

a = scores("/tmp/run-a.json")
b = scores("/tmp/run-b.json")
assert a.keys() == b.keys(), "the runs cover different queries"
ids = sorted(a)
result = paired_bootstrap([a[i] for i in ids], [b[i] for i in ids])  # b against a
print(result.mean_delta, result.ci_low, result.ci_high, result.p_value, result.n)
```

* `paired_bootstrap(base, cand, *, n_resamples=10_000, confidence=0.95, seed=0)` draws
  query indices with replacement once per replicate and applies them to both runs. The
  interval is the percentile interval of the replicate mean deltas and `p_value` is
  `2 * min(share of replicates <= 0, share >= 0)`, capped at 1.0. It is a multiple of
  `1 / n_resamples`, and 0.0 means no replicate reached zero. The same `seed` gives the
  same numbers. It resamples queries only: freeze the plans first (above), or each run's
  per-query scores are one draw of a planner that re-plans every time.
* `mde(sigma_d, n)` is `2.8 * sigma_d / sqrt(n)`: the smallest true mean difference that a
  two-sided 5 percent test over `n` queries detects 80 percent of the time, where
  `sigma_d` is the standard deviation of one paired per-query difference. The constant
  2.8 is `1.96 + 0.84`, the normal quantiles for 5 percent two-sided error (1.96) and
  80 percent power (0.84). With `sigma_d = 0.2` over the 54 queries it is 0.076.
* `holm(p_values, alpha=0.05)` is the Holm-Bonferroni step-down for several comparisons at
  once (one per area, say): a list of booleans, in input order, true where the hypothesis
  is rejected.

An interval that includes 0 is not evidence that the runs are equal, and a delta inside
the `mde` is not evidence of anything; see "What a difference has to be before it means
anything" above, where the same idea is called the MDD.

### Pre-registration and `trelix eval-compare`

`trelix eval-compare` is the only command that produces a PASS. It judges a candidate run
against a baseline run by a rule that was written down before anyone looked at a number:

```bash
trelix eval-compare BASE.json CAND.json --prereg EXP.yaml
```

It needs no git, no index and no embedder. The verdict is a pure function of the three files:
queries are paired by `id`, fed to the bootstrap in sorted-`id` order with a fixed seed
(`seed=0`, 10,000 resamples), and the output is identical for identical files.

#### The results file (`results.json`, schema_version 1)

One file per run. It is **not** the `--per-query-out` file: that one also says
`schema_version: 1` but names no repository, commit or golden file, so `eval-compare`
refuses it (`not a results.json`). The format lives in `trelix.eval.results`, which has the
writer (`build_results`, `write_results`) beside the reader (`load_results`) so the two cannot
drift; `trelix eval-suite ... --arm ARM --out PATH` (below) writes one per arm. Written with sorted
keys and mode 0600, like the per-query file:

```json
{
  "aggregate": {"mrr": 1.0, "n_queries": 1.0, "ndcg@10": 1.0, "recall@10": 1.0},
  "arm": "baseline",
  "embedder": {"dimension": 384, "library_version": "5.1.0", "model": "sentence-transformers/all-MiniLM-L6-v2", "provider": "local"},
  "pipeline": {"config": {"retrieval": {"top_k": 10}}, "flare_enabled": false, "hyde_fallback_enabled": false, "multi_query_enabled": false, "plans": "replayed", "rerank": false, "rerank_summary": "disabled"},
  "records": [
    {"error": null, "id": "q0001", "kind": null, "lang": null, "mrr": 1.0, "ndcg": 1.0, "recall": 1.0, "repo": "demo", "split": null, "top10": ["src/app.py"]}
  ],
  "run": {"created_at": "2026-10-05T12:00:00+00:00"},
  "schema_version": 1,
  "suite": {"golden_sha256": "<64 hex>", "golden_version": "v1", "license": "MIT", "name": "demo", "plans_sha256": "<64 hex>", "repo_sha": "<40 hex>", "repo_url": "https://example.invalid/demo.git"},
  "trelix_version": "3.4.3"
}
```

| Key | Rule |
|---|---|
| top level | exactly `aggregate, arm, embedder, pipeline, records, run, schema_version, suite, trelix_version`; another key, or a missing one, is refused |
| `arm` | `^[a-z0-9][a-z0-9_-]{0,62}$`; the label the pre-registration's `comparison_id` binds to |
| `suite` | exactly `name, golden_version, repo_url, repo_sha, license, golden_sha256, plans_sha256`: which repository, commit, golden file and plans file the run measured. `repo_sha` is 40 hex, the hashes 64 hex |
| `embedder` | exactly `provider, model, dimension` (integer, 1 or more) and `library_version` (a string, or `null` when the package is not installed; never an error) |
| `pipeline` | `rerank, hyde_fallback_enabled, multi_query_enabled, flare_enabled` (booleans), `plans`, `rerank_summary` (strings) and `config` (an object: the effective configuration, which `eval-compare` reports but never refuses on). Keys it does not know are accepted and ignored, so a later field needs no schema bump |
| `aggregate` | exactly `mrr, n_queries, ndcg@10, recall@10`, compared with `==` against `aggregate_metrics(records)` recomputed from the records as they stand in the file: a hand-edited file cannot keep a stale aggregate |
| `records` | a non-empty list; each record has exactly the ten `QueryRecord` keys, scores are numbers in 0 to 1 (a bool is not a number), `top10` is at most 10 strings, `split` is `null`, `dev` or `test`, `error` is `null` or a non-empty string (a message of only whitespace counts: the query still raised), `id` is unique, and `repo` equals `suite.name` |
| `run` | an object with `created_at` (a string). The one non-deterministic datum: two runs of the same code agree on everything but `run`, and no verdict reads it. Keys it does not know are ignored |

A file over 32 MiB, one that is not UTF-8 JSON, and one that contains `NaN` or `Infinity` are
refused; every problem found in a readable file is listed (the first 20). `write_results` applies
the same rules before it writes, so it refuses a document carrying a `NaN`, an infinity or a value
that is not JSON (`pipeline.config` is the one place a value is free-form).

#### The pre-registration

A small YAML file, written and committed before the run. **All ten keys are required**, there are
no defaults, and unknown keys, duplicate keys, YAML anchors and aliases are refused (it is read with
PyYAML's safe loader only):

```yaml
schema_version: 1
experiment_id: EXP-declaration-boost
comparison_id: baseline..declaration-boost
primary_metric: ndcg@10
direction: increase
expected_effect: 0.03
alpha: 0.05
family_size: 3
min_queries: 30
cost_class: flag
```

| Key | Rule |
|---|---|
| `schema_version` | the integer `1` |
| `experiment_id` | `EXP-` followed by letters, digits and `. _ -` |
| `comparison_id` | `<baseline arm>..<candidate arm>`: two different arm names, which must be the `arm` of BASE and of CAND in that order (swapped arguments are refused) |
| `primary_metric` | exactly `ndcg@10` |
| `direction` | exactly `increase` |
| `expected_effect` | a number above 0 and at most 1: the gain the experiment expects, compared with the minimum detectable effect |
| `alpha` | a number above 0 and at most 0.05, and `alpha / family_size` at least 0.001 (below that a tail of the 10,000-replicate bootstrap holds fewer than 5 replicates and `1 - alpha / family_size` can round to 1.0) |
| `family_size` | an integer from 1 to 50: how many comparisons against this baseline the experiment makes, fixed before the first run. Every interval is taken at confidence `1 - alpha / family_size` (Bonferroni) |
| `min_queries` | an integer of at least 20, compared with the size of the **decision set** (below), not with the number of records |
| `cost_class` | `flag` (hurdle 0.01), `index` (0.02) or `heavy` (0.03): the smallest nDCG@10 gain worth shipping. A fixed table, so a file cannot pick its own hurdle |

`min_queries` matters on a golden file with `split` labels: it is compared with the test split, so
a file with 54 queries of which 38 are `test` can never satisfy `min_queries: 54`.

#### The decision set and the rule

The decision set `D` comes from the **baseline** file's labels, so a results file cannot choose its
own: with no `split` on any baseline record `D` is every record; with one on every baseline record
`D` is the `test` records (dev queries never enter a decision); a mixture is refused. The two files
must give every query the same `split`, `kind` and `lang`. With `n = |D|`, `delta` the mean paired
nDCG@10 difference (`cand - base`) and `[L, H]` its interval at confidence `1 - alpha / family_size`;
`[Lr, Hr]` the same for recall@10; `h` the hurdle; `G = -0.02`; and
`MDE = 2.8 * sigma_d / sqrt(n)` with `sigma_d` the sample standard deviation (`n - 1`) of the paired
nDCG@10 differences on `D`:

| Row | Condition | Outcome |
|---|---|---|
| 1 | the candidate has a record with an `error` (checked over all its records) | FAIL |
| 2 | `n < min_queries` (no statistics are computed) | INCONCLUSIVE |
| 3 | `H < 0`: nDCG@10 is confidently worse | FAIL |
| 4 | `Hr < G`: recall@10 is confidently down by more than 0.02 | FAIL |
| 5 | `expected_effect < MDE`: exploratory, cannot change a default | INCONCLUSIVE |
| 6 | `L <= 0`: the improvement is not demonstrated | INCONCLUSIVE |
| 7 | `delta < h`: below the hurdle | INCONCLUSIVE |
| 8 | `Lr < G`: the recall guard is not resolved | INCONCLUSIVE |
| 9 | none of the above | PASS |

Rows 1 and 2 stop the evaluation. After them, any match among rows 3 and 4 makes the verdict FAIL,
else any match among rows 5 to 8 makes it INCONCLUSIVE, and every matching row of the winning class
is printed as a `reason:` line. Recall@10 is a guard, not a second hypothesis. "No demonstrated gain"
(identical arms give the interval `[0, 0]`) is INCONCLUSIVE, not FAIL: an interval containing 0 is not
an equivalence result. A recall point estimate below -0.02 whose interval still reaches 0 is also
INCONCLUSIVE (row 8), not FAIL.

Four limits, stated so nobody reads more into a PASS than it holds. **Holm is not applied**:
one invocation judges one pair, so `family_size` widens the intervals by Bonferroni (never less
conservative than Holm) and `stats.holm` is not called. **`stats.mde` uses the fixed constant
2.8** (5 percent two-sided error, 80 percent power), so row 5 does not tighten when
`alpha / family_size` shrinks although the interval of row 6 does. **`sigma_d` is measured on the pair
being judged**, so a noisy candidate raises its own MDE. The bootstrap draws from
`numpy.random.default_rng(0)`, whose stream numpy does not promise across releases (the tests here
were measured with numpy 2.5.0); a verdict within a few percent of a tail could flip after an
upgrade, which is why the interval is always printed. **Memory**: each bootstrap allocates
`10,000 x n` indices and then the deltas gathered with them, so its peak is about
`2 x 8 bytes x 10,000 x n` (about 56 MB at `n = 350`, 800 MB at `n = 5,000`; measured with
`tracemalloc`) and `eval-compare` does not chunk them, so a decision set of several thousand
queries needs that first.

#### Refusals, output and exit codes

Anything that makes the comparison invalid rather than unfavourable is REFUSED, with every
reason listed: a file that is unreadable or malformed; a `--prereg` that is missing or invalid;
suites that differ in `name, golden_version, repo_url, repo_sha, license, golden_sha256` or
`plans_sha256`; a run with `rerank`, `hyde_fallback_enabled`, `multi_query_enabled` or
`flare_enabled` on, or whose `plans` is not `replayed`; runs that cover different queries, or label
them differently, or a baseline with mixed `split` labels; a `comparison_id` that does not match the
two arms; and a baseline with an `error` (a bad baseline is a bad instrument). Differences that are
not refusals are printed as `note:` lines: a different `trelix_version`, a different `embedder`, and
every leaf of `pipeline.config` that differs (at most 20; or `note: pipeline.config identical`).

| Exit code | Verdict | Last line of stdout |
|---|---|---|
| 0 | PASS | `verdict: PASS` |
| 1 | FAIL | `verdict: FAIL` |
| 2 | INCONCLUSIVE | `verdict: INCONCLUSIVE` |
| 3 | REFUSED (stderr gets one `refused: ...` line per reason) | `verdict: REFUSED` |

Only PASS exits 0, also when a reader stops reading: `trelix eval-compare ... | head -n 20` under
`pipefail` drops the lines `head` did not take and still exits with the verdict's code, where
`stats` and the other commands that print through the shared consoles (telemetry, graph,
review, the search tables) exit 0 on a closed pipe. A click usage error (an unknown option, a
missing argument) also exits 2 but prints no `verdict:` line, so a script that must tell
INCONCLUSIVE from a mistyped command reads the `verdict:` line; the most likely mistake, leaving
out `--prereg`, is a refusal (3), not a usage error. Text that comes from the files (ids, error
messages, config keys) is printed literally: Rich markup is escaped, control bytes are dropped and
a line break becomes a space, so a value cannot start a line of its own and pass for a `verdict:`
line. Worked example: 20 queries, baseline nDCG@10 0.5 on all of them, a candidate at 0.5625 on
all of them, and a pre-registration like the one above but with
`comparison_id: baseline..flag-on`, `family_size: 1` and `min_queries: 20`:

```
comparison: baseline..flag-on
experiment: EXP-example
suite: demo golden_version v1 repo_sha 0123456789abcdef0123456789abcdef01234567
runs: base 2026-10-05T12:00:00+00:00 cand 2026-10-05T12:30:00+00:00
queries: 20 in the decision set (no split labels: all queries), min_queries 20
note: pipeline.config identical
ndcg@10: base 0.5000 cand 0.5625 delta +0.0625 ci [+0.0625, +0.0625] at 95.00% confidence
recall@10: base 0.5000 cand 0.5000 delta +0.0000 ci [+0.0000, +0.0000] at 95.00% confidence
mde: 0.0000 (sigma_d 0.0000, n 20); expected_effect 0.0300
hurdle: 0.0100 (cost_class flag)
verdict: PASS
```

Lines before the `reason:` lines are informational and may gain fields; the `reason:` and
`verdict:` lines are the contract. The same base run against a candidate at 0.75 on half of the
queries and 0.5 on the other half gives delta +0.1250, interval `[+0.0750, +0.1750]`, `sigma_d`
0.1282 and MDE 0.0803: an `expected_effect` of 0.05 or 0.079 is INCONCLUSIVE (row 5 only, with the
reason `expected_effect ... is below the minimum detectable effect 0.0803`), and 0.10 is PASS.

### Suites and `trelix eval-suite`

A suite is the committed definition of one measurement: ONE repository at ONE pinned commit, the
golden file of queries about it, and the file of frozen plans recorded for those queries (above).
`results.json` names the suite it measured, and `trelix eval-compare` refuses two runs of different
suites. `trelix eval-suite` verifies the suite and its pinned clone, builds ONE index for ONE arm
from that clone, replays the frozen plans and writes a `results.json`; `--prepare-only` stops after
the verification, and given together with `--arm` or `--out` it is refused rather than silently
winning (`--prepare-only runs nothing and takes no --arm or --out: ...`). No suite is committed yet
(the first one needs its plans recorded by someone with an LLM key; see "Recording the plans" below).

```bash
trelix eval-suite eval/suites/NAME/suite.json --arm ARM --out results.json [--cache-dir DIR]
trelix eval-suite eval/suites/NAME/suite.json --prepare-only [--cache-dir DIR]
make eval-suite EVAL_SUITE=eval/suites/NAME/suite.json EVAL_ARM=baseline   # EVAL_RESULTS defaults to .trelix/eval-suite/results.json
```

#### `suite.json`

```json
{
  "schema_version": 1,
  "name": "trelix-self",
  "golden_version": "2026-10-r1",
  "repo": {"url": "https://github.com/OWNER/REPO.git", "sha": "<40 hex>", "license": "MIT"},
  "golden": {"path": "golden.jsonl", "sha256": "<64 hex>"},
  "plans": {"path": "plans.jsonl", "sha256": "<64 hex>"}
}
```

Every key is required; an unknown or a repeated key, `NaN` and a `schema_version` other than `1`
are refused, and every problem found is listed, not the first. `name` is `[a-z0-9][a-z0-9_-]{0,62}`
(it is a directory name of the clone and the `repo` label of every record), `golden_version` a label
of `[A-Za-z0-9._-]`, `repo.sha` a full lower-case 40-hex commit id (not an abbreviation, a branch or
a tag), `repo.license` an SPDX identifier that is recorded and not enforced, and `golden.path` and
`plans.path` bare file names of files beside `suite.json` (a symlink to a file elsewhere is refused).
`repo.url` is `https://github.com/<owner>/<repo>` or the same ending in `.git`, and nothing else: no
`http`, `ssh` or `git@` form, no credentials, port, query, fragment or further path, no other host
(`localhost`, an IP literal, a lookalike such as `github.com.evil.example`, a trailing dot and an
upper-case host are all refused), and `<owner>` and `<repo>` must have the characters GitHub allows
(an owner is letters, digits and single hyphens, 39 at most; a repo is letters, digits, `-`, `_` and
`.`, 100 at most, and not `.` or `..`). A committed `suite.json` can arrive in a pull request, and a
CI job that runs `eval-suite` would otherwise send a git request to whatever host it names (server-side
request forgery from CI). The allowed hosts are the one constant `ALLOWED_REPO_HOSTS` in
`trelix/eval/suite.py`, so adding a host is a one-line change for review, and the refusal names them.
A local path or a `file://` URL is accepted only by the Python API, through the keyword `allow_local=True`
of `load_suite`, `ensure_clone` and `prepare_suite` (the tests use it); the command line never sets it,
so a `suite.json` cannot select a local or an internal remote. The sha256 values are over the raw bytes
of the files as committed (`shasum -a 256 golden.jsonl`): one changed byte, a CRLF rewrite or a missing
final newline refuses the suite, and the message prints both digests.

#### What `--prepare-only` checks, in this order

Nothing is cloned, and the cache directory is not created, until the first three are proven.

1. `suite.json` is valid and both hashes match.
2. The golden file validates as `trelix eval-validate` validates it (schema, duplicates), without its
   repository and without the v2 strata thresholds (those decide whether a golden file may be committed,
   not whether a suite may run). A `<stem>-metadata.json` beside it is read by the loader for `area`
   labels but is covered by neither sha256: one that is not valid JSON, or that cannot be read (a
   directory, no permission), is a refusal; any other shape yields no labels. No label from it reaches
   a record: `--prepare-only` uses none, and a run reads a copy of the golden file in the run
   directory, where no sidecar exists (below).
3. The plans file holds a plan for every golden query. It is read by the planner's own cache class and
   asked the way the planner asks, so `"How Does Login Work "` is covered by a plan recorded for
   `"how does login work"` exactly when a run would cover it. A plans file with no plans is refused: the
   planner would read it as "record" and call the LLM for every query. Record plans once, with
   `trelix eval . --golden GOLDEN --plan-cache-file PLANS` as above.
4. The clone at `<cache>/clones/<sha>/<name>/` exists, or is made: cloned into `<name>.partial`,
   `git checkout --detach <sha>`, `HEAD` equal to the pin, then one rename. An existing clone is
   re-verified and never repaired or deleted: `HEAD` is the pin, the worktree is pristine (ignored files
   count, because the walker indexes what git ignores), the origin URL is the suite's, and the clone is
   neither shallow nor partial (git could complete either from the network). A `.partial` directory left
   by a killed run is refused with its path and never deleted, also when it sits beside a clone that
   verifies (the same refusal, on every run until it is removed); remove it by hand.
5. Every gold path exists at the pin and is in the set `FileWalker` would index with
   `walker.follow_symlinks = false`. `trelix eval-validate` proves a path exists in git; a file that
   exists in git and is not indexed scores 0 in every arm, which looks exactly like a retrieval miss. The
   default walker ignores a directory called `packages`, and 2 of the 54 queries of `eval/golden.jsonl`
   (lines 43 and 46) have gold files only there, so a suite over this repository has to leave those two
   out (its golden file, and so its hash, differ from the shipped one). The refusal names the first
   five paths and counts the rest. The walk uses the walker settings in effect, `TRELIX_WALKER_*`
   environment included, apart from `follow_symlinks`, which is forced. Before the walk, a tracked
   symlink named `.gitignore` or `package.json` is refused (see Security below).

The cache is the directory given with `--cache-dir` (an empty value counts as not given), else
`$XDG_CACHE_HOME/trelix/eval-suites` (a relative value is ignored), else
`~/.cache/trelix/eval-suites`; with no home directory and no `--cache-dir` the command says to
pass it.

Output of `--prepare-only` on success (exit 0), one line each: `suite:`, `repository:`, `sha:`,
`golden:` (its sha256, the number of queries and of distinct gold files), `plans:` (its sha256),
`clone:` and a last `prepared:` line. Any refusal is one `refused: ...` line per reason on stderr and
exit 1.

#### Running an arm

`--arm ARM --out results.json` runs the five checks above and then, in this order:

0. Before any of them, `--out` is checked: its directory must exist and be writable, and it must not
   be a directory itself. A missing directory should fail in a second, not after the index is built.
1. The run directory `<cache>/arms/<sha>/<name>/<arm>/` is created and is the claim on the arm. One
   that exists is refused: `arm 'X' already has a run directory at PATH: choose another --arm or delete
   it`. **One index per arm, never reused, never deleted by eval-suite.** An index built by other trelix
   code (a candidate branch with the same version string, a changed chunker) would otherwise be
   measured as if it were current, which is the quiet way to run an experiment that measures nothing.
   Sharing an index between arms that differ only in retrieval-time flags needs a cache key that sees
   the chunker and the embedder; until that is designed, every arm builds its own. Nothing under the
   cache is ever deleted by eval-suite, so a refused or interrupted arm is deleted by hand, and every
   refusal after this step ends with one more line saying so: `the run directory PATH is left as it
   is; delete it before running this arm again`.
2. The verified golden and plans bytes are copied into the run directory as `golden.jsonl` and
   `plans.jsonl`, each re-checked against its sha256 as it is read, and THOSE copies are what the
   harness reads. The committed files are never written to, the bytes replayed are the bytes hashed,
   and the copy has no `<stem>-metadata.json` beside it. A copy that cannot be written (disk full)
   is a refusal naming the file.
3. The configuration is `IndexConfig(repo_path=<clone>)` with these forced to literal values, whatever
   `TRELIX_*` or `~/.config/trelix/env` says: `file_summaries_enabled=false`, `use_batch_api=false`,
   `embedder.provider=local`, `walker.follow_symlinks=false`, `chunker.contextual=false`,
   `store.backend=sqlite` with the index at `<run dir>/index.db`, `retrieval.rerank`,
   `hyde_fallback_enabled`, `multi_query_enabled` and `flare_enabled` `false`, and
   `retrieval.plan_cache_file` the plans copy. Everything else (the embedder's `local_model`, every
   ranking knob of `RetrievalConfig`, the walker's ignore lists) is taken from the environment and
   recorded, not forced: see `pipeline.config` below.
4. The index is built with `Indexer(config, quiet=True)` and, if it reports any error, the run is
   refused and no results are written (whether a parse error on an exotic file should be tolerated is
   an owner decision, once the self-index is measured). An index that cannot be built or opened at
   all (an `OSError`, or the `ImportError` of a missing `local` extra) is one refusal too, not a
   traceback. The build and the query run happen with the operator's git configuration switched off
   for the whole process (`GIT_CONFIG_GLOBAL` and `GIT_CONFIG_SYSTEM` the null device,
   `GIT_CONFIG_NOSYSTEM=1`, `GIT_ALLOW_PROTOCOL=https`, `GIT_TERMINAL_PROMPT=0`,
   `GIT_LFS_SKIP_SMUDGE=1`, `GIT_OPTIONAL_LOCKS=0`, put back afterwards) and with `GIT_DIR`,
   `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_COMMON_DIR` and `GIT_OBJECT_DIRECTORY` removed for the
   same span (put back too), because the Indexer's own `git rev-parse` and `git status` in the clone
   inherit the process environment and run with `cwd=` and no `-C`: an exported `GIT_DIR` (a git
   hook's shell has one) would make the index's provenance rows describe another repository. The
   index and the per-query debug traces live in the run directory, so the clone stays pristine (the
   next arm re-verifies it).
5. The queries run through the real `EvalHarness` over the plans copy. A golden query the plans do not
   cover at run time (`PlanCacheMissError`, only possible if the pre-flight was wrong) stops the run,
   is printed as a refusal, writes no `results.json` and exits 1; it is never scored as a miss.
6. `results.json` is written, even when queries raised (the file names every failed query and the
   command then exits 1, listing the first five on stderr, as `trelix eval` does).

Output on success (exit 0): the six `--prepare-only` lines, then `arm:`, `run directory:`, `index:
built, embedding dimension N`, one line with `ndcg@10`, `recall@10`, `mrr`, the number of queries and
the rerank summary, and `results: <path>`. No PASS or FAIL: only `trelix eval-compare` judges.

**Recording the plans.** A plan is a function of the query text alone (the planner is given no
repository), so `plans.jsonl` can be recorded from ANY indexed checkout, once, by someone with an LLM
key: `trelix eval . --golden eval/suites/NAME/golden.jsonl --plan-cache-file
eval/suites/NAME/plans.jsonl`, then `shasum -a 256` of both files into `suite.json`. eval-suite never
records: an empty plans file is refused before anything is cloned.

**What the results file records about the pipeline.** Besides the four forced retrieval flags,
`plans: "replayed"` and the rerank summary, `pipeline.config` holds, for each of `walker`, `parser`,
`chunker`, `store`, `retrieval`, `indexer` and `sparse`, the section's effective values, with every
field whose name contains `key`, `secret`, `token`, `password`, `endpoint`, `url` or `uri` removed (a
results file is a pull request artifact, and an operator's `store.qdrant_url` or `store.lance_uri` may
carry credentials although neither backend is used by a run; the word `token` would also drop token
counts, so `chunker.max_tokens_per_chunk`, `retrieval.context_token_budget` and `sparse.top_k_tokens`
are kept by name because each changes the index or the ranking) and with `store.db_path` and `retrieval.plan_cache_file` removed (they differ per arm
by construction). `embedder` holds the provider, `local_model`, the dimension the index was built with
and the `sentence-transformers` version (`null` when it is not installed). `trelix eval-compare` never
refuses on any of these; it prints `note: pipeline.config identical` or up to 20 `note: pipeline.config
differs: <section>.<field> (base X, cand Y)` lines (then one line counting the rest), and a `note:` for a differing
`trelix_version` or `embedder`, so a reviewer of a result sees exactly what the arms changed, and
sees an ambient setting that the arms did not mean to change.

**Two judgement calls `eval-compare` makes, stated here because the plan does not define them.** (i) A
recall@10 change whose point estimate is below -0.02 but whose interval reaches 0 (say -0.3 on 2 of
20 queries: mean -0.03, interval about [-0.075, 0]) is INCONCLUSIVE by the recall-guard row, not
FAIL, although the plan's wording "Recall@10 CI lower bound >= -0.02" can be read as a hard guard;
FAIL is reserved for a recall interval entirely below -0.02. (ii) The plan says "Holm-adjusted across
experiments in one batch"; `eval-compare` sees one pair at a time and widens the interval to
`1 - alpha / family_size` instead (Bonferroni), which is never less conservative than Holm and needs
no other arm's p-value. Both are open for the owner to change.

#### Security

The repository is untrusted input. Nothing from it is executed, imported, built or installed: it is
cloned, checked out, read, hashed, and (`.gitignore` and `package.json`) parsed as data.

* git gets a built environment, not the operator's: `GIT_CONFIG_GLOBAL` and `GIT_CONFIG_SYSTEM` are the
  null device, `GIT_CONFIG_NOSYSTEM=1`, `GIT_ALLOW_PROTOCOL=https` (`https:file` for the clone command
  alone, and only when the Python API is called with `allow_local=True`), `GIT_TERMINAL_PROMPT=0`,
  `GIT_LFS_SKIP_SMUDGE=1`, `GIT_OPTIONAL_LOCKS=0`, and nothing else named
  `GIT_*`; only `PATH`, a temp directory, the proxy and CA-bundle variables (`HTTPS_PROXY`,
  `SSL_CERT_FILE`, ...) and, on Windows, `SYSTEMROOT` are passed on. A global `url.*.insteadOf`,
  `core.hooksPath`, `init.templateDir`,
  `filter.lfs.required` or an exported `GIT_DIR` therefore cannot shape a clone. While the index is
  built and the queries run, the process itself gets the same configuration variables and has
  `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_COMMON_DIR` and `GIT_OBJECT_DIRECTORY` removed,
  so the Indexer's own git describes the clone and nothing else. The price is that there are no
  credentials, so only public repositories can be used.
* Every command is `git -C <directory>` on a directory this command created, with `-c
  core.hooksPath=<null device>` and the URL after `--`; no submodule is fetched. A clone has 1800
  seconds and any other command 120. A missing `git`, a timeout or any other OS error is a one-line
  refusal.
* The walker is confined to the clone (`follow_symlinks = false`). Its default follows symlinks and
  does not resolve them, so a tracked symlink to a file outside the clone would be read, hashed and
  indexed under a name that looks like it is inside. The tests plant one, with a canary file outside the
  clone, and show the default walker reads it and this configuration does not. That setting covers
  the files the walker iterates. It also opens two by name, `.gitignore` in every directory it enters
  and `package.json` beside a `packages` or `bin` directory, and reads them through a symlink wherever
  it points: a `.gitignore` linked to a regular file is read whole and its lines shape the walk; a
  `package.json` is read with no check of what it is, so a link to `/dev/zero` or to a FIFO is never
  read to its end. The name is matched in any case: a case-insensitive filesystem (macOS, Windows)
  resolves the fixed name `.gitignore` the walker asks for to a tracked `.GITIGNORE`. So a tracked
  symlink with either name, at any depth and whatever it points at, is refused before the walk starts:
  one `refused: the clone tracks N symlink(s) named .gitignore or package.json, ...` line naming the
  first five. A `.gitignore` line pathspec rejects or compiles into an invalid regex, or a `package.json`
  nested too deep to decode, makes the walker raise; that is one `refused: the walker failed on the
  clone: ...` line, on every run.
* The planted-file test: `conftest.py`, `setup.py` and `sitecustomize.py` in the pinned commit write a
  marker if they are ever run; a control runs one to show the marker fires, and none runs here.

* A run keeps its index, its input copies and its debug traces in the run directory, outside the
  clone; the clone is re-verified pristine before the next arm. The tests plant `conftest.py`,
  `setup.py` and `sitecustomize.py` (a control runs one) and a symlinked `src/leak.py` with a canary
  word outside the clone, run the real `Indexer`, and show the markers never appear, the clone's
  `git status --porcelain --untracked-files=all --ignored` is empty, and no `files` row or chunk
  carries the leak.

Not here, on purpose: reusing an index between arms (`--reuse-index`, `--index-from`, a stamp), an
embedder fingerprint, other embedders, more than one repository per suite (several repositories are
several suite directories), and a committed suite (the first one lands with its own data PR).

## `golden_synthesis_sample.jsonl` — synthesis quality

Input to `trelix eval-synthesis`, which scores answer faithfulness and completeness
GroUSE-style rather than measuring retrieval. The sample has four lines: three answerable
queries carrying `expected_citations`, and one unanswerable query.

### Golden v2 fields for synthesis

A line carries `query`, `expected_answer_fragments` and `expected_symbols` as before
(`relevant_files` is not read by this harness) and may add:

| Field | Allowed values | Meaning |
|---|---|---|
| `answerable` | `true`, `false`; absent means `true` | Whether the repository answers the query; the right answer to an unanswerable one is an `INSUFFICIENT_EVIDENCE:` line |
| `gold_answer` | non-empty string | A reference answer, stored for a future judge and never scored |
| `expected_citations` | list of non-empty repo-relative paths (may be empty) | The files a correct answer cites; validated and stored here, counted by the next change |

A field present with another type (any of them `null`), or a line that is JSON but not an
object, is refused with its line number and no query runs; a line that is not JSON is still
skipped, as it always was.

### What `trelix eval-synthesis` reports

One record per golden line (`SynthesisEvalHarness.run_detailed`); the seven printed values
are means and counts over those records:

| Key | Over which records |
|---|---|
| `hallucination_rate`, `completeness`, `faithfulness`, `overall` | means over the records whose line is answerable; a record whose query raised keeps the placeholder scores (hallucination 1.0, the rest 0.0) and is included, so a v1 file (every line answerable) returns exactly the numbers the harness returned before records existed; 0.0 when there is none |
| `n_queries` | every record |
| `unscoreable` | records whose query raised in retrieval or scoring (a model failure is an empty answer, not an error) |
| `n_unanswerable` | records whose line is unanswerable and whose query did not raise |

Three v1 lines where the second query's retrieval raises and the other two answer with no
hallucination: `hallucination_rate 0.3333`, `completeness 0.6667`, `unscoreable 1.0`,
`n_unanswerable 0.0`, `n_queries 3.0`.

`--per-query-out FILE` writes the records and the aggregate as JSON; `docs/CLI_REFERENCE.md`
(`trelix eval-synthesis`) lists the document's keys and the record fields.

`TRELIX_RETRIEVAL_PLAN_CACHE_FILE` applies to the retrieval half of every query, so two runs over
a frozen plan retrieve the same context; the answers still come from the model, and on this path
they are capped at 2,048 tokens (`synthesize()` hands `_stream_response` an `EmbedderConfig`,
which has no `synthesis_max_tokens`) where plain `trelix ask` streams at 12,000: pre-existing.

With `TRELIX_RETRIEVAL_CITATIONS=true` the synthesis prompt also asks the model to abstain with one
line starting `INSUFFICIENT_EVIDENCE:` when the retrieved context does not answer the question.
GraphRAG answers (large contexts under the `openai` or `azure` embedder, reached through
`synthesize()` only) get the `[C#]` cite lines but not the abstention sentence, so they cannot
abstain by protocol: an abstention count over such a run is structurally lower.

When an answer's markers are verified (`trelix.retrieval.citations.verify_citations`; nothing in
the eval reads them yet), `line_out_of_range` means the cited chunk ends past the file's line
count, where lines are counted as the extractors count them, `count("\n") + 1`: a newline-terminated
7-line file has 8 lines and its Python `<module>` symbol spans `1-8`, so a whole-file citation on a
fresh index is `valid`.

## Related tooling

`scripts/measure_index_hygiene.py` answers a different question from `trelix eval`: not
"did we rank the right files" but "how much of what we ranked was never source code to
begin with". It needs no golden set, so it works on any index:

```bash
python scripts/measure_index_hygiene.py . --json > docs/reports/index-hygiene-after.json
```
