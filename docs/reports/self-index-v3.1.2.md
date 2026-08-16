# Self-index report — trelix v3.1.2

Indexing this repository with trelix, then checking whether each dimension of the
resulting index was actually populated.

Everything below was measured on this machine. Where a number could not be measured
honestly it is marked as such rather than estimated.

---

## What the index looked like before

The index at `.trelix/index.db` had been built on 14 Aug with the **local** 384-dim
embedder, while `.env` had specified `azure` (3072-dim) — so the index and the configured
provider disagreed. More importantly, most of it was not this repository:

| | Count | Share |
|---|---|---|
| Files | 915 | |
| …from `workspace-vscode/.vscode-test/` | **543** | **59.3%** |
| Chunks | 32,337 | |
| …from that bundle | **23,865** | **73.8%** |
| Tokens embedded | 3,929,977 | |
| …spent on that bundle | 2,267,576 | 57.7% |
| `packages/` sub-packages indexed | **0** | |

`@vscode/test-electron` fills `.vscode-test/` with a **2.6 GB VS Code application
bundle**. `workspace-vscode/.gitignore` excludes it; the walker never read that file.

Several dimensions were empty: `file_summaries`, `def_use_edges`, `sparse_embeddings`,
`sub_chunks`, `taint_flows`, `artifacts`, `generic_edges` — and no `graph_*` table
existed at all.

### What the pollution actually cost

Corpus damage and retrieval damage turned out to be very different numbers, and the
distinction matters:

- **Corpus**: 73.8% of chunks were noise, and 57.7% of embedding spend bought it.
- **Retrieval**: mean precision@10 across a 10-query set was **95%** — the RRF fusion
  pipeline masks the noise for most queries.

The damage concentrates where a query's vocabulary overlaps the junk, and there it is
severe. *"How does the file walker filter ignored directories"* returned **6 of 10
results from the bundle** — minified single-letter identifiers from GitHub Copilot's
bundled JavaScript — and pushed `FileWalker` itself down to **rank 4**.

An earlier draft of this report generalised from that one query to "every query is
degraded". The measurement did not support it. The honest claim is narrower and still
serious: most queries are fine, and the ones that are not are badly broken.

---

## What the index looks like now

`trelix index` via `scripts/self-index.sh`, azure `text-embedding-3-large`,
1154 s wall clock, exit 0, **zero errors or warnings**.

| Dimension | Before | After |
|---|---|---|
| vec0 declared dimension | `FLOAT[384]` | **`FLOAT[3072]`** |
| Files | 915 (543 junk) | **454 (0 junk)** |
| `packages/` sub-packages | 0 | **35** |
| Symbols / chunks | 32,337 | 10,423 |
| Chunk vectors | 32,337 | **10,423 (1:1, no orphans)** |
| `file_summaries` | **0** | **428** |
| `calls` | 19,107 | 19,558 |
| `imports` | 2,345 | 2,714 |
| `def_use_edges` | **0** | **109,902** |
| `graph_metadata` | table absent | **10,423 rows** |
| Corpus noise | 59.3% files / 73.8% chunks | **0% / 0%** |
| Mean precision@10 | 95.0% | **100.0%** |

All 22 gates in `scripts/verify-index.sh` pass, including the two integrity gates that
matter most: **0 chunks without an embedding**, and chunk-vector count exactly equal to
chunk count.

**Caveat on precision@10.** That 95% → 100% figure conflates three changes: junk
removal, the 384 → 3072 embedder upgrade, and the sentinel-exclusion fix. It is not a
measurement of the walker fix alone. The corpus numbers above *are* attributable to the
walker fix and the `packages/` inclusion, because they are structural.

### Retrieval quality

`trelix eval . --golden eval/golden.jsonl`, over a 50-query golden set authored for this
repository (10 per subsystem, graded by difficulty). All 50 evaluated — matching the
file's line count, so nothing was silently skipped.

Each configuration was run twice, because the shipped one is not deterministic.

| Metric | Shipped `.env` | | HyDE + FLARE off | |
|---|---|---|---|---|
| | run 1 | run 2 | run 1 | run 2 |
| nDCG@10 | 0.6282 | 0.6228 | **0.6539** | **0.6539** |
| Recall@10 | 0.8300 | 0.8300 | 0.8300 | 0.8300 |
| MRR | 0.5889 | 0.5847 | **0.6353** | **0.6353** |

These are the first numbers this harness has produced that are inside its documented
`[0, 1]` range; see the eval fix below. **They are not comparable with any nDCG or
recall figure published before v3.1.2.**

Three things fall out of the repeated runs:

**HyDE and FLARE are the sole source of nondeterminism.** With both disabled, two
independent runs produced bit-identical scores. With them enabled, nDCG moved by 0.005
between runs. Anyone tuning retrieval against this harness should disable them first,
or they are measuring a moving target.

**They cost ranking quality here, and the effect is larger than the noise.** Disabling
them improved nDCG by 0.026–0.031 and MRR by 0.046–0.051 — **5–6× the observed
run-to-run variance of ±0.005**, in the same direction in both pairs. That is a real
effect rather than sampling noise, though 50 queries on one repository is not grounds
for changing a default.

**They do not change *which* files are found.** Recall@10 was 0.8300 in all four runs,
to four decimal places. Whatever HyDE and FLARE are doing on this repository, it is
re-ordering the same result set — not widening it.

### All of the above was measured with the query planner switched off

Which nobody had asked for. Telemetry from the eval runs showed all 219 rows classified
as a single intent, `feature_flow`, out of eight `IntentType` values. That is
`default_plan()`'s hard-coded fallback — the planner's LLM client was `None`, so every
query took it.

The cause is one missing line, described under defect 7 below. Once fixed, the same
50-query set was measurably different:

| Deterministic config | nDCG@10 | Recall@10 | MRR |
|---|---|---|---|
| Planner inert (as shipped, credentials in `.env`) | 0.6539 | **0.8300** | 0.6353 |
| Planner active | **0.6808** | 0.7900 | **0.6754** |
| Δ | **+0.027** | **−0.040** | **+0.040** |

Every delta is 5–8× the ±0.005 noise floor established above, so none of it is sampling
noise. The planner is a genuine **precision-for-recall trade**: intent-based leg
selection narrows which retrieval legs run, so it surfaces slightly fewer relevant files
but ranks the ones it finds noticeably better. MRR moving +0.040 means the first useful
result arrives sooner, which is the part a user feels.

The point is not which configuration is better. It is that the number depended on
whether credentials happened to be exported into the process environment or read from
`.env` — an invisible difference, with no log line at the CLI's default level to
distinguish them. Any retrieval tuning done against this harness before the fix was
measuring a planner that was not running.

---

## Defects found

Every entry was reproduced before being written down. The recurring shape is a **feature
that was switched on and doing nothing** — and in each case the failure was invisible
from outside, because it logged at DEBUG while the CLI runs at WARNING, or was swallowed
by a bare `except`, or surfaced as a number nobody had reason to doubt.

### Fixed

| # | Defect | Evidence |
|---|---|---|
| 1 | Nested `.gitignore` files never read, though the docstring claimed they were | 543/915 files from an ignored 2.6 GB bundle |
| 2 | Taint parser read three fields wrong; every real flow discarded | 0 flows on a repo semgrep finds flows in; now 3/3 with correct source, sink, severity |
| 3 | nDCG@10 and Recall@10 could exceed 1.0 | recall@10 = 10.0 measured against a documented `[0,1]` |
| 4 | Sentinel rows stole slots from the vector search top-k | sentinels took the top 2 of 5 on an in-memory vec0 |
| 5 | PageRank boost had never once fired, and logged nothing | 0 `graph_*` tables; `OperationalError` caught at DEBUG |
| 6 | A failed file summary permanently cost a file its vectors | `chunks_total=0, chunks_embedded=0` for a file recorded as indexed |
| 7 | The query planner discarded its credentials, collapsing 8 intents to 1 | 8 textbook queries → 1 distinct intent; 8/8 after the fix |

Two of these deserve emphasis because the *obvious* fix was also wrong, and only
measurement showed it:

**Defect 2 shipped because a test asserted the bug.** The existing fixture was
hand-written with `severity` at the top level and `taint_sink` as a `{"location": …}`
dict. Real semgrep puts severity under `extra` and makes `taint_sink` a tagged
**list**, `["CliLoc", [location, code]]`. Calling `.get()` on that list raised
`AttributeError`, which the bare `except` turned into a dropped finding. A fabricated
fixture is worse than no test: it froze a wrong contract and gave the parser a green
light every release.

**Defect 4 could not be fixed in SQL.** `WHERE embedding MATCH ? AND chunk_id > 0 …
LIMIT ?` is *rejected* by sqlite-vec 0.1.9 — "A LIMIT or 'k = ?' constraint is required
on vec0 knn queries" — because the extra predicate stops its planner recognising the
`LIMIT`. The `k = ?` form is accepted but **silently wrong**: it filters after the ANN
cut, so `k=5` returned 4 rows, trading a polluted top-k for a short one. The fix filters
in Python and over-fetches by the exact sentinel count.

**Defect 7 is a missing line, and its siblings prove it.** `QueryPlanner.__init__` builds
its `LLMConfig` with `_env_file=None`, which disables dotenv loading. Two other sites use
that same shim — `Synthesizer` and `graph_rag` — and both follow it with
`model_copy(update={"azure_api_key": …, "azure_endpoint": …})` to carry the credentials
across from the `EmbedderConfig`; `graph_rag` even labels the block
`# Carry over credentials`. The planner omitted it. So credentials supplied via `.env` —
the documented route, and the reason `.env.example` exists — never reached its client,
`_plan_direct` fell back to `default_plan()` on every call, and eight intents became one.

This is also why synthesis was unaffected: `trelix ask` produced correct answers
throughout, because `Synthesizer` accepts an explicit `llm_config` *and* copies
credentials in its shim. Only the planner had neither.

Every existing planner test supplies **no** credentials, so all of them pass whether the
copy is there or not. That is the gap that let it ship, and the new tests close it by
asserting a client is built from credentials on the config with the process environment
explicitly cleared.

### Found, not fixed

Reported rather than changed, because each is outside the scope agreed for this work.
All were reproduced.

| Severity | Defect | Location |
|---|---|---|
| **Critical** | `trelix eval-synthesis` cannot ever produce a non-empty answer. `SynthesisEvalHarness` passes an `IndexConfig` where `Synthesizer` expects an `EmbedderConfig`; the resulting `AttributeError` is swallowed into `answer = ""`, and every query then scores a constant | `eval/synthesis.py` |
| **Critical** | `migrate-vectors --reset` cannot fix a dimension change, which is the one thing the tool recommends it for. It deletes rows and the recorded dimension but leaves the `FLOAT[n]` declaration; because the recorded dimension is now gone, `DimensionGuard` has nothing to compare against, so the next run **pays for a full embedding pass** before failing on the first insert | `cli/main.py`, `store/dimension_guard.py` |
| High | `trelix index` never prunes files that have vanished from the walk. `delete_file_by_path` is called only by the watchers, so deleted files stay indexed and searchable forever, and a polluted index cannot be cleaned by re-indexing — only by deleting it | `indexing/indexer.py` |
| High | `SparseConfig.model` defaults to `naver-splab/splade-code-distil`, which is not a resolvable HuggingFace id. `SparseEmbedder` returns empty dicts, so `sparse_embeddings` stays at 0 rows whether the flag is set or not | `core/config.py` |
| Medium | `trelix graph --visualize --json` silently writes no HTML. The `--json` branch returns before the `if visualize:` block, so the flag is accepted and ignored | `cli/main.py` |
| Medium | `GitLinker` drops every merge commit — `git log --name-only` with no `-m`/`--diff-merges` yields no file list for a merge | `indexing/git_linker.py` |
| Medium | The default ticket pattern `[A-Z]+-\d+` has no word boundary, so it matches `UTF-8`, `SHA-256` and `HTTP-400` as ticket references | `core/config.py` |
| Low | `trelix taint` reports a **clean scan** as `"No taint flows found. Ensure semgrep is installed"`, conflating "your code is clean" with "the tool is missing". The comment above that message already records two previously-fixed defects in it | `cli/main.py` |
| Low | `sub_chunks` has no foreign key and no delete path anywhere in `src/`, so rows orphan on every re-index | `store/db.py` |

### Not defects

Recorded because they look like defects and are not — the documentation gets these right,
and an earlier draft of this report wrongly listed the first as critical.

- **`TRELIX_WALKER_*`, `TRELIX_PARSER_*`, `TRELIX_CHUNKER_*` and `TRELIX_SPARSE_*` are
  ignored in `.env`.** Those four config groups declare `env_prefix` without `env_file`,
  so they read the process environment only. This is deliberate and
  `docs/CONFIGURATION.md` states it explicitly. It is why `scripts/self-index.sh` exists.
- **Multi-granularity chunking is Python-only and off by default** — also documented.

---

## Dimensions deliberately left empty

Each of these could have been made non-zero. Doing so would have produced rows that were
wrong, unmeasurable, or bought nothing, so they were left alone and recorded instead.

| Dimension | Why not |
|---|---|
| `sub_chunks` | Phase 2.6 embeds once **per symbol**, bypassing token batching and the TPM limiter — thousands of serial un-throttled round-trips. The table has no FK and no delete path, so rows orphan on every re-index. Python-only. And it is retro-blind (it iterates changed-or-new symbols), so it cannot be added later without a full re-index. To measure it, use a throwaway DB: `TRELIX_STORE_DB_PATH=/tmp/x.db TRELIX_CHUNKER_MULTI_GRANULARITY=true trelix index .` |
| `sparse_embeddings` | The default SPLADE model id does not resolve. Setting the flag writes 0 rows either way |
| `taint_flows` | **Genuinely clean.** semgrep scanned all 140 source files with `config/semgrep-taint.yaml`: 0 findings, 0 errors. Verified as a true negative, not broken rules — the same rules fire twice on a planted `os.environ` → `subprocess.run(shell=True)` flow |
| `artifacts` | Requires a live Jira/Linear connector sync. There is no offline route |
| `generic_edges` | This repository's git history contains **no real ticket references** — across 825 commits the default pattern matches 8 strings, all false positives (`UTF-8`, `SHA-256`, `HTTP-400`, …). Running `link-tickets` would be a no-op reported as a step |
| `graph_concepts` | `--concepts` samples `symbols[0:200]` from a query with no `ORDER BY`, so the sample is arbitrary. The LLM spend buys nothing interpretable |

## A coverage limitation, not a defect

`EXTENSION_MAP` has no entry for `.sh`, `.bash`, `.sql`, `.proto` or `Dockerfile`. So
`scripts/self-index.sh`, `e2e_test.sh` and the `Dockerfile` are invisible to the indexer.
The 21 supported languages are documented and shell is not among them — but for a
code-intelligence tool, CI glue and entrypoints are exactly what people search for.

---

## Reproducing this

```bash
scripts/self-index.sh --dry-run                     # what would be indexed, embeds nothing
scripts/self-index.sh                               # full index (needs Azure credentials)
scripts/verify-index.sh                             # 22 gates, including expected-empty
trelix graph . --visualize                          # graph_metadata + graph.html
                                                    # (omit --json: it skips the export)
trelix eval . --golden eval/golden.jsonl            # nDCG@10 / Recall@10 / MRR
trelix taint . --rules config/semgrep-taint.yaml    # offline, pinnable taint scan
python scripts/measure_index_hygiene.py . --json    # corpus + retrieval noise
```

Raw measurements: `docs/reports/index-hygiene-before.json`,
`docs/reports/index-hygiene-after.json`.

The pre-existing 384-dim index was preserved rather than deleted, at
`.trelix/index.db.pre-v3.1.2-384dim.bak` (103 MB, gitignored). Delete it when the
before/after comparison is no longer wanted.
