# Migrating from trelix v2.x to v3.x

**Scope:** v2.12.0 → v3.1.2 (the current shipping version). If you are on a v2 release
older than v2.12.0, read the earlier
[v2.8.0 and v2.4.0 breaking changes](../BACKWARDS_COMPATIBILITY.md#v280-breaking-changes)
first — this guide does not repeat them.

**What this costs you:** one forced re-index, which is mandatory. Nothing else is required
to keep working — but two things are worth five minutes each: an env var rename that is
still optional until v4.0.0, and one `IndexConfig` kwarg that stopped being silently
ignored. Both are below.

---

## Why this guide exists at all

v3.0.0's own release note says "upgrading from v2.12.0 needs no reindex and no
migration". That was true of the **schema** and is still true of the **API**, but it was
wrong about your data, and v3.0.1 corrected it in writing:

> **A reindex is required to obtain a correct call graph.** This corrects v3.0.0's
> release note […] — true of the schema, but the graph built by the old parser is wrong.

So the honest summary of v2 → v3 is not "nothing to do". It is "one thing to do, and the
obvious command does not do it". That second half is the reason this file is not a stub.

---

## Required: force a full re-parse

### What went wrong in v2

The Python extractor recorded symbol indices during the AST walk and then did
`symbols.insert(0, <module>)` afterwards when the file had a module docstring, shifting
every element by one. Three index families resolved to a **valid but wrong** row:
`Symbol.parent_id`, `CallEdge.caller_id`, and `TypeEdge.from_symbol_id`. It never raised,
never logged, and never produced a null — which is why it shipped in v2.12.0 and again in
v3.0.0.

Measured on trelix's own 139 source files: **8,815 of 8,815 index references (100%) were
wrong** — 7,190 call edges, 1,525 parent links, 100 type edges. 434 parent links named
`<module>` instead of the declaring class, and 77 call edges were fabricated
self-recursion.

Fixed in v3.0.1 by reserving `symbols[0]` for the `<module>` symbol *before* the walk, in
`PythonExtractor` (`src/trelix/indexing/parser/extractors/python.py`; the comment there says
"Do not turn this back into `symbols.insert(0, ...)`"). **The fix applies to newly parsed
files only. It cannot repair rows already in your index.**

### The trap: `trelix index` alone is a no-op here

`IndexConfig.incremental` defaults to `True`, and the incremental pre-filter in
`Indexer.index()` (`src/trelix/indexing/indexer.py` — grep for `if self.config.incremental:`)
skips any file whose content hash is unchanged:

```python
if self.config.incremental:
    to_parse = [f for f in files if self.db.get_file_hash(f.rel_path) != f.hash]
```

Upgrading trelix does not change your source files, so every hash still matches. A plain
`trelix index .` after the upgrade selects **zero** files, prints
`Nothing to index — all files up to date.`, exits 0, and leaves the wrong call graph
exactly where it was. Nothing warns you.

### The command that actually works

```bash
TRELIX_INCREMENTAL=false trelix index .
```

The `else` branch sets `to_parse = files`, forcing a re-parse of everything. Notes:

- **There is no CLI flag for this.** The env var is the only route; `trelix index` exposes
  no `--no-incremental`.
- **Do not delete the index instead.** A forced re-parse is cheaper and safer: `_insert_one`
  diffs symbols by qualified-name + content hash, so unchanged symbols are not rewritten,
  and you keep the `query_telemetry` history that a delete would throw away.
- `trelix update-index` and `trelix watch` go through `index_file()` and will **not** do
  this for you.

### What changes afterwards

Expect different — and correct — results from call-graph expansion, blast radius,
PageRank centrality, the knowledge graph, and symbol hierarchy. On trelix's own index
**100% of PageRank values and 95.1% of ranks change** (`CachingPlanner.plan` moves from
rank 1975 to 34). If you have dashboards or golden files pinned to v2 ranking output,
they will move. That is the fix landing, not a regression.

---

## Not breaking, despite what you might expect

Each row below was checked against the code at v3.1.2, because a guide that lists a break
that did not happen is worse than no guide.

| Change in v3.x | Breaking? | Verified how |
|---|---|---|
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` removal (was scheduled for v3.0.0) | **No — deferred to v4.0.0** | The legacy alias is still live in the `AliasChoices(...)` on `RetrievalConfig.flare_max_retries`; setting it yields `flare_max_retries=3` plus a `DeprecationWarning` |
| `RetrievalConfig.context_token_budget` typed `int` → `int \| None` | **No** | A widening, and the default is still `12_000` — the exact v2.12.0 number |
| `complete()` / `stream()` gained a `thinking` argument | **No** | Added as `thinking: bool = False`, so every existing call site is unaffected. `tool_call()` did not get it |
| `declaration_boost_enabled` FTS5 reweighting | **No** | Default weight `1.0` is a verified no-op — byte-identical to the old unweighted ranking |
| Audit trail / OIDC SSO tables | **No** | They live in a separate `audit.db`, created only when auditing or SSO is switched on. The index schema is untouched |
| Index/DB schema | **No** | Additive and idempotent; no `ALTER` that drops or retypes a column. This is why the required action above is a *re-parse*, not a *migration* |

Also non-breaking, and off by default, so they cannot change your behaviour until you
opt in: SeleCom context compression (`TRELIX_RETRIEVAL_COMPRESSION`), extended thinking
(`TRELIX_LLM_THINKING_ENABLED`), auto-derived context budget
(`TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null`), and `TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET`.

---

## One behaviour change worth knowing about

`IndexConfig` was the only aliased-config class in `core/config.py` missing
`populate_by_name=True`. Until v3.0.0, passing these two by **field name** was silently
ignored — the value fell back to the env var or the default, with no error, because
`model_config` also sets `extra="ignore"`:

```python
# v2.12.0: silently ignored, index built WITHOUT file summaries
IndexConfig(repo_path=".", file_summaries_enabled=True)   # -> False

# v3.x: honoured
IndexConfig(repo_path=".", file_summaries_enabled=True)   # -> True
```

Same for `telemetry_enabled`. This is a bug fix, not a removal, but it is the one change
here that can alter behaviour on upgrade **without** you editing anything: if you were
passing either kwarg by field name and unknowingly getting the default, the feature now
actually turns on. File summaries require LLM API access and cost money per file, so
check for this before your first v3 index run if you construct `IndexConfig` from Python.
Passing by alias (`TRELIX_FILE_SUMMARIES_ENABLED=`) worked in v2 and still works.

---

## Still deprecated — deadline moved, not lifted

| Symbol | Deprecated in | Now removed in | Replacement |
|---|---|---|---|
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` env var | v2.4.0 | **v4.0.0** (was v3.0.0) | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` |

It still works in every v3.x release and emits a `DeprecationWarning` at
`RetrievalConfig()` instantiation. Rename it now rather than at the v4.0.0 upgrade:

```bash
-export TRELIX_RETRIEVAL_FLARE_MAX_ITER=2
+export TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=2
```

The field `flare_max_iterations` is a separate, older matter and is **already gone** —
it was renamed to `flare_max_retries` in v2.4.0. Only the env-var alias survives.

---

## Upgrade checklist

- [ ] `pip install --upgrade trelix` (extras are unchanged; `sso` is new and optional)
- [ ] `TRELIX_INCREMENTAL=false trelix index .` — **not** a plain `trelix index`
- [ ] Rename `TRELIX_RETRIEVAL_FLARE_MAX_ITER` → `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES`
- [ ] If you build `IndexConfig` in Python, confirm `file_summaries_enabled` /
      `telemetry_enabled` kwargs say what you actually want — they are honoured now
- [ ] Re-baseline anything pinned to v2 call-graph or PageRank output

---

## Not verified here

- **The forced re-parse was verified by reading the code path, not by running a v2 → v3
  index on a large repo.** `TRELIX_INCREMENTAL=false` demonstrably flips
  `IndexConfig.incremental` to `False`, and the `else` branch demonstrably sets
  `to_parse = files`; the end-to-end wall-clock cost and the exact row-level delta on
  *your* repo are not measured.
- **The blast radius is Python-only, but re-parse everything anyway.** The v3.0.1 fix and
  its 100%-wrong measurement are specific to the Python extractor. The other three
  extractors were checked and never had the equivalent defect: `typescript.py`,
  `javascript.py`, and `csharp.py` all add the `<module>` symbol to the list *before* calling
  `_walk_top_level()`, so their recorded indices were always correct for the final layout —
  the same shape the Python extractor was rewritten into. If your repository has no Python,
  your call graph was not affected. The forced re-parse is still the right move: it costs
  little, and it is cheaper than auditing which of your files the old parser touched.
- **Line numbers are deliberately omitted throughout this guide.** `core/config.py` and
  `indexing/indexer.py` both moved by 70+ lines while this guide was being written; every
  reference here names a symbol or a greppable string instead. Two docs in this repo already
  carry `config.py` cites that now point at unrelated fields.
- **`packages/trelix-langchain` and `packages/trelix-llama-index` are still stamped
  2.4.0** while declaring `trelix>=3.0.0`. If you install a pinned adapter version, see
  [Integration Package Policy](../BACKWARDS_COMPATIBILITY.md#integration-package-policy)
  in BACKWARDS_COMPATIBILITY.md — the fix is not in this guide's scope.

---

## See also

- [CHANGELOG.md](../../CHANGELOG.md) — `[3.0.0]` through `[3.1.2]` in full
- [BACKWARDS_COMPATIBILITY.md](../BACKWARDS_COMPATIBILITY.md) — stability guarantees and
  the deprecation clock
- [ROADMAP.md](../ROADMAP.md) — what v4.0.0 removes
