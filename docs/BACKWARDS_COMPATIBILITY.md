# trelix Backwards Compatibility Policy

trelix follows **Semantic Versioning** (SemVer 2.0.0): `MAJOR.MINOR.PATCH`.

---

## Guarantees

### What we guarantee (stable API surface)

| Component | Stability | Details |
|-----------|-----------|---------|
| `IndexConfig` constructor kwargs | **Stable** | All existing kwargs preserved across minor versions |
| `Retriever(config).retrieve(query)` | **Stable** | Signature and `RetrievedContext` return type |
| `Indexer(config).index()` | **Stable** | Signature and stats dict return type |
| `FederatedRetriever` public methods | **Stable** | `retrieve()`, `cache_stats()`, `clear_cache()` |
| `DiffReviewer.review()` | **Stable** | Both `hunks=` and `diff_text=` params |
| CLI commands | **Stable** | All flags documented in CLI_REFERENCE.md |
| MCP tool signatures | **Stable** | Tools, resources, prompts and their parameters |
| `query_telemetry` DB schema | **Stable** | Existing columns never removed (only additive) |
| Environment variable names | **Stable** | Old names emit `DeprecationWarning` before removal |

### What we do NOT guarantee (private/internal)

- Private methods (prefix `_`)
- Internal dataclass field order (use keyword arguments)
- Debug trace JSON format (`.trelix/debug/`)
- Vector store internal format (re-index required on major version)

---

## Deprecation Policy

1. **Announce** — deprecation documented in CHANGELOG, `DeprecationWarning` emitted at runtime
2. **Grace period** — minimum **2 minor versions** *and* minimum **3 months**, whichever
   lands later
3. **Migration guide** — CHANGELOG includes exact rename/replacement
4. **Remove** — only on MAJOR version bump

**This file is authoritative for the grace period.** CONTRIBUTING.md's "Deprecation
policy" section repeats the two numbers so a contributor never has to leave the file
they are in, and links here for the reasoning; if the two ever diverge again, this one
wins. They did diverge: CONTRIBUTING.md said "at least **one** minor version", which
would have authorised a removal this document forbids, so the answer a contributor got
depended on which file they happened to open.

The two clocks are not redundant, and the calendar one is what actually binds. trelix
ships minors fast — v2.4.0 (2026-07-04) through v2.12.0 (2026-08-03) is eight minor
releases in 30 days — so "2 minor versions" can elapse in under a fortnight, which
protects nobody's pinned dependency. The live deprecation below is the worked example:
`TRELIX_RETRIEVAL_FLARE_MAX_ITER` was deprecated in v2.4.0 and v3.0.0 shipped 40 days
later on 2026-08-13, so removing it there would have cleared the minor-version count
and broken the 3-month floor.

### Current deprecations

*(None at this time — the one tracked deprecation, `TRELIX_RETRIEVAL_FLARE_MAX_ITER`, was removed in v3.3.0; see "v3.3.0 Breaking Changes" below.)*

---

## Breaking Changes

Breaking changes are only made in a release explicitly flagged for them in CHANGELOG
under a `### Breaking Changes` heading. Historically that meant reserving them for a
MAJOR bump (v3.0.0's own bump was major, though it shipped no breaking changes — see
below); starting with this cycle, breaking changes are instead reserved for a
deliberately elevated **MINOR** bump (v3.3.0) rather than a MAJOR one — trelix's version
number is a risk signal for upgraders, not strict semver.

Before a flagged breaking release:
- All breaking changes are listed in CHANGELOG under a `### Breaking Changes` heading,
  with the exact old → new form
- A minimum 3-month deprecation period for any removed feature
- A standalone guide at `docs/migration/v{N}-to-v{N+1}.md` (or, for a flagged MINOR
  bump like v3.3.0, `docs/migration/v{N.M}-to-v{N.M+1}.md`) **only when that release
  actually removes or changes something** — see immediately below

**`docs/migration/v2-to-v3.md` exists, and v3.0.0 is not what earned it.** v3.0.0
(2026-08-13) shipped no breaking changes at all; its CHANGELOG entry says so outright —
"Everything here is additive and OFF by default. No breaking changes: upgrading from
v2.12.0 needs no reindex and no migration" — because the one queued removal
(`TRELIX_RETRIEVAL_FLARE_MAX_ITER`, retargeted to v3.3.0 in the section below) was
deferred, so the major bump bought a feature surface rather than an incompatibility. A
guide written on that basis would have had "nothing to do" as its only honest content,
and would have cost a reader their trust in the directory — precisely the trust v3.3.0
will need when it puts something real there. What earned the guide was v3.0.1 retracting
the "no reindex" claim: the Python extractor's off-by-one made 8,815 of 8,815 index
references wrong, and the obvious remediation is a silent no-op — a plain `trelix index`
selects zero files, prints "Nothing to index — all files up to date.", and exits 0. So
the guide's scope line reads "v2.12.0 → v3.1.2", its one mandatory step is
`TRELIX_INCREMENTAL=false trelix index .` (which no CLI flag exposes), and for the
adapter stamps it links back to the Integration Package Policy below rather than
restating it. The "Create migration guide at `docs/migration/v2-to-v3.md`" box in
[v3-0-0-breaking-changes.md](superpowers/plans/v3-0-0-breaking-changes.md) is still
unchecked and now lags the tree: the file it names was written for v3.0.1's reason, not
v3.0.0's.

The earlier wording promised the file unconditionally, which made this document false
the moment v3.0.0 tagged.

### Behaviour changes shipped as fixes: `trelix review` exit codes

Two exit codes of `trelix review` were added in patch releases on purpose, as correctness fixes
rather than flagged breaking changes: `3` in 3.4.2 (the review could not run, which used to
exit `0` and print "No issues found."), and `4` in 3.4.3 (the review ran but
left hunks unreviewed, which used to exit `0`; 3.4.2's own CHANGELOG entry recorded that
limitation). Both replace a false "clean" result, and a caller that treats any non-zero exit as
failure will now fail where it used to pass, which is the point. The escape hatch for `4` is
`TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION=1`, which restores the old exit `0` for a partial review;
there is none for `3`, because nothing had been reviewed. The CHANGELOG entry for `4` says so in
bold. The default of `4` is a judgement call, not a settled one: shipping it opt-in first and
flipping the default in a flagged MINOR release would follow this policy more strictly.

A `--base` or `--head` that git cannot resolve is the same kind of change: it used to print
"No changes found in diff." and exit `0`, and now exits `1` (the default `HEAD~1` in a repository
with a single commit included). So does any other `git diff` failure between refs that each resolve.
There is no escape hatch, for the same reason as for `3`: no diff had been read.

### Behaviour changes shipped as fixes: the MCP `resources.subscribe` capability

`trelix-mcp` used to force `resources.subscribe: true` into the capabilities it returns to a
client at connect time (both the `initialize` result and `server/discover`), although it does
not serve the `resources/subscribe` request: a client that sent it got JSON-RPC "Method not
found". The forced value is removed, so the server now reports `resources.subscribe: false`,
which is what it does. A client that gated on the capability now sees `false`, which is the
truth; a client that ignored the capability sees no difference. The `subscribe_resource` and
`unsubscribe_resource` tools work as before (only the `subscribe_resource` description text that
`tools/list` sends was reworded, because it told models to call the tool after seeing
`resources.subscribe: true`), and there is no escape hatch because none is needed: a client that
wants them calls the tools by name. What did not work before, and still does not, is delivery:
nothing starts a file watcher inside the `trelix-mcp` process, and `trelix watch` runs in a
separate process whose subscription registry is empty, so no MCP client receives
`notifications/resources/updated`. This was shipped as a fix rather than a flagged breaking
change because the old value was a false statement about the server, not a feature a working
client could depend on; the tool names, parameters and return values are untouched.

### Behaviour changes shipped as fixes: read surfaces on a repository with no index

Pointed at a repository that was never indexed, these used to answer with an empty result and
leave an empty `.trelix/index.db` behind, which made the repository look indexed (a later
`trelix search` said "no results" instead of "not indexed"). 3.4.2 fixed the CLI read commands;
the rest now refuse too, with the CLI's words (`No index found at <repo>/.trelix/index.db. Run
trelix index <repo> first.`) and without creating `.trelix/`:

| Surface | Before | Now |
|---------|--------|-----|
| MCP tools `search_code`, `get_symbol`, `blast_radius`, `build_knowledge_graph`, `graph_search_mcp`, `ask_agent`, `agent_list_sessions`, `agent_clear_session` | an empty result (`get_symbol` and `blast_radius`: `null` and `[]`) | a tool error (`isError: true`, the session carries on) |
| MCP resource handlers in `trelix_mcp.resources` | zero counts, an empty manifest, "Symbol not found" | `{"error": "No index found at ..."}` |
| MCP `federation_search_all` | an empty page | the unindexed repo is skipped and not counted in `repos_searched`; `error` is set only when none of the queried repos is indexed |
| REST `/search`, `/ask`, `/stats`, `/graph`, `/graph/communities`, `/graph/visualize`, `/graph/search` | `200` with zero results, zero counts or an empty graph | `400` with `{"detail": "No index found at ..."}` |
| `TrelixRetriever` (LangChain), `TrelixIndexRetriever` (LlamaIndex) | `[]` | raise `trelix.core.index_check.IndexNotFoundError`, a `FileNotFoundError` |
| `FederatedRetriever.retrieve` and `trelix search-all` | the unindexed repo contributed nothing | the same, but it is skipped before it is opened (so no index, and for a registered path that does not exist no directory, is created); `search-all` names it on stderr and exits `1` when every repo it would query is unindexed |

A caller that treated the empty answer as "nothing found" now gets an error where it had none,
which is the point: the old answer claimed a search that never happened. There is no toggle; the
remedy is `trelix index <repo>` (or the `index_codebase` tool, or `POST /index`), which still
creates the index. `trelix review` is deliberately not in the table: it keeps working on a
repository with no index, from the diff alone, and only stops creating the empty index.
`trelix eval`, `eval-synthesis`, `telemetry`, `agent sessions` and `taint` do not check yet
(unchanged).

`trelix-mcp`, `trelix-langchain` and `trelix-llama-index` now import `trelix.core.index_check`,
which the core ships from the release that contains this change. An older core under a newer
package fails on import (`trelix-mcp` at start-up, the adapters on their first query), so the
release that ships this raises their `trelix>=` floors to that release.

### Behaviour change shipped as a fix: `Synthesizer.stream()` on an empty retrieval

When retrieval found no results, `Synthesizer.synthesize()` (FLARE, `eval-synthesis`) already
answered with `[trelix] No relevant code found — cannot synthesize an answer.` and made no LLM
call, but `Synthesizer.stream()` (plain `trelix ask` with a non-local embedder, REST `GET /ask`)
sent the model `No relevant code found.` as the whole code context and streamed whatever it said.
That answer could not be grounded in the repository, so `stream()` now does what `synthesize()`
does:

| Surface | Before | Now |
|---------|--------|-----|
| `trelix ask <repo> <question> --provider openai` (any non-local embedder; FLARE and agentic mode off) | an LLM answer written without any retrieved code, exit `0` | the notice line on stdout, exit `0`, no LLM call |
| REST `GET /ask` | `data: <token>` frames of that answer, then `data: [DONE]` | `data: [trelix] No relevant code found — cannot synthesize an answer.` then `data: [DONE]`, no LLM call |
| `trelix ask` with `TRELIX_RETRIEVAL_FLARE=true`; `trelix eval-synthesis` | the notice, no LLM call | unchanged |
| `trelix ask --provider local` (context only) | the assembler's `No relevant code found.` | unchanged |
| `trelix ask --provider openai` with no LLM key configured, whatever retrieval found | `Synthesis failed: LLM not configured` on stderr, exit `1` | unchanged: `stream()` checks the key before the retrieval result, as `synthesize()` always did |

There is no toggle: no answer existed to lose, only one the retrieved code did not support. A
caller that parsed the streamed text now sees the notice literal (the same one `synthesize()` has
always returned) in its place; `Synthesizer.last_error` stays `None`, as it did for `synthesize()`.

Additive: `Synthesizer.last_abstain_reason` (`"no_results"`, `"insufficient_evidence"` or `None`)
says why the last call abstained, and `trelix.retrieval.citations` gains `NO_RESULTS_MESSAGE`,
`ABSTAIN_PREFIX`, `AbstainReason` and `is_abstention()`. With `TRELIX_RETRIEVAL_CITATIONS=true`
the synthesis prompt also asks the model to reply with exactly one line starting
`INSUFFICIENT_EVIDENCE:` when the context does not contain what the question needs; that line
streams as an answer (exit `0`) and FLARE treats it as an uncertainty phrase. With the flag off (the
default) the prompts are unchanged.

### Behaviour change: MCP list results are bounded

`trelix-mcp` list tools used to return as much as the caller asked for. Measured through an
in-process client with 100,000-character bodies, `search_code` with `k=100` was about 97,000
characters of text and 197,000 on the wire (FastMCP sends a dict result as a text block, with every
quote escaped, and again as `structuredContent`), past Claude Code's 25,000-token cap for one tool
result; `graph_search_mcp` had no upper bound on `k`, and `blast_radius` returned every dependent
file (about 127 characters each with short paths, more with long ones). Tool names and parameters
are unchanged and every new argument has a default that keeps the old meaning, but these results now
differ for a caller that asked for a lot:

| Before | Now | Escape hatch |
|--------|-----|--------------|
| `k` (and `agent_list_sessions`'s `limit`) of any size was honoured | clamped to `1..50`; the response says what was used (`page_size`) | `TRELIX_MCP_MAX_K=<n>` |
| A list result of any size | cut to fit `TRELIX_MCP_MAX_RESULT_CHARS` (15,000) by dropping the tail: the budget counts both copies a client is sent, so the response is at most 30,000 characters and its text under 15,000. `truncated` and `omitted` say so and `next_cursor` continues from the first dropped result. A bare array (`blast_radius`, `graph_search_mcp`) keeps the JSON array in the first text block and adds a note block and `_meta.trelix`. One result is always kept, so a single result longer than the budget is returned whole | `TRELIX_MCP_MAX_RESULT_CHARS=0` turns the cut off |
| `blast_radius` returned every dependent | at most 100 by default; `limit` raises that up to 500, which is the most it can return (it has no offset), and the character cut still applies (about 108 dependents fit with paths like `src/callers/caller_module_0000.py`, about 84 with 72-character ones) | `limit=500` with `TRELIX_MCP_MAX_RESULT_CHARS=0` |
| `get_symbol` returned the whole body | a body over 20,000 characters is cut and `body_truncated` is `true` | `max_body_chars=0` |
| `agent_list_sessions` returned each session's most recent prompt (`query`) whole | a `query` over 300 characters is cut to its first 300 and that session gets `query_truncated: true`, so one long prompt cannot push a response past 30,000 characters | none: the full prompt is only in the caller's latest `ask_agent` call for that session |
| A negative `cursor` sliced from the end of the result list | an error result (`isError: true`) | none: use `0` or the previous `next_cursor` |

Everything else is additive: `search_code`, `federation_search_all` and `agent_list_sessions` responses
gain `page_size`, `truncated` and `omitted`, `get_symbol` gains `body_truncated`, and the new
arguments are `detail` (`concise` or `detailed`, default `detailed`, on `search_code`,
`graph_search_mcp` and `federation_search_all`), `limit` (`blast_radius`) and `max_body_chars`
(`get_symbol`). The first text block of every result keeps the keys it had, which is what the VS Code
extension reads; it also reads `_meta.trelix.total_available`, so its "N dependents" lens and its `@trelix /impact` chat
command show the real count and say how many they list when a long list was cut (`150 dependents (showing 100)`).
An extension build that predates this reads only the array and shows the cut length with no sign that the list
was cut. The `federation_search_all` error and empty-registry responses keep their shorter shape.
A value of `TRELIX_MCP_MAX_K` or `TRELIX_MCP_MAX_RESULT_CHARS` that is not an integer (at least 1 for
the first, at least 0 for the second) stops `trelix-mcp` at start-up with exit code 2.

### Additive: MCP tool annotations, instructions, tool order and `--tools`

`trelix-mcp` now sends tool annotation hints, server `instructions` and a five-minute cache hint, lists its tools in a fixed order (the two subscription tools, which came first, now come last) and accepts `--tools core|full` (default `full`, every tool); this is additive, because no tool name, parameter or result changes.

### v3.3.0 Breaking Changes

The following deprecated item was removed in v3.3.0. Its `AliasChoices`/`DeprecationWarning` backward-compat shim had been active since v2.4.0.

> **Retargeted from v3.0.0.** v3.0.0 shipped on 2026-08-13 without this
> removal and the shim stayed live, so the original deadline passed. Per
> the policy above — remove only in a release flagged for breaking changes,
> now a deliberately elevated MINOR bump rather than a MAJOR one — the next
> opportunity was v3.3.0.

| Item | Deprecated in | Removed in | Old name | New name |
|------|--------------|-----------|----------|----------|
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` env var | v2.4.0 | v3.3.0 | `TRELIX_RETRIEVAL_FLARE_MAX_ITER` | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` |

**Migration**: Set `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` instead of `TRELIX_RETRIEVAL_FLARE_MAX_ITER` in your environment or config files. As of v3.3.0 the old name is silently ignored — it no longer sets `flare_max_retries` and no longer emits `DeprecationWarning`.

See [v3-0-0-breaking-changes.md](superpowers/plans/v3-0-0-breaking-changes.md) for the complete v3.0.0 deprecation audit and removal schedule.

---

### v2.8.0 Breaking Changes
- **`AgentLoop.run()`** — signature changed from `run(query: str) -> str` to `run(query: str, session_id: str | None = None) -> tuple[str, str]`, to support persisted multi-turn agent sessions (`agent_sessions`/`agent_turns` tables). See CHANGELOG for migration.

This was a deliberate exception to the "breaking changes only in MAJOR versions" rule above, made in an otherwise non-breaking minor release. `AgentLoop` is not listed in the stable-API-surface table in the Guarantees section and is not exported from trelix's top-level `__all__` (`src/trelix/__init__.py`), so it was judged not to be a documented-stable public API surface.

---

### v2.4.0 Breaking Changes
- **`search_code` MCP tool** — return type changed from `list[dict]` → `{results, next_cursor, total_available}`. See CHANGELOG for migration.

---

## Integration Package Policy

**The rule: all three integration packages (`trelix-langchain`, `trelix-llama-index`,
`trelix-mcp`) carry the core version, and are released only by a core tag.** When
upstream frameworks (LangChain, LlamaIndex) release breaking changes, we:

1. Support the previous major version for 1 minor trelix release
2. Add the new version support in the same or next minor release
3. Drop old version support only on a trelix minor or major version bump

### Why lockstep, and not "independent cadence"

CONTRIBUTING.md used to claim the opposite — that `trelix-langchain` and
`trelix-llama-index` "version independently … on their own cadence" — and reality
matched neither document: `trelix-mcp` tracked the core version, while the other two sat
frozen at 2.4.0. Both adapters now carry 3.1.2 with the core, and the record of what
closed that gap is below. Lockstep wins on mechanism, not on preference:

- **There is no independent cadence to be on.** `.github/workflows/release.yml` triggers
  on `push: tags: v*` — a core tag — and a single run builds all four distributions in its
  `build-distributions` job and uploads all four in its `publish` job. No workflow, and no
  `workflow_dispatch`, can release an adapter by itself. "Own cadence" described machinery
  that does not exist.
- **A frozen version number cannot ship a fix.** Every publish step passes
  `skip-existing: true` (deliberately, so re-running a partially failed release is
  safe), so rebuilding an already-published version uploads nothing and still goes
  green. `trelix-langchain` sat at 2.4.0 while declaring `trelix>=3.0.0` and carrying the
  `license`/classifier metadata that left PyPI showing "License: UNSPECIFIED" for seven
  releases — none of which could reach users while the stamp stayed there. This has
  already happened once at this exact seam: `docs/v2.4.0-world-release-report.md`
  records both adapters stranded at 2.0.0 while the docs advertised 2.4.0, making
  `pip install trelix-langchain==2.4.0` a 404.
- **The number misinformed.** A package stamped 2.4.0 that required `trelix>=3.0.0` told
  a reader the opposite of the truth about which core it pairs with. 3.1.2 against that
  same, unchanged `trelix>=3.0.0` floor reads as what it is: an adapter on the core's
  version line that works with core 3.0.0 and up.
- **One contract cannot have two version lines.** `TrelixRetriever` and
  `TrelixIndexRetriever` are already listed in the Guarantees table above and in
  CONTRIBUTING.md's stable-API list, i.e. under *core's* SemVer promise. If the adapter
  versions independently, "not without a major version bump" has no referent — whose
  major?

### What had to change to comply, and what closed it

All four distributions are now gated against the release tag. `release.yml`'s
`verify-version` job checks **twelve** stamps: root `pyproject.toml`,
`src/trelix/__init__.py`, `helm/trelix/Chart.yaml` `appVersion`, `helm/trelix/values.yaml`
`image.tag`, `trelix-mcp`'s `pyproject.toml` and both of its `server.json` stamps, and —
new here — `trelix_mcp.__version__` plus each adapter's `pyproject.toml` version and its
runtime `__version__`. That is one check per stamp, with no exceptions: `trelix_mcp`'s
runtime stamp had been documented as the one the gate skipped, to be verified by hand.
Before this change only `trelix-mcp`'s dist and `server.json` stamps were compliant. The
three items below are the audit trail of what was wrong; each is now closed.

1. **Bump `packages/trelix-langchain/pyproject.toml` and
   `packages/trelix-llama-index/pyproject.toml` from `2.4.0` to the core version.**
   Closed: both read `3.1.2`, as do both `__version__` constants and both adapters'
   `tests/test_retriever.py` assertions. That releases the `trelix>=3.0.0` floor and the
   license metadata, neither of which any publish could carry while the stamp was frozen.
2. **Add both to `verify-version` in `.github/workflows/release.yml`, which checked
   neither** — that omission is what let the drift persist silently. Closed: five
   `check()` calls added — four adapter stamps plus `trelix_mcp.__version__` — taking the
   job from seven stamps to twelve. Run with `GITHUB_REF_NAME=v3.1.2`, all twelve print
   "ok" and it exits 0; with `v3.2.0`, all twelve emit an `::error file=` annotation and it
   exits 1 — the gate reports the whole set, not the first mismatch.
3. **Re-stamp the `==2.4.0` install pins in `docs/FAQ.md` and
   `docs/LANGCHAIN_LLAMAINDEX_GUIDE.md`, and retire `docs/FAQ.md`'s "independent release
   cadence" claim.** Closed in this same change: `docs/FAQ.md` now pins `==3.1.2` and
   states outright that the two are **not** on an independent cadence, and the guide's two
   install lines dropped the pin entirely, so they cannot go stale at the next tag.

The same workflow's `test` job now installs both adapters and runs all four suites, as
separate `pytest` invocations — both adapter `tests/` directories hold an `__init__.py`
and a `test_retriever.py`, so one collection over both aborts on "import file mismatch".
Before this the adapter suites ran only in `ci.yml`, which never fires on a tag, so the
release path could gate a stamp it had never executed a test against.

Two things this deliberately did not change. The dependency floor stays `trelix>=3.0.0`
in both adapters, because 3.0.0 is the lowest published core verified to expose every
name they import (`packages/trelix-langchain/pyproject.toml:35`). Lockstep governs the
version stamp, which is identity; the floor is a compatibility contract, and raising one
for a release-cadence reason is exactly the mistake CHANGELOG's v2.7.1 entry reverted
("Unjustified dependency-floor bumps reverted", after re-checking every import). And the
stamp encodes no behaviour change: `git diff v2.4.0..HEAD -- packages/trelix-*/src` is 13
insertions and 3 deletions, all type annotations. 2.4.0 → 3.1.2 is a re-alignment onto
the core's version line, not eight minors of adapter change.

---

## Database / Index Compatibility

`.trelix/index.db` schema upgrades are **always additive** and **idempotent** within a major version series:
- New columns added with `ALTER TABLE ADD COLUMN ... DEFAULT NULL`
- New tables added with `CREATE TABLE IF NOT EXISTS`
- Existing data never deleted by upgrade

Across MAJOR versions, re-indexing may be required (announced in CHANGELOG).
