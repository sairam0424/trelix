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

| Symbol | Deprecated in | Removal target | Replacement |
|--------|--------------|----------------|-------------|
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` env var | v2.4.0 | v4.0.0 | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` |

---

## Breaking Changes

Breaking changes are only made in **MAJOR** versions (v3.0.0, v4.0.0, etc.).

Before a MAJOR release:
- All breaking changes are listed in CHANGELOG under a `### Breaking Changes` heading,
  with the exact old → new form
- A minimum 3-month deprecation period for any removed feature
- A standalone guide at `docs/migration/v{N}-to-v{N+1}.md` **only when that major
  actually removes or changes something** — see immediately below

**`docs/migration/` does not exist, and v3.0.0 has no v2-to-v3 guide.** That is not an
omission to be patched with a stub. v3.0.0 (2026-08-13) shipped no breaking changes at
all; its CHANGELOG entry says so outright — "Everything here is additive and OFF by
default. No breaking changes: upgrading from v2.12.0 needs no reindex and no migration"
— because the one queued removal (`TRELIX_RETRIEVAL_FLARE_MAX_ITER`, retargeted to
v4.0.0 in the section below) was deferred, so the major bump bought a feature surface
rather than an incompatibility. A
guide whose only honest content is "nothing to do" costs a reader their trust in the
directory, which is precisely the trust v4.0.0 will need when it puts something real
there. The unchecked "Create migration guide at `docs/migration/v2-to-v3.md`" box in
[v3-0-0-breaking-changes.md](superpowers/plans/v3-0-0-breaking-changes.md) is unchecked
for the same reason, and should stay unchecked.

The earlier wording promised the file unconditionally, which made this document false
the moment v3.0.0 tagged.

### v4.0.0 Breaking Changes (planned)

The following deprecated items will be removed in v4.0.0. All have `DeprecationWarning` or `AliasChoices` backward-compat shims active since the version listed.

> **Retargeted from v3.0.0.** v3.0.0 shipped on 2026-08-13 without this
> removal and the shim is still live, so the old deadline has already
> passed. Per the policy above — remove only on a MAJOR bump — the next
> opportunity is v4.0.0. The env var continues to work in all v3.x
> releases, with a `DeprecationWarning`.

| Item | Deprecated in | Old name | New name | File:line |
|------|--------------|----------|----------|-----------|
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` env var | v2.4.0 | `TRELIX_RETRIEVAL_FLARE_MAX_ITER` | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `src/trelix/core/config.py:577` |

**Migration**: Set `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` instead of `TRELIX_RETRIEVAL_FLARE_MAX_ITER` in your environment or config files. The old name emits `DeprecationWarning` at `RetrievalConfig()` instantiation and will be removed in v4.0.0.

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
matched neither document: `trelix-mcp` tracks the core version, while the other two are
frozen at 2.4.0. Lockstep wins on mechanism, not on preference:

- **There is no independent cadence to be on.** `.github/workflows/release.yml` triggers
  on `push: tags: v*` — a core tag — and its `publish` job builds and uploads all four
  distributions in one go. No workflow, and no `workflow_dispatch`, can release an
  adapter by itself. "Own cadence" described machinery that does not exist.
- **A frozen version number cannot ship a fix.** Every publish step passes
  `skip-existing: true` (deliberately, so re-running a partially failed release is
  safe), so rebuilding an already-published version uploads nothing and still goes
  green. The tree's `trelix-langchain` 2.4.0 now declares `trelix>=3.0.0` and adds the
  `license`/classifier metadata that left PyPI showing "License: UNSPECIFIED" for seven
  releases — and none of it can reach users while the stamp stays 2.4.0. This has
  already happened once at this exact seam: `docs/v2.4.0-world-release-report.md`
  records both adapters stranded at 2.0.0 while the docs advertised 2.4.0, making
  `pip install trelix-langchain==2.4.0` a 404.
- **The number misinforms.** A package stamped 2.4.0 that requires `trelix>=3.0.0` tells
  a reader the opposite of the truth about which core it pairs with.
- **One contract cannot have two version lines.** `TrelixRetriever` and
  `TrelixIndexRetriever` are already listed in the Guarantees table above and in
  CONTRIBUTING.md's stable-API list, i.e. under *core's* SemVer promise. If the adapter
  versions independently, "not without a major version bump" has no referent — whose
  major?

### What reality has to change to comply

Compliant today: `trelix-mcp`, whose `pyproject.toml` and `server.json` stamps are
gated against the release tag by `release.yml`'s `verify-version` job.

Outstanding, and not fixable in this document:

1. Bump `packages/trelix-langchain/pyproject.toml` and
   `packages/trelix-llama-index/pyproject.toml` from `2.4.0` to the core version. Until
   then their raised `trelix>=3.0.0` floor and their license metadata are unpublishable.
2. Add both to `verify-version` in `.github/workflows/release.yml`, which currently
   checks neither — that omission is what let the drift persist silently.
3. Re-stamp the `==2.4.0` install pins in `docs/FAQ.md` and
   `docs/LANGCHAIN_LLAMAINDEX_GUIDE.md`; `docs/FAQ.md` also still asserts the
   "independent release cadence" retired here.

Until (1) and (2) land, the adapters are out of policy. That is a statement about the
tree, not a licence to keep the docs disagreeing.

---

## Database / Index Compatibility

`.trelix/index.db` schema upgrades are **always additive** and **idempotent** within a major version series:
- New columns added with `ALTER TABLE ADD COLUMN ... DEFAULT NULL`
- New tables added with `CREATE TABLE IF NOT EXISTS`
- Existing data never deleted by upgrade

Across MAJOR versions, re-indexing may be required (announced in CHANGELOG).
