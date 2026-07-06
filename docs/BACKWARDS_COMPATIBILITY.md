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
2. **Grace period** — minimum **2 minor versions** (e.g. deprecated in v2.4 → removed in v3.0)
3. **Migration guide** — CHANGELOG includes exact rename/replacement
4. **Remove** — only on MAJOR version bump

### Current deprecations

| Symbol | Deprecated in | Removal target | Replacement |
|--------|--------------|----------------|-------------|
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` env var | v2.4.0 | v3.0.0 | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` |

---

## Breaking Changes

Breaking changes are only made in **MAJOR** versions (v3.0.0, v4.0.0, etc.).

Before a MAJOR release:
- All breaking changes are listed in CHANGELOG with migration guides
- A migration guide doc is created at `docs/migration/v{N}-to-v{N+1}.md`
- A minimum 3-month deprecation period for any removed feature

### v3.0.0 Breaking Changes (planned)

The following deprecated items will be removed in v3.0.0. All have `DeprecationWarning` or `AliasChoices` backward-compat shims active since the version listed.

| Item | Deprecated in | Old name | New name | File:line |
|------|--------------|----------|----------|-----------|
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` env var | v2.4.0 | `TRELIX_RETRIEVAL_FLARE_MAX_ITER` | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `src/trelix/core/config.py:434` |

**Migration**: Set `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` instead of `TRELIX_RETRIEVAL_FLARE_MAX_ITER` in your environment or config files. The old name emits `DeprecationWarning` at `RetrievalConfig()` instantiation and will be removed in v3.0.0.

See [v3-0-0-breaking-changes.md](docs/superpowers/plans/v3-0-0-breaking-changes.md) for the complete v3.0.0 deprecation audit and removal schedule.

---

### v2.4.0 Breaking Changes
- **`search_code` MCP tool** — return type changed from `list[dict]` → `{results, next_cursor, total_available}`. See CHANGELOG for migration.

---

## Integration Package Policy

The integration packages (`trelix-langchain`, `trelix-llama-index`, `trelix-mcp`) follow the core version. When upstream frameworks (LangChain, LlamaIndex) release breaking changes, we:

1. Support the previous major version for 1 minor trelix release
2. Add the new version support in the same or next minor release
3. Drop old version support only on a trelix minor or major version bump

---

## Database / Index Compatibility

`.trelix/index.db` schema upgrades are **always additive** and **idempotent** within a major version series:
- New columns added with `ALTER TABLE ADD COLUMN ... DEFAULT NULL`
- New tables added with `CREATE TABLE IF NOT EXISTS`
- Existing data never deleted by upgrade

Across MAJOR versions, re-indexing may be required (announced in CHANGELOG).
