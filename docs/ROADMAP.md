# trelix Roadmap

> **Status:** Living document — updated with each release.
> **Version:** 3.1.2 (current)

This roadmap describes planned features, research directions, and long-term vision for trelix. Items are organized by phase; specific timelines are intentionally loose to reflect research-driven development.

---

## ✅ Shipped (v2.0 – v3.1.2)

| Version | Feature |
|---------|---------|
| v2.0.0 | BGE-Code-v1 embedder, PLAID reranker, LanceDB backend, REST API, Knowledge Graph |
| v2.1.0 | Beast-mode retrieval: FLARE loop, HyDE fallback, multi-query expansion, query telemetry, CoIR eval |
| v2.2.0 | Agentic ReAct loop, data-flow analysis, SPLADE sparse retrieval, multi-granularity indexing |
| v2.3.0 | DimensionGuard, MultiQueryExpander wiring, MCP Resources + Prompts, DiffReviewer, FederatedRetriever |
| v2.4.0 | flare_max_retries rename, expansion observability, federation cache, GitHub PR review, watch-all, MCP pagination |
| v2.5.0 | Multi-query expansion wiring (TRELIX_RETRIEVAL_MULTI_QUERY), DimensionGuard at watch startup, MCP resource subscriptions, v3.0.0 deprecation audit |
| v2.6.0 | Incremental Louvain, Short-query lexical fallback, XTR reranker (experimental), GroUSE synthesis eval harness |
| v2.7.0 | Watch bridge wired ✅, DB index ✅, AdaptiveRouter fix ✅, Cross-repo symbols ✅, Diff embeddings ✅, Streaming indexing ✅, VS Code extension ✅, GitHub App PR review ✅ |
| v2.7.1 | Release pipeline asset-collision fix, Linux binary in CI, reverted unjustified dependency floors, trelix-mcp tests wired into CI, doc/env-var corrections |
| v2.7.2 | Qdrant Cloud gRPC/timeout readiness, incremental per-symbol embedding on partial re-index, opt-in parallel BM25 read pool, Linux ARM64 binary release, SparseEmbedder/MCP-stdout/BM25 concurrency race fixes, FK-repoint on partial re-index, qdrant-client 1.18 migration |
| v2.7.3 | README end-to-end audit + rewrite (fixed 15+ factual bugs, redrew the retrieval-pipeline diagram, deduplicated content into `docs/`), backfilled the empty v2.2.0 CHANGELOG entry, migrated Troubleshooting entries into `docs/TROUBLESHOOTING.md` |
| v2.8.0 | Multi-repo support in MCP (4 new federation tools), persistent agent (ReAct loop) session memory (3 new MCP tools + CLI `agent sessions` sub-app), fixed 2 latent `FederatedRetriever` bugs (lost repo-provenance tagging, unused `RepoEntry.weight`) |
| v2.8.1 | Pre-push audit hardening follow-up (issue #69): MCP `config_path` path confinement, federation repo-count/fan-out caps (`TRELIX_FEDERATION_MAX_REPOS`), stable `federation_search_all` pagination (fixed fetch width independent of cursor), test-hardening (deterministic weight-pairing test, stronger `ask_agent` assertion) |
| v2.9.0 | Python 3.13 support (`tree-sitter-language-pack` migration), OpenTelemetry tracing for the retrieval pipeline, typed REST API response models + `/search` cursor pagination, `@trelix/sdk` TypeScript client, official Docker image + Helm chart, VS Code extension hardening (XSS fix, bundler) + live search refinement, GitHub App GA hardening (installation-token auth, webhook signature verification, Check-annotation posting). All additive/opt-in — no breaking changes. |
| v2.10.0 | REST API-key auth (`TRELIX_API_AUTH_TOKEN`) + HTTP-layer OpenTelemetry spans, `/parse` endpoint + per-source context budgeting, leg-level path filtering + `intent_hint`/`hyde_snippet_hint` on `/search`, cross-source `generic_edges` table + git-log ticket linker (`trelix link-tickets`), Jira/TestRail artifact connectors (`trelix connector sync`), and a PageRank eval showing +0.040 recall@5 / +0.031 NDCG@10 from the new bidirectional graph edges. All new surface additive/opt-in — no breaking changes. |
| v2.11.0 | Connector-fetched artifacts auto-link into `generic_edges` on sync (`ArtifactLinker`, `trelix link-artifacts`), opt-in Personalized PageRank (`TRELIX_RETRIEVAL_PAGERANK_PERSONALIZATION`), a unified retry/backoff contract (`tenacity`-based) shared by every LLM backend/embedder/connector/reranker/PR client, structured JSON logging with OTel trace correlation, and two new artifact connectors — Xray Cloud and Linear (`trelix connector sync <repo> xray\|linear`). Plus two real bugs found by live-testing Jira against a production site (ADF descriptions silently dropped to empty bodies; bad credentials silently reported as success) and a `.env`-leakage test-isolation fix. All new surface additive/opt-in — no breaking changes. |
| v2.11.1 | Fixed a path-containment bug in `/graph/visualize`'s `output` query param: a raw string-prefix check accepted a sibling directory sharing the same prefix as `<repo>/.trelix` (e.g. `<repo>/.trelix-evil`), letting a caller write an arbitrary-named HTML file outside the intended directory. Switched to `Path.is_relative_to()`, mirroring `/parse`'s existing correct check in the same file. Found via a full production dry-run of v2.11.0's REST API surface. |
| v2.12.0 | Fixed a real call-graph/type-edge resolver bug (same-named methods across classes silently wired to the wrong symbol; ~2,400 provably-wrong edges removed on trelix's own self-index), activated per-leg RRF weight config (`TRELIX_RETRIEVAL_LEG_WEIGHT_<LEG>`) at the primary retrieval call site, clamped an unbounded `Retry-After` header that could crash the retry loop, capped unbounded Jira ADF recursion, documented `trelix eval-synthesis`, and extended CI to lint/type-check `trelix-mcp`/`trelix-langchain`/`trelix-llama-index` — which surfaced and fixed a missing PEP 561 `py.typed` marker on the core `trelix` package and an unresolvable `mcp`/`fastmcp` dependency-floor landmine. |
| v3.0.0 | Six feature areas, all additive and OFF by default: Anthropic extended thinking as a per-call-site request parameter, a model-aware context budget (`llm/context_windows.py`), an actionable VS Code extension (code lenses + `@trelix` chat participant + hover provider), a hash-chained audit trail in its own `audit.db`, OIDC SSO with an asymmetric-only algorithm allowlist, and SeleCom query-conditioned context compression. Plus opt-in FTS5 declaration-boost ranking. Six classes of live defect fixed, including an `AttributeError` that crashed every Anthropic response containing a thinking block and an unauthenticated `MarkupError` DoS of `trelix audit list`. A major bump for the size of the feature surface, **not** for an incompatibility — the one queued removal was deferred to v4.0.0. |
| v3.0.1 | The Python extractor's `symbols.insert(0, <module>)` shifted every recorded index by one in any file with a module docstring: **8,815 of 8,815 index references (100%) wrong** on trelix's own source — 7,190 call edges, 1,525 parent links, 100 type edges. Silent in v2.12.0 and v3.0.0 because a pre-insert index always resolved to a valid-but-wrong row. **A forced re-parse is required** — see [migration/v2-to-v3.md](migration/v2-to-v3.md), because a plain `trelix index` skips unchanged hashes and fixes nothing. Also restored 177 byte-identical backcompat assertions that had gone dark and unblocked the integration suite. |
| v3.1.0 | A correctness release for everything trelix renders — terminal, machine-readable pipe, and LLM prompt. Eight defects in one class; no config, schema, or reindex change, and output byte-identical wherever no bug was tripped (51 new regression tests, 2,394 → 2,445). |
| v3.1.1 | A security-documentation release. `SECURITY.md` had claimed since v2.x that trelix "does not follow symlinks outside the repo boundary" — `FileWalker` had no symlink handling at all, so an out-of-tree file was indexed and reported as if it sat inside. Adds opt-in `TRELIX_WALKER_FOLLOW_SYMLINKS=false` to make the boundary real, plus three smaller document defects found in the same audit. |
| v3.1.2 | Self-index findings plus fixes: a symlink cycle that multiplied one file into dozens of copies, all three MCP prompts failing at `prompts/get`, and a deprecation deadline that had already passed (the `TRELIX_RETRIEVAL_FLARE_MAX_ITER` retarget to v4.0.0). Documentation version stamps advanced to 3.1.2 across 14 files, and `CONTRIBUTING.md` gained a release checklist naming the five real version sites. |

---

## 🐛 v2.5.1 — Backlog (bugs / hardening from v2.5.0)

- [x] **SparseEmbedder TOCTOU under parallel multi-query** — add `threading.Lock` around lazy-init path hit by `ThreadPoolExecutor` workers ✅ (shipped v2.5.1)
- [x] **`send_resource_notification` stdout isolation** — fix asyncio transport conflict when FastMCP writes notifications to stdout concurrently ✅ (shipped v2.5.1)
- [x] **`SubscriptionRegistry` max-subscriber cap / TTL eviction** — unbounded subscription growth; add configurable cap and TTL-based cleanup ✅ (shipped v2.5.1)
- [x] **Watch bridge: wire `notify_file_changed` into `FileWatcher._do_reindex` callback** — MCP subscribers not notified after file-change re-index completes ✅ (shipped v2.7.0)

---

## 🏗️ v2.6.0 — Scale & Performance (Q4 2026)

**Goal:** Handle 1M+ symbol codebases without degradation.

### Plan A: Incremental Louvain Community Detection — ✅ Shipped (v2.6.0)
- [x] **DF Louvain frontier heuristic** — Maintains prior partition, reprocesses only affected-vertex frontier
- [x] `compute_affected_frontier(G, seed_nodes, partition)` — Computes affected nodes
- [x] `detect_communities_incremental()` — Incremental Louvain with >50% frontier fallback
- [x] **GraphUpdater** — Stores `_prev_partition`, uses incremental detection on file changes

### Remaining backlog
- [x] **Cross-repo symbol resolution** — SCIP-style IDs, FederatedRetriever.resolve_symbol() ✅ (shipped v2.7.0)
- [x] **Semantic diff embeddings** — CCRep-style before/after body pair embeddings ✅ (shipped v2.7.0)
- [x] **Streaming indexing** — generator + bounded Queue, try/finally sentinel ✅ (shipped v2.7.0)
- [x] **Qdrant Cloud integration** — first-class remote vector store with auto-migration ✅ (shipped v2.6.x)
- [x] **Incremental embedding** — only re-embed changed symbols on partial re-index ✅ (shipped v2.6.x)
- [x] **Parallel BM25 shard** — FTS5 read-only shards for read-heavy deployments ✅ (shipped v2.6.x)
- [x] **Binary releases** — single-file executable for Linux ARM64 ✅ (shipped v2.6.x; Windows ARM64 excluded — tree-sitter-languages/sqlite-vec publish no win_arm64 wheel or sdist)

---

## 🌐 v4.0.0 — Breaking Changes & Ecosystem (H1 2027)

**Goal:** Clean API surface + first-class cloud deployment.

- [ ] **Remove deprecated** — the `TRELIX_RETRIEVAL_FLARE_MAX_ITER` **env-var alias**
      is removed (deprecated in v2.4). Naming precision matters here, because this
      item previously named `flare_max_iterations`: that *field* was renamed to
      `flare_max_retries` in v2.4.0 and is already gone — `test_config.py:621`
      asserts `not hasattr(cfg, "flare_max_iterations")`. What survives, and what
      v4.0.0 actually removes, is only the legacy env name.
      Verified live at 3.1.2: the alias is the second entry in the
      `AliasChoices(...)` on `RetrievalConfig.flare_max_retries` in
      `src/trelix/core/config.py`, and the warning comes from the
      `_warn_deprecated_flare_iter_env` model validator directly below it;
      setting `TRELIX_RETRIEVAL_FLARE_MAX_ITER=3` still yields
      `flare_max_retries == 3` plus a `DeprecationWarning`.
      **Grep those two names; do not trust a line number.** `config.py` is 1,564
      lines and moves every release, and both existing cites for this deprecation
      are already wrong — the removal checklist in
      `docs/superpowers/plans/v3-0-0-breaking-changes.md` says `config.py:429-452`
      and BACKWARDS_COMPATIBILITY.md says `config.py:577`; each now lands on an
      unrelated field.
      Everything else originally scoped under v3.0.0 shipped
      additively in v2.9.0/v2.10.0 instead (see the Shipped table above).
      **This slipped past v3.0.0**, which shipped 2026-08-13 with the alias still
      live; since removal is only permitted on a MAJOR bump, it moves to v4.0.0.
      See [BACKWARDS_COMPATIBILITY.md](BACKWARDS_COMPATIBILITY.md) and
      [migration/v2-to-v3.md](migration/v2-to-v3.md).
- [ ] **MCP `InputRequiredResult` pattern** — adopt SEP-2322's
      `InputRequiredResult` pattern for `ask_agent`'s input-wait behavior.
      This is a real behavioral/protocol change, not a dependency-floor
      edit — needs its own design pass. (The `mcp`/`fastmcp` version-floor
      bump this item used to bundle shipped in v2.12.0 as
      `mcp>=1.24.0,<2.0`/`fastmcp>=3.4.0`.)
- [ ] **MCP streaming** — true streaming tool responses once MCP spec supports it
- [ ] **GitHub App Marketplace listing** — the App itself (`infra/github-app/`)
      is installable and hardened as of v2.9.0 (signature verification,
      installation-token auth, Check-annotation posting), with REST API auth
      (`TRELIX_API_AUTH_TOKEN`) added in v2.10.0; Marketplace paid-app
      verification requires ≥100 installations before GitHub will even review
      it — an adoption/business gate, not engineering scope.

---

## 🔭 Candidates (not yet committed)

*Nothing is currently queued here.* The one item this section used to track has
shipped — see below.

- [x] **VS Code chat participant + hover providers** — the original Phase 3 Plan A
  spec (`docs/superpowers/plans/2026-07-08-phase3-vscode-github-app.md`)
  described a `@trelix` chat participant and hover providers, and for a while
  neither was delivered (only the `trelix.search`/`trelix.ask` QuickPick/Webview
  commands shipped in v2.9.0). **Both shipped in v3.0.0** ✅ —
  `workspace-vscode/src/chat-participant.ts` + `chat-handler.ts` register the
  `trelix.chat` participant, and `hover-provider.ts` shows signature/docstring via
  the `get_symbol` MCP tool. This section described them as unbuilt for three
  releases after they landed; it was stale, not a scope cut.

---

## 🔧 Phase 3 — Developer Tools & Integration (Q3 2026)

**Goal:** native IDE integration + automated PR review.

| Item | Status |
|------|--------|
| VS Code extension scaffolded | ✅ Phase 3 Plan A |
| GitHub App Actions workflow | ✅ Phase 3 Plan B |
| JetBrains plugin (IntelliJ/PyCharm) | 📋 backlog |
| Multi-repo workspace support in MCP | ✅ shipped v2.8.0 |

---

## 💡 Research Backlog (no timeline)

Ideas being researched but not yet committed to a release:

- **CodeBERT fine-tuning** — domain-adapted embedding model trained on trelix's own telemetry data
- ~~**Semantic diff** — diff-aware retrieval (weight recently-changed symbols higher)~~ ✅ shipped in Phase 2 Plan B
- ~~**IDE plugins** — VS Code extension with inline search~~ ✅ shipped in Phase 3 Plan A
- ~~**GitHub App** — PR review comments posted automatically via GitHub App auth~~ ✅ shipped in Phase 3 Plan B
- **Multi-modal** — index diagrams, comments referencing architecture docs
- ~~**Agent memory** — persist AgentLoop history across sessions~~ ✅ shipped v2.8.0

---

## How We Decide What to Build

1. **CoIR benchmark** — does it move nDCG@10?
2. **User telemetry** — what queries fail today?
3. **Integration requests** — LangChain/LlamaIndex ecosystem needs
4. **Security requirements** — supply chain, audit trail

File issues or start Discussions to influence the roadmap.
