# trelix Roadmap

> **Status:** Living document — updated with each release.
> **Version:** 2.8.1 (current) | **Next:** 3.0.0

This roadmap describes planned features, research directions, and long-term vision for trelix. Items are organized by phase; specific timelines are intentionally loose to reflect research-driven development.

---

## ✅ Shipped (v2.0 – v2.8.1)

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

## 🌐 v3.0.0 — Breaking Changes & Ecosystem (H1 2027)

**Goal:** Clean API surface + first-class cloud deployment.

- [ ] **Remove deprecated** — `flare_max_iterations` removed (deprecated in v2.4)
- [ ] **MCP streaming** — true streaming tool responses once MCP spec supports it
- [ ] **Python 3.13 support** — test matrix expansion
- [ ] **OpenTelemetry integration** — spans for every retrieval leg
- [ ] **Helm chart** — production Kubernetes deployment for `trelix serve`
- [ ] **TypeScript SDK** — native SDK matching Python API surface
- [ ] **VS Code extension improvements** — inline search refinement, snippet preview
- [ ] **GitHub App GA** — public marketplace listing, production hardening

---

## 🔭 v3.1.0 — Candidates (not yet committed)

- [ ] **VS Code chat participant + hover providers** — the original Phase 3 Plan A
  spec (`docs/superpowers/plans/2026-07-08-phase3-vscode-github-app.md`)
  described a `@trelix` chat participant and hover providers; neither was
  actually delivered (only the `trelix.search`/`trelix.ask` QuickPick/Webview
  commands shipped). Deferred again in the v3.0.0 VS Code extension pass
  (search refinement + snippet preview) to keep that item bounded — this is
  a real, once-planned scope cut, not a silently-dropped gap. Revisit as its
  own PR if/when picked up.

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
