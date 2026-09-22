# Post-v3.3.0 Research Sweep — 2026-09-16

15 parallel `/deep-research` runs (each: 5 search angles → fetch top sources → 3-vote adversarial claim verification → cited synthesis), commissioned immediately after the v3.3.0 release shipped (2026-09-12), covering upgrade paths, new-feature opportunities, optimization, and best practices. Every finding below was cross-checked against trelix's actual live source/docs where applicable, not just external research — several changed conclusion after that check (a claimed bug turned out to be a safely-handled fallback; a claimed gap turned out to already be covered).

---

## Tier 1 — Real findings worth fixing (doc accuracy, small UX gaps)

### 1.1 `sqlite-vec` has never shipped HNSW — trelix's docs claim it does, everywhere

Two independent research runs (vector-index landscape, SQLite scaling ceilings) both flagged this, and it was confirmed against trelix's live source. `sqlite-vec` has **never** implemented HNSW in any release, stable or alpha — the maintainer explicitly rejected it years ago (tracking issue #25) as a poor fit for SQLite's shadow-table storage model, building DiskANN/IVF/rescore instead (alpha only, first landed 2026-03-31, still alpha as of today). trelix's `+hnsw(m=N, ef_construction=N)` syntax in `src/trelix/store/vector.py`'s `_try_create_hnsw_table()` has **never** been valid sqlite-vec syntax.

**The runtime is safe**: the `CREATE VIRTUAL TABLE ... +hnsw(...)` call is wrapped in `try/except sqlite3.OperationalError`, correctly falling back to flat `vec0` scan with a logged warning — the same graceful-degradation pattern used throughout trelix. Results are always correct (brute-force is exact, not approximate). But this means `hnsw_active` has been unconditionally `False` for every SQLite-backend user, in every release including v3.3.0 — not an edge case, the permanent universal state.

**The docs are wrong, widely:**
- `README.md` — architecture bullet ("sqlite-vec HNSW + FTS5 BM25"), the Mermaid pipeline diagram, the schema table, and the env-var table (`TRELIX_STORE_HNSW`, `TRELIX_STORE_HNSW_M`, `TRELIX_STORE_HNSW_EF_SEARCH`)
- `docs/USER_GUIDE.md` — sample CLI output literally prints `"sqlite-vec HNSW, M=16"`
- `docs/architecture.md` — full HNSW param table (M=16, ef_construction=200, ef_search=50), none of which sqlite-vec has ever accepted
- `src/trelix/store/vector.py`'s own docstrings ("sqlite-vec >= 0.1.6 supports the +hnsw(...) syntax")

**Recommendation:** correct the docs to describe the actual, always-current behavior (flat/brute-force `vec0` scan with correct-but-not-approximate results), and either remove the `TRELIX_STORE_HNSW*` env vars (dead knobs) or reframe them as reserved-for-when-sqlite-vec's-alpha-ANN-stabilizes. Low risk, no behavior change, pure accuracy fix — the same category of issue this project has fixed repeatedly in past releases.

### 1.2 VS Code chat participant: missing `stream.anchor()`

`@trelix`'s chat handler already correctly uses `stream.reference()` for citing search/blast-radius results as context provenance (VS Code's documented pattern), but never calls `stream.anchor()` — so when the synthesized answer mentions a symbol by name inline, it renders as plain text instead of a clickable inline link to that symbol's editor location. Small, well-scoped fix; doesn't touch the already-correct `reference()` provenance list.

### 1.3 Blast-radius code lens uses the wrong CodeLens pattern

Microsoft's own docs use "a list of references to a method, shown in a popup above the CodeLens line" as the canonical example of a **Visual CodeLens**. trelix's `$(references) N dependents` / `$(references) Blast radius` lenses instead open a global `showQuickPick` — an **Invokable CodeLens** pattern — for exactly the use case Microsoft's docs use to illustrate the other pattern. Worth reimplementing as an inline popup.

### 1.4 Minor CLI polish
Progress bars show a percentage (`TaskProgressColumn`) rather than the literal "X of Y files" count that `clig.dev`/Evil Martians recommend as the default for measurable step-by-step work. Everything else on the CLI side (Rich spinners, TTY/CI detection, `--json` stderr routing) already exceeds the documented baseline.

---

## Tier 2 — Validated non-issues (confirmed adequate or already ahead — don't chase)

- **Embedding models**: no open-weight/self-hostable model beats trelix's current `BGE-Code-v1`/`voyage-code-3`/`text-embedding-3` options on code-retrieval benchmarks. Nomic Embed Code wins inconsistently (3-2-1 split vs voyage-code-3); LateOn-Code uses a different architecture with no direct comparison; voyage-code-4 is API-only, doesn't fit the `local` provider.
- **Rerankers**: no newer late-interaction/LLM-as-reranker/listwise technique beats trelix's cross-encoder/Cohere/PLAID/XTR options on a direct benchmark. A new benchmark (CoREB) shows off-the-shelf rerankers as a category are fragile on code-specific tasks, which is a caution about reranker choice generally, not a specific upgrade path.
- **Fresh LLM SDK drift**: LiteLLM v1.101.0 (shipped 2 days after v3.3.0) is a real breaking release, but traced against trelix's actual `litellm_backend.py` and `retry.py` — neither code path touches the changed surfaces (new `cost` field, Anthropic error-envelope reshaping is already covered by `retry.py`'s explicit degrade-to-non-retryable design). No action needed.
- **tree-sitter parsing**: ABI stable across 0.26→0.27 (no regeneration needed). One real accuracy bug (`tree-sitter-language-pack` v1.15.0, ~15 languages silently getting the wrong grammar's query files) predates trelix's floor (`>=1.17.0`, installed `1.18.0`) — never reachable.
- **Competitive landscape**: no retrieval-quality gap vs Cursor/Copilot/Cody/Continue/Aider. Gaps found are orchestration-scope (Cursor's persistent cross-session agent memory, cross-repo *change* orchestration) — a different product category from trelix's read-only cross-repo *search*. Continue.dev actually deprecated its own indexed retrieval in favor of pure agentic tool-use — moving away from trelix's thesis, not ahead of it.
- **Cross-repo/federated symbol resolution**: neither Meta's Glean nor Google's Kythe (the two dominant production systems) document a federation/TTL-caching layer comparable to trelix's `FederatedRetriever` — trelix may already be ahead of the documented state of the art here.
- **Supply-chain hygiene**: trelix already meets the new PyPI-recommended baseline — Trusted Publishing (`id-token: write`, no stored API token) is live, and `attestations: true` is set on all 4 PyPI publish steps (PEP 740 Sigstore attestations generated automatically). Remaining gaps: no SBOM generation, no GitHub native `actions/attest@v4` build-provenance for binaries/Docker images — lower priority.

---

## Tier 3 — Real future opportunities (evidence-backed, not urgent)

### 3.1 Dynamic per-query dependency graphs (strongest architectural finding of the sweep)
**DyRetriever/DyCoder** (ASE 2026) builds a dependency graph *per query, on demand*, then discards it — instead of maintaining one static, pre-built, whole-repo graph (trelix's current call-graph/type-edge/import-edge layer + PageRank/Louvain). It beats the static-graph pattern by **+4–18 Pass@1/EM points while indexing ~7.4x faster**, and includes a concrete self-correction step trelix's agent loop may lack: every LLM-proposed dependency gets checked against the real repo and discarded if it doesn't exist, rather than trusted. Worth a real design pass, not urgent — trelix's static graph is a deliberate architectural choice with its own tradeoffs (always-available, no per-query latency), and the comparison wasn't run against trelix specifically.

### 3.2 Adjacent retrieval-path ideas
**CodeRAG** (EMNLP 2025) adds a third retrieval path — a dataflow graph traced from the unfinished file to the completion point — beating dataflow-only and iterative-retrieval baselines by ~7-8 EM points; conceptually adjacent to trelix's existing graph leg. **cAST**'s recursive AST split-then-merge chunking beats naive fixed-size chunking, but was never benchmarked against per-symbol chunking (what trelix already does), so it doesn't prove trelix's chunking is behind.

### 3.3 Embedding cost optimization
Quantized embeddings (int8/binary) are now a mainstream native feature on Cohere and Voyage AI — int8 retains ~99.99% recall at 4x smaller storage/~30% faster search; binary retains 90-98% at 32x smaller/~40x faster. trelix has no direct Voyage/Cohere embedder backend today, so this mainly matters if one gets built. Separately, **OpenAI's and Voyage's Batch APIs** (50%/33% cost discount on async large-scale embedding jobs) apply to trelix's *existing* OpenAI/Azure backends today with zero quality cost — the cleanest, lowest-risk win found in this sweep, purely additive to existing code.

### 3.4 Call-graph resolution
**PoTo/PoToCG** (ECOOP 2025, open-source) is a production-ready static call-graph tool beating PyCG on completeness/precision/scale with zero LLM cost — a candidate if trelix wants to push Python call-graph resolution beyond declared-type-hint matching. A rigorous EMSE 2025 study found classical static tools decisively beat LLMs at call-graph construction (84.9% vs 60.3% completeness) while LLMs decisively beat classical tools at *type inference* (93.7% exact match) — the two tasks favor opposite approaches.

### 3.5 GNN-based ranking — not ready
Graph neural network approaches to code-ranking (GNN-Coder, AcbertGraphC) show modest gains (1-10% MRR) but have no open-source release and only small-scale validation — research-only, not adoptable today.

### 3.6 MCP protocol tracking
**SEP-2577** ("Deprecate Roots, Sampling, and Logging") went Final in the same 2026-07-28 spec release as SEP-2322 — 12-month compat window, no urgent action. **SEP-2998** (streaming tool results) is confirmed still Draft/unsponsored, exactly matching trelix's own roadmap. The spec shipped a new, real (non-draft) `subscriptions/listen` method replacing the old `resources/subscribe` RPC that trelix's MCP server still implements — worth tracking for a future `trelix-mcp` bump, not urgent (old mechanism still works for backward compat).

### 3.7 OTel GenAI semantic conventions — wait
The spec has not reached stable/1.0 — every `gen_ai.*` attribute across chat, retrieval, and agent spans remains tagged "Development" (lowest OTel maturity tier), and the spec relocated to a new repo in June 2026 with zero tagged releases. Not worth adopting yet (attribute names still churning), but the shape is informative for a future tracing alignment: retrieval spans define `gen_ai.retrieval.documents` (id+score), `.query.text`, `.top_k`; agent spans define lifecycle operations (`invoke_agent`, `plan`, `execute_tool`).

---

## Tier 4 — Already-known risk, now externally validated

**Prompt injection via indexed content.** trelix's own `SECURITY.md` already discloses this honestly: *"Status: documented, not mitigated... trelix ships no defence against prompt injection."* This sweep's security research confirms the severity is real, not theoretical: MCP "tool poisoning" has been demonstrated at scale against 45 real production MCP servers (36.5% average attack-success rate, up to 72.8%); separately, poisoning as little as **0.7% of a code-retrieval corpus achieves 80-93% attack success** via top-ranked retrieval (CodePoisonRAG / "Beyond the Payload," both Sept 2026), with defenses only partially mitigating (40-71% residual success). trelix's path-confinement/read-only-tool design correctly blocks unauthorized filesystem access but does nothing about content-level manipulation of an already-indexed codebase — exactly the gap already flagged. These citations are worth attaching to the existing `SECURITY.md` disclosure whenever the team prioritizes building a real mitigation.

---

## Phase 6 spike findings — 2026-09-17

Four independent research spikes evaluating the bigger architectural candidates surfaced above, each run against trelix's actual current code (not just the source papers). No code shipped from this phase — these are written findings only, feeding future plans.

### 6.1 Dynamic per-query dependency graphs (DyRetriever/DyCoder)
**Confidence: HIGH — don't build it as framed.** trelix already runs this exact pattern at the retrieval leg that matters: `retrieval/graph.py`'s `expand_with_call_graph`/`expand_with_imports`/`expand_with_type_edges` + `rank_by_pagerank` build a fresh, tiny per-query subgraph from indexed SQL lookups and discard it — the paper's "build a graph per query, on demand, then discard it" pattern, just implemented as raw SQL BFS rather than a named abstraction. Measured directly against trelix's own self-index: the per-query path costs 3-28ms (warmed) vs ~1.2s for the full static Louvain+PageRank build — 2-3 orders of magnitude cheaper already. The one place trelix tried the paper's "expensive" full-graph-per-query rebuild (`graph_search_enabled`) is off by default. The real, scoped opportunity found: the per-query subgraph's edge topology (which symbol called which, how many hops) is currently discarded before reaching the assembled LLM context — only flattened, independently-scored snippets survive. Serializing that topology into the prompt for structural intents (`DEPENDENCY_MAP`, `BLAST_RADIUS`, `FEATURE_FLOW`) reuses 100% of existing extraction machinery and is where a real Pass@1/EM gain plausibly lives.

### 6.2 CodeRAG-style dataflow retrieval leg
**Confidence: HIGH — additive, but not the highest-leverage next move.** Verified: none of trelix's 6 RRF-fused legs, nor the separate call/import/type expansion functions, nor the optional graph-BFS leg encode value movement (what data crosses an edge) — only call/import/type/reference *topology*. A CodeRAG-style dataflow leg would be genuinely new signal, not redundant. But trelix already has an unused intra-procedural def-use extractor (`analysis/defuse.py`'s `DataFlowExtractor`, gated behind `dataflow_enabled: bool = False`) that populates a `def_use_edges` table with **zero readers anywhere in the codebase** — not wired into `CodeGraph`, not fused, not graphed. Wiring that up and extending it across resolved `CALLS` edges for inter-procedural flow is a much smaller lift than a new module, and reuses the existing schema/fusion/dedup machinery entirely. Also flagged: CodeRAG's ~7-8 EM benchmark is on a fill-in-the-middle completion task, which doesn't map directly onto trelix's QA-style retrieval intents — any gain claim should be re-measured against trelix's own eval harness, not assumed to transfer.

### 6.3 PoTo/PoToCG for Python call-graph resolution
**Confidence: HIGH — not adoptable, and corrects an overstatement in this doc's own §3.4.** Located and read the real ECOOP 2025 paper and GitHub repo. PoTo/PoToCG is real, but: no PyPI package, no packaging at all, Python-3.9-pinned, requires a hand-authored non-pytest driver per target (its own README says it "cannot interpret pytest tests"), *executes arbitrary target code* during analysis (explicit safety warning in its own README), runs for hours on packages trelix indexes in seconds, and its own PyCG-comparison reproduction script is broken in the published repo (missing `utils.py`). The "beats PyCG on completeness/precision/scale" claim is real but narrower than it sounds — PoToCG actually finds *fewer* reachable functions than PyCG in every benchmark; the precision win is scoped to the intersection of jointly-reached functions only. §3.4 above calling it "production-ready" does not hold up against the repo's actual state — correcting that here. Lower-risk lever if trelix wants better call-graph recall: widen the existing `_extract_param_types` declared-type-hint extraction (track `self.attr` assignments, simple local-variable-to-return-type propagation) rather than importing a whole-program points-to analysis that conflicts with trelix's fast/incremental indexing model.

### 6.4 Direct Voyage/Cohere embedder backend + quantized output
**Confidence: split — HIGH for CohereEmbedder, MEDIUM for quantization; verdict differs sharply between the two.** A direct `CohereEmbedder` (alongside the existing direct `VoyageEmbedder`; today Cohere is only reachable via Bedrock, a different API surface) is small and mechanical — the codebase has now done this exact shape of provider addition 8 times over. Quantized int8/binary output does **not** integrate cleanly with the current sqlite-vec flat-scan store: verified live against the installed `sqlite-vec` extension that `INT8[N]`/`BIT[N]` columns and `vec_quantize_int8()`/`vec_quantize_binary()` exist and work, but trelix's `_pack()`/raw-INSERT pattern is float32-only and would silently misinterpret quantized bytes without using SQL-level `vec_int8()`/`vec_bit()` wrapping; `DimensionGuard` only compares scalar dimension and is blind to a dtype-only mismatch; two code paths (`search_file_summaries`/`search_sub_chunks`) bypass the vec0 engine entirely with manual Python float-distance math. More fundamentally, quantization's headline recall numbers are benchmarked as an *ANN-index* feature (smaller compressed index + rescore) — trelix's sqlite path is a deliberate exact flat scan, so quantization there would only buy storage/CPU savings, not the "faster approximate search" the vendor numbers assume. This fits the Qdrant backend (real HNSW) much better than sqlite.



Each of the 15 topics ran as an independent `deep-research` workflow (5 search-angle agents → source fetch → 3-vote adversarial verification per claim → synthesis), scoped with trelix-specific context rather than generic prompts. Total: ~1,500 sub-agents, ~150M tokens, ~12,000 tool calls across the sweep. Every finding in Tiers 1 and 4 was independently re-verified against trelix's actual live source/docs (not just the external research) before being included here.
