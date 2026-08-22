# trelix v3.1.6 — Frequently Asked Questions

> Last updated: 2026-08-17 — covers trelix 3.1.5, trelix-mcp 3.1.5, trelix-langchain 3.1.5, and trelix-llama-index 3.1.5.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Search and Retrieval](#2-search-and-retrieval)
3. [MCP and Integrations](#3-mcp-and-integrations)
4. [Embedding Providers](#4-embedding-providers)
5. [v2.4.0 Specific Features](#5-v240-specific-features)
6. [Production Deployment](#6-production-deployment)

---

## 1. Getting Started

### Do I need an API key to use trelix?

No. The `local` embedding provider runs entirely offline using `sentence-transformers/all-MiniLM-L6-v2`, which is downloaded once and cached locally. No API key is required to index, search, or use `trelix query`.

```bash
pip install "trelix[local]"
trelix index ./my-repo
trelix search ./my-repo "authentication middleware"
```

An API key is only required when you want LLM-synthesized answers via `trelix ask` (needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or equivalent) or when using a cloud embedding provider such as `openai`, `voyage`, or `bedrock-cohere`.

---

### What programming languages does trelix support?

trelix uses tree-sitter for full AST-level parsing of **20+ languages**:

**Code (AST — functions, classes, methods, call edges, imports):**
Python, TypeScript, TSX, JavaScript, JSX, Go, Java, Rust, C, C++, C#, Kotlin, Ruby

**.NET / Razor:**
Razor Components (`.razor`), Razor MVC Views (`.cshtml`), MSBuild projects (`.csproj`)

**Config (key-path extraction):**
JSON, JSONC, TOML, YAML (multi-document)

**Markup:**
Markdown (heading sections), HTML (custom elements), CSS, SCSS

For unsupported file types trelix skips the file without error. To add a new language, see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

### How large can my codebase be?

The default SQLite backend handles repositories up to roughly **100,000 chunks** (typically a 200–400k line codebase) without noticeable slowdown. For larger repos:

| Scale | Recommended backend | Install |
|-------|---------------------|---------|
| Up to ~100k chunks | SQLite (default) | included |
| 100k–500k chunks | LanceDB — 3–5x faster insert | `pip install "trelix[lance]"` |
| 500k+ chunks / multi-team | Qdrant | `pip install "trelix[qdrant]"` |

Switch backends with:
```bash
TRELIX_STORE_BACKEND=lance trelix index ./my-repo   # LanceDB
TRELIX_STORE_BACKEND=qdrant trelix index ./my-repo  # Qdrant (set QDRANT_URL too)
```

---

### Does trelix work on Windows, macOS, and Linux?

Yes. trelix is a Python package and runs on any OS that supports Python 3.11+. Standalone binaries are provided for **macOS ARM64**, **Windows x64**, and **Linux x64** on the GitHub Release page. `pip install trelix` also works on all three.

macOS users can also install via Homebrew:
```bash
brew tap sairam0424/trelix && brew install trelix
```

---

### How long does indexing take?

Indexing speed depends on the embedding provider and codebase size:

| Provider | ~1,000 files | ~10,000 files |
|----------|-------------|---------------|
| `local` (sentence-transformers) | 15–30 s | 3–6 min |
| `openai` / `azure` | 20–45 s | 5–10 min |
| `voyage` (API) | 20–45 s | 5–10 min |
| `bge-code` (local, larger model) | 60–120 s | 10–20 min |

Indexing is a one-time cost. After that, `trelix watch` incrementally re-indexes only changed files on every save, typically completing in under 1 second per file.

---

## 2. Search and Retrieval

### What is the difference between `trelix search` and `trelix ask`?

| Command | Retrieval | LLM | Output | Offline |
|---------|-----------|-----|--------|---------|
| `trelix search` | Hybrid (vector + BM25 + grep) | Planner only — no synthesis | Ranked code chunks in a table | Only with no chat credential set |
| `trelix ask` | Hybrid + synthesis — no reranking, same as `search` | Yes | Synthesized natural-language answer | Requires API key |

The `search` row used to read "LLM: No / Offline: Yes". That contradicted
[What is `trelix query`?](#what-is-trelix-query) eleven lines below it, which says the planner
call applies to `search` too. See that section for the two ways to get zero LLM calls.

The `ask` row used to read "Hybrid + reranking + synthesis". It does not rerank: `ask`
constructs `RetrievalConfig(rerank=False)` at `src/trelix/cli/main.py:1269`, exactly as
`search` does at `:1173`, and an init keyword outranks `TRELIX_RETRIEVAL_RERANK` in
pydantic-settings — so no environment setting switches it on. Reranking is on by default
only on the surfaces that leave `retrieval` at its default: the MCP tools, the REST API,
the Python API, and the `eval` / `eval-synthesis` / `review` / `search-all` commands.

Use `trelix search` when you want to browse the raw matches and decide yourself. Use `trelix ask` when you want a direct answer to a question about the codebase. `trelix ask` calls `trelix search` internally and then sends the top results to an LLM.

---

### What is `trelix query`?

`trelix query` runs a structured query over the index — keyword and semantic matching — and performs **no LLM synthesis**. It is faster than `trelix ask` and never writes a generated answer.

It is not, however, unconditionally LLM-free. When a chat credential is resolvable from the environment, retrieval draws a **query plan** from the LLM: one call per distinct query, made by `QueryPlanner.plan()` from inside `Retriever.retrieve()`, so it applies to `trelix search` as well. `--provider local` does not prevent it — that selects the *embedder*, while the planner reads the chat credential independently. If the call fails, retrieval logs a warning and falls back to `default_plan()`.

Two ways to get the offline, deterministic, zero-cost behaviour this section used to promise unconditionally:

- **Unset the chat credential.** With no `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / Azure equivalent in the environment, retrieval makes **zero** LLM calls. (The planner object is still constructed — that happens unconditionally — but with no resolvable credential it holds no usable client, so it short-circuits to `default_plan()` without calling out. Measured: 0 calls.)
- **Freeze the plans.** Set `TRELIX_RETRIEVAL_PLAN_CACHE_FILE` to a path. The first pass records each plan (one call per distinct query); every later pass replays from the file and makes **zero** calls. Commit that file to get a CI run that is both free and byte-for-byte reproducible.

So `trelix query` is safe for CI — but for cost and determinism you must either withhold the credential or commit a recorded plan cache. A CI box that has a key set for `trelix ask` will otherwise pay one planner call per distinct query here too.

```bash
trelix query ./my-repo "rate limiting middleware"
```

---

### How does hybrid search work?

trelix combines three complementary retrieval legs and merges their results with Reciprocal Rank Fusion (RRF):

1. **Vector (dense) search** — ANN over sentence embeddings stored in sqlite-vec HNSW. Captures semantic similarity even when exact terms differ.
2. **Contextual BM25** — Full-text search over FTS5. Fast and precise for identifier names, function signatures, and exact tokens.
3. **Grep search** — Regex/exact pattern matching over raw source. Required for case-sensitive symbol lookups.

Results from all three legs are merged by RRF before optional reranking. The adaptive query router decides which legs to activate based on the classified intent of the query.

Optional additional legs available via feature flags:
- **4th leg** — Knowledge Graph BFS (`TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED`). **Leave this one off.** Measured on trelix's own 54-query golden set it costs **−0.2084 / −0.2322 nDCG@10** across two independent query-plan draws (MRR −0.28 / −0.31) — 3.4–3.8x the harness's ±0.061–0.080 detection band, with 34–35 of 54 queries moving. Graph hits are scored `0.5^hop` against post-fusion scores of 0.016–0.029 and then merged by raw score, so they take **411 of 540** top-10 slots. It defaults off in code and should stay off until that merge fuses in rank space. `docs/USER_GUIDE.md` section 4, Leg 5 carries the detail
- **5th leg** — File-summary semantic search (`TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=true`)
- **6th leg** — SPLADE-Code sparse vectors (`TRELIX_RETRIEVAL_SPARSE=true`)
- **7th leg** — Multi-granularity block+statement indexing (`TRELIX_CHUNKER_MULTI_GRANULARITY=true`)

---

### What is RRF (Reciprocal Rank Fusion)?

Reciprocal Rank Fusion is a rank aggregation formula that merges multiple ranked result lists without requiring calibrated scores. For each document, its RRF score is:

```
score = sum(1 / (k + rank_i))  for each list i
```

where `k=60` (trelix default). Documents that appear near the top of multiple retrieval legs receive a high combined score. RRF is robust to score scale differences between legs — vector cosine similarity and BM25 TF-IDF scores are not comparable directly, but ranks are.

---

### How does call-graph expansion work?

After RRF produces an initial candidate set, trelix expands it through the code property graph:

1. For each seed result, trelix looks up its symbol in the `call_graph` and `imports` tables.
2. BFS traversal (configurable depth, default 2) surfaces callers, callees, and direct importers.
3. New symbols found by BFS are added to the context window.

This ensures that if you search for `verify_jwt_token`, the decorator `require_auth` that calls it also appears — even if it was not in the original top-k.

Call-graph expansion is active for the `symbol_lookup` and `feature_flow` retrieval intents. It is **not** active for the `blast_radius` intent, whose strategy sets `expand_depth=0` (`retrieval/planner/models.py`) and runs grep/vector/bm25 only — impact analysis is served instead by the `blast_radius` MCP tool, which reads the call and import edges directly rather than going through retrieval. Enable the full knowledge graph for deeper graph traversal: `pip install "trelix[knowledge-graph]"` then `trelix graph ./my-repo`.

---

### Can I search across multiple repositories?

Yes. trelix v2.3.0 introduced **federated search**:

```bash
# Register repos
trelix federation add backend ./services/api
trelix federation add frontend ./apps/web

# Search across all registered repos
trelix search-all "database connection pooling"
```

`FederatedRetriever` fans out queries to all registered repos in parallel via `ThreadPoolExecutor`, merges results with RRF, and deduplicates by `(file_path, symbol_id)`. Results are annotated with the source repo alias.

In v2.4.0, `FederatedRetriever` gained a **TTL cache** (SHA-256-keyed, thread-safe) so repeated queries during a debugging session resolve in under 1 ms instead of re-querying every repo. See [v2.4.0 Specific Features](#5-v240-specific-features) for details.

---

### What does the `score` field in search results mean?

The `score` is the RRF aggregate score after all active retrieval legs have been merged. Higher is better. The scale is not fixed — it depends on how many legs contributed a rank for that result. A score above 0.05 typically indicates a strong multi-leg match; a score near 0.016 (= 1/62) indicates the result came from exactly one leg at rank 1.

If a reranker is active (`TRELIX_RETRIEVAL_RERANK_PROVIDER=cohere` or `cross_encoder` — underscore; the hyphenated `cross-encoder` is rejected), the final score is the reranker's relevance score (0.0–1.0) rather than the RRF score.

---

## 3. MCP and Integrations

### How do I add trelix to Claude Code?

```bash
pip install trelix-mcp
claude mcp add trelix -- trelix-mcp
```

After restarting Claude Code, the trelix tools are available in every conversation. You can verify with `/mcp` in Claude Code — trelix should appear in the list of active servers.

---

### How do I add trelix to Cursor?

1. Install the MCP package: `pip install trelix-mcp`
2. Open Cursor Settings → MCP → Add Server.
3. Set the command to `trelix-mcp` (no arguments needed).
4. Save and restart Cursor.

Cursor will discover the trelix tools automatically via the MCP stdio protocol.

---

### What MCP tools does trelix expose?

trelix-mcp v3.1.5 exposes **15 tools**:

| Tool | Description |
|------|-------------|
| `search_code` | Hybrid code search with optional cursor-based pagination (v2.4.0) |
| `index_codebase` | Index or re-index a repository; sends progress notifications during stages |
| `get_symbol` | Get full source of a symbol by qualified name |
| `blast_radius` | Find what depends on a symbol |
| `build_knowledge_graph` | Build the Code Property Graph for a repo |
| `graph_search_mcp` | Graph BFS search from a seed symbol |
| `subscribe_resource` | Subscribe to file change notifications (v2.5.0+) |
| `unsubscribe_resource` | Unsubscribe from file change notifications (v2.5.0+) |
| `federation_list_repos` | List all repos registered for federated search (v2.8.0+) |
| `federation_add_repo` | Register a repo for federated search (v2.8.0+) |
| `federation_remove_repo` | Unregister a repo by alias (v2.8.0+) |
| `federation_search_all` | Search across all registered repos with RRF-weighted fusion (v2.8.0+) |
| `ask_agent` | Multi-turn ReAct Q&A with persistent session memory (v2.8.0+) |
| `agent_list_sessions` | List recent agent sessions for a repo (v2.8.0+) |
| `agent_clear_session` | Delete a persisted agent session (v2.8.0+) |

Plus the following MCP **Resources** (application-controlled, URI-addressable):
- `trelix://index/stats` — aggregate index statistics
- `trelix://repo/{repo_path}/manifest` — indexed file list
- `trelix://repo/{repo_path}/symbols/{qualified_name}` — symbol source code

And **Prompts** (reusable LLM interaction templates):
- `trelix-search`, `trelix-explain`, `trelix-blast-radius`

---

### What changed in `search_code` in v2.4.0?

**This is a breaking change.** The `search_code` MCP tool now returns a **pagination envelope** instead of a bare list.

**Before (v2.3.0):**
```python
results = search_code(query="auth", repo_path="/repo")
# results is list[dict]
for r in results:
    print(r["file"], r["symbol"])
```

**After (v2.4.0):**
```python
response = search_code(query="auth", repo_path="/repo")
# response is {"results": [...], "next_cursor": int|null, "total_available": int}
for r in response["results"]:
    print(r["file"], r["symbol"])

# Paginate to the next page
next_page = search_code(query="auth", repo_path="/repo", cursor=response["next_cursor"])
```

The `next_cursor` field is `null` when all results have been returned. `total_available` tells you how many results exist before pagination. This change enables large result sets to be streamed without hitting MCP payload limits.

---

### Does trelix work with LangChain?

Yes. Install `trelix-langchain` and use `TrelixRetriever` as a standard LangChain `BaseRetriever`:

```bash
pip install trelix-langchain
```

```python
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo", top_k=5)
docs = retriever.invoke("how does authentication work?")

# In a LangChain chain
chain = retrieval_chain | llm | output_parser
```

---

### Does trelix work with LlamaIndex?

Yes. Install `trelix-llama-index` and use `TrelixIndexRetriever`:

```bash
pip install trelix-llama-index
```

```python
from trelix_llama_index import TrelixIndexRetriever

retriever = TrelixIndexRetriever(repo_path="/path/to/repo")
nodes = retriever.retrieve("rate limiting middleware")
```

The retriever returns LlamaIndex `NodeWithScore` objects compatible with all LlamaIndex query engines, response synthesizers, and pipelines.

---

### Can I search across multiple repos from an AI assistant, not just the CLI?

Yes. trelix v2.8.0 added four federation MCP tools that expose the `RepoRegistry` and `FederatedRetriever` to any MCP client (Claude Desktop, Cursor, VS Code, JetBrains):

- `federation_list_repos` — list all registered repos
- `federation_add_repo` — register a repo with an alias and weight
- `federation_remove_repo` — unregister a repo by alias
- `federation_search_all` — search across all registered repos with weighted RRF merging

Your AI assistant can call these tools directly — no need to leave the conversation to run CLI commands. Results are annotated with the source repo alias. For full details and parameter reference, see [docs/MCP_GUIDE.md](MCP_GUIDE.md) and [docs/FEDERATION_GUIDE.md](FEDERATION_GUIDE.md).

---

### Does trelix remember previous questions in an agent conversation?

Yes. The agentic loop (activated via `trelix ask --agentic` or the `ask_agent` MCP tool) now persists turn history to the repo's `.trelix/index.db`, keyed by a client-supplied or auto-generated `session_id`.

On the first call to `ask_agent`, trelix generates a new `session_id` and returns it in the response. Pass this `session_id` back on follow-up calls to resume with full prior context — all previous thoughts, actions, and observations are loaded and prepended to the current turn's history.

Sessions auto-evict after `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS` of inactivity (default 7 days; set to `0` to disable eviction). You can also explicitly clear a session via the `agent_clear_session` MCP tool or the `trelix agent sessions clear` CLI command.

For usage examples and full session lifecycle documentation, see [docs/MCP_GUIDE.md](MCP_GUIDE.md) and [docs/USER_GUIDE.md](USER_GUIDE.md).

---

## 4. Embedding Providers

### Which embedding provider gives the best quality?

For code-specific retrieval quality, ranked highest to lowest:

| Provider | Model | CoIR Score | Notes |
|----------|-------|-----------|-------|
| `bge-code` | BAAI/bge-code-v1 (1536-dim) | SOTA 2025 | Local, no API key, ~8GB RAM |
| `voyage` | voyage-code-3 (256–2048-dim, Matryoshka) | Very High | Best API option; set `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512` for 2x faster HNSW |
| `local-code` | SFR-Embedding-Code-2B_R (4096-dim) | Very High | Local, large model, requires GPU or ~8GB RAM |
| `nomic-code` | nomic-ai/nomic-embed-code (768-dim) | High | Local, good for large repos |
| `openai` / `azure` | text-embedding-3-large (3072-dim) | High | General-purpose, strong on mixed code+prose |
| `bedrock-cohere` | cohere.embed-english-v3 (1024-dim) | High | AWS ecosystem |
| `local` | all-MiniLM-L6-v2 (384-dim) | Baseline | Default, fully offline, excellent for getting started |

CoIR benchmark scores are from [archersama.github.io/coir](https://archersama.github.io/coir/) (ACL 2025).

For teams on a budget: `nomic-code` or `bge-code` offer near-API quality at zero cost after the one-time model download.

---

### Can I switch embedding providers after I have already indexed a repo?

Yes, but you must reset and re-index. Embedding vectors from different providers have different dimensions and are not compatible. Mixing providers silently produces wrong results, which is why trelix v2.3.0 introduced **DimensionGuard** (see below).

Migration steps:

```bash
# 1. Clear existing vectors and dimension metadata (REPO is a required positional)
trelix migrate-vectors ./my-repo --reset

# 2. Set the new provider
export TRELIX_EMBEDDER_PROVIDER=voyage
export VOYAGE_API_KEY=your_key

# 3. Re-index
trelix index ./my-repo
```

Without `--reset`, trelix raises `DimensionMismatchError` at startup and shows the exact recovery command to run.

---

### What is DimensionGuard?

DimensionGuard is a startup check introduced in v2.3.0 that detects when the embedding provider has changed since the last index run.

When you run `trelix index`, trelix records the embedding dimension in a new `index_metadata` table (e.g. `3072` for `openai`). On the next run, `Retriever.__init__` reads this metadata and compares it against the current provider's output dimension. If they differ, it raises `DimensionMismatchError` with a clear message:

```
DimensionMismatchError: Index was built with dimension 3072 (azure) but current
provider produces dimension 384 (local). Run:
  trelix migrate-vectors --reset ./your-repo
then re-index to continue.
```

This prevents the silent wrong-results bug that previously occurred when users switched providers without clearing the index.

---

## 5. v2.4.0 Specific Features

### What is `flare_max_retries` and how does it differ from `flare_max_iterations`?

`flare_max_retries` is the v2.4.0 rename of the `flare_max_iterations` config field in `RetrievalConfig`. FLARE (Forward-Looking Active REtrieval) is the confidence-gated re-retrieval feature that detects low-confidence synthesis spans and re-queries before finalising the answer.

Both the old and new environment variable names are accepted for backward compatibility:

```bash
TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=2   # new name (preferred)
TRELIX_RETRIEVAL_FLARE_MAX_ITER=2      # old name (emits DeprecationWarning; removed in v4.0.0)
```

**Important constraint added in v2.4.0:** The field now enforces `ge=1, le=3`. If you previously set `TRELIX_RETRIEVAL_FLARE_MAX_ITER` to a value greater than 3 (for example, 5 or 10), you must lower it to 3 or below before upgrading. Otherwise pydantic raises `ValidationError` at process startup:

```
pydantic_core.ValidationError: 1 validation error for RetrievalConfig
flare_max_retries
  Input should be less than or equal to 3 [type=less_than_equal, ...]
```

---

### What is `trelix watch-all`?

`trelix watch-all` is a new CLI command in v2.4.0 that watches all federated repositories simultaneously in a single process. Previously, you had to run a separate `trelix watch <repo>` process for each repository.

```bash
trelix watch-all
```

Internally, `MultiRepoWatcher` calls `watchfiles.awatch(*all_paths)` with all registered repo paths in one call. A SHA-256 hash guard prevents re-index cascade loops (the same file change does not trigger multiple index runs). Deleted files are removed from both the SQLite index and the vector store. Press `Ctrl+C` to exit cleanly — per-repo statistics are printed on shutdown.

Prerequisites: at least one repo must be registered via `trelix federation add <alias> <path>`.

---

### How does the FederatedRetriever cache work in v2.4.0?

`FederatedRetriever` now accepts a `cache_ttl` parameter (seconds, default `120.0`):

```python
from trelix.federation import FederatedRetriever, RepoRegistry

registry = RepoRegistry.load()
retriever = FederatedRetriever(registry, cache_ttl=120)
results = retriever.retrieve("authentication flow", k=10)
```

The cache is:
- **Keyed by SHA-256** of `(query, k, extra_params)` — different queries never collide.
- **Thread-safe** via `threading.Lock` — safe for concurrent MCP tool calls.
- **TTL-evicted** — entries older than `cache_ttl` seconds are discarded on the next cache read.
- **Inspectable** via `retriever.cache_stats()` — returns `{"hits": N, "misses": N, "size": N}`.
- **Clearable** via `retriever.clear_cache()` — useful after a manual re-index.

Set `cache_ttl=0` to disable caching entirely.

Expected cache hit rate: ~90% for typical debugging-session query patterns (same question asked multiple times while iterating on code).

---

### How do I review a GitHub PR with trelix?

trelix v2.4.0 adds `GitHubPRClient` and the `trelix review --pr` CLI flag:

```bash
# Review a PR without posting comments (prints to stdout)
trelix review --pr owner/repo#42

# Fetch the diff AND post review findings back to GitHub as a batched review
trelix review --pr owner/repo#42 --post-comments
```

Requirements:
- Your local trelix index for the relevant repo must already be built.
- `GITHUB_TOKEN` environment variable must be set (required for private repos; public repos may work without it but rate-limits apply).

What it does:
1. Fetches the PR file diffs from the GitHub REST API.
2. For each changed hunk, runs the hybrid retrieval pipeline to find semantically related code.
3. The LLM generates `ReviewComment` objects referencing existing patterns and potential regressions.
4. With `--post-comments`, all findings are posted back as a single batched GitHub review.

The reviewer handles all 7 GitHub file status values (`added`, `modified`, `removed`, `renamed`, `copied`, `changed`, `unchanged`) and emits a warning if the PR exceeds 3,000 files.

---

### What is the `search_code` breaking change in v2.4.0?

The `search_code` MCP tool now returns a **pagination envelope** instead of a bare `list[dict]`. This is the only breaking change in v2.4.0.

See the detailed answer in [MCP and Integrations](#what-changed-in-search_code-in-v240) above for the before/after code samples and migration guidance.

---

## 6. Production Deployment

### Is trelix suitable for production use?

Yes. As of v2.4.0, the core `trelix` package and `trelix-mcp` have:
- 1,703 tests (1,621 unit + 82 MCP) at 100% pass rate on Python 3.11, 3.12, and 3.13.
- `mypy --strict` clean.
- `ruff` lint and format clean.
- No hardcoded secrets; all credentials sourced from environment variables.
- Parameterized SQL throughout (no injection risk).
- MCP pagination following the spec-approved cursor pattern.

`trelix-langchain` and `trelix-llama-index` are **not** on an independent release cadence — all four distributions carry the core version and ship on the same tag. `.github/workflows/release.yml` triggers only on `push: tags: v*` (a core tag); its `build-distributions` job builds core, `trelix-mcp`, and both adapters, and its `publish` job uploads all four. No workflow, and no `workflow_dispatch`, can release an adapter by itself. The `verify-version` job fails the release unless all twelve version stamps in the tree equal the tag — five of them newly gated here: the two adapters' `pyproject.toml` versions, their runtime `__version__` values, and `trelix_mcp.__version__`. For why lockstep, and not "independent cadence", is the policy, see [docs/BACKWARDS_COMPATIBILITY.md](BACKWARDS_COMPATIBILITY.md#why-lockstep-and-not-independent-cadence).

So pin all four to the same version in your `requirements.txt`:

```
trelix==3.1.6
trelix-mcp==3.1.6
trelix-langchain==3.1.6
trelix-llama-index==3.1.6
```

The version stamp and the dependency floor are separate facts, and a reader pinning versions needs both. Both adapters at 3.1.5 still declare `trelix>=3.0.0`; the floor was deliberately not raised to match the stamp, because a floor is an API compatibility contract rather than a statement about release cadence. `packages/trelix-langchain/pyproject.toml` records the reasoning: 3.0.0 is the lowest published core verified to expose every name `retriever.py` reads. So `trelix-langchain` 3.1.5 resolving against core 3.0.0 is supported and intended — the four-way 3.1.5 pin above is the combination CI installs and tests, not the only one that works.

---

### What is the recommended setup for a team?

A typical team setup for a shared codebase:

1. **Shared index in CI.** Use the [trelix-index-action](https://github.com/sairam0424/trelix-index-action) to build and cache the index on every push. The index file (`.trelix/index.db`) can be committed or shared via a CI artifact.

2. **REST API for shared access.** Run `trelix serve` on a team server so every developer queries the same index:
   ```bash
   pip install "trelix[serve]"
   trelix serve ./my-repo --port 8765
   ```
   The REST API exposes `/search`, `/ask` (SSE streaming), `/index`, `/health`, and `/stats`.

3. **MCP for individual developers.** Each developer installs `trelix-mcp` locally and points it at either their own local index or the team REST server.

4. **Voyage or BGE-Code embeddings.** For best retrieval quality across a shared index, standardize on one embedding provider. `voyage-code-3` (API) or `bge-code` (local, SOTA 2025) are recommended over the default `local` provider for teams.

5. **Federation for multi-repo monorepos.** Register all service repos and use `trelix search-all` or `FederatedRetriever` for cross-repo search.

---

### How do I deploy `trelix serve`?

`trelix serve` starts a lightweight FastAPI REST server:

```bash
pip install "trelix[serve]"
trelix serve ./my-repo --port 8765
```

For production:

```bash
# With gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker "trelix.server:create_app(repo_path='./my-repo')" --bind 127.0.0.1:8765

# With Docker
docker run -v $(pwd)/my-repo:/repo -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -p 127.0.0.1:8765:8765 trelix serve /repo --host 0.0.0.0 --port 8765
```

Both forms bind loopback on the **host** deliberately. The API is **open by
design when no token is set**, so binding a non-loopback interface serves
unauthenticated code search over the indexed repository to anything that can
reach the box. Reach a wider interface — a different `--bind` address for
gunicorn, or a `-p` mapping without the `127.0.0.1:` prefix for Docker — only
together with `TRELIX_API_AUTH_TOKEN`, which makes every route require an
`X-Trelix-Api-Key` header. (`--host 0.0.0.0` in the Docker command is the
*container's* bind, which a published port needs in order to reach the process;
the `-p` mapping is the exposure decision, and `docker-compose.yml` at the repo
root carries the same split.)

Available endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Returns `{"status": "ok"}` |
| `GET /stats` | GET | Index statistics (chunk count, file count, languages) |
| `GET /search` | GET | Hybrid search: `?q=<query>&top_k=5` |
| `GET /ask` | GET | Streaming synthesis via SSE: `?question=<question>` |
| `POST /index` | POST | Trigger a full or incremental re-index |
| `GET /graph` | GET | Knowledge graph stats: `?repo=<path>` |
| `GET /graph/communities` | GET | Community summary list |
| `GET /graph/visualize` | GET | Export Pyvis HTML, return path |
| `GET /graph/search` | GET | BFS from symbol: `?repo=<path>&symbol_id=<id>&depth=2` |

The server is stateless — all state is in `.trelix/index.db`. Horizontal scaling is possible by mounting the index file as a read-only volume on multiple server instances.

---

*For additional help, open an issue at [github.com/sairam0424/trelix](https://github.com/sairam0424/trelix) or consult the other docs in this directory: [GETTING_STARTED.md](GETTING_STARTED.md), [USER_GUIDE.md](USER_GUIDE.md), [architecture.md](architecture.md), and [GLOSSARY.md](GLOSSARY.md).*
