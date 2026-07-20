# Getting Started with Trelix

**Time to first search: ~5 minutes**

You installed `trelix`. Here is what it can do and how to use it.

---

## 1. Install Trelix

```bash
pip install "trelix[local]"
```

The `local` extra bundles `sentence-transformers` so embeddings run entirely on your machine — no API key required to index and search.

Trelix requires **Python 3.11+** and supports over 20 programming languages via tree-sitter parsers (Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, Swift, Kotlin, and more).

Verify the install:

```bash
trelix --version
# trelix 2.8.1
```

---

## 2. Index Your First Repository

Point `trelix index` at any local directory. Trelix parses every supported source file, extracts symbols and doc chunks, embeds them with the default local embedder, and writes the index to `<repo>/.trelix/index.db` — a single SQLite file, zero external infrastructure.

```bash
trelix index ./my-repo
```

Expected output:

```
[trelix] Scanning ./my-repo...
[trelix] Found 1,247 files (312 supported)
[trelix] Parsing symbols...  ████████████████████ 312/312
[trelix] Embedding chunks... ████████████████████ 4,891/4,891
[trelix] Index written to ./my-repo/.trelix/index.db
[trelix] Done. 4,891 chunks indexed in 18.4s
```

The `.trelix/` directory is self-contained. You can check it in or add it to `.gitignore` depending on whether you want the index shared with your team.

To see index statistics at any time:

```bash
trelix stats ./my-repo
```

```
Repository:  ./my-repo
Index:       ./my-repo/.trelix/index.db
Chunks:      4,891
Files:       312
Languages:   Python (204), TypeScript (68), YAML (40)
Embedder:    local (sentence-transformers/all-MiniLM-L6-v2)
Last index:  2026-07-05 09:32:11 UTC
```

---

## 3. Search Your Code

Trelix offers three interfaces for search: the CLI, the Python API, and the MCP server for AI assistants.

### CLI

Basic semantic search:

```bash
trelix search ./my-repo "authentication middleware"
```

```
1. src/middleware/auth.py:42  — verify_jwt_token()
   Verifies JWT signature and expiry. Returns decoded payload or raises AuthError.

2. src/middleware/auth.py:78  — require_auth()
   Decorator that calls verify_jwt_token and attaches user context to request.

3. src/api/routes/users.py:15 — UserRouter
   FastAPI router; all routes decorated with @require_auth.

4. tests/test_auth.py:31      — test_expired_token_raises()
   Asserts AuthError is raised when JWT exp is in the past.

5. docs/auth.md:1             — Authentication Overview
   Describes the JWT-based auth flow and token rotation policy.
```

Search returns ranked results with file path, line number, symbol name, and a one-line description.

### Python API

```python
from trelix import TrelixClient

client = TrelixClient(repo_path="./my-repo")

results = client.search("authentication middleware", top_k=5)
for r in results:
    print(r.file, r.line, r.symbol, r.score)
```

### MCP (for Claude and other AI assistants)

Install the MCP package and register it with Claude:

```bash
pip install trelix-mcp
claude mcp add trelix -- trelix-mcp
```

Once registered, Claude can call `search_code`, `ask_agent`, `build_knowledge_graph`, and other tools directly against your indexed repos. The MCP server supports pagination for large result sets and MCP resource subscriptions.

---

## 4. Dashboard Tour

All trelix capabilities are exposed as CLI subcommands. Here is the full reference:

| Command | What it does |
|---------|-------------|
| `trelix index ./my-repo` | Parse, embed, and write the index to `.trelix/index.db` |
| `trelix search ./my-repo "query"` | Hybrid semantic + keyword search, returns ranked chunks |
| `trelix query ./my-repo "question"` | Structured query over the index with no LLM — fast, offline, deterministic |
| `trelix ask ./my-repo "question"` | Retrieval-augmented answer using an LLM (requires `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or equivalent); `--session <id>` resumes a persisted agentic session (implies `--agentic`) |
| `trelix agent sessions list/show/clear ./my-repo` | List, inspect, or delete persisted agentic (ReAct) sessions |
| `trelix stats ./my-repo` | Index statistics: chunk count, file count, language breakdown, embedder, timestamp |
| `trelix graph ./my-repo` | Print the knowledge graph: symbol dependencies, call chains, import relationships |
| `trelix telemetry ./my-repo` | Index health: embedding coverage, stale file count, missing symbols |
| `trelix watch ./my-repo` | Watch a single repository for file changes and incrementally re-index on save |
| `trelix watch-all` | Watch all federated repositories simultaneously (v2.4.0) |
| `trelix review ./my-repo` | Run a code review against the local diff (uncommitted changes) |
| `trelix review --pr owner/repo#N` | Fetch and review a GitHub PR by number (v2.4.0) |
| `trelix federation add <alias> <path>` | Register a repo under a short alias for cross-repo search |
| `trelix search-all "query"` | Search across all federated repos in one command |
| `trelix serve ./my-repo --port 8765` | Start a REST API server for the repo's index |

**Embedding providers** (`--embedder` flag or `TRELIX_EMBEDDER` env var):

| Provider | Key required | Best for |
|----------|-------------|---------|
| `local` | No | Default; works fully offline |
| `openai` | `OPENAI_API_KEY` | General-purpose; high quality |
| `azure` | `AZURE_OPENAI_*` | Enterprise Azure deployments |
| `voyage` | `VOYAGE_API_KEY` | Code-optimized; strong ranking |
| `bge-code` | No | Local code-specialized model |
| `nomic-code` | No | Local; good on large repos |

---

## 5. Common Workflows

### Code Archaeology

When you need to understand how a feature was built or trace the history of a decision through the codebase:

1. Index the repo if you have not already: `trelix index ./my-repo`
2. Search for the feature area: `trelix search ./my-repo "rate limiting"`
3. Inspect the knowledge graph for a key symbol: `trelix graph ./my-repo` — look for the function or class and its callers.
4. Ask a natural-language question: `trelix ask ./my-repo "How does rate limiting interact with the authentication middleware?"` (requires LLM key).
5. Use `trelix query` for the same question offline if you do not have an LLM key available — you get structured results without a synthesized answer.

This workflow is faster than grep for cross-cutting concerns because hybrid search finds semantically related code even when the exact terms differ.

### Impact Analysis Before Refactor

Before changing a shared module, use trelix to understand what else depends on it:

1. Search for the module or symbol: `trelix search ./my-repo "UserService"`
2. Pull the graph: `trelix graph ./my-repo` — look at the `UserService` node's incoming edges (callers and importers).
3. Ask for the blast radius: `trelix ask ./my-repo "Which files would break if I change the UserService.get_by_id signature?"`
4. Check telemetry to see if the index is current: `trelix telemetry ./my-repo`
5. Keep the index fresh during the refactor with `trelix watch ./my-repo` running in a terminal — every save triggers an incremental re-index.

Run `trelix search ./my-repo "UserService"` again after the refactor to confirm all call sites were updated.

### PR Review Workflow

Use trelix to add code-intelligence context to any PR review:

**Local diff review** (your branch, not yet pushed):

```bash
trelix review ./my-repo
```

Trelix computes a diff against `HEAD`, retrieves context for every changed symbol, and produces a structured review with impact notes.

**GitHub PR review** (v2.4.0):

```bash
trelix review --pr owner/repo#42
```

Trelix fetches the PR diff from GitHub, runs the same retrieval pipeline against your local index, and outputs a review that references existing code patterns and potential regressions. Requires `GITHUB_TOKEN` for private repos.

Both modes output:
- Changed symbols and their call sites
- Semantically similar code that might need parallel updates
- Potential test gaps (symbols with no corresponding test chunk)

---

## 6. Useful Commands

### Index management

```bash
# Re-index after large changes (full rebuild)
trelix index ./my-repo

# Incremental watch mode — re-indexes only changed files on save
trelix watch ./my-repo

# Watch all federated repos at once
trelix watch-all

# Check index health and coverage
trelix telemetry ./my-repo

# View statistics (chunk count, languages, age)
trelix stats ./my-repo
```

### Direct REST API

Start a REST server for integrations that cannot use the CLI:

```bash
trelix serve ./my-repo --port 8765
```

```bash
# Search via HTTP
curl "http://localhost:8765/search?q=authentication+middleware&top_k=5"

# Ask a question via HTTP
curl -X POST http://localhost:8765/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does rate limiting work?"}'

# Index stats via HTTP
curl http://localhost:8765/stats
```

### Federation shortcuts

```bash
# Register repos under aliases
trelix federation add backend ./services/api
trelix federation add frontend ./apps/web
trelix federation add infra ./infrastructure

# Search across all registered repos simultaneously
trelix search-all "database connection pooling"

# List registered repos
trelix federation list
```

### LangChain and LlamaIndex integrations

```bash
# LangChain
pip install trelix-langchain
```

```python
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="./my-repo", top_k=5)
docs = retriever.get_relevant_documents("rate limiting middleware")
```

```bash
# LlamaIndex
pip install trelix-llama-index
```

```python
from trelix_llama_index import TrelixIndexRetriever

retriever = TrelixIndexRetriever(repo_path="./my-repo")
nodes = retriever.retrieve("rate limiting middleware")
```

---

## 7. Next Steps

- **[architecture.md](./architecture.md)** — internals: how hybrid search combines dense and sparse retrieval, the multi-granularity chunking model, and the agentic loop
- **[integrations/](./integrations/)** — deep-dive guides for MCP, LangChain, LlamaIndex, and the REST API
- **[superpowers/](./superpowers/)** — advanced capabilities: federation cache, cross-repo graph, telemetry schema
- **[v2.4.0-world-release-report.md](./v2.4.0-world-release-report.md)** — full changelog for v2.4.0 (watch-all, review --pr, federation cache, MCP pagination)
