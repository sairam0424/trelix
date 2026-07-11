# trelix User Guide — v2.7.1

**Audience:** Developers, tech leads, and engineering teams who want to understand, navigate, and interrogate their codebases faster.
**Time to read:** ~30 minutes (or jump directly to the section you need).

---

## Table of Contents

1. [What Is Code Intelligence?](#1-what-is-code-intelligence)
2. [Why Use trelix?](#2-why-use-trelix)
3. [How trelix Works](#3-how-trelix-works)
4. [The Retrieval Pipeline — All 7 Legs](#4-the-retrieval-pipeline--all-7-legs)
5. [Step-by-Step: Index Your First Repo](#5-step-by-step-index-your-first-repo)
6. [Search Modes](#6-search-modes)
7. [Understanding Results](#7-understanding-results)
8. [Common Workflows](#8-common-workflows)
9. [MCP Integration](#9-mcp-integration)
10. [Embedding Providers](#10-embedding-providers)
11. [Understanding Query Telemetry](#11-understanding-query-telemetry)
12. [Troubleshooting Common Issues](#12-troubleshooting-common-issues)
13. [CLI Flags and Configuration Reference](#13-cli-flags-and-configuration-reference)
14. [API Quick Reference](#14-api-quick-reference)

---

## 1. What Is Code Intelligence?

### The library analogy

Imagine a large public library with 50,000 books. A basic text search engine can find the word "authentication" in every book that contains it. That is keyword search — fast, literal, and shallow.

A *code intelligence* system is something deeper. It understands that `authenticate_user()` in `auth/middleware.py` is *called by* `handle_request()` in `api/server.py`, which is *imported by* `main.py`. It knows that the `UserService` class *implements* the `IUserRepository` interface, and that both live in separate files. It can answer: "If I change how tokens are validated, which files will I need to update?" — and give you actual file names, line numbers, and callee signatures.

trelix is a **search engine for your codebase's meaning, not just its text**.

### Why keyword search is not enough

When you open a large unfamiliar codebase and search for `"JWT"` in your editor, you get 47 matches across 23 files. Some are in tests, some in comments, some are unrelated token validation helpers. You spend 20 minutes manually reading through them to understand the one function that actually creates and signs tokens.

trelix answers: *"how does JWT authentication work in this codebase?"* in under 3 seconds with a synthesized answer grounded in the actual code — not a hallucination.

### The three pillars of code intelligence

**Syntax understanding** — trelix uses [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) to parse your code into an Abstract Syntax Tree (AST). It does not tokenize strings. It understands the difference between a function definition and a function call, between a class declaration and an instantiation, between an import statement and a usage. It parses 20+ languages: Python, TypeScript, JavaScript, Go, Java, Rust, C, C++, C#, Kotlin, Ruby, and more.

**Semantic understanding** — trelix embeds every code symbol (function, class, method, variable) into a vector space where *meaning* is preserved. The embedding for `validate_password()` and `check_credentials()` sit close together even though they share no words. This is what makes semantic search work.

**Structural understanding** — trelix stores the full call graph (who calls whom), import graph (who imports what), and type hierarchy (who extends/implements whom) and uses this graph to expand every search result. When you ask about `process_payment()`, trelix also surfaces the callers that trigger it and the helpers it depends on.

---

## 2. Why Use trelix?

Here are the five situations where trelix is the right tool:

### 1. Pre-refactor blast radius analysis

You want to rename or restructure `UserRepository.find_by_email()`. Before touching a line, you need to know: which files call this method? What are its callers' callers? Are there tests that stub it?

Without trelix, this is a combination of `grep -r`, manual reading, and hope. With trelix:

```bash
trelix search ./my-repo "UserRepository.find_by_email callers"
# or, more precisely:
trelix ask ./my-repo "what calls UserRepository.find_by_email and what would break if its signature changed?"
```

trelix uses its `blast_radius` retrieval intent to expand in the *reverse import direction* — finding everything that depends on the symbol, not just the symbol itself. You get a prioritized list of affected files with line numbers before you write a single line of code.

### 2. PR code review

Your team receives a 400-line PR that touches the payment processing layer. Before approving, you need to understand what the changed functions depend on and whether the change is safe. trelix can review the PR diff directly:

```bash
trelix review --pr owner/repo#142
```

This fetches the diff from GitHub, runs code intelligence on the changed symbols, and produces a structured review report highlighting blast radius, callee depth, and potential breakage points — in the time it takes you to read one file manually.

### 3. Onboarding to a new codebase

You joined the team two weeks ago. The codebase has 150,000 lines across 800 files. Your task is to add a new payment provider. You have no idea where payment routing lives.

```bash
trelix ask ./my-repo "walk me through how a payment is processed end to end, from the API endpoint to the database"
```

trelix activates its multi-step decomposition mode, breaks this into sub-queries (HTTP handler → service layer → repository → database write), retrieves the relevant symbols for each step, and synthesizes a narrative explanation with the actual function names and file paths filled in. Ten minutes of trelix replaces a week of code reading.

### 4. Debugging unknown code

You are on call. An exception trace says `line 342 in validate_token_claims`. You have never seen this function. You need to understand what it does, what calls it, and what values it expects — right now.

```bash
trelix search ./my-repo "validate_token_claims"
```

The first result is the function definition with its signature and docstring. The second is the caller that passes it token data. The third is the test that covers the happy path. In 30 seconds you have enough context to diagnose the bug.

### 5. Architecture understanding

You are the new tech lead. You need to present the system architecture to the team. You need to understand how the ten microservices relate to each other, which service owns which data model, and what the request lifecycle looks like for the main user-facing API.

```bash
TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED=true trelix ask ./my-repo "explain the overall architecture: which services exist, what each one does, and how they communicate"
```

With the knowledge graph layer active, trelix builds a Code Property Graph from all call, import, and type edges, runs Louvain community detection to identify architectural modules, and synthesizes a high-level description with specific file references. You have a working architecture narrative in under a minute.

---

## 3. How trelix Works

trelix operates in two separate phases: **indexing** (offline, one-time per repo) and **retrieval** (online, per query). Understanding this split is essential for knowing what to expect from each command.

### Phase 1 — Indexing (`trelix index`)

Think of indexing as building a very sophisticated card catalog for your codebase. It runs once, takes a few minutes for most repos, and produces a single SQLite file at `.trelix/index.db`.

```
Repository files
       |
       v
  [FileWalker]
  Walks every file, skips .gitignore entries,
  computes SHA-256 hashes for change detection
       |
       v
  [Tree-sitter Parser]
  Parses each file into an AST.
  Extracts: functions, classes, methods, their signatures, docstrings, bodies.
  Extracts: call edges (who calls whom), import edges, type hierarchy edges.
       |
       v
  [ContextualChunker]
  Wraps each symbol's body in a breadcrumb header:
  "File: auth/jwt.py | Class: JWTValidator | Method: validate_claims"
  Optionally generates a 2-3 sentence LLM summary per symbol.
       |
       v
  [Embedder]
  Converts each chunk text into a dense vector (384–4096 dimensions
  depending on the embedding provider).
       |
       v
  [SQLite store]
  Saves everything: symbols, call_graph, imports, type_edges,
  BM25 FTS5 index, HNSW vector index.
  Zero external infrastructure — one file.
```

After `trelix index ./my-repo` completes, all retrieval happens locally against `.trelix/index.db`. Nothing is sent to an external server unless you use an API-based embedding provider.

### Phase 2 — Retrieval (`trelix search` / `trelix ask`)

Retrieval is the live, per-query phase. Every time you run a search, trelix:

1. Classifies your query (is this a symbol lookup? a feature flow question? a project overview?).
2. Selects the appropriate retrieval strategy (which search legs to use, how deep to traverse the graph).
3. Runs all selected legs in parallel.
4. Fuses their results using Reciprocal Rank Fusion (explained below).
5. Expands results via the call and import graph.
6. Optionally reranks the final list for precision.
7. For `ask` queries: synthesizes a natural language answer from the top results using an LLM.

### The 3-tier query planner

Before retrieval starts, trelix classifies your query into one of three tiers:

**Tier 1 — Direct:** Simple factual patterns like "what is X" or "define X". trelix skips retrieval entirely and lets the LLM answer from its training knowledge. These are answered in milliseconds.

**Tier 2 — Single-step (default):** Most code queries. trelix classifies the query into one of 8 intents (symbol lookup, feature flow, blast radius, etc.) and applies a pre-baked retrieval strategy optimized for that intent.

**Tier 3 — Multi-step:** Complex queries with phrases like "walk me through", "end-to-end", or "step by step". The LLM decomposes the query into 2–3 focused sub-queries, retrieves each independently, merges the results, then synthesizes a unified answer.

### Reciprocal Rank Fusion (RRF)

Different retrieval legs return different ranked lists. Vector search ranks by cosine similarity. BM25 ranks by keyword frequency. The call-graph leg ranks by graph distance. How do you combine these into one list?

Reciprocal Rank Fusion uses the formula: `score(d) = Σ 1/(k + rank_i(d))` for each leg `i` that returns document `d`, with `k=60` (a constant that prevents any one very-high-ranked result from dominating). This is the same technique used in production search systems at major tech companies. It is simple, robust, and empirically outperforms learned fusion for code retrieval.

The practical meaning: a result that appears at rank 3 in the vector leg AND rank 2 in the BM25 leg AND rank 5 in the grep leg gets a much higher fused score than a result that appears at rank 1 in only one leg. Breadth of agreement across legs signals genuine relevance.

---

## 4. The Retrieval Pipeline — All 7 Legs

trelix v2.7.1 supports up to 7 parallel retrieval legs. Three are always active; four are opt-in. All results are fused via RRF, then graph-expanded, then optionally reranked.

```
User Query
    |
    v
[Query Enhancement Layer]  ←  optional: HyDE + multi-query expansion
    |
    v
┌───────────────────────────────────────────────────────────────┐
│  PARALLEL RETRIEVAL (all enabled legs run simultaneously)     │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│ Leg 1:       │ Leg 2:       │ Leg 3:       │ Leg 4:           │
│ Vector       │ BM25         │ Grep         │ File Summary     │
│ (always on)  │ (always on)  │ (always on)  │ (opt-in)         │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ Leg 5:       │ Leg 6:       │ Leg 7:       │                  │
│ CodeGraph    │ Sparse       │ Sub-chunk    │                  │
│ BFS (opt-in) │ (opt-in)     │ (opt-in)     │                  │
└──────────────┴──────────────┴──────────────┴──────────────────┘
    |
    v
[RRF Fusion — all legs, k=60]
    |
    v
[Graph Expansion — call_graph + import_graph + type_edges]
    |
    v
[Reranker — Cohere | cross-encoder | PLAID] ← optional
    |
    v
[Post-Rerank Enhancement: PageRank Boost + FLARE Loop] ← optional
    |
    v
[Context Assembler]
    |
    v
[LLM Synthesis] ← only for `trelix ask`
```

> **v2.5.0 — Multi-query expansion is now wired and active.** Previously built but disconnected, `MultiQueryExpander` is now integrated into the standard retrieval path. Enable it with `TRELIX_RETRIEVAL_MULTI_QUERY=true` (default: `false`). Tune the number of generated variants with `TRELIX_RETRIEVAL_MULTI_QUERY_COUNT=3` (default: `2`, range: `1–4`). When enabled, the primary query is rephrased into N variants by the LLM; each variant runs all retrieval legs in parallel via `ThreadPoolExecutor`, and all results are RRF-merged (`k=60`) before synthesis. Falls back silently to the original query when no LLM is available. HyDE and multi-query can be enabled simultaneously.

### Leg 1 — Vector (always active)

**What it does:** Converts your query into a dense vector using the same embedding model used at index time, then runs an Approximate Nearest Neighbor (ANN) search over all chunk vectors using an HNSW index (O(log n) lookup).

**Why it matters:** This is the semantic leg. It finds code that *means the same thing* as your query even when no words match. "User credential verification" will find `authenticate_user()` even if neither word appears in the function.

**When it leads:** Abstract natural-language queries, questions about behavior, anything phrased in terms of concepts rather than identifiers.

**Configuration:**
```bash
TRELIX_STORE_HNSW=true          # Enable HNSW (default)
TRELIX_STORE_HNSW_M=16          # HNSW M parameter (higher = more accurate, more memory)
TRELIX_STORE_HNSW_EF_SEARCH=50  # ef_search (higher = more accurate, slower)
```

### Leg 2 — BM25 (always active)

**What it does:** Runs a keyword frequency search over the `symbols_fts` FTS5 virtual table, which indexes function names, qualified names, docstrings, bodies, and (when contextual chunking is enabled) LLM-generated context summaries.

**Why it matters:** BM25 is unbeatable for identifier-exact queries. When you type `authenticate_user`, BM25 finds every file containing that exact string with perfect precision. Semantic search would find semantically similar code too — which helps for discovery, but hurts for precision lookups.

**When it leads:** Exact function name lookups, searching for a specific class or method name, finding configuration keys.

**Key difference from grep:** BM25 is ranked (most relevant results first) and understands term frequency and document length. Grep is unranked and returns everything that matches literally.

### Leg 3 — Grep (always active)

**What it does:** Direct SQL `LIKE` or regex match on `symbols.name` and `symbols.qualified_name` columns.

**Why it matters:** Zero latency, no embedding required, highest precision for exact symbol lookups. When the query is a function name or class name typed exactly, grep finds it in microseconds.

**When it leads:** `symbol_lookup` intent — any query that looks like a Python or TypeScript identifier.

**Example queries that activate grep first:**
- `"UserService.create_user"`
- `"validate_token_claims"`
- `"PaymentProcessor"`

### Leg 4 — File Summary (opt-in, `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=true`)

**What it does:** Runs ANN search over per-file LLM-generated 2–4 sentence summaries. This is the RAPTOR technique (Recursive Abstractive Processing for Tree-Organized Retrieval): summarize at multiple granularities, retrieve the right granularity for the question.

**Why it matters:** Some questions are inherently about files, not symbols. "What does the authentication module do?" or "explain the database access layer" are answered better by file-level summaries than by individual function chunks. File summaries give trelix a higher-altitude view of the codebase.

**Prerequisite:** Requires `TRELIX_FILE_SUMMARIES_ENABLED=true` at index time.

```bash
TRELIX_FILE_SUMMARIES_ENABLED=true trelix index ./my-repo
TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=true trelix ask ./my-repo "what does the auth module do?"
```

**When it leads:** Project overview queries, architecture questions, onboarding questions about what a module does.

### Leg 5 — CodeGraph BFS (opt-in, `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED=true`)

**What it does:** After the first three legs produce seed results, BFS (breadth-first search) traverses the Code Property Graph starting from those seeds. Nodes one hop away get score `0.5`, two hops get `0.25`, etc. This surfaces callers, callees, imported modules, and type ancestors that no text or embedding search would find.

**Why it matters:** Imagine you search for `process_payment` and get the function body. But you also need to know what calls it, and what that caller depends on. Graph BFS surfaces this structural neighborhood automatically. It finds code that is *related by structure* rather than by text or meaning.

**Prerequisite:** Requires `trelix graph ./my-repo` to be run once to build the Code Property Graph.

```bash
pip install trelix[knowledge-graph]
trelix graph ./my-repo
TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED=true trelix ask ./my-repo "how does checkout work?"
```

**When it leads:** Feature flow questions, blast radius queries, any question requiring a chain of related symbols.

### Leg 6 — Sparse (opt-in, `TRELIX_RETRIEVAL_SPARSE=true`)

**What it does:** Uses SPLADE-Code, a learned sparse model, to produce a sparse vector of (token_id, weight) pairs at index time. At query time, dot-product aggregation over the SQLite inverted index finds the most relevant chunks. Think of this as a learned BM25 that understands subword relationships.

**Why it matters:** BM25 breaks down when identifiers use camelCase or snake_case. `userRepository` and `user_repository` are different tokens to BM25. SPLADE-Code learns that they are the same concept and handles subword fragmentation gracefully.

**When it leads:** Queries involving camelCase identifiers, TypeScript/Java codebases with long compound names.

### Leg 7 — Sub-chunk / Multi-granularity (opt-in, `TRELIX_CHUNKER_MULTI_GRANULARITY=true`)

**What it does:** Indexes each symbol at two granularities simultaneously: the whole symbol (function body) and its individual statements. Retrieval can therefore return a single line of code as a result, not just the whole function.

**Why it matters:** Some queries are best answered at statement granularity. "Where is the database connection opened?" is answered by a single line inside a larger function. Sub-chunk indexing lets trelix surface that line directly rather than returning the entire 80-line function.

**When it leads:** Specific operational queries ("where is X set?", "when is Y called?").

---

## 5. Step-by-Step: Index Your First Repo

This walkthrough takes you from a fresh install to your first working query against a real repository.

### Prerequisites

- Python 3.11 or 3.12 installed
- A code repository you want to explore (use your own or any public GitHub repo)

### Step 1 — Install trelix

Choose the installation that matches your available API keys:

**Option A — Local embeddings, no API key needed:**
```bash
pip install "trelix[local]"
```
This uses `all-MiniLM-L6-v2` for embeddings (384-dim) and skips LLM synthesis. You can search but not ask questions. Good for getting started.

**Option B — OpenAI for best-quality search and synthesis:**
```bash
pip install trelix
export OPENAI_API_KEY=sk-...
```
Uses `text-embedding-3-large` (3072-dim) for embeddings and `gpt-4o` for synthesis. Excellent quality. Small cost per query.

**Option C — Voyage AI for best code-specific embeddings:**
```bash
pip install "trelix[voyage]"
export VOYAGE_API_KEY=...
```
Uses `voyage-code-3` (code-specialized, 1024-dim Matryoshka) for embeddings. Best retrieval quality for code. Requires a Voyage API key.

**Option D — Offline local code-specialized embeddings (no API key, needs ~8GB RAM):**
```bash
pip install "trelix[local-code]"
```
Uses `SFR-Embedding-Code-2B_R` (4096-dim). CoIR score 67.41 — excellent quality with zero API cost. Requires a machine with 8GB+ RAM.

### Step 2 — Index the repository

Navigate to the parent directory of the repository you want to index, then run:

```bash
trelix index ./my-repo
```

You will see output like this:

```
trelix v2.7.1 — indexing ./my-repo
✓ FileWalker: 243 files found (.gitignore applied)
  Phase 1/4 — Parse
    [████████████████████] 243/243 files  3.2s
  Phase 2/4 — Write symbols
    ✓ 4,891 symbols written (functions: 2,104, classes: 312, methods: 1,847, other: 628)
    ✓ 8,204 call edges resolved (3,441 exact qualified-name, 2,108 type-hint, 2,655 name)
    ✓ 1,024 import edges stored
    ✓ 312 type edges (extends: 89, implements: 187, trait_impl: 36)
  Phase 3/4 — Embed
    [████████████████████] 9,241 chunks  18.4s  (avg 2.0ms/chunk)
  Phase 4/4 — Resolve cross-file edges
    ✓ Cross-file call edges: 1,832
─────────────────────────────────────────────
  Total time:    24.1s
  Index size:    48.3 MB  (.trelix/index.db)
  Files:         243
  Symbols:       4,891
  Chunks:        9,241
  Vectors:       9,241  (sqlite-vec HNSW, M=16)
─────────────────────────────────────────────
Done. Run: trelix search ./my-repo "your query"
```

The index is stored at `./my-repo/.trelix/index.db`. This file is safe to commit to `.gitignore` (it is generated, not authored). You can regenerate it at any time with the same command.

### Step 3 — Run your first search

```bash
trelix search ./my-repo "authentication"
```

Expected output (Rich table in terminal):

```
 trelix search — ./my-repo — "authentication"

 #   Score   Source                              Symbol                    Lines   Lang
 ─── ─────── ─────────────────────────────────── ──────────────────────── ─────── ──────────
 1   0.924   src/auth/middleware.py               authenticate_request      42-67   python
 2   0.891   src/auth/jwt.py                      JWTValidator.validate      18-44   python
 3   0.867   src/users/service.py                 UserService.login         89-112   python
 4   0.821   src/auth/tokens.py                   generate_access_token      31-52   python
 5   0.794   tests/test_auth.py                   test_invalid_token        78-93   python
 6   0.776   src/auth/middleware.py                AuthMiddleware            12-40   python
 7   0.743   src/users/repository.py              UserRepository.find_by_email 44-61  python
 8   0.712   src/config/settings.py               JWT_SECRET_KEY             14-14   python
 9   0.688   src/api/routes.py                    login_route                55-71   python
 10  0.641   src/auth/exceptions.py               AuthenticationError        1-12    python
```

### Step 4 — Ask a question

If you installed with OpenAI or another LLM provider, you can ask natural language questions:

```bash
trelix ask ./my-repo "how does the authentication middleware work?"
```

Expected output (streams tokens live):

```
trelix ask — ./my-repo — "how does the authentication middleware work?"

The authentication middleware (src/auth/middleware.py:12-67) intercepts every
incoming HTTP request before it reaches a route handler. It reads the
Authorization header, extracts the Bearer token, and passes it to
JWTValidator.validate() (src/auth/jwt.py:18-44).

JWTValidator checks three things: the token signature (using the
JWT_SECRET_KEY from src/config/settings.py:14), the expiry timestamp, and the
issuer claim. If any check fails, it raises AuthenticationError
(src/auth/exceptions.py:1-12), which is caught by the FastAPI exception handler
and returns a 401 response.

If the token is valid, JWTValidator returns the decoded payload, which the
middleware attaches to the request context as request.state.user. Route handlers
downstream can access the authenticated user object from there without repeating
the token check.

The login flow that generates tokens lives in UserService.login
(src/users/service.py:89-112): it validates credentials, calls
generate_access_token() (src/auth/tokens.py:31-52) which signs a JWT with a
15-minute expiry, and returns the token to the client.

Sources used: src/auth/middleware.py, src/auth/jwt.py, src/users/service.py,
src/auth/tokens.py, src/config/settings.py
```

### Step 5 — View index statistics

```bash
trelix stats ./my-repo
```

```
trelix stats — ./my-repo

  Index:        ./my-repo/.trelix/index.db
  Version:      2.7.0
  Last indexed: 2026-07-05 10:32:14 UTC

  Files:        243
  Symbols:      4,891
    functions:  2,104
    classes:    312
    methods:    1,847
    other:      628
  Chunks:       9,241
  Vectors:      9,241  (sqlite-vec HNSW)
  Call edges:   8,204
  Import edges: 1,024
  Type edges:   312

  Index size:   48.3 MB
  Embed dims:   384  (provider: local)
```

---

## 6. Search Modes

trelix has three primary query commands. Each serves a different purpose. Knowing when to use which one saves time.

### `trelix search` — Retrieval only, no synthesis

```bash
trelix search ./my-repo "<query>"
```

**What it does:** Runs the full retrieval pipeline (all enabled legs, RRF fusion, graph expansion, optional reranking) and returns a ranked table of results. No LLM synthesis. No external API call (if using a local embedding provider).

**Output:** A Rich table with score, file path, symbol name, line numbers, and language.

**When to use it:**
- You want specific file and line references, not a narrative answer.
- You are working offline or want to avoid API costs.
- You are doing exploratory navigation: "what files are involved in X?"
- You want to see raw retrieval quality before asking a full question.

**Key options:**
```bash
trelix search ./my-repo "JWT validation" --top-k 20      # return 20 results (default: 10)
trelix search ./my-repo "auth" --lang python             # filter by language
trelix search ./my-repo "database" --file src/db/        # filter by file path prefix
trelix search ./my-repo "login" --rerank cohere          # apply Cohere reranker
```

### `trelix ask` — Retrieval + LLM synthesis

```bash
trelix ask ./my-repo "<question>"
```

**What it does:** Runs the full retrieval pipeline, assembles a context window from the top results, and sends the context to an LLM to synthesize a natural language answer. Streams tokens live as they are generated. For large contexts (>8,000 tokens or >20 results), activates GraphRAG map-reduce to avoid context window overflows.

**Output:** Streaming natural language answer with source citations at the end.

**When to use it:**
- You want to understand *how something works*, not just find where it is.
- You are onboarding and need an explanation of a module.
- You are debugging and need a summary of what a chain of functions does.
- You want an architecture overview.

**Requires:** An LLM API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_API_KEY`, etc.) or a local Ollama instance via LiteLLM.

**Key options:**
```bash
trelix ask ./my-repo "how does auth work?" --stream          # explicit streaming flag
trelix ask ./my-repo "explain the request lifecycle" --top-k 30  # more context
TRELIX_RETRIEVAL_FLARE=true trelix ask ./my-repo "complex question"  # enable FLARE re-retrieval
```

**GraphRAG map-reduce:** When the number of relevant results exceeds 20 or total context tokens exceed 8,000, trelix automatically switches to map-reduce synthesis:
1. **Map:** Each group of ~10 results is sent to the LLM separately to produce a partial answer.
2. **Reduce:** All partial answers are sent to the LLM together for final synthesis.

This prevents context window truncation on large codebases without sacrificing answer quality.

### `trelix query` — Structured JSON output

```bash
trelix query ./my-repo "<query>" --json
```

**What it does:** Runs the same retrieval pipeline as `trelix search` but outputs JSON to stdout instead of a Rich table. Useful for scripting, CI pipelines, or feeding results into other tools.

**Output:**
```json
{
  "query": "JWT validation",
  "results": [
    {
      "score": 0.924,
      "file": "src/auth/jwt.py",
      "symbol": "JWTValidator.validate",
      "qualified_name": "src.auth.jwt.JWTValidator.validate",
      "line_start": 18,
      "line_end": 44,
      "language": "python",
      "source": "vector+bm25",
      "body": "def validate(self, token: str) -> dict:\n    ..."
    }
  ],
  "total": 10,
  "latency_ms": 43.2,
  "legs_used": ["vector", "bm25", "grep"]
}
```

**When to use it:**
- Scripting: feeding trelix output into a CI step, a report generator, or a custom tool.
- Building integrations: connecting trelix to your own dashboard or notification system.
- Debugging the retrieval pipeline: seeing exactly which legs contributed to which results.

---

## 7. Understanding Results

Every result in a `trelix search` table (and every entry in the JSON output) has the same fields. Here is what each one means.

### `score`

**Type:** Float, 0.0–1.0 (approximately).

The fused relevance score after RRF fusion, graph expansion, and optional reranking. Higher means more relevant.

**Practical interpretation:**
- `0.90+` — Almost certainly what you are looking for. The symbol is highly relevant to your query across multiple retrieval legs.
- `0.70–0.89` — Very likely relevant. Worth reading.
- `0.50–0.69` — Possibly relevant. May be a related concept, a caller, or a test file.
- `< 0.50` — Marginally relevant. The retrieval system surfaced it due to structural proximity (graph expansion), not direct relevance.

Note: scores are relative to the results in the current query. A score of 0.7 in a query with tight relevance is better than a score of 0.9 in a query where all results are loosely related.

### `source`

**Type:** String, e.g. `"vector"`, `"bm25"`, `"grep"`, `"vector+bm25"`, `"graph"`.

Which retrieval legs contributed to this result. Multiple legs separated by `+` means the result appeared in multiple legs and its RRF score reflects that breadth.

**Why this matters:**
- `"grep"` — exact match on the symbol name; highest precision for identifier lookups.
- `"vector"` — semantic match; the symbol means something similar to your query even if words differ.
- `"bm25"` — keyword frequency match; the symbol contains your query terms with high frequency.
- `"graph"` — surfaced via BFS graph expansion from a seed result; structurally related to relevant code.
- `"vector+bm25+grep"` — appeared in all three primary legs; extremely high confidence result.

### `symbol`

**Type:** String, e.g. `"JWTValidator.validate"`.

The short display name of the code symbol. For top-level functions it is just the function name. For methods it is `ClassName.method_name`.

### `qualified_name`

**Type:** String, e.g. `"src.auth.jwt.JWTValidator.validate"`.

The fully-qualified identifier including the module path. This is the permanent, unique identifier for the symbol in trelix's index. When using the API or MCP tools, use `qualified_name` rather than `symbol` to avoid ambiguity.

### `line_start` and `line_end`

**Type:** Integer.

The 1-indexed line range of the symbol in the file. Use these to jump directly to the relevant code in your editor.

### `language`

**Type:** String, e.g. `"python"`, `"typescript"`, `"go"`.

The language detected by Tree-sitter for this file. Useful when filtering results across a polyglot codebase.

### `file`

**Type:** String, relative path from repo root.

The file containing this symbol. Always relative to the repo root that was indexed. Combine with `line_start` to get a full click-to-open reference.

### `body`

**Type:** String (present in JSON output, truncated in Rich table).

The source code body of the symbol. In JSON output (`--json` flag), the full body is included. In the terminal table, it is shown as a truncated snippet.

---

## 8. Common Workflows

### 8a. Pre-refactor blast radius analysis

**Scenario:** You want to change the signature of `UserRepository.get_by_email(email: str)` to `get_by_email(email: str, active_only: bool = True)`. Before making the change, you need to know every call site.

**Step 1 — Find direct callers:**
```bash
trelix search ./my-repo "UserRepository.get_by_email callers" --top-k 20
```

Look at results with `source` containing `"grep"` or `"bm25"` — these are exact matches and likely direct call sites.

**Step 2 — Ask for blast radius analysis:**
```bash
trelix ask ./my-repo "what calls UserRepository.get_by_email and what would break if I added an active_only parameter?"
```

The `blast_radius` intent activates reverse import traversal. trelix finds `get_by_email` call sites, then checks what imports the callers, expanding outward to show the full dependency chain.

**Step 3 — Use the MCP tool for programmatic blast radius (in Claude Code or Cursor):**
```
trelix: blast_radius(symbol_name="get_by_email", repo_path="./my-repo")
```

Returns a JSON list of all symbols that directly or indirectly depend on the target symbol.

**Step 4 — Enable graph search for deeper traversal:**
```bash
TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED=true trelix ask ./my-repo "blast radius of UserRepository.get_by_email — what would need to change?"
```

With the knowledge graph active, graph BFS from `get_by_email` surfaces callers, their callers, and imported helpers — the full impact surface.

**What good output looks like:**
```
The following files directly call UserRepository.get_by_email:

1. src/auth/service.py:156  — AuthService.authenticate_by_email
2. src/users/api.py:89      — UserEndpoints.get_user_profile
3. src/admin/views.py:44    — AdminUserView.lookup_user
4. tests/test_auth.py:112   — test_login_with_email

Adding active_only=True would require updating all 4 call sites. The
authentication flow (1, 4) is most sensitive — test coverage looks good
(test_login_with_email covers the primary path). The admin view (3) has no
test coverage for this specific call path.
```

---

### 8b. Onboarding to a new codebase in 10 minutes

**Scenario:** You joined the team yesterday. The repository has 200 files and 60,000 lines. Your first task is to understand the request lifecycle for the main API.

**Minute 1 — Index:**
```bash
pip install "trelix[local]"  # no API key needed for basic navigation
trelix index ./my-repo
```

**Minute 2 — Get the high-level architecture:**
```bash
trelix ask ./my-repo "what are the main modules and what does each one do?"
```

If you have an LLM API key configured, this gives you a module-by-module summary. Without an LLM, use:
```bash
trelix search ./my-repo "main module architecture overview" --top-k 15
```

This surfaces the top-level entry points and key organizational units.

**Minutes 3–5 — Trace a request:**
```bash
trelix ask ./my-repo "walk me through a user login request from the HTTP endpoint to the database"
```

The multi-step query planner decomposes this into: HTTP handler lookup → service layer → repository → database write. Each sub-query is retrieved independently and the results are synthesized into a coherent narrative with file paths.

**Minutes 6–8 — Understand the data model:**
```bash
trelix search ./my-repo "User model schema database" --top-k 10
trelix ask ./my-repo "what are the main data models and how do they relate to each other?"
```

**Minutes 9–10 — Find your entry point for the task:**
```bash
trelix search ./my-repo "payment processing" --top-k 10
trelix ask ./my-repo "if I need to add a new payment provider, which files would I need to change?"
```

The `blast_radius` intent combined with graph expansion gives you the files that own the payment provider abstraction and the interfaces you need to implement.

**Building the knowledge graph for architecture visualization:**
```bash
pip install trelix[knowledge-graph]
trelix graph ./my-repo --visualize
```

This produces `./my-repo/.trelix/graph.html` — an interactive Pyvis visualization of the entire codebase's call, import, and type relationship graph, with community-colored nodes showing architectural module boundaries. Open it in a browser to see the structure at a glance.

---

### 8c. PR review with GitHub integration

**Scenario:** A teammate submitted a 320-line PR touching the authentication layer. You need to understand what changed, what it depends on, and whether anything could break.

**Step 1 — Review the PR:**
```bash
export GITHUB_TOKEN=ghp_...  # your GitHub personal access token

trelix review --pr owner/repo#142
```

Expected output:

```
trelix review — owner/repo#142

PR: "feat(auth): add OAuth2 provider support"
Files changed: 8  |  Additions: +284  Deletions: -36

── Changed files ────────────────────────────────────────────────────
  MODIFIED  src/auth/providers.py        (+142 -12)
  MODIFIED  src/auth/middleware.py       (+31 -8)
  ADDED     src/auth/oauth2.py           (+89 +0)
  MODIFIED  src/users/service.py         (+12 -8)
  MODIFIED  tests/test_auth.py           (+10 -8)
─────────────────────────────────────────────────────────────────────

── Blast radius analysis ─────────────────────────────────────────────
  src/auth/providers.py → imported by:
    src/auth/middleware.py (already in diff ✓)
    src/api/app.py (NOT in diff — verify no interface breakage)
    src/config/dependencies.py (NOT in diff — verify no interface breakage)

  AuthProvider (modified) → implemented by:
    src/auth/jwt_provider.py (NOT in diff)
    src/auth/oauth2.py (in diff ✓)

── Findings ──────────────────────────────────────────────────────────
  ⚠  src/api/app.py references AuthProvider.authenticate() — verify
     the method signature is backward-compatible with this call site.

  ⚠  No test coverage added for OAuthProvider.revoke_token() in oauth2.py.

  ✓  AuthMiddleware test coverage covers the modified validation path.
─────────────────────────────────────────────────────────────────────
```

**Step 2 — Post findings back to GitHub:**
```bash
trelix review --pr owner/repo#142 --post-comments
```

This posts a single batched GitHub review comment with the blast radius analysis and findings. One API call, no rate limit issues.

**Step 3 — Deep-dive on specific changed symbols:**
```bash
trelix ask ./my-repo "explain what AuthProvider.authenticate does and what calls it"
```

---

### 8d. Federated multi-repo search

**Scenario:** Your organization has a microservices architecture across 6 repositories. You want to find which service handles user subscription management, without knowing which repo it lives in.

**Step 1 — Index all repos:**
```bash
trelix index ./service-a
trelix index ./service-b
trelix index ./service-c
# ... repeat for all repos
```

**Step 2 — Register a multi-repo federation:**

Create a `trelix.registry.json` in your workspace root:
```json
{
  "repos": [
    {"path": "./service-a", "name": "auth-service"},
    {"path": "./service-b", "name": "billing-service"},
    {"path": "./service-c", "name": "user-service"},
    {"path": "./service-d", "name": "notification-service"},
    {"path": "./service-e", "name": "api-gateway"},
    {"path": "./service-f", "name": "analytics-service"}
  ]
}
```

**Step 3 — Query across all repos simultaneously:**
```bash
trelix search . "subscription management" --federated
trelix ask . "which service handles user subscription upgrades and downgrades?" --federated
```

The `FederatedRetriever` fans out the query to all registered repos simultaneously (parallel), then merges and re-ranks the results. A TTL cache (120 seconds by default) ensures that repeated queries in the same debugging session return instantly without re-querying all repos.

**Cache stats during a debugging session:**
```python
from trelix.retrieval.federated import FederatedRetriever
retriever = FederatedRetriever.from_registry("trelix.registry.json")
print(retriever.cache_stats())
# {'hits': 47, 'misses': 6, 'size': 6}  ← ~88% cache hit rate
```

**Watch all repos simultaneously for live reindexing:**
```bash
trelix watch-all
```

Uses `watchfiles.awatch()` to monitor all registered repos in a single OS-level watch call. When any file changes, only the affected repo's index is updated. Includes an MD5 hash guard to skip no-op saves (common with IDEs that auto-save on focus change).

```
trelix watch-all — monitoring 6 repos

  [auth-service]        watching ./service-a  (1,204 files)
  [billing-service]     watching ./service-b  (892 files)
  [user-service]        watching ./service-c  (1,441 files)
  [notification-service] watching ./service-d  (203 files)
  [api-gateway]         watching ./service-e  (654 files)
  [analytics-service]   watching ./service-f  (387 files)

Ctrl+C to stop

14:23:11  MODIFIED  ./service-b/src/billing/subscriptions.py
14:23:11  → re-indexed 3 symbols in billing-service  (0.8s)
14:25:44  MODIFIED  ./service-c/src/users/service.py
14:25:44  → re-indexed 7 symbols in user-service  (1.1s)
```

---

### 8e. Watch mode for live reindexing

**Scenario:** You are actively developing in a codebase and want trelix's search results to always reflect your latest code changes, without running `trelix index` after every save.

**Step 1 — Start watch mode:**
```bash
trelix watch ./my-repo
```

> **v2.5.0 — DimensionGuard at startup.** `trelix watch` now checks for embedding dimension mismatches before starting the observer. If the embedding provider was changed since the last `trelix index` run, watch fails immediately with a `DimensionMismatchError` rather than silently re-embedding changed files with wrong dimensions and corrupting the index. The error message includes an exact migration hint: `trelix migrate-vectors --reset`. This is a safe no-op when the index has never been built.

On startup, trelix runs a full index. Then it starts a watchdog observer that monitors every file in the repo.

```
trelix watch — ./my-repo

Full index complete: 4,891 symbols in 24.1s
Watching for changes... (Ctrl+C to stop)

14:31:05  MODIFIED  src/auth/jwt.py
14:31:05  → re-indexed: JWTValidator.validate, JWTValidator._decode  (0.3s)

14:31:42  CREATED   src/auth/refresh.py
14:31:42  → indexed: refresh_token, revoke_token  (0.2s)

14:32:15  DELETED   src/auth/old_tokens.py
14:32:15  → removed 4 symbols from index
```

**Key behaviors:**
- **Debounce:** 500ms debounce prevents index cascades when your IDE auto-formats a file on save.
- **Incremental:** Only the changed file is re-processed; the rest of the index is untouched.
- **Graph sync:** When `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED=true` and the graph has been built, `trelix watch` also patches the Code Property Graph on every file change. You do not need to re-run `trelix graph` manually.
- **Delete handling:** When a file is deleted, trelix removes its symbols, chunks, vectors, and call edges from the index atomically.

**Step 2 — Query while watching:**

Open a second terminal and run queries normally. The watch process keeps the index current in the background.

---

## 9. MCP Integration

Model Context Protocol (MCP) allows AI coding assistants — Claude Code, Cursor, Windsurf, Continue.dev — to call trelix as a tool directly from the assistant's context. Instead of copying code into chat, the assistant queries trelix in the background and receives grounded, codebase-specific context.

### Installation

```bash
pip install trelix-mcp
```

### Claude Code

```bash
claude mcp add trelix -- trelix-mcp
```

That is the entire setup. trelix-mcp launches as a stdio MCP server. Claude Code can now use trelix tools in any conversation.

### Cursor

Add to your Cursor MCP configuration file (`~/.cursor/mcp.json` or `.cursor/mcp.json` in your project):

```json
{
  "mcpServers": {
    "trelix": {
      "command": "trelix-mcp",
      "args": [],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Restart Cursor. trelix tools appear in the Cursor tool panel.

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "trelix": {
      "command": "trelix-mcp",
      "args": [],
      "transport": "stdio"
    }
  }
}
```

### Continue.dev

Add to your Continue config file:

```json
{
  "models": [...],
  "tools": [
    {
      "type": "mcp",
      "transport": {
        "type": "stdio",
        "command": "trelix-mcp"
      }
    }
  ]
}
```

### Available MCP tools

Once connected, your AI assistant has access to six trelix tools:

**`search_code`** — Hybrid code search with pagination (v2.4.0).
```
search_code(query="JWT validation", repo_path="./my-repo", k=10, cursor=0)

Returns:
{
  "results": [
    {"score": 0.924, "file": "src/auth/jwt.py", "symbol": "JWTValidator.validate", ...}
  ],
  "next_cursor": 10,   ← null on last page
  "total_available": 23
}
```

To fetch the next page: `search_code(cursor=10)`. Paginate until `next_cursor` is `null`.

**BREAKING CHANGE (v2.4.0):** If you upgraded from v2.3.x, replace `for item in response` with `for item in response["results"]`.

**`index_codebase`** — Trigger indexing programmatically, with progress notifications.
```
index_codebase(repo_path="./my-repo", provider="local")
```
During indexing, the MCP server emits `ctx.report_progress()` notifications that your assistant's UI can display as a progress indicator.

**`get_symbol`** — Look up a specific symbol by qualified name.
```
get_symbol(qualified_name="src.auth.jwt.JWTValidator.validate", repo_path="./my-repo")
```
Returns full symbol metadata including body, docstring, line range, and call edges.

**`blast_radius`** — Find everything that depends on a symbol.
```
blast_radius(symbol_name="UserRepository.get_by_email", repo_path="./my-repo")
```
Returns a list of dicts describing all callers and importers of the named symbol, ordered by dependency depth.

**`build_knowledge_graph`** — Build the Code Property Graph.
```
build_knowledge_graph(
  repo_path="./my-repo",
  detect_communities=true,
  extract_concepts=false
)
```
Returns stats: `{node_count, edge_count, community_count, build_time_ms}`.

**`graph_search_mcp`** — BFS graph search from a seed query.
```
graph_search_mcp(query="auth middleware", repo_path="./my-repo", depth=2, max_results=15)
```
Returns structurally adjacent symbols to the query's top results, traversed via the Code Property Graph.

> **v2.5.0 — MCP resource subscriptions.** trelix-mcp now advertises `resources.subscribe=True` in server capabilities and exposes two new tools:
>
> - **`subscribe_resource(uri, subscription_id)`** — Subscribe to a `trelix://` resource URI. When the underlying file changes and the index is updated, the server fires a `notifications/resources/updated` notification (URI + `subscriptionId` in `_meta`).
> - **`unsubscribe_resource(subscription_id)`** — Cancel an active subscription by ID.
>
> URI scheme: `trelix://repo/{repo_path}/manifest`. The wire protocol is: `resources/subscribe` → `notifications/resources/updated` → `resources/read`. This is the standard MCP subscription flow and allows assistants to react to live code changes without polling.

### Example Claude Code session

With trelix-mcp active, you can ask Claude Code questions grounded in your actual codebase:

```
You: "Explain how authentication works in this repo"

Claude Code: [calls search_code(query="authentication flow", repo_path=".")] 
             [calls get_symbol(qualified_name="src.auth.middleware.authenticate_request")]
             [calls blast_radius(symbol_name="authenticate_request")]

Claude Code: "Based on trelix's analysis of your codebase, here's how authentication works:
             The entry point is authenticate_request() in src/auth/middleware.py:42..."
```

The key difference from asking Claude Code without trelix: every file reference, line number, and function name in the answer is verified against your actual code, not generated from training data.

---

## 10. Embedding Providers

Choosing the right embedding provider is the single highest-leverage configuration decision you can make. It affects retrieval quality, speed, cost, and whether you need internet access.

### Provider comparison table

| Provider | Model | Dimensions | CoIR Score | Cost | Needs API key | Best for |
|----------|-------|-----------|-----------|------|--------------|---------|
| `local` | all-MiniLM-L6-v2 | 384 | baseline | free | no | Getting started, offline |
| `openai` | text-embedding-3-large | 3072 | ~45 | ~$0.13/1M tokens | yes | General-purpose, high quality |
| `azure` | text-embedding-3-large | 3072 | ~45 | ~$0.13/1M tokens | yes | Enterprise, existing Azure setup |
| `voyage` | voyage-code-3 (Matryoshka) | 256–2048 | **56.26** | ~$0.06/1M tokens | yes | Best semantic quality for code |
| `local-code` | SFR-Embedding-Code-2B_R | 4096 | **67.41** | free | no | Best offline, needs 8GB RAM |
| `bge-code` | BAAI/bge-code-v1 | 1536 | **63.10** | free | no | SOTA 2025, balanced size/quality |
| `nomic-code` | nomic-embed-code | 768 | **58.40** | free | no | No API key, good quality |
| `bedrock-titan` | amazon.titan-embed-text-v2:0 | 256–1024 | — | AWS pricing | AWS creds | AWS-native deployments |
| `bedrock-cohere` | cohere.embed-english-v3 | 1024 | — | AWS pricing | AWS creds | AWS + best asymmetric retrieval |

CoIR scores from the Code Information Retrieval benchmark (ACL 2025). Higher is better. Baseline (`local`) score is ~30.

### When to use which provider

**Starting out / evaluating trelix:**
```bash
pip install "trelix[local]"
TRELIX_EMBEDDER_PROVIDER=local trelix index ./my-repo
```
No API key, no cost, fast. Quality is adequate for basic navigation and learning how trelix works. Do not use in production for semantic search quality.

**Production deployment, best-in-class quality:**
```bash
pip install "trelix[bge-code]"
TRELIX_EMBEDDER_PROVIDER=bge-code trelix index ./my-repo
```
BGE-Code-v1 is the 2025 CoIR leader. It runs fully offline after the first model download (~3GB), needs no API key, and outperforms all API providers on code retrieval benchmarks. If you have a GPU, it is fast. On CPU it is slower (plan for 30–60 seconds for a 10k-chunk index, vs. 3 seconds with local).

**Best API-based quality for code:**
```bash
pip install "trelix[voyage]"
export VOYAGE_API_KEY=va-...
TRELIX_EMBEDDER_PROVIDER=voyage trelix index ./my-repo
```
Voyage AI's `voyage-code-3` is the best API-based code embedding model. The Matryoshka architecture means you can choose your dimension tradeoff:
```bash
TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512  # 2× faster HNSW, minimal quality loss
TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=1024 # full quality (default)
```

**AWS-native enterprise deployment:**
```bash
pip install "trelix[bedrock]"
# Uses existing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere trelix index ./my-repo
```
Bedrock-Cohere uses asymmetric retrieval: `search_document` at index time, `search_query` at query time. This is the correct setup for production retrieval systems and gives better precision than symmetric embedders.

### Matryoshka dimensions (Voyage)

Voyage AI's `voyage-code-3` uses Matryoshka Representation Learning: the model trains a single embedding that can be truncated to any dimension without retraining. This means you can trade off quality vs. speed:

```bash
# Full quality: 1024 dims, slower HNSW
TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=1024

# 2× faster HNSW, ~2% quality loss: 512 dims
TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512

# 4× faster HNSW, ~5% quality loss: 256 dims (very large repos only)
TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=256
```

For repos under 100k chunks, stay at 1024. For larger repos where HNSW latency becomes noticeable, try 512.

### Changing providers (requires re-index)

If you change the embedding provider after indexing, you **must** re-index:

```bash
TRELIX_EMBEDDER_PROVIDER=bge-code trelix index ./my-repo  # re-runs all 4 phases
```

Vector dimensions differ between providers (384 for local vs. 1536 for bge-code). The new index overwrites the old one. The SQLite schema is compatible; only the vector table needs to be rebuilt.

---

## 11. Understanding Query Telemetry

Query telemetry gives you a per-query performance and quality breakdown. It is opt-in and has zero overhead when disabled.

### Enabling telemetry

```bash
TRELIX_TELEMETRY_ENABLED=true trelix ask ./my-repo "how does auth work?"
```

Or add to your `.env` file:
```bash
echo "TRELIX_TELEMETRY_ENABLED=true" >> .env
```

Telemetry rows are written to `./my-repo/.trelix/index.db` in the `query_telemetry` table. They persist across sessions.

### Viewing telemetry

```bash
trelix telemetry ./my-repo             # last 20 queries
trelix telemetry ./my-repo --limit 100 # last 100 queries
trelix telemetry ./my-repo --json      # JSON output for dashboards
```

Example output:

```
trelix telemetry — ./my-repo — last 10 queries

 #   Query                                 Legs used          Latency   Results  RRF p50  Reranker
 ─── ──────────────────────────────────── ─────────────────── ───────── ──────── ──────── ────────
 1   how does authentication work?         vector+bm25+graph   243ms     12       0.71     0.88
 2   JWT validation                        grep+bm25+vector    41ms      8        0.84     0.92
 3   UserRepository.get_by_email callers   grep+bm25           28ms      5        0.91     —
 4   walk me through the request lifecycle vector+bm25+graph   891ms     22       0.63     0.81
 5   what does the billing module do?      file_summary+vector 312ms     9        0.76     0.84
```

### What each telemetry field tells you

**`legs_used`** — The JSON array of retrieval legs that contributed results to this query.

*What to look for:* If a query consistently uses only one leg (e.g., `vector` only), check that your BM25 and graph legs are enabled and populated. A query hitting all three primary legs typically produces better results.

**`rrf_score_p50`** — The median RRF score across all fused results, before reranking.

*What to look for:* A high p50 (>0.75) means most results are strongly relevant. A low p50 (<0.50) means the query returned many marginally relevant results — the retrieval pool had weak alignment with the query. Consider rephrasing or enabling more legs.

**`reranker_top1_score`** — The score assigned to the top result by the reranker.

*What to look for:* A top-1 score above 0.85 means the reranker is confident about the best result. Below 0.70 suggests ambiguity — multiple results have similar relevance. Enable multi-query expansion or FLARE re-retrieval for ambiguous queries.

**`latency_ms`** — Wall-clock time from query start to results returned (not including synthesis).

*Typical ranges:*
- `grep+bm25`: 10–50ms
- `vector`: 30–100ms (HNSW query)
- `vector+bm25+grep` (RRF): 40–150ms
- `+graph` (BFS expansion): add 50–200ms depending on graph size
- `+reranker` (Cohere): add 300–800ms (network round trip)
- `+reranker` (PLAID): add 50–200ms (local inference)

**`flare_iterations`** — Number of times FLARE triggered a re-retrieval loop.

*What to look for:* 0 = no uncertainty. 1–2 = synthesis was uncertain on some spans and re-retrieved. >2 = the query may be too broad or the codebase coverage is sparse. If FLARE is consistently triggering 2+ times, consider re-indexing with contextual chunking enabled.

### Multi-query expansion telemetry (v2.4.0)

When multi-query expansion is enabled (`TRELIX_RETRIEVAL_MULTI_QUERY=true`), three additional telemetry columns are populated:

| Column | Type | Meaning |
|--------|------|---------|
| `expansion_used` | BOOLEAN | Whether expansion ran for this query |
| `expansion_variants` | INTEGER | How many variant queries were generated |
| `expansion_elapsed_ms` | REAL | LLM call duration for expansion |

Use this to tune the cost/quality tradeoff: if expansion_elapsed_ms is high and expansion_used is true on 90% of queries, consider pre-filtering to only expand complex queries.

---

## 12. Troubleshooting Common Issues

### sqlite-vec not loading on macOS

**Symptom:**
```
ImportError: sqlite-vec requires SQLite ≥ 3.45 with loadable extensions
```

**Cause:** macOS ships with an old system SQLite (typically 3.37) that disables loadable extensions for security reasons. trelix's vector search requires sqlite-vec, which needs the loadable extension feature.

**Fix:**
```bash
# Install current SQLite via Homebrew
brew install sqlite

# Reinstall trelix against the Homebrew SQLite
LDFLAGS="-L/opt/homebrew/opt/sqlite/lib" \
CPPFLAGS="-I/opt/homebrew/opt/sqlite/include" \
pip install --force-reinstall "trelix[local]"
```

**Verify:**
```bash
python -c "import sqlite_vec; print('sqlite-vec loaded successfully')"
```

---

### Empty or low-quality results

**Symptom:** `trelix search` returns few results or results with score < 0.5.

**Possible causes and fixes:**

*Index is stale* — your code changed since last index:
```bash
trelix index ./my-repo  # re-index
```

*Provider mismatch* — you indexed with one provider and are querying with another:
```bash
# Check which provider was used at index time
trelix stats ./my-repo | grep "provider"

# Re-index with the correct provider
TRELIX_EMBEDDER_PROVIDER=openai trelix index ./my-repo
```

*Query is too abstract for the embedding model* — try `trelix ask` instead of `trelix search` for high-level conceptual questions. Or enable file summaries:
```bash
TRELIX_FILE_SUMMARIES_ENABLED=true trelix index ./my-repo
TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=true trelix ask ./my-repo "your abstract question"
```

*Repo uses a language not yet supported:* Check with `trelix stats ./my-repo` — unsupported file types are skipped during indexing. See the supported languages list.

---

### Bedrock ValidationException on model inference

**Symptom:**
```
ValidationException: Invocation of model ID anthropic.claude-sonnet-4-6 with on-demand throughput isn't supported
```

**Cause:** AWS Bedrock requires inference profile IDs (with regional `us.*` prefixes), not bare model IDs.

**Fix:**
```bash
TRELIX_LLM_BEDROCK_PRIMARY_MODEL=us.anthropic.claude-sonnet-4-6
TRELIX_LLM_BEDROCK_FALLBACK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

trelix automatically falls back to the fallback model on ValidationException — the fallback is transparent to the caller.

---

### Bedrock Cohere embeddings fail on large chunks

**Symptom:**
```
ValidationException: expected maxLength: 2048
```

**Cause:** Bedrock's Cohere endpoint rejects input texts longer than 2,048 characters before truncation occurs at the API level. This is a Bedrock-specific limitation, not a Cohere API limitation.

**Fix:** Upgrade to trelix v0.7.1 or later — pre-truncation was added in that release.
```bash
pip install --upgrade "trelix[bedrock]"
```

---

### tree-sitter FutureWarning spam

**Symptom:** Terminal flooded with messages like `FutureWarning: TreeSitter.Language` during indexing.

**Cause:** tree-sitter 0.21.x changed its Language API and the old API emits deprecation warnings. Harmless, but noisy.

**Fix:**
```bash
PYTHONWARNINGS=ignore::FutureWarning trelix index ./my-repo
```

Or add to your `.env`:
```bash
PYTHONWARNINGS=ignore::FutureWarning
```

---

### HuggingFace token warning

**Symptom:**
```
huggingface_hub.utils._headers: UserWarning: 'HF_TOKEN' is not set
```

**Cause:** sentence-transformers checks for a HuggingFace API token on startup. The local embedding models (MiniLM, BGE-Code, Nomic) are cached locally after first download and do not need authentication.

**Fix:** Harmless, but suppress with:
```bash
HF_HUB_DISABLE_SYMLINKS_WARNING=1 trelix index ./my-repo
```

Or for the first download (to avoid the warning entirely):
```bash
export HF_TOKEN=hf_...  # optional — only needed for private models
```

---

### MCP search_code returns wrong type after v2.4.0 upgrade

**Symptom:** Code that worked with v2.3.x breaks with `TypeError: 'dict' object is not iterable` or `KeyError: 'score'`.

**Cause:** v2.4.0 is a breaking change for `search_code`. It now returns a pagination envelope `{"results": [...], "next_cursor": int|null, "total_available": int}` instead of a bare list.

**Fix:**
```python
# Before (v2.3.x and earlier):
results = search_code(query, repo_path)
for item in results:
    print(item["score"])

# After (v2.4.0+):
response = search_code(query, repo_path)
for item in response["results"]:
    print(item["score"])

# To paginate:
cursor = 0
while cursor is not None:
    response = search_code(query, repo_path, cursor=cursor)
    process(response["results"])
    cursor = response["next_cursor"]
```

---

### Graph build is slow

**Symptom:** `trelix graph ./my-repo` takes several minutes on a large repo.

**Expected timings:**
- `local` codebase (4,599 nodes, 4,945 edges): **0.34 seconds**
- Medium repo (20k nodes): **2–5 seconds**
- Large repo (100k nodes): **20–60 seconds**

**If significantly slower:**
```bash
# Use label_prop algorithm (O(n) vs O(n log n) for Louvain)
TRELIX_GRAPH_COMMUNITY_ALGORITHM=label_prop trelix graph ./my-repo

# Skip concept extraction (adds one LLM call per batch)
trelix graph ./my-repo  # omit --concepts flag

# Build graph in background while continuing to work
trelix graph ./my-repo &
```

---

### `trelix ask` synthesis is slow or times out

**Symptom:** `trelix ask` takes more than 30 seconds or times out.

**Cause:** GraphRAG map-reduce is activated when results > 20 or total context > 8,000 tokens. Each map step is a separate LLM call.

**Fixes:**
```bash
# Reduce context budget (default: 12000 tokens)
TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=4000 trelix ask ./my-repo "question"

# Reduce top-k results
trelix ask ./my-repo "question" --top-k 5

# Use a faster/cheaper LLM for synthesis
TRELIX_LLM_MODEL=gpt-4o-mini trelix ask ./my-repo "question"

# If using Anthropic, switch to Haiku for fast answers
TRELIX_LLM_PROVIDER=anthropic TRELIX_LLM_MODEL=claude-haiku-4-5 trelix ask ./my-repo "question"
```

---

## 13. CLI Flags and Configuration Reference

### Global flags (apply to all commands)

```bash
trelix <command> ./repo --help          # full help for any command
trelix <command> ./repo --provider local|openai|azure|voyage|bge-code|nomic-code
trelix <command> ./repo --top-k N       # return N results (default: 10)
trelix <command> ./repo --json          # output as JSON
trelix <command> ./repo --verbose       # verbose logging
```

### `trelix index` flags

```bash
trelix index ./repo                     # index entire repo
trelix index ./repo --provider voyage   # use specific embedding provider
trelix index ./repo --workers 8         # parallel parse workers (default: 4)
trelix index ./repo --no-call-graph     # skip call graph extraction (faster)
trelix index ./repo --force             # re-index all files (ignore SHA-256 cache)
trelix index ./repo --include "*.py"    # index only Python files
trelix index ./repo --exclude "tests/"  # skip tests directory
```

### `trelix search` flags

```bash
trelix search ./repo "query"
  --top-k N          # number of results (default: 10)
  --lang python       # filter by language
  --file src/auth/    # filter by file path prefix
  --rerank cohere     # apply Cohere reranker
  --rerank plaid      # apply PLAID ColBERT reranker
  --json              # JSON output
  --no-graph          # skip graph expansion
  --legs vector,bm25  # use only specified legs (comma-separated)
```

### `trelix ask` flags

```bash
trelix ask ./repo "question"
  --top-k N           # number of retrieval results (default: 10)
  --stream            # force streaming output (default: true)
  --no-stream         # buffer complete answer, print at end
  --provider bedrock  # override LLM provider for this query
  --model gpt-4o-mini # override LLM model for this query
  --json              # return JSON with answer + sources
```

### `trelix graph` flags

```bash
trelix graph ./repo
  --visualize         # export Pyvis HTML to .trelix/graph.html
  --json              # machine-readable stats to stdout
  --concepts          # run LLM concept extraction (adds latency)
  --algorithm louvain|girvan_newman|label_prop  # community detection algorithm
```

### `trelix review` flags

```bash
trelix review --pr owner/repo#N
  --post-comments     # post findings as GitHub review comment
  --token TOKEN       # GitHub token (overrides GITHUB_TOKEN env var)
  --json              # JSON output instead of terminal table
  --top-k N           # blast radius depth (default: 10)
```

### `trelix serve` flags

```bash
trelix serve ./repo
  --port 8765         # HTTP port (default: 8765)
  --host 0.0.0.0      # bind address (default: 127.0.0.1)
  --workers 4         # uvicorn workers
  --reload            # auto-reload on code changes (dev mode)
```

### `trelix watch` flags

```bash
trelix watch ./repo
  --provider openai   # embedding provider for incremental updates
  --debounce 500      # debounce ms (default: 500)
  --graph             # also maintain Code Property Graph in watch mode
```

### `trelix telemetry` flags

```bash
trelix telemetry ./repo
  --limit N           # number of rows (default: 20)
  --json              # JSON output
  --since "2026-07-01" # filter by date
  --query-filter "auth" # filter by query text substring
```

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Embedding provider |
| `TRELIX_LLM_PROVIDER` | `openai` | LLM provider for synthesis |
| `TRELIX_LLM_MODEL` | `gpt-4o` | LLM model override |
| `TRELIX_STORE_BACKEND` | `sqlite` | Vector store: `sqlite`, `qdrant`, `lance` |
| `TRELIX_PARSE_WORKERS` | `4` | Parallel parse threads |
| `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` | `12000` | Max context tokens to LLM |
| `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED` | `false` | Enable CodeGraph BFS leg |
| `TRELIX_GRAPH_SEARCH_DEPTH` | `2` | BFS depth from seed nodes |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG` | `false` | Enable file-summary retrieval leg |
| `TRELIX_FILE_SUMMARIES_ENABLED` | `false` | Generate LLM summaries at index time |
| `TRELIX_RETRIEVAL_HYDE_FALLBACK` | `false` | Enable HyDE query expansion |
| `TRELIX_RETRIEVAL_FLARE` | `false` | Enable FLARE confidence-gated re-retrieval |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `2` | Max FLARE re-retrieval loops |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST` | `false` | Enable PageRank centrality boost |
| `TRELIX_RETRIEVAL_SPARSE` | `false` | Enable SPLADE-Code sparse leg |
| `TRELIX_CHUNKER_MULTI_GRANULARITY` | `false` | Enable sub-chunk indexing |
| `TRELIX_RETRIEVAL_AGENTIC` | `false` | Enable ReAct multi-turn agentic loop |
| `TRELIX_PARSER_DATAFLOW` | `false` | Enable def-use data-flow analysis |
| `TRELIX_TELEMETRY_ENABLED` | `false` | Enable query telemetry |
| `TRELIX_CHUNKER_CONTEXTUAL` | `false` | Enable LLM context summaries per chunk |
| `TRELIX_RETRIEVAL_RERANK_PROVIDER` | — | Reranker: `cohere`, `cross-encoder`, `plaid` |

---

## 14. API Quick Reference

trelix exposes a REST API when running as a server. Start it with:

```bash
pip install "trelix[serve]"
trelix serve ./my-repo --port 8765
```

All endpoints accept and return JSON. The server binds to `http://127.0.0.1:8765` by default.

### Health check

```bash
curl http://localhost:8765/health
```

```json
{"status": "ok", "version": "2.7.1", "repo": "./my-repo"}
```

### Index statistics

```bash
curl http://localhost:8765/stats
```

```json
{
  "files": 243,
  "symbols": 4891,
  "chunks": 9241,
  "vectors": 9241,
  "call_edges": 8204,
  "import_edges": 1024,
  "type_edges": 312,
  "index_size_mb": 48.3,
  "last_indexed": "2026-07-05T10:32:14Z",
  "embed_provider": "local",
  "embed_dims": 384
}
```

### Hybrid code search

```bash
curl -s -X POST http://localhost:8765/search \
  -H "Content-Type: application/json" \
  -d '{"query": "JWT validation", "repo_path": "./my-repo", "k": 10}' \
  | jq .
```

```json
{
  "query": "JWT validation",
  "results": [
    {
      "score": 0.924,
      "file": "src/auth/jwt.py",
      "symbol": "JWTValidator.validate",
      "qualified_name": "src.auth.jwt.JWTValidator.validate",
      "line_start": 18,
      "line_end": 44,
      "language": "python",
      "source": "vector+bm25",
      "body": "def validate(self, token: str) -> dict:\n    \"\"\"Validate JWT token signature and claims.\"\"\"\n    ..."
    }
  ],
  "total": 10,
  "latency_ms": 43.2,
  "legs_used": ["vector", "bm25", "grep"]
}
```

**Request body fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Search query |
| `repo_path` | string | yes | Path to indexed repo |
| `k` | integer | no | Number of results (default: 10) |
| `lang` | string | no | Filter by language |
| `file_prefix` | string | no | Filter by file path prefix |
| `rerank` | string | no | `cohere`, `cross-encoder`, or `plaid` |

### Streaming ask (SSE)

```bash
curl -s -N -X POST http://localhost:8765/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "how does auth work?", "repo_path": "./my-repo"}'
```

The response is a Server-Sent Events stream. Each `data:` event is one token. The final event is `data: [DONE]`.

```
data: The
data:  authentication
data:  middleware
data:  (src/auth/middleware.py)
data:  intercepts
...
data: [DONE]
```

**In JavaScript (browser or Node.js):**

```javascript
const response = await fetch('http://localhost:8765/ask', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: "how does auth work?", repo_path: "./my-repo"})
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  // Parse SSE events: each line starting with "data: " is a token
  for (const line of chunk.split('\n')) {
    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
      process.stdout.write(line.slice(6));
    }
  }
}
```

### Trigger re-indexing

```bash
curl -s -X POST http://localhost:8765/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "./my-repo", "provider": "openai"}' \
  | jq .
```

```json
{
  "status": "indexed",
  "files": 243,
  "symbols": 4891,
  "chunks": 9241,
  "duration_seconds": 24.1
}
```

### Knowledge graph endpoints

**Get graph stats:**
```bash
curl "http://localhost:8765/graph?repo=./my-repo" | jq .
```

```json
{
  "node_count": 4599,
  "edge_count": 4945,
  "community_count": 47,
  "top_nodes": [
    {"name": "parse", "degree": 438},
    {"name": "index_file", "degree": 211},
    {"name": "retrieve", "degree": 189}
  ]
}
```

**Get community summary:**
```bash
curl "http://localhost:8765/graph/communities?repo=./my-repo" | jq .
```

```json
[
  {"community_id": 0, "size": 234, "sample_symbols": ["authenticate_request", "JWTValidator", "AuthMiddleware"]},
  {"community_id": 1, "size": 189, "sample_symbols": ["UserRepository", "UserService", "create_user"]},
  {"community_id": 2, "size": 156, "sample_symbols": ["process_payment", "PaymentGateway", "charge_card"]}
]
```

**BFS graph search:**
```bash
curl "http://localhost:8765/graph/search?repo=./my-repo&query=auth+middleware&depth=2" | jq .
```

```json
[
  {
    "symbol": "authenticate_request",
    "file": "src/auth/middleware.py",
    "hop": 0,
    "score": 1.0,
    "edge_type": "seed"
  },
  {
    "symbol": "JWTValidator.validate",
    "file": "src/auth/jwt.py",
    "hop": 1,
    "score": 0.5,
    "edge_type": "CALLS"
  },
  {
    "symbol": "UserRepository.get_by_id",
    "file": "src/users/repository.py",
    "hop": 2,
    "score": 0.25,
    "edge_type": "CALLS"
  }
]
```

**Export visualization:**
```bash
curl "http://localhost:8765/graph/visualize?repo=./my-repo" -o graph.html
open graph.html  # opens in browser: interactive Pyvis visualization
```

### LangChain integration

```python
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(
    repo_path="./my-repo",
    k=10,
    provider="openai"  # embedding provider
)

# Returns list[Document] with page_content = chunk body, metadata = {file, symbol, score}
docs = retriever.invoke("how does authentication work?")

for doc in docs:
    print(f"{doc.metadata['file']}:{doc.metadata['line_start']} — {doc.metadata['symbol']}")
    print(doc.page_content[:200])
    print()
```

### LlamaIndex integration

```python
from trelix_llama_index import TrelixIndexRetriever
from llama_index.core import QueryBundle

retriever = TrelixIndexRetriever(
    repo_path="./my-repo",
    similarity_top_k=10
)

# Returns list[NodeWithScore]
nodes = retriever.retrieve(QueryBundle("JWT validation"))

for node in nodes:
    print(f"Score: {node.score:.3f}  File: {node.node.metadata['file']}")
    print(node.node.text[:200])
```

### GitHub Actions CI integration

Add trelix indexing to your CI pipeline so every PR has a fresh, searchable index:

```yaml
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Index codebase with trelix
        uses: sairam0424/trelix-index-action@v1
        with:
          provider: local        # no API key needed for CI
          cache-key: ${{ github.sha }}
      
      - name: Run tests
        run: pytest
        
      # The index is available at .trelix/index.db for downstream steps
      - name: Upload index artifact
        uses: actions/upload-artifact@v4
        with:
          name: trelix-index
          path: .trelix/index.db
```

The action handles Python setup, caches the index keyed to the commit SHA (so unchanged code reuses the cached index), and exposes the index path as an output for downstream steps.

---

## 15. Synthesis Quality Evaluation (GroUSE Harness)

trelix ships a GroUSE-inspired synthesis quality harness that measures how well the full retrieve+synthesize pipeline answers questions. Use it to catch generator failure modes that nDCG and MRR miss.

### Why a separate synthesis harness?

The existing `trelix eval` command measures *retrieval* quality (nDCG@10, Recall@10, MRR). Retrieval quality tells you whether the right files were retrieved, but not whether the LLM synthesized a correct, faithful, complete answer from those files.

GroUSE (arXiv:2409.06595, COLING 2025) identifies 7 generator failure modes that GPT-4 judges systematically miss and standard NL benchmarks do not cover. This harness adds code-specific extensions for symbol hallucination and stale line references.

### Metrics

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `hallucination_rate` | Fraction of expected symbols mentioned without retrieval support | Lower is better |
| `completeness` | Fraction of expected text fragments present in the answer | Higher is better |
| `faithfulness` | Fraction of answer tokens that appear in retrieved context (lexical heuristic) | Higher is better |
| `overall` | Weighted composite: `(1-hallucination)*0.4 + completeness*0.4 + faithfulness*0.2` | Higher is better |

### Golden file format

Create a `golden_synthesis.jsonl` file with one JSON object per line:

```json
{"query": "how does JWT authentication work", "relevant_files": ["src/auth/middleware.py"], "expected_answer_fragments": ["jwt", "decode", "bearer"], "expected_symbols": ["AuthMiddleware.verify"]}
{"query": "what does the database connection pool do", "relevant_files": ["src/db/pool.py"], "expected_answer_fragments": ["pool", "connection", "max"], "expected_symbols": []}
{"query": "explain the retry logic", "relevant_files": ["src/utils/retry.py"], "expected_answer_fragments": ["retry", "backoff", "attempt"], "expected_symbols": ["retry_with_backoff"]}
```

A sample file is provided at `eval/golden_synthesis_sample.jsonl`. Copy and adapt it for your codebase.

**Fields:**
- `query` (required) — the question to ask
- `relevant_files` (optional) — files that should be retrieved (not currently used by the harness scorer, but useful for documentation)
- `expected_answer_fragments` (optional) — strings that MUST appear in a correct answer (case-insensitive)
- `expected_symbols` (optional) — qualified names that should appear WITHOUT being hallucinated

### Running the evaluation

```bash
# Run synthesis eval against a golden file
trelix eval-synthesis ./my-repo --golden ./eval/golden_synthesis.jsonl
```

Output:
```
                     Synthesis Quality Results (GroUSE-style)
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Metric              ┃  Score ┃ Direction         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Hallucination rate  │ 0.0000 │ lower = better    │
│ Completeness        │ 0.8333 │ higher = better   │
│ Faithfulness        │ 0.7250 │ higher = better   │
│ Overall             │ 0.8783 │ higher = better   │
│ Queries evaluated   │      3 │                   │
└─────────────────────┴────────┴───────────────────┘
```

### Python API

```python
from trelix.eval.synthesis import evaluate_synthesis, SynthesisEvalHarness
from trelix.core.config import IndexConfig

# Score a single answer
result = evaluate_synthesis(
    query="how does JWT authentication work",
    answer="The AuthMiddleware.verify method decodes the jwt bearer token...",
    retrieved_context="def verify(token): return jwt.decode(token, SECRET)",
    retrieved_symbols=["AuthMiddleware.verify"],
    expected_symbols=["AuthMiddleware.verify"],
    expected_fragments=["jwt", "decode", "bearer"],
)
print(result.scores)
# {'hallucination': 0.0, 'completeness': 1.0, 'faithfulness': 0.87, 'overall': 0.97}

# Run full harness
config = IndexConfig(repo_path="./my-repo")
harness = SynthesisEvalHarness(config)
metrics = harness.run("./eval/golden_synthesis.jsonl")
print(f"Overall: {metrics['overall']:.3f}")
```

## Phase 1–3 Features (v2.7.0)

- **Watch bridge MCP notifications** — file-system watcher now emits real-time MCP events when the index is updated, enabling editors and agents to subscribe to live re-indexing signals.
- **Cross-repo symbol resolution** — symbol lookups can now resolve definitions that live in a sibling or dependency repository; configure via `cross_repo_paths` in `trelix.config.json`.
- **Streaming indexing** (`TRELIX_INDEXER_STREAMING=1`) — large repositories are indexed incrementally in a streaming pass, reducing peak memory usage and enabling partial results during initial index builds.
- **VS Code extension** (`workspace-vscode/`) — first-party extension adds inline symbol hover, semantic search palette, and a Trelix side-panel directly inside VS Code.
- **GitHub App PR review workflow** — install the Trelix GitHub App to get automated code-intelligence comments (symbol impact analysis, cross-repo call-graph diffs) on every pull request.

---

*trelix v2.7.1 — For changelog, see [CHANGELOG.md](../CHANGELOG.md). For architecture details, see [architecture.md](architecture.md). For contribution guide, see [CONTRIBUTING.md](../CONTRIBUTING.md).*
