# trelix

[![CI](https://github.com/trelix-dev/trelix/actions/workflows/ci.yml/badge.svg)](https://github.com/trelix-dev/trelix/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trelix)](https://pypi.org/project/trelix/)
[![Python](https://img.shields.io/pypi/pyversions/trelix)](https://pypi.org/project/trelix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fast, reliable code indexing and retrieval.** Given a user query and a repository, trelix finds the most relevant code — using an intelligent query planner, hybrid search (semantic + keyword + grep), call-graph expansion, reranking, and LLM synthesis.

```
trelix index  ./my-repo
trelix ask    ./my-repo "how does authentication work?"
trelix search ./my-repo "JWT validation"
trelix stats  ./my-repo
```

---

## Features

- **Tree-sitter parsing** for 20+ languages — functions, classes, methods, call edges, imports
- **Hybrid search** — vector (ANN) + BM25 (FTS5) + grep combined via Reciprocal Rank Fusion
- **Call-graph + import expansion** — PageRank-weighted graph traversal pulls in callers/callees
- **8-intent query planner** — LLM classifies queries into `symbol_lookup`, `blast_radius`, `feature_flow`, etc. for optimal retrieval strategy
- **Reranking** — Cohere or cross-encoder reranker for final precision
- **LLM synthesis** — `trelix ask` assembles context and calls OpenAI/Azure for a direct answer
- **Zero-infra** — single SQLite file (`.trelix/index.db`) with sqlite-vec vectors + FTS5 BM25
- **Works offline** — `--provider local` uses sentence-transformers, no API key needed

---

## Quick Start

```bash
# Install (local embeddings — no API key needed)
pip install "trelix[local]"

# Index a repository
trelix index ./my-repo

# Search for code (returns a Rich table)
trelix search ./my-repo "database connection pooling"

# Ask a question (local mode: prints context; OpenAI mode: calls LLM)
trelix ask ./my-repo "how does the authentication middleware work?"

# Show index statistics
trelix stats ./my-repo

# Re-index a single file after editing
trelix update-index ./my-repo src/auth/middleware.py
```

### End-to-end example

```
$ trelix index ./my-repo --provider local
╭────────────────────────────────╮
│ Indexing ./my-repo             │
╰────────────────────────────────╯
  Phase 1/3: parsing 7 files (2 workers)…
  Phase 2/3: inserting symbols & building chunks…
  Phase 3/3: embedding 42 chunks (8,701 tokens)…
┌──────────────────────┬────────┐
│ Metric               │  Value │
├──────────────────────┼────────┤
│ Files found          │      7 │
│ Files indexed        │      5 │
│ Files skipped        │      2 │
│ Symbols extracted    │     26 │
│ Chunks embedded      │     42 │
│ Elapsed              │   18.2s│
└──────────────────────┴────────┘

$ trelix search ./my-repo "authentication"
┌──────────────┬──────────────────────────┬───────┬────────┐
│ File         │ Symbol                   │ Lines │  Score │
├──────────────┼──────────────────────────┼───────┼────────┤
│ auth.py      │ AuthService              │  1-67 │ 0.8812 │
│ auth.py      │ AuthService.login        │ 15-31 │ 0.8134 │
│ auth.py      │ AuthService.validate_... │ 43-52 │ 0.7901 │
│ main.py      │ run_auth_flow            │ 38-52 │ 0.7245 │
│ api.py       │ login_route              │ 34-49 │ 0.6883 │
└──────────────┴──────────────────────────┴───────┴────────┘

$ trelix stats ./my-repo
╭──────────────────────────────────╮
│ Index Stats: ./my-repo           │
╰──────────────────────────────────╯
┌─────────────────┬──────────┐
│ Metric          │    Value │
├─────────────────┼──────────┤
│ Files indexed   │        5 │
│ Symbols         │       26 │
│ Chunks          │       42 │
│ DB size         │  516.0 KB│
└─────────────────┴──────────┘
```

---

## Installation

```bash
# Minimal — local embeddings only (no API key)
pip install "trelix[local]"

# With OpenAI embeddings + query planner + synthesis
pip install trelix
export OPENAI_API_KEY=sk-...

# With best-quality Cohere reranker
pip install "trelix[rerank]"
export COHERE_API_KEY=...

# Everything
pip install "trelix[all]"
```

---

## Configuration

All settings via environment variables or a `.env` file in the working directory.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | `local` \| `openai` \| `azure` |
| `OPENAI_API_KEY` | — | OpenAI API key (embeddings + planner + synthesis) |
| `OPENAI_MODEL` | `gpt-4o` | Chat model for planner + synthesis |
| `AZURE_API_KEY` | — | Azure OpenAI API key |
| `AZURE_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `COHERE_API_KEY` | — | Cohere API key (reranker) |
| `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` | `12000` | Max context tokens sent to LLM |
| `TRELIX_PARSE_WORKERS` | `4` | Parallel threads for parsing phase |

See `.env.example` for the full reference.

---

## Supported Languages

### Code (Tree-sitter AST)
Python, TypeScript/TSX, JavaScript/JSX, Go, Java, Rust, C, C++, C#, Kotlin, Ruby

### .NET / Razor
Razor Components (`.razor`), Razor MVC Views (`.cshtml`), MSBuild projects (`.csproj`)

### Config (key-path extraction)
JSON/JSONC, TOML, YAML (multi-document)

### Markup
Markdown (heading sections), HTML (custom elements), CSS/SCSS

---

## How it works

```mermaid
flowchart TD
    subgraph INDEXING["INDEXING  (offline — trelix index)"]
        A[Repository] --> B[FileWalker\n.gitignore-aware\nSHA-256 change detection]
        B --> C[Tree-sitter Parser\nper-language extractors\nsymbols + call/import/type edges]
        C --> D[Chunker\ncontext-header breadcrumbs\ntiktoken token count]
        D --> E[Embedder\nazure | openai | local\nsentence-transformers]
        E --> F[(sqlite-vec\nANN vector store)]
        C --> G[(SQLite DB\nfiles, symbols,\ncall_graph, imports,\nFTS5 BM25)]
    end

    subgraph RETRIEVAL["RETRIEVAL  (per query — trelix search / ask)"]
        H[User Query] --> I[QueryPlanner\noptional LLM\n8 intents → RetrievalStrategy]
        I --> J[Vector Search\nHyDE snippet → ANN]
        I --> K[BM25 Search\nFTS5 pre-cleaned tokens]
        I --> L[Grep Search\nexact / regex names]
        J --> M[RRF Fusion\nReciprocal Rank Fusion k=60]
        K --> M
        L --> M
        M --> N[Graph Expansion\ncall_graph + import_graph + type_edges]
        N --> O[Reranker\nCohere | cross-encoder]
        O --> P[Context Assembler\ngreedy | breadth_first\ntoken budget]
        P --> Q[LLM Synthesis\ntrelix ask — optional]
    end

    F --> J
    G --> K
    G --> L
    G --> N
```

### Indexing phases

| Phase | What | Parallelism |
|-------|------|-------------|
| 1 — Parse | Tree-sitter AST traversal per file | ThreadPoolExecutor (parse_workers=4) |
| 2 — Write | Symbol + chunk insertion, parent_id remapping | Sequential (DB consistency) |
| 3 — Embed | Token-aware batch embedding + TPM rate limiting | Batch API calls |
| 4 — Resolve | Cross-file call edges, import paths, type edges | Sequential |

### 8 retrieval intents

| Intent | Legs | Graph expansion | Assembly |
|--------|------|-----------------|----------|
| `symbol_lookup` | grep + BM25 + vector | call (depth 1) | greedy |
| `file_overview` | file-direct | none | greedy |
| `feature_flow` | vector + BM25 | call+import (depth 2) | greedy |
| `project_overview` | file-direct | none | greedy |
| `comparison` | all 3 | call+import (depth 1) | greedy |
| `config_lookup` | file-direct + grep | none | greedy |
| `dependency_map` | vector + BM25 | import forward (depth 2) | breadth_first |
| `blast_radius` | grep + vector + BM25 | import reverse (depth 1) | breadth_first |

### Store layout

Single SQLite file (`.trelix/index.db`) — zero external infrastructure.

| Table | Purpose |
|-------|---------|
| `files` | Indexed files with SHA-256 hash for incremental updates |
| `symbols` | Extracted symbols (function, class, method…) with line spans |
| `call_graph` | Directed call edges (caller_id → callee_id) |
| `imports` | File-level import edges |
| `type_edges` | Inheritance / implements / trait edges |
| `chunks` | Embeddable text (context header + symbol body) |
| `symbols_fts` | FTS5 virtual table for BM25 full-text search |
| `vec_chunks` | sqlite-vec vector table for ANN search |

---

## Eval Results

Recall measured on `tests/fixtures/mini_repo` — a 7-file synthetic repo with Python auth, user, utils, api, and main modules.

**Provider**: `local` (sentence-transformers `all-MiniLM-L6-v2`, no API key)
**Metric**: Recall@5 — expected file appears in top-5 results

| Query | Expected file | Result | Top-1 file |
|-------|--------------|--------|------------|
| how does authentication work | auth.py | PASS | auth.py |
| user repository get by id | user.py | PASS | user.py |
| hash password function | utils.py | PASS | utils.py |
| login method | auth.py | PASS | auth.py |
| validate token | auth.py | PASS | api.py |
| User dataclass | user.py | PASS | user.py |
| main entry point | main.py | PASS | main.py |
| delete user | user.py | PASS | user.py |
| verify password | utils.py | PASS | auth.py |
| create user | user.py | PASS | user.py |

**Recall@5: 10/10 = 100%**

Run the eval harness yourself:

```bash
pip install "trelix[local,dev]"
python -m pytest tests/integration/test_recall.py -v
```

---

## Development

```bash
git clone https://github.com/trelix-dev/trelix
cd trelix
make install-dev
make test
make lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide including how to add a new language parser.

---

## License

MIT — see [LICENSE](LICENSE).
