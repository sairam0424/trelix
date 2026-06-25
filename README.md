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

# Search for code (returns JSON)
trelix search ./my-repo "database connection pooling"

# Ask a question (requires OPENAI_API_KEY for synthesis)
trelix ask ./my-repo "how does the authentication middleware work?"

# Show index statistics
trelix stats ./my-repo

# Re-index a single file after editing
trelix update-index ./my-repo src/auth/middleware.py
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

## Architecture

```
INDEXING (offline)
  Repository → FileWalker → Tree-sitter Parser → Chunker → Embedder → sqlite-vec
                                                         ↘ SQLite FTS5 (BM25)

RETRIEVAL (per query)
  Query → [LLM Query Planner] → Vector + BM25 + Grep legs
                             → RRF Fusion → Graph Expansion
                             → Reranker → Context Assembler
                             → [LLM Synthesis]
```

See [docs/architecture.md](docs/architecture.md) for the full pipeline diagrams.

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
