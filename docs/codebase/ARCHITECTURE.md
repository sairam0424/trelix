# Architecture

## 1) Architectural Style

- Primary style: domain-separated Python package with a staged indexing pipeline and a retrieval pipeline; CLI, REST, MCP, and framework adapters expose those core services.
- Evidence: `src/trelix/indexing/`, `src/trelix/retrieval/`, `src/trelix/store/`, and `src/trelix/api/` are separate modules. `src/trelix/indexing/indexer.py` documents the staged processing and `src/trelix/retrieval/retriever.py` documents query routing.
- Primary constraints:
  - Local SQLite-backed operation is the default storage path; remote vector backends and model providers are optional.
  - Index contents are updated incrementally using file hashes and persisted symbol IDs.
  - Optional model, serving, and integration dependencies are installed through package extras or adapter packages.

## 2) System Flow

```text
CLI / Python API / REST / MCP
  -> config + Indexer or Retriever
  -> parse and index / plan and retrieve
  -> SQLite metadata + vector store
  -> ranked context / synthesized answer
```

Indexing flow:

1. `FileWalker` discovers repository files and determines language; the parser registry selects a Tree-sitter-backed extractor (`src/trelix/indexing/walker.py`, `src/trelix/indexing/parser/registry.py`).
2. `Indexer` parses files concurrently, then writes symbols and chunks in a controlled database phase so local symbol references can be remapped to persistent IDs (`src/trelix/indexing/indexer.py`).
3. Optional file summaries can be generated; chunk embeddings are produced in concurrent batches and saved through the selected vector store (`src/trelix/indexing/indexer.py`, `src/trelix/store/vector.py`).
4. A cross-file pass resolves call and import relationships after file insertion (`src/trelix/indexing/indexer.py`).

Query flow:

1. `Retriever` obtains or accepts a query plan; `QueryPlanner` and routing models choose a retrieval strategy (`src/trelix/retrieval/retriever.py`, `src/trelix/retrieval/planner/`).
2. The selected retrieval legs search vector, BM25/FTS, grep, and configured optional indexes (`src/trelix/retrieval/`).
3. Reciprocal-rank fusion merges results; graph-based call, import, and type expansion and optional reranking enrich or reorder candidates (`src/trelix/retrieval/fusion.py`, `src/trelix/retrieval/graph.py`, `src/trelix/retrieval/reranker.py`).
4. The assembler fits evidence to a context budget, then `Synthesizer` or the agent loop can produce an answer (`src/trelix/retrieval/assembler.py`, `src/trelix/retrieval/synthesizer.py`, `src/trelix/agent/loop.py`).

## 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `core/` | Shared configuration, data models, retry, and confinement helpers | Interface-specific rendering | `src/trelix/core/` |
| `indexing/` | Walk, parse, chunk, index, watch, and source connectors | Query ranking | `src/trelix/indexing/indexer.py` |
| `embedder/`, `llm/` | Provider-neutral model interfaces and provider implementations | Repository traversal or database schema | `src/trelix/embedder/`, `src/trelix/llm/` |
| `store/` | Metadata schema and vector-storage backends | API and CLI policy | `src/trelix/store/db.py`, `src/trelix/store/vector.py` |
| `retrieval/` | Query planning, candidate gathering, fusion, ranking, context assembly, synthesis | Interface transport | `src/trelix/retrieval/retriever.py` |
| `api/`, `cli/` | HTTP and command entry surfaces | A separate indexing/retrieval implementation | `src/trelix/api/app.py`, `src/trelix/cli/main.py` |
| `packages/` | MCP, LangChain, and LlamaIndex wrappers | Ownership of core data and algorithms | Package `pyproject.toml` files |

## 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| Registry/factory | Parser, connector, vector-store, embedder, and LLM modules | Select implementations from language/provider/config without duplicating pipeline code |
| Adapter | MCP, LangChain, and LlamaIndex packages | Integrate the same core retrieval API into different consumer protocols |
| Incremental processing | `FileWalker`, `Indexer`, and database file hashes | Avoid redoing work for unchanged files |
| Retry/backoff | `src/trelix/core/retry.py` and provider/connector call sites | Share transient HTTP retry behavior |
| Settings objects | `src/trelix/core/config.py` | Validate environment-derived config before use |

## 5) Known Architectural Risks

- `src/trelix/cli/main.py` contains a large set of CLI commands and was the largest Python source file in the repository scan; changes to command wiring and user-visible output can affect many operations.
- Several alternate providers, stores, and integrations increase the configuration and test matrix; the project uses separate extras and CI jobs to cover selected combinations (`pyproject.toml`, `.github/workflows/ci.yml`).
- Default SQLite vector search is a local, zero-infrastructure option; the README and code provide scale advisories and optional remote stores (`docs/README.md`, `src/trelix/store/scale_advisory.py`).

## 6) Evidence

- `src/trelix/indexing/indexer.py`
- `src/trelix/retrieval/retriever.py`
- `src/trelix/store/db.py`
- `src/trelix/core/config.py`
- `docs/architecture.md`
