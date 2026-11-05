# Codebase Structure

## 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `src/trelix/` | Core Python library and CLI, split by indexing, retrieval, storage, graph, API, and related domains | `pyproject.toml`, `src/trelix/` |
| `packages/` | Separately installable MCP, LangChain, and LlamaIndex adapters | `packages/trelix-mcp/pyproject.toml`, `packages/trelix-langchain/pyproject.toml`, `packages/trelix-llama-index/pyproject.toml` |
| `tests/` | Core unit, integration, and end-to-end suites, plus fixtures and evaluation tests | `tests/unit/`, `tests/integration/`, `tests/e2e/` |
| `infra/github-app/` | Standalone TypeScript GitHub App that accepts webhooks and runs PR reviews | `infra/github-app/README.md`, `infra/github-app/src/server.ts` |
| `workspace-vscode/` | VS Code extension | `workspace-vscode/package.json`, `docs/integrations/vscode-plugin.md` |
| `plugins/trelix/` | Claude Code plugin assets and scripts | `plugins/trelix/README.md`, `.claude-plugin/marketplace.json` |
| `docs/` | User guides, reference documentation, architecture, roadmap, and reports | `docs/README.md` |
| `helm/trelix/`, `Dockerfile`, `docker-compose.yml` | Container and Kubernetes deployment definitions | `helm/trelix/Chart.yaml`, `Dockerfile`, `docker-compose.yml` |
| `eval/` | Golden retrieval datasets and evaluation metadata | `eval/README.md`, `eval/golden.jsonl` |
| `scripts/`, `config/` | Development, build, measurement, and security-analysis utilities | `Makefile`, `config/semgrep-taint.yaml` |

## 2) Entry Points

- Main Python CLI: `trelix.cli.main:app`, registered by `[project.scripts]` in `pyproject.toml`.
- MCP server: `trelix_mcp.server:main`, registered in `packages/trelix-mcp/pyproject.toml`.
- REST API: launched by the CLI's `serve` command and implemented in `src/trelix/api/app.py`.
- GitHub App: `infra/github-app/src/server.ts`, compiled to `dist/server.js`; npm start script runs the compiled server.
- VS Code extension: package metadata and extension commands live under `workspace-vscode/`; implementation entry is specified by its `package.json`.

## 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `src/trelix/indexing/` | File walking, language parsing, chunking, indexing, watchers, and connectors | Query-time ranking or HTTP routing |
| `src/trelix/retrieval/`, `embedder/`, `llm/` | Query planning, retrieval legs, ranking, embedding, chat backends, and synthesis | CLI-specific presentation |
| `src/trelix/store/` | SQLite schema and vector-store implementations | Provider-specific query planning |
| `src/trelix/api/`, `cli/` | HTTP and command-line interfaces over the library | Duplicated persistence or parser implementations |
| `src/trelix/graph/`, `analysis/` | Code graph construction, graph querying, and static analyses | User-interface code |
| `packages/` | Optional protocol/framework adapters built on the core package | Core pipeline ownership |
| `infra/github-app/` | Hosted webhook intake, review job coordination, and GitHub Check publication | Core parser or retrieval algorithms |

## 4) Naming and Organization Rules

- Python module and test filenames use `snake_case` (for example, `retriever.py`, `test_retriever_core.py`); test files are primarily in `tests/unit`, with `test_*.py` names.
- Python package directories are organized by responsibility (`indexing`, `retrieval`, `store`, `graph`, `api`), rather than one flat package.
- No general Python import alias is configured in the root package; source uses `trelix.*` imports. The package root explicitly exports its supported convenience API in `src/trelix/__init__.py`.
- The TypeScript App uses `.ts` modules and `.js` import suffixes in source imports, as shown in `infra/github-app/src/server.ts`.

## 5) Evidence

- `pyproject.toml`
- `src/trelix/`
- `packages/`
- `tests/`
- `infra/github-app/package.json`
- `docs/README.md`
