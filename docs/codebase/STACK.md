# Technology Stack

## 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python for the core; TypeScript for the GitHub App and VS Code extension | `pyproject.toml`, `infra/github-app/package.json`, `workspace-vscode/package.json` |
| Runtime + version | Python `>=3.12` (CI tests 3.12, 3.13, and 3.14); Node.js `>=24` for the GitHub App | `pyproject.toml`, `.github/workflows/ci.yml`, `infra/github-app/package.json` |
| Package manager | Python packages are installed with pip; the GitHub App uses npm (`package-lock.json`) | `Makefile`, `infra/github-app/package-lock.json` |
| Module/build system | Hatchling builds Python wheels; the core is a `src/` layout. The GitHub App compiles TypeScript with `tsc`. | `pyproject.toml`, `infra/github-app/package.json` |

## 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| Tree-sitter and tree-sitter-language-pack | `>=0.26.0`, `>=1.17.0` | Parse source files and extract symbols and relationships | `pyproject.toml`, `src/trelix/indexing/parser/` |
| SQLite, sqlite-vec, FTS5 | sqlite-vec `>=0.1.9,<0.1.10` | Local metadata, vector, and full-text storage | `pyproject.toml`, `src/trelix/store/db.py`, `src/trelix/store/vector.py` |
| Typer and Rich | `>=0.26.0`, `>=15.0.0` | CLI command and terminal UI | `pyproject.toml`, `src/trelix/cli/main.py` |
| Pydantic Settings | `>=2.14.2` | Typed configuration from environment and operator-owned dotenv files | `pyproject.toml`, `src/trelix/core/config.py` |
| OpenAI SDK | `>=2.20.0,<3.0.0` | Core LLM client and supported embedding path | `pyproject.toml`, `src/trelix/llm/` |
| NetworkX | `>=3.3.0` | Code graph and graph algorithms | `pyproject.toml`, `src/trelix/graph/` |
| FastAPI and Uvicorn (optional `serve` extra) | FastAPI `>=0.111.0`; Uvicorn `>=0.29.0` | Optional REST API | `pyproject.toml`, `src/trelix/api/app.py` |
| FastMCP and MCP SDK | Optional `trelix-mcp` package dependencies | MCP server over stdio or Streamable HTTP | `packages/trelix-mcp/pyproject.toml`, `packages/trelix-mcp/src/trelix_mcp/server.py` |
| Express and Octokit | Express `^5.2.1`; Octokit dependencies in the App manifest | Webhook-driven GitHub App for PR reviews | `infra/github-app/package.json`, `infra/github-app/src/server.ts` |

Other optional integrations include local sentence-transformer embeddings, Cohere and Voyage providers, AWS Bedrock and Vertex AI, Qdrant and LanceDB stores, OIDC, and OpenTelemetry. Their extras and version constraints are defined in `pyproject.toml`.

## 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| pytest / pytest-cov | Python unit, integration, and coverage tests | `pyproject.toml`, `Makefile` |
| Ruff | Python linting and formatting | `pyproject.toml`, `.github/workflows/ci.yml` |
| mypy | Strict Python type checking | `pyproject.toml`, `.github/workflows/ci.yml` |
| Vitest / TypeScript compiler | GitHub App tests and build/typecheck | `infra/github-app/package.json` |
| pre-commit | Local lint, format, and secret-scanning hooks | `.pre-commit-config.yaml` |
| GitHub Actions | CI, release, packaging, security, and deployment workflows | `.github/workflows/` |

## 4) Key Commands

```bash
pip install -e ".[local,dev]"
make test
make test-e2e
make lint
make typecheck
make build
```

`make test` runs core unit tests and `trelix-mcp` tests. `make test-e2e` runs subprocess and distribution-install checks separately.

## 5) Environment and Config

- Configuration is defined in typed settings classes in `src/trelix/core/config.py`; environment variables use `TRELIX_`-prefixed settings names.
- `.env.example` documents optional provider credentials and runtime options. The core resolves an explicit `TRELIX_CONFIG_FILE` or an operator config under `$XDG_CONFIG_HOME/trelix/env` / `~/.config/trelix/env`; it does not automatically load a `.env` from the current repository directory.
- Local embedding and search can run without provider credentials. Remote embeddings, LLM synthesis/planning, connectors, and hosted integrations require the corresponding credentials.
- Container deployment uses a Python 3.14 slim image; the GitHub App runs on Node 24 or newer. See `Dockerfile`, `docker-compose.yml`, and `infra/github-app/package.json`.

## 6) Evidence

- `pyproject.toml`
- `Makefile`
- `src/trelix/core/config.py`
- `infra/github-app/package.json`
- `.github/workflows/ci.yml`
