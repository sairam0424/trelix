# Changelog

All notable changes to trelix are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-06-25

### Added
- Ruby parser — completes all 20 language extractors
- PyInstaller spec (`trelix.spec`) — produces `dist/trelix` single-file binary
- `scripts/build-binary.sh` — local binary build script
- `make binary` / `make binary-clean` / `make binary-install` Makefile targets
- GitHub Actions `build-binaries.yml` — macOS arm64 + Windows x64 matrix
- Release workflow attaches `trelix` / `trelix.exe` binaries to GitHub Releases
- `docs/integrations/vscode-plugin.md` — guide for embedding trelix in a VS Code extension

## [0.1.0] — 2025-06-25

### Added
- Initial release — Tree-sitter AST indexing for 20+ languages
- Hybrid search: vector (ANN, sqlite-vec) + BM25 (FTS5) + grep
- RRF fusion + call-graph / import / type-edge expansion with PageRank
- 8-intent LLM query planner (symbol_lookup, feature_flow, blast_radius, …)
- Cohere + cross-encoder reranker support
- Intent-aware context assembler with greedy/breadth_first packing
- LLM synthesis via OpenAI or Azure OpenAI (`trelix ask`)
- CLI: `index`, `search`, `ask`, `query`, `stats`, `update-index`
- Three embedding providers: `local` (no API key), `openai`, `azure`
- Zero-infra store: single SQLite file with sqlite-vec + FTS5

[Unreleased]: https://github.com/trelix-dev/trelix/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/trelix-dev/trelix/releases/tag/v0.2.0
[0.1.0]: https://github.com/trelix-dev/trelix/releases/tag/v0.1.0
