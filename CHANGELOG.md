# Changelog

All notable changes to trelix are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-06-25

### Added

#### Binary Distribution
- macOS arm64 binary (`codeindex`) via PyInstaller
- Windows x64 binary (`codeindex.exe`) via GitHub Actions
- Drop-in compatible with aava-core-vscode-ide-plugin

#### VS Code Plugin Compatibility
- `CODEINDEX_STORE_DB_PATH` env var accepted as fallback when `TRELIX_STORE_DB_PATH` is not set
- `--provider aava` flag supported — routes to `AavaPlatformEmbedder`
- `AavaPlatformEmbedder` ported back from aava-core for Aava platform embedding service
- Aava-specific config fields: `EMBEDDING_BEARER_TOKEN`, `EMBEDDING_BASE_URL`, `EMBEDDING_SERVICE`, `EMBEDDING_MODEL_REF`

#### Language Support
- Ruby parser added (Tree-sitter grammar)

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

[Unreleased]: https://github.com/trelix-dev/trelix/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/trelix-dev/trelix/releases/tag/v0.1.0
