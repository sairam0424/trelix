# trelix v3.1.7 Documentation

Welcome to the trelix documentation hub. This page indexes every file under `docs/` and all five
top-level Markdown docs, with one deliberate exception: the 35 implementation plans in
[superpowers/plans/](superpowers/plans/) and the 6 design specs in
[superpowers/specs/](superpowers/specs/) are linked as directories, not enumerated. They are
dated working artifacts — a per-file list here would go stale the next time a plan is written.
Everything else is listed below by name.

---

## Quick Links

| Goal | Document |
|------|----------|
| Getting started | [GETTING_STARTED.md](GETTING_STARTED.md) |
| Why trelix? | [WHY_TRELIX.md](WHY_TRELIX.md) |
| Install options | [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) |
| MCP integration | [MCP_GUIDE.md](MCP_GUIDE.md) |
| Latest release | [../CHANGELOG.md](../CHANGELOG.md) |

---

## What Documentation Do I Need?

```
First time using trelix?
  → GETTING_STARTED.md

Want to understand the system in depth?
  → USER_GUIDE.md

Setting up trelix as an MCP server?
  → MCP_GUIDE.md

Using trelix from Python (LangChain / LlamaIndex)?
  → LANGCHAIN_LLAMAINDEX_GUIDE.md

Confused about config options?
  → CONFIGURATION.md

Connecting multiple trelix instances?
  → FEDERATION_GUIDE.md

Locking down the HTTP API (OIDC bearer auth, audit trail)?
  → SSO.md, then AUDIT.md

Using a specific provider (OpenAI, Ollama, etc.)?
  → PROVIDERS.md

Something broken?
  → TROUBLESHOOTING.md

Looking for a specific term?
  → GLOSSARY.md
```

---

## User Documentation

| File | Description | Audience |
|------|-------------|----------|
| [../README.md](../README.md) | Project entry point: what trelix is, install one-liners, feature tour | Everyone |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Five-minute quickstart: install, configure, run your first query | New users |
| [WHY_TRELIX.md](WHY_TRELIX.md) | Design rationale, architectural decisions, and use-case fit | Evaluators, decision-makers |
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | All install paths: pip, Docker, from source, CI environments | All users |
| [USER_GUIDE.md](USER_GUIDE.md) | Full feature walkthrough — hybrid search, agentic loop, configuration | All users |
| [GLOSSARY.md](GLOSSARY.md) | Definitions for trelix-specific terms and concepts | All users |
| [FAQ.md](FAQ.md) | Answers to common questions | All users |

---

## Reference

| File | Description | Audience |
|------|-------------|----------|
| [architecture.md](architecture.md) | Internal architecture: components, data flow, extension points | Contributors, integrators |
| [CONFIGURATION.md](CONFIGURATION.md) | Every config key, environment variable, and default value | Operators, power users |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Full CLI command reference with flags and examples | CLI users |
| [OBSERVABILITY.md](OBSERVABILITY.md) | OpenTelemetry tracing: what's traced, how to enable, stability caveats | Operators, SRE |

---

## Security & Operations

Both features below are new in v3.0.0 and fully opt-in: with their enabling env var unset
(`TRELIX_OIDC_ENABLED`, `TRELIX_AUDIT_ENABLED`) no verifier is built and no middleware is
registered, and the HTTP API behaves exactly as it did in v2.12.0.

| File | Description | Audience |
|------|-------------|----------|
| [SSO.md](SSO.md) | OIDC bearer-token verification for the HTTP API. OIDC only — trelix is a resource server, and SAML is not supported | Operators, platform engineers |
| [AUDIT.md](AUDIT.md) | Hash-chained append-only audit trail: one entry per API request, backed by stdlib `sqlite3` | Operators, compliance |

---

## Integrations

| File | Description | Audience |
|------|-------------|----------|
| [MCP_GUIDE.md](MCP_GUIDE.md) | Running trelix as a Model Context Protocol server | Agent / tool builders |
| [LANGCHAIN_LLAMAINDEX_GUIDE.md](LANGCHAIN_LLAMAINDEX_GUIDE.md) | Using trelix from Python with LangChain and LlamaIndex | Python developers |
| [PROVIDERS.md](PROVIDERS.md) | Provider configuration: OpenAI, Anthropic, Ollama, Azure, and more | All users |
| [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) | Connecting multiple trelix instances for federated search | Platform engineers |
| [integrations/vscode-plugin.md](integrations/vscode-plugin.md) | VS Code extension setup and usage | VS Code users |

---

## Ecosystem & Discoverability

| File | Description |
|------|-------------|
| [discoverability/AWESOME-LIST-SUBMISSIONS.md](discoverability/AWESOME-LIST-SUBMISSIONS.md) | Submission templates for awesome-list entries |
| [discoverability/ECOSYSTEM-ROADMAP.md](discoverability/ECOSYSTEM-ROADMAP.md) | Planned ecosystem integrations and plugin roadmap |

---

## Support

| File | Description |
|------|-------------|
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Diagnosis steps for common errors and failure modes |
| [FAQ.md](FAQ.md) | Frequently asked questions |
| [../SUPPORT.md](../SUPPORT.md) | How to get help: GitHub Discussions, issues, community channels |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guide: dev setup, PR process, coding standards |
| [../SECURITY.md](../SECURITY.md) | Security policy and responsible disclosure process |

---

## Release

| File | Description |
|------|-------------|
| [../CHANGELOG.md](../CHANGELOG.md) | Full version history with breaking changes and migration notes |
| [BACKWARDS_COMPATIBILITY.md](BACKWARDS_COMPATIBILITY.md) | SemVer policy: which API surfaces are stable, and the deprecation grace period before removal |
| [ROADMAP.md](ROADMAP.md) | Living roadmap — what shipped per version, and planned research directions |
| [v2.4.0-world-release-report.md](v2.4.0-world-release-report.md) | v2.4.0 release readiness audit: what shipped, benchmarks, blockers found before tagging |

---

## Measurement Reports

Everything under `reports/` is measured output from a specific machine and release, kept for
provenance. These are not guides — do not read a number here as a documented guarantee.

| File | Description |
|------|-------------|
| [reports/self-index-v3.1.2.md](reports/self-index-v3.1.2.md) | v3.1.2 self-index run: indexing trelix with trelix, then checking which index dimensions were actually populated. Numbers that could not be measured honestly are marked as such rather than estimated |
| [reports/index-hygiene-before.json](reports/index-hygiene-before.json) | Raw before-state for that run: 915 files, 59.3% of them `.vscode-test/` noise, indexed with the 384-dim local embedder |
| [reports/index-hygiene-after.json](reports/index-hygiene-after.json) | Raw after-state: 454 files, 0% noise, indexed with the configured `azure` provider |

---

## Internal / Development

Dated working artifacts. These are the only files in `docs/` this index does not enumerate
per-file — see the note at the top.

| Path | Description |
|------|-------------|
| [superpowers/plans/](superpowers/plans/) | 35 implementation plans, one per feature or release |
| [superpowers/specs/](superpowers/specs/) | 6 design specs backing the earliest plans |

---

*Last updated: 2026-08-17 — trelix v3.1.7*
