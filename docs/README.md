# trelix v2.7.1 Documentation

Welcome to the trelix documentation hub. This page is a complete index of every documentation file in the project.

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
  → CONFIGURATION.md (root)

Connecting multiple trelix instances?
  → FEDERATION_GUIDE.md

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
| [CONFIGURATION.md](../CONFIGURATION.md) | Every config key, environment variable, and default value | Operators, power users |
| [CLI_REFERENCE.md](../CLI_REFERENCE.md) | Full CLI command reference with flags and examples | CLI users |

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
| [v2.4.0-world-release-report.md](v2.4.0-world-release-report.md) | v2.4.0 release readiness audit: what shipped, benchmarks, blockers found before tagging |
| [superpowers/CHANGELOG.md](superpowers/CHANGELOG.md) | Superpowers module changelog |

---

## Internal / Development

| Path | Description |
|------|-------------|
| [superpowers/plans/](superpowers/plans/) | Implementation plans for upcoming features |
| [superpowers/specs/](superpowers/specs/) | Feature specifications |

---

*Last updated: 2026-07-10 — trelix v2.7.1*
