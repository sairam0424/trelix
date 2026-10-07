# trelix plugin for Claude Code

This directory is the `trelix` plugin that the marketplace at the repository root
(`.claude-plugin/marketplace.json`) lists. Installing it gives Claude Code two things:

1. **The `trelix-mcp` server**, registered as `plugin:trelix:trelix` and launched exactly as
   `.mcp.json` says: `uvx --from trelix-mcp==3.4.3 trelix-mcp`. Its tools are callable as
   `mcp__plugin_trelix_trelix__<tool>` (for example `mcp__plugin_trelix_trelix__search_code`).
2. **The skill `/trelix:use-trelix-index`** (`skills/use-trelix-index/SKILL.md`): when to search
   with trelix, how to page through results, and when to fall back to grep.

The plugin pins the newest *published* `trelix-mcp` release, so it trails this repository by one
release and never points at a version PyPI does not have. The current pin is in `.mcp.json`;
`plugin.json`'s `version` is that pin (or `<pin>.N` for a plugin-only change).

## Prerequisites

- `uv` on your `PATH` (<https://docs.astral.sh/uv/>; the plugin was written against uv 0.10.7).
  `uvx` downloads a Python >= 3.12 if none is available.
- Network access on the first start: `uvx` downloads `trelix-mcp==3.4.3` and its dependencies
  (on one macOS machine: about a minute and 277 MB of downloads; later starts take about two
  seconds from the cache). Run the warm-up command below once so the first session does not wait.
- An embedding provider (see [Embeddings](#embeddings)): the plugin's server has no local model.

## Install

```bash
claude plugin marketplace add sairam0424/trelix
claude plugin install trelix@trelix
uvx --from trelix-mcp==3.4.3 trelix-mcp --version   # one-time warm-up; prints "trelix-mcp 3.4.3"
```

The server registers as `plugin:trelix:trelix` (`/mcp` in a session lists plugin servers), and
`/trelix:use-trelix-index` loads the skill. The server's tools answer meaningfully only for a
repository that has an index (without one they answer empty, not with an error; see
[What the server reads and writes](#what-the-server-reads-and-writes)): ask Claude to run
`index_codebase` with the absolute path of your project first (it embeds every symbol, so it
takes minutes on a large tree and costs money with an API embedder).

## Every command the plugin executes

| When | Command | Purpose |
|---|---|---|
| Session start, by Claude Code | `uvx --from trelix-mcp==3.4.3 trelix-mcp` | The MCP server over stdio |

That is the whole list: this version has no hooks and runs nothing else. The plugin sets no
environment variables and carries no secrets; the server reads provider keys from your shell or
from `~/.config/trelix/env`, exactly as `trelix-mcp` does outside the plugin. The plugin itself
makes no network calls; `uvx` downloads from PyPI on the first start.

## What the server reads and writes

The server works on the repository you pass as `repo_path`, inside `<repo_path>/.trelix/`:
`index_codebase` creates or updates `index.db` (and a `.gitignore` beside it), and
`build_knowledge_graph` / `graph_search_mcp` write graph metadata into it. On the pinned 3.4.3 a
call to `search_code`, `get_symbol` or `blast_radius` against a repository with **no** index
creates an empty, gitignored `.trelix/index.db` and answers empty (`search_code` with
`results: []`, `blast_radius` with `[]`, `get_symbol` with `null`); nothing else is written.

## Embeddings

trelix's default embedder is `local` (sentence-transformers), which the plugin's server does
**not** install: the `trelix[local]` extra pulls PyTorch, which would turn the first start into a
multi-gigabyte download on some platforms. Pick one:

- **An API embedder.** Set `TRELIX_EMBEDDER_PROVIDER` (for example `openai`) and its key in your
  shell or in `~/.config/trelix/env`; see `docs/CONFIGURATION.md` and `docs/PROVIDERS.md`.
- **Local embeddings, by registering the server yourself.** Run

  ```bash
  claude mcp add --transport stdio trelix -- uvx --from trelix-mcp==3.4.3 --with "trelix[local]==3.4.3" trelix-mcp
  ```

  then toggle the plugin's server `plugin:trelix:trelix` off in `/mcp`: Claude Code stops
  connecting to it, records the choice per project in `~/.claude.json` under `disabledMcpServers`,
  and keeps the plugin with its skill (<https://code.claude.com/docs/en/mcp>, "Disable a server
  without removing it"). The skill's tool names and pre-approval refer to the plugin's server, so
  the server you registered answers under `mcp__trelix__*` and keeps its permission prompts.
  Disabling the whole plugin (`claude plugin disable trelix@trelix`, which also drops the skill) or
  keeping both servers (distinct prefixes: `mcp__trelix__*` and `mcp__plugin_trelix_trelix__*`)
  also works.

A tool error that names `sentence-transformers` means neither option is in place yet.

## Permissions

The skill's `allowed-tools` pre-approves `search_code`, `get_symbol` and `blast_radius`: the three
tools annotated `readOnlyHint` on this repository's `develop` branch
(`packages/trelix-mcp/tests/test_tool_readonly.py`). On the pinned 3.4.3 a call to any of them
against an unindexed repository creates an empty `.trelix/index.db` (see above); nothing else is
written. The grant is kept because the skill forbids searching before indexing, and it lasts one
turn (Claude Code clears it at your next message). Every other tool keeps its permission prompt:
`index_codebase` (writes the index, spends embedding money), `ask_agent` (spends LLM money,
stores sessions), `build_knowledge_graph` and `graph_search_mcp` (write graph metadata), and the
federation, session and subscription tools.

## Update

Auto-update is off for third-party marketplaces, so updates are explicit:

```bash
claude plugin marketplace update trelix
claude plugin update trelix@trelix
```

A new version loads in your next session (or after `/reload-plugins`). Claude Code keeps you on a
cached copy until `plugin.json`'s `version` changes, which is why every change under this
directory bumps it.

## Uninstall

```bash
claude plugin uninstall trelix@trelix
claude plugin marketplace remove trelix   # from the last scope that declares it: also uninstalls its plugins
```

## Decisions taken for this version

- **No `trelix[local]` in `.mcp.json`** (see Embeddings): opt in with the command above.
- **Pinned to the published `trelix-mcp==3.4.3`**, which accepts no `--tools` flag and sends no
  server instructions; the skill carries that guidance. A later change moves the pin once the
  next release is on PyPI.
- **`plugin.json` `version` equals the pin** (a literal in `tests/unit/test_claude_plugin_manifest.py`).
  A hash over this directory's contents makes every edit here fail that test until its literal
  is updated; the failure message asks for the `version` bump, which no test can see.
- **`claude plugin validate --strict` stays a manual step, and the pin check is offline.** CI runs
  the offline tests, not the validator (it would need an npm install of Claude Code); the tests
  check that the pin has a released CHANGELOG section and is not newer than the trelix-mcp stamp,
  and PyPI itself is checked by hand in the pin-bump PR body (CONTRIBUTING.md).

Full guide: `docs/integrations/claude-code-plugin.md`.
