# trelix plugin for Claude Code

This directory is the `trelix` plugin that the marketplace at the repository root
(`.claude-plugin/marketplace.json`) lists. Installing it gives Claude Code three things:

1. **The `trelix-mcp` server**, registered as `plugin:trelix:trelix` and launched exactly as
   `.mcp.json` says: `uvx --from trelix-mcp==3.4.3 trelix-mcp`. Its tools are callable as
   `mcp__plugin_trelix_trelix__<tool>` (for example `mcp__plugin_trelix_trelix__search_code`).
2. **The skill `/trelix:use-trelix-index`** (`skills/use-trelix-index/SKILL.md`): when to search
   with trelix, how to page through results, and when to fall back to grep.
3. **A SessionStart hook** (`hooks/hooks.json`, `scripts/session_start.py`): one line of context
   at the start of every session (startup, resume, `/clear`) that says whether the project has a
   trelix index, how old it is, and what to pass as `repo_path`; see
   [What Claude sees at session start](#what-claude-sees-at-session-start).

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
takes minutes on a large tree and costs money with an API embedder). The SessionStart line
tells Claude which case applies.

## Every command the plugin executes

| When | Command | Purpose |
|---|---|---|
| Session start, by Claude Code | `uvx --from trelix-mcp==3.4.3 trelix-mcp` | The MCP server over stdio |
| Session start (`startup`, `resume`, `clear`), by Claude Code, 15 s timeout | `uv run --no-project --script "${CLAUDE_PLUGIN_ROOT}/scripts/session_start.py"` | One line on stdout: whether this project has a trelix index |
| Inside that script, only when the index records the commit it was built from | `git -C <project> rev-list --count <commit>..HEAD` (5 s timeout, `GIT_TERMINAL_PROMPT=0`, `GIT_OPTIONAL_LOCKS=0`) | How many commits HEAD is ahead of the index |

That is the whole list. The plugin sets no environment variables and carries no secrets; the
server reads provider keys from your shell or from `~/.config/trelix/env`, exactly as
`trelix-mcp` does outside the plugin. The plugin itself makes no network calls; `uvx` downloads
from PyPI on the first start.

The hook script is standard-library Python (a test pins its import roots and that the name
`urlopen` does not appear in it). It reads `CLAUDE_PROJECT_DIR` (or the hook's stdin `cwd` when that is unset or does
not name a directory) and opens `<project>/.trelix/index.db` through a read-only `mode=ro` SQLite
URI: a missing index is not created and a write would fail. SQLite may leave empty
`index.db-wal` / `index.db-shm` files beside an existing index, as every read-only open of a WAL
database does (when `.trelix/` is not writable and those files are absent, it cannot open a WAL
index at all and the hook prints nothing); nothing else is written. It exits 0 on every path and
prints at most 1,000 characters; on an error it prints nothing and writes one `trelix
session_start: skipped (<reason>)` line to stderr, which Claude Code keeps in its debug log only.
Hooks run with your permissions, outside any sandbox.

## What the server reads and writes

The server works on the repository you pass as `repo_path`, inside `<repo_path>/.trelix/`:
`index_codebase` creates or updates `index.db` (and a `.gitignore` beside it), and
`build_knowledge_graph` / `graph_search_mcp` write graph metadata into it. On the pinned 3.4.3 a
call to `search_code`, `get_symbol` or `blast_radius` against a repository with **no** index
creates an empty, gitignored `.trelix/index.db` and answers empty (`search_code` with
`results: []`, `blast_radius` with `[]`, `get_symbol` with `null`); nothing else is written.

## What Claude sees at session start

The hook prints exactly one of three lines (the absolute project path in place of `<repo>`), or
nothing when it has no project directory or hits an error:

- `trelix: no index at <repo>/.trelix/index.db. Call index_codebase(repo_path="<repo>") before search_code; until then, use grep.`
- `trelix: the index at <repo>/.trelix/index.db is empty (0 files). Call index_codebase(repo_path="<repo>") before search_code; until then, use grep.`
  (the index a search on the pinned release leaves behind on an unindexed repository)
- `trelix: <repo> is indexed: 1204 files, 9817 symbols; built 2026-10-01T00:00:00+00:00 from commit 0123456789ab (HEAD is 2 commits ahead); embedder local. Pass repo_path="<repo>" to every trelix tool.`

In the third line the distance reads `(= HEAD)` when nothing is newer than the index (HEAD is the
indexed commit or an ancestor of it), `(HEAD is 1 commit ahead)`, `(HEAD is N commits ahead)`, or
`(distance from HEAD unknown)` when git cannot say (no git, a commit this clone does not have, a
timeout). An index written before trelix recorded provenance reads `built at an unknown time
(re-index to record provenance)` with no commit and no embedder. The embedder is the provider
name, never the model. A line longer than 1,000 characters is cut to 999 and `…`.

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
- **The SessionStart hook prints plain text** (which Claude Code adds to the session context),
  not the JSON `hookSpecificOutput` form; it uses only the standard library, opens the index
  read-only, and exits 0 on every path so that it can never block a session. Hooks on other
  events are a later roadmap item.
- **`claude plugin validate --strict` runs in CI and before committing; the pin check is offline.**
  CI's `TypeScript SDK` job runs Claude Code's validator over the marketplace and this directory
  through a pinned release of `@anthropic-ai/claude-code` (`.github/workflows/ci.yml`; the version
  is a test literal, never written here), with warnings treated as errors; the tests check that
  the pin has a released CHANGELOG section and is not newer than the trelix-mcp stamp, and PyPI
  itself is checked by hand in the pin-bump PR body (CONTRIBUTING.md).

Full guide: `docs/integrations/claude-code-plugin.md`.
