# Claude Code plugin

trelix ships as a Claude Code plugin from this repository's own marketplace
(`.claude-plugin/marketplace.json`). The plugin lives in `plugins/trelix/`; its README is the
authoritative list of every command it runs. This page is the user guide.

## What you get

| Component | What it is | How you see it |
|---|---|---|
| MCP server `trelix` | The published `trelix-mcp` release that `plugins/trelix/.mcp.json` pins, launched by `uvx` | Server `plugin:trelix:trelix`; tools `mcp__plugin_trelix_trelix__<tool>` |
| Skill `use-trelix-index` | When to search with trelix, how to page, when to fall back to grep | `/trelix:use-trelix-index` |
| SessionStart hook | One line of context at session start: whether the project has a trelix index, how old it is, which `repo_path` to pass | A line starting `trelix:` in the session context (`uv run --no-project --script` runs `plugins/trelix/scripts/session_start.py`) |

The server is the same `trelix-mcp` that `claude mcp add trelix -- trelix-mcp` registers (see
[MCP_GUIDE.md](../MCP_GUIDE.md)); the plugin adds the pinned launch, the skill and the
SessionStart line.

## Prerequisites

- `uv` on your `PATH` (the server launch and the SessionStart hook both run through it). `uvx`
  fetches a Python >= 3.12 if none is available.
- Network access on the first start, when `uvx` downloads the pinned `trelix-mcp` and its
  dependencies (about a minute on one machine; a few seconds afterwards).
- An embedding provider: the plugin's server does not install the `local` embedder (see
  [Embeddings](#embeddings)).

## Install

```bash
claude plugin marketplace add sairam0424/trelix
claude plugin install trelix@trelix
```

To warm the `uvx` cache before the first session, run the server once with `--version`, using
the pin from `plugins/trelix/.mcp.json`:

```bash
uvx --from trelix-mcp==<pin> trelix-mcp --version
```

Then open a session in your project. `/mcp` lists plugin servers; the trelix one is
`plugin:trelix:trelix`.

## First use

trelix answers from an index at `<repo>/.trelix/index.db`. Ask Claude to index the repository
first (`index_codebase` with the absolute path); it embeds every symbol, so it takes minutes on
a large tree and costs money with an API embedder. The SessionStart line (next section) tells
Claude which case applies, and the skill tells it to ask you before indexing.

On the pinned release a search against a repository with no index does not fail: it answers
`results: []` and leaves an empty `.trelix/index.db` behind. The skill therefore forbids
concluding "not found" from an empty result alone. (The next release, `develop` today after
#467, answers `No index found at <path>. Run trelix index <repo> first.` instead; the plugin
picks that up with the next pin bump.)

## What the SessionStart line looks like

At every session start (`startup`, `resume`, `clear`) Claude Code runs
`uv run --no-project --script "<plugin root>/scripts/session_start.py"` with a 15 s timeout and
adds its stdout to the session context. The script prints exactly one of three lines, with the
absolute project path (`CLAUDE_PROJECT_DIR`, or the hook's `cwd` when that is unset or does not name
a directory) in place of
`<repo>`:

- `trelix: no index at <repo>/.trelix/index.db. Call index_codebase(repo_path="<repo>") before search_code; until then, use grep.`
- `trelix: the index at <repo>/.trelix/index.db is empty (0 files). Call index_codebase(repo_path="<repo>") before search_code; until then, use grep.`
- `trelix: <repo> is indexed: 1204 files, 9817 symbols; built 2026-10-01T00:00:00+00:00 from commit 0123456789ab (HEAD is 2 commits ahead); embedder local. Pass repo_path="<repo>" to every trelix tool.`

The distance is `(= HEAD)` when nothing is newer than the index, `(HEAD is 1 commit ahead)`,
`(HEAD is N commits ahead)`, or `(distance from HEAD unknown)` when git cannot say; an index
written before trelix recorded provenance reads `built at an unknown time (re-index to record
provenance)` with no commit and no embedder. The embedder is the provider name, never the model.
The line is at most 1,000 characters.

The script is standard-library Python: it opens the index through a read-only SQLite URI (a
missing index is not created; SQLite may leave empty `-wal`/`-shm` files beside an existing
one, and when `.trelix/` is not writable and those files are absent it cannot open a WAL index
at all, so the hook prints nothing), runs one `git rev-list --count` when the index records a
commit, makes no network call, and exits 0 on every path. When it has nothing to say (no project
directory, an unreadable index) it prints nothing and writes `trelix session_start: skipped
(<reason>)` to stderr, which Claude Code keeps in its debug log (`claude --debug`), not in the
session. The plugin README lists every command it runs.

## The skill

`/trelix:use-trelix-index` loads `plugins/trelix/skills/use-trelix-index/SKILL.md`. It names the
tools in the order to use them (`search_code`, `get_symbol`, `blast_radius`, `graph_search_mcp`,
`build_knowledge_graph`, `ask_agent`), says that `repo_path` is always the absolute project
root, how to follow `next_cursor`, and when grep is the better tool. Its `allowed-tools`
pre-approves `search_code`, `get_symbol` and `blast_radius` for the turn that invokes it; every
other tool keeps its permission prompt. For another agent, [AGENTS_SNIPPET.md](AGENTS_SNIPPET.md)
is the same guidance as a paste-able block.

## Embeddings

The server's default embedder is `local` (sentence-transformers), which the plugin does not
install: that extra pulls PyTorch. Either set an API embedder (`TRELIX_EMBEDDER_PROVIDER` and
its key, in your shell or `~/.config/trelix/env`; see [CONFIGURATION.md](../CONFIGURATION.md)
and [PROVIDERS.md](../PROVIDERS.md)), or register the server yourself with the `[local]` extra:

```bash
claude mcp add --transport stdio trelix -- uvx --from trelix-mcp==<pin> --with "trelix[local]==<pin>" trelix-mcp
```

then toggle the plugin's server `plugin:trelix:trelix` off in `/mcp`: Claude Code stops connecting
to it, records the choice per project in `~/.claude.json` under `disabledMcpServers`, and keeps the
plugin with its skill (<https://code.claude.com/docs/en/mcp>, "Disable a server without removing
it"). The skill's tool names and pre-approval refer to the plugin's server, so the one you
registered answers under `mcp__trelix__*` and keeps its permission prompts. Disabling the whole
plugin (`claude plugin disable trelix@trelix`, which also drops the skill) or keeping both servers
(distinct prefixes, `mcp__trelix__*` and `mcp__plugin_trelix_trelix__*`) also works.

## Updating

Auto-update is off for third-party marketplaces. To update:

```bash
claude plugin marketplace update trelix
claude plugin update trelix@trelix
```

A new version loads in the next session or after `/reload-plugins`. Claude Code keeps you on a
cached copy until `plugin.json`'s `version` changes.

## Uninstalling

```bash
claude plugin uninstall trelix@trelix
claude plugin marketplace remove trelix   # from the last scope that declares it: also uninstalls its plugins
```

## How the pin relates to releases

`plugins/trelix/.mcp.json` pins the newest *published* `trelix-mcp` with `==`. A release PR never
touches it, so `main` never points at a version PyPI does not have yet; after PyPI shows a new
release, a separate PR moves the pin and `plugin.json`'s `version`. The plugin therefore trails
this repository by one release. A test fails when the pin is newer than the repository's own
stamp or names a version with no `## [X.Y.Z]` CHANGELOG section. Marketplaces are fetched from
the repository's default branch, so a change becomes installable when it reaches `main`. CI also
runs Claude Code's own `claude plugin validate --strict` (a pinned release of
`@anthropic-ai/claude-code`, through `npx`) over the marketplace and the plugin on every push and
pull request, so a manifest Claude Code would warn on does not merge.

## Troubleshooting

- **The server is not listed, or shows as failed, in the first session.** The cold start can
  take a minute while `uvx` downloads; run the warm-up command above and start a new session.
- **A tool error names `sentence-transformers`.** No embedder is configured; see
  [Embeddings](#embeddings).
- **Empty results on a repository you expected to be indexed.** Check for
  `<repo>/.trelix/index.db`; see [First use](#first-use).
- **No `trelix:` line at session start.** `uv` is not on `PATH` (the hook runs
  `uv run --no-project --script`), or the hook had nothing to say: `claude --debug` shows a
  `trelix session_start: skipped (<reason>)` line, `no project directory` when neither
  `CLAUDE_PROJECT_DIR` nor the hook's `cwd` named a directory, an exception type (such as
  `DatabaseError`) when `.trelix/index.db` could not be read.
- **Two trelix servers.** You also registered `claude mcp add trelix -- trelix-mcp`; both work,
  with distinct tool prefixes. Toggle one off in `/mcp` (see [Embeddings](#embeddings)) or remove
  it if the duplicate tool list bothers you.
- More in [MCP_GUIDE.md](../MCP_GUIDE.md), section 18.
