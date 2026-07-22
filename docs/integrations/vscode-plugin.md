# trelix in VS Code

## Overview

The trelix VS Code extension (`workspace-vscode/`, published as `trelix-vscode`)
is a thin MCP client. It spawns `trelix-mcp` as a child process over stdio and
calls its tools/prompts directly — there is no bundled binary and no
PyInstaller step involved. The extension itself ships as one bundled
`dist/extension.js` (via `esbuild`), but the intelligence layer it talks to is
the same `trelix-mcp` server used by Claude Desktop/Cursor/any other MCP
client (see `docs/integrations/mcp.md`).

```
VS Code extension  --stdio JSON-RPC-->  trelix-mcp (child process)  -->  trelix core
(dist/extension.js)                     (search_code, trelix-search prompt)
```

---

## Prerequisites

`trelix-mcp` must be installed and importable wherever VS Code runs (locally,
or inside a devcontainer/remote host if you use one):

```bash
pip install trelix-mcp
trelix-mcp --help
```

The extension spawns it by name (`trelix-mcp`, no args) — it must be on
`PATH` for whichever Python environment the VS Code process resolves.

You'll also need a repo already indexed (`trelix index <path>`) before
`trelix.search`/`trelix.ask` return anything.

---

## Commands

| Command | What it does |
|---|---|
| `trelix: Search Codebase` (`trelix.search`) | Prompts for a query, calls the `search_code` MCP tool, shows results in a `QuickPick`. Picking a result opens the file and jumps to (and highlights) the matched symbol's line range. |
| `trelix: Ask about Code` (`trelix.ask`) | Prompts for a question, calls the `trelix-search` MCP prompt, renders the answer in a read-only Webview panel. |

## Settings

| Setting | Default | Purpose |
|---|---|---|
| `trelix.indexPath` | `""` (falls back to the first workspace folder) | Absolute path to the trelix-indexed repository to query. |

---

## Data flow — `trelix.search`

`search_code`'s real response envelope (see
`packages/trelix-mcp/src/trelix_mcp/server.py`) is:

```json
{
  "results": [
    {"file": "...", "symbol": "...", "kind": "...", "lines": "10-25",
     "score": 0.92, "source": "vector", "body": "...", "language": "python"}
  ],
  "next_cursor": 10,
  "total_available": 42
}
```

`src/mcp-client.ts`'s `TrelixMcpClient.search()` parses this exactly (result
keys are `symbol`/`file`/`kind`/`lines`/`score`/`source`/`body`/`language` —
there is no `symbol_name`/`file_path`), returning `{results, nextCursor,
totalAvailable}`. `lines` is a `"start-end"` 1-indexed string; `extension.ts`
parses it into a 0-indexed `vscode.Range` used both as the `showTextDocument`
selection and to reveal the range in the editor.

## Security notes

- The Webview created for `trelix.ask`'s answer sets `enableScripts: false`
  and an explicit CSP (`default-src 'none'`) with the answer text
  HTML-escaped before interpolation — the LLM-generated answer is untrusted
  content and is never treated as executable/renderable markup.
- `search_code` and `trelix-search` run against whatever `repo_path`/index
  the workspace is configured for — the extension does not sandbox or
  validate that path beyond what `trelix-mcp` itself enforces server-side.

---

## Building and testing locally

```bash
cd workspace-vscode
npm install
npm run typecheck   # tsc --noEmit
npm run build        # tsc (emits src/test/**) + esbuild (bundles dist/extension.js)
npm test             # downloads a VS Code test instance, runs the Mocha suite
```

`npm run watch` runs esbuild in watch mode for iterating on `src/extension.ts`
during development (press `F5` in VS Code to launch an Extension Development
Host against it).

## Packaging

```bash
npm run package   # vsce package -> trelix-vscode-<version>.vsix
```

`.vscodeignore` excludes `src/`, `tests/`, and source maps from the packaged
`.vsix` — only the bundled `dist/extension.js` (plus `package.json`,
`README.md`, etc.) ships to end users.
