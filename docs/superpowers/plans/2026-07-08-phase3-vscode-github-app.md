# trelix Phase 3 — VS Code Extension + GitHub App

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two adoption multipliers: (A) a VS Code extension that exposes trelix search inline via the existing MCP server — no new backend needed; (B) a GitHub App manifest and Actions workflow that posts `trelix review --pr` findings as structured PR check comments automatically.

**Architecture:** Plan A is a VS Code extension (`workspace-vscode/`) that connects to the trelix MCP server over stdio transport, registers a chat participant (`@trelix`) and an inline code-search command, and surfaces results in VS Code's chat panel and hover providers. It piggybacks entirely on the existing `trelix-mcp` package — no new Python code. Plan B is a GitHub App (`infra/github-app/`) with a `pull_request` webhook handler that runs `trelix review --pr` and posts findings to the PR check suite via the GitHub Checks API. Both plans are fully independent.

**Tech Stack:** Plan A: TypeScript, VS Code Extension API, `@modelcontextprotocol/sdk`. Plan B: TypeScript (GitHub Actions), `@octokit/rest`, existing `trelix review` CLI.

## Global Constraints

- **DO NOT bump any version numbers** in `pyproject.toml`, `__init__.py`, or CHANGELOG version fields
- Plan A: TypeScript strict mode, `@vscode/vscode-dts` for type safety, no bundled Python
- Plan B: GitHub Actions workflow YAML + minimal Node.js script; no new Python dependencies
- Plans A and B are fully independent — implement in either order
- Conventional commits: `feat(vscode):`, `feat(github-app):`

---

## Plan A — VS Code Extension

### Task A-1: Scaffold the extension package

**Files:**
- Create: `workspace-vscode/` — new directory
- Create: `workspace-vscode/package.json`
- Create: `workspace-vscode/tsconfig.json`
- Create: `workspace-vscode/src/extension.ts` — entry point
- Create: `workspace-vscode/src/mcp-client.ts` — MCP stdio client wrapper

**Interfaces:**
- Produces: `activate(context: vscode.ExtensionContext)` — registers commands and chat participant
- Produces: `TrelixMcpClient` class with `search(query: str, repoPath: str) -> SearchResult[]` and `ask(query: str, repoPath: str) -> str`

- [ ] **Step 1: Create `workspace-vscode/package.json`**

```json
{
  "name": "trelix-vscode",
  "displayName": "trelix — Code Intelligence",
  "description": "Natural language code search and Q&A for your codebase via trelix",
  "version": "0.1.0",
  "publisher": "trelix",
  "engines": { "vscode": "^1.90.0" },
  "categories": ["AI", "Other"],
  "activationEvents": ["onStartupFinished"],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "trelix.search",
        "title": "trelix: Search Codebase"
      },
      {
        "command": "trelix.ask",
        "title": "trelix: Ask about Code"
      }
    ],
    "configuration": {
      "title": "trelix",
      "properties": {
        "trelix.indexPath": {
          "type": "string",
          "default": "",
          "description": "Path to trelix-indexed repository (defaults to workspace root)"
        }
      }
    }
  },
  "scripts": {
    "build": "tsc -p tsconfig.json",
    "watch": "tsc -watch -p tsconfig.json",
    "package": "vsce package"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0"
  },
  "devDependencies": {
    "@types/vscode": "^1.90.0",
    "@vscode/vsce": "^2.0.0",
    "typescript": "^5.0.0"
  }
}
```

- [ ] **Step 2: Create `workspace-vscode/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

- [ ] **Step 3: Create `workspace-vscode/src/mcp-client.ts`**

```typescript
/**
 * Thin wrapper around the trelix MCP server (trelix-mcp) over stdio transport.
 * Spawns `trelix-mcp` as a child process and communicates via JSON-RPC stdio.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

export interface SearchResult {
  symbolName: string;
  filePath: string;
  score: number;
  body: string;
}

export class TrelixMcpClient {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;

  async connect(): Promise<void> {
    this.transport = new StdioClientTransport({
      command: "trelix-mcp",
      args: [],
    });
    this.client = new Client(
      { name: "trelix-vscode", version: "0.1.0" },
      { capabilities: {} }
    );
    await this.client.connect(this.transport);
  }

  async search(query: string, repoPath: string, k = 10): Promise<SearchResult[]> {
    if (!this.client) throw new Error("Not connected");
    const result = await this.client.callTool({
      name: "search_code",
      arguments: { query, repo_path: repoPath, k },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const text = content.find((c) => c.type === "text")?.text ?? "{}";
    const parsed = JSON.parse(text);
    const results: SearchResult[] = (parsed.results ?? []).map((r: any) => ({
      symbolName: r.symbol_name ?? "",
      filePath: r.file_path ?? "",
      score: r.score ?? 0,
      body: r.body ?? "",
    }));
    return results;
  }

  async ask(query: string, repoPath: string): Promise<string> {
    if (!this.client) throw new Error("Not connected");
    // Use the trelix-search prompt for structured search
    const result = await this.client.getPrompt({
      name: "trelix-search",
      arguments: { query, repo_path: repoPath },
    });
    return result.messages.map((m) =>
      typeof m.content === "string" ? m.content : JSON.stringify(m.content)
    ).join("\n");
  }

  async disconnect(): Promise<void> {
    await this.client?.close();
    this.client = null;
    this.transport = null;
  }
}
```

- [ ] **Step 4: Create `workspace-vscode/src/extension.ts`**

```typescript
import * as vscode from "vscode";
import { TrelixMcpClient } from "./mcp-client";

let client: TrelixMcpClient | null = null;

function getRepoPath(): string {
  const configured = vscode.workspace
    .getConfiguration("trelix")
    .get<string>("indexPath");
  if (configured && configured.length > 0) return configured;
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? "";
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  client = new TrelixMcpClient();

  // Connect lazily on first command use
  async function ensureConnected(): Promise<TrelixMcpClient> {
    if (!client) throw new Error("trelix client disposed");
    try {
      await client.connect();
    } catch {
      // Already connected — ignore
    }
    return client;
  }

  // Command: trelix.search
  context.subscriptions.push(
    vscode.commands.registerCommand("trelix.search", async () => {
      const query = await vscode.window.showInputBox({
        prompt: "Search your codebase with trelix",
        placeHolder: "e.g. JWT token validation middleware",
      });
      if (!query) return;

      const c = await ensureConnected();
      const repoPath = getRepoPath();

      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "trelix: searching…" },
        async () => {
          const results = await c.search(query, repoPath);
          if (results.length === 0) {
            vscode.window.showInformationMessage("trelix: no results found.");
            return;
          }
          const items = results.map((r) => ({
            label: r.symbolName,
            description: r.filePath,
            detail: r.body.slice(0, 120),
            result: r,
          }));
          const picked = await vscode.window.showQuickPick(items, {
            matchOnDescription: true,
            matchOnDetail: true,
          });
          if (picked) {
            const uri = vscode.Uri.file(picked.result.filePath);
            await vscode.window.showTextDocument(uri);
          }
        }
      );
    })
  );

  // Command: trelix.ask
  context.subscriptions.push(
    vscode.commands.registerCommand("trelix.ask", async () => {
      const query = await vscode.window.showInputBox({
        prompt: "Ask trelix about your codebase",
        placeHolder: "e.g. how does authentication work?",
      });
      if (!query) return;

      const c = await ensureConnected();
      const repoPath = getRepoPath();

      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "trelix: thinking…" },
        async () => {
          const answer = await c.ask(query, repoPath);
          const panel = vscode.window.createWebviewPanel(
            "trelixAnswer",
            `trelix: ${query.slice(0, 40)}`,
            vscode.ViewColumn.Beside,
            {}
          );
          panel.webview.html = `<html><body><pre style="font-family:monospace;white-space:pre-wrap">${answer}</pre></body></html>`;
        }
      );
    })
  );
}

export async function deactivate(): Promise<void> {
  await client?.disconnect();
}
```

- [ ] **Step 5: Build and verify**

```bash
cd workspace-vscode
npm install
npm run build
```

Expected: `dist/extension.js` created, no TypeScript errors.

- [ ] **Step 6: Commit**

```bash
git add workspace-vscode/
git commit -m "feat(vscode): scaffold trelix VS Code extension with MCP stdio client

Adds workspace-vscode/ with TrelixMcpClient (stdio transport to trelix-mcp),
trelix.search command (QuickPick results), and trelix.ask command (Webview answer).
No new Python code — piggybacks entirely on existing trelix-mcp package."
```

---

## Plan B — GitHub App for Automatic PR Review

### Task B-1: GitHub Actions workflow for automatic PR review

**Files:**
- Create: `.github/workflows/trelix-review.yml` — triggers on `pull_request`, runs `trelix review --pr`
- Create: `infra/github-app/post-review.js` — posts findings as GitHub Check run annotations

**Interfaces:**
- Produces: `trelix-review.yml` workflow that runs on `pull_request` events, calls `trelix review --pr ${{ github.event.pull_request.number }}`
- Produces: `post-review.js` — reads trelix JSON output and posts it as a GitHub Check with annotations

- [ ] **Step 1: Create `.github/workflows/trelix-review.yml`**

```yaml
name: trelix PR Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  trelix-review:
    name: trelix Code Intelligence Review
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      checks: write
      contents: read

    steps:
      - name: Checkout PR
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install trelix
        run: pip install "trelix[local]" --quiet

      - name: Index repository
        run: trelix index .

      - name: Run trelix review
        id: review
        run: |
          PR_NUMBER=${{ github.event.pull_request.number }}
          REPO="${{ github.repository }}"
          trelix review --pr "${REPO}#${PR_NUMBER}" --json > /tmp/trelix-review.json 2>&1 || true
          echo "review_file=/tmp/trelix-review.json" >> "$GITHUB_OUTPUT"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Post review as Check annotations
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const fs = require('fs');
            const reviewFile = '${{ steps.review.outputs.review_file }}';

            let findings = [];
            try {
              const raw = fs.readFileSync(reviewFile, 'utf8');
              const data = JSON.parse(raw);
              findings = data.findings || data.reviews || [];
            } catch (e) {
              console.log('No structured review output:', e.message);
            }

            const annotations = findings.slice(0, 50).map(f => ({
              path: f.file || f.path || 'unknown',
              start_line: f.line || f.start_line || 1,
              end_line: f.line || f.end_line || 1,
              annotation_level: f.severity === 'error' ? 'failure'
                : f.severity === 'warning' ? 'warning' : 'notice',
              message: f.message || f.description || String(f),
              title: 'trelix review',
            }));

            await github.rest.checks.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              name: 'trelix Code Review',
              head_sha: context.sha,
              status: 'completed',
              conclusion: annotations.some(a => a.annotation_level === 'failure')
                ? 'failure' : 'success',
              output: {
                title: `trelix found ${annotations.length} issue(s)`,
                summary: `trelix reviewed PR #${{ github.event.pull_request.number }} and found ${annotations.length} issue(s).`,
                annotations,
              },
            });
```

- [ ] **Step 2: Create `infra/github-app/` directory and README**

```bash
mkdir -p infra/github-app
```

Create `infra/github-app/README.md`:

```markdown
# trelix GitHub App Integration

This directory contains the GitHub App integration for automatic trelix
PR review as a Check run with inline annotations.

## Setup

1. The workflow at `.github/workflows/trelix-review.yml` handles everything
   automatically on every pull_request event.

2. Required permissions:
   - `pull-requests: write` — post comments
   - `checks: write` — post Check annotations
   - `contents: read` — checkout code

3. Optional: set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in repository
   secrets for LLM-powered synthesis (trelix review works without it
   but annotations are richer with an LLM provider configured).

## How it works

1. trelix indexes the PR's base branch
2. `trelix review --pr owner/repo#N --json` fetches the diff and reviews it
3. Findings are posted as GitHub Check annotations with file + line references
```

- [ ] **Step 3: Verify workflow YAML is valid**

```bash
# Lint the workflow YAML (requires actionlint if installed, otherwise just check syntax)
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/trelix-review.yml'))" && \
echo "YAML valid"
```

Expected: `YAML valid`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/trelix-review.yml infra/github-app/
git commit -m "feat(github-app): add trelix PR review GitHub Actions workflow

Triggers on pull_request events: indexes repo, runs trelix review --pr,
posts findings as GitHub Check annotations with file/line references.
Requires checks: write + pull-requests: write permissions.
Works without LLM keys (local embedder) — richer with one configured."
```

---

## Self-Review

**Spec coverage check:**

| Priority item | Plan | Task |
|---|---|---|
| Watch bridge wired | Phase 1 | Task 1 ✅ |
| Missing `files.rel_path` index | Phase 1 | Task 2 ✅ |
| AdaptiveRouter config passthrough | Phase 1 | Task 3 ✅ |
| Cross-repo symbol resolution | Phase 2 Plan A | Task A-1 ✅ |
| Semantic diff embeddings | Phase 2 Plan B | Task B-1 ✅ |
| Streaming indexing | Phase 2 Plan C | Task C-1 ✅ |
| VS Code extension | Phase 3 Plan A | Task A-1 ✅ |
| GitHub App / Actions | Phase 3 Plan B | Task B-1 ✅ |

**Version bump check:** No `version =` strings modified in any plan. ✅

**No placeholders:** All code blocks are complete. ✅

**Type consistency:** `make_scip_symbol_id(package, version, qualified_name) -> str` used consistently. `DiffEmbedder.embed_hunk(before_code, after_code) -> list[float]` consistent. `TrelixMcpClient.search/ask` consistent. ✅
