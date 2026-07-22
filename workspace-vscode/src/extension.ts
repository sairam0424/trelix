import * as path from "path";
import * as vscode from "vscode";
import { TrelixMcpClient, SearchResult } from "./mcp-client";

let client: TrelixMcpClient | null = null;

function getRepoPath(): string {
  const configured = vscode.workspace
    .getConfiguration("trelix")
    .get<string>("indexPath");
  if (configured && configured.length > 0) return configured;
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? "";
}

/** Parses the "start-end" line range search_code returns (1-indexed) into a 0-indexed vscode.Range. */
function parseLineRange(lines: string): vscode.Range | undefined {
  const match = /^(\d+)-(\d+)$/.exec(lines);
  if (!match) return undefined;
  const start = Math.max(0, parseInt(match[1], 10) - 1);
  const end = Math.max(0, parseInt(match[2], 10) - 1);
  return new vscode.Range(start, 0, end, 0);
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
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

  async function openResult(result: SearchResult, repoPath: string): Promise<void> {
    const absolutePath = vscode.Uri.file(
      path.isAbsolute(result.file) ? result.file : path.join(repoPath, result.file)
    );
    const range = parseLineRange(result.lines);
    const editor = await vscode.window.showTextDocument(absolutePath, {
      selection: range,
    });
    if (range) {
      editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
    }
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
          const page = await c.search(query, repoPath);
          if (page.results.length === 0) {
            vscode.window.showInformationMessage("trelix: no results found.");
            return;
          }
          const items = page.results.map((r) => ({
            label: r.symbol,
            description: `${r.file}:${r.lines} (${r.kind})`,
            detail: r.body.slice(0, 120),
            result: r,
          }));
          const picked = await vscode.window.showQuickPick(items, {
            matchOnDescription: true,
            matchOnDetail: true,
          });
          if (picked) {
            await openResult(picked.result, repoPath);
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
            { enableScripts: false, localResourceRoots: [] }
          );
          const nonce = String(Date.now());
          panel.webview.html = `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'nonce-${nonce}';">
<style nonce="${nonce}">body { font-family: monospace; white-space: pre-wrap; }</style>
</head>
<body>${escapeHtml(answer)}</body>
</html>`;
        }
      );
    })
  );
}

export async function deactivate(): Promise<void> {
  await client?.disconnect();
}
