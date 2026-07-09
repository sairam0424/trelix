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
