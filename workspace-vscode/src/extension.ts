import * as path from "path";
import * as vscode from "vscode";
import { TrelixMcpClient, SearchResult } from "./mcp-client";
import { SnippetPreviewProvider } from "./preview";
import { SearchController } from "./search-controller";

const SEARCH_DEBOUNCE_MS = 250;
const PAGE_SIZE = 10;

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

type ResultItem = vscode.QuickPickItem & { itemKind: "result"; result: SearchResult };
type LoadMoreItem = vscode.QuickPickItem & { itemKind: "loadMore" };
type SearchQuickPickItem = ResultItem | LoadMoreItem;

function toResultItem(r: SearchResult): ResultItem {
  return {
    itemKind: "result",
    label: r.symbol,
    description: `${r.file}:${r.lines} (${r.kind})`,
    detail: r.body.slice(0, 120).replace(/\s+/g, " "),
    result: r,
  };
}

const LOAD_MORE_ITEM: LoadMoreItem = {
  itemKind: "loadMore",
  label: "$(ellipsis) Load more results…",
  alwaysShow: true,
};

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  client = new TrelixMcpClient();

  const previewProvider = new SnippetPreviewProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(SnippetPreviewProvider.scheme, previewProvider),
    previewProvider
  );

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

  async function previewResult(result: SearchResult): Promise<void> {
    const uri = previewProvider.uriFor(result);
    await vscode.window.showTextDocument(uri, {
      preview: true,
      preserveFocus: true,
      viewColumn: vscode.ViewColumn.Beside,
    });
  }

  // Command: trelix.search — live-narrowing QuickPick: search-as-you-type
  // (debounced), snippet preview on highlight, "load more" pagination using
  // search_code's real cursor/next_cursor fields. The debounce/pagination
  // state machine lives in SearchController (testable without the Extension
  // Host); this closure is just vscode.QuickPick wiring on top of it.
  context.subscriptions.push(
    vscode.commands.registerCommand("trelix.search", async () => {
      const c = await ensureConnected();
      const repoPath = getRepoPath();

      const qp = vscode.window.createQuickPick<SearchQuickPickItem>();
      qp.placeholder = "Search your codebase with trelix — e.g. JWT token validation middleware";
      qp.matchOnDescription = true;
      qp.matchOnDetail = true;

      const controller = new SearchController({
        debounceMs: SEARCH_DEBOUNCE_MS,
        pageSize: PAGE_SIZE,
        search: (query, k, cursor) => c.search(query, repoPath, k, cursor),
        onItems: (results, hasMore) => {
          const items = results.map(toResultItem);
          qp.items = hasMore ? [...items, LOAD_MORE_ITEM] : items;
        },
        onError: (err) => {
          vscode.window.showErrorMessage(`trelix search failed: ${err instanceof Error ? err.message : String(err)}`);
        },
        onBusyChange: (busy) => {
          qp.busy = busy;
        },
      });

      qp.onDidChangeValue((value) => controller.onValueChange(value));

      qp.onDidChangeActive((active) => {
        const item = active[0];
        if (item && item.itemKind === "result") {
          void previewResult(item.result);
        }
      });

      qp.onDidAccept(async () => {
        const picked = qp.selectedItems[0];
        if (!picked) return;
        if (picked.itemKind === "loadMore") {
          await controller.loadMore();
          return;
        }
        qp.hide();
        await openResult(picked.result, repoPath);
      });

      qp.onDidHide(() => {
        controller.dispose();
        qp.dispose();
      });

      qp.show();
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
