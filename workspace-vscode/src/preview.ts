import * as vscode from "vscode";
import { SearchResult } from "./mcp-client";

/**
 * Serves search-result snippet bodies as read-only virtual documents, so
 * showTextDocument({preview:true}) renders them with real VS Code syntax
 * highlighting (inferred from the URI's fake path/extension) instead of a
 * hand-rolled Webview highlighter.
 */
export class SnippetPreviewProvider implements vscode.TextDocumentContentProvider {
  static readonly scheme = "trelix-preview";

  private readonly bodies = new Map<string, string>();
  private readonly onDidChangeEmitter = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.onDidChangeEmitter.event;

  /** Registers (or replaces) the body for one result and returns its virtual document URI. */
  uriFor(result: SearchResult): vscode.Uri {
    // Keep the real file's extension in the path so VS Code's
    // language-detection-by-extension picks the right grammar; disambiguate
    // same-named files across results via the query string.
    const uri = vscode.Uri.from({
      scheme: SnippetPreviewProvider.scheme,
      path: `/${result.file}`,
      query: `symbol=${result.symbol}&lines=${result.lines}`,
    });
    const key = uri.toString();
    const changed = this.bodies.get(key) !== result.body;
    this.bodies.set(key, result.body);
    if (changed) this.onDidChangeEmitter.fire(uri);
    return uri;
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.bodies.get(uri.toString()) ?? "";
  }

  dispose(): void {
    this.onDidChangeEmitter.dispose();
    this.bodies.clear();
  }
}
