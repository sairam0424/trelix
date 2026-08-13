import * as vscode from "vscode";
import { TrelixMcpClient, Symbol } from "./mcp-client";

/**
 * Known limitation (documented, not fixed here): get_symbol does an
 * ambiguous bare-name lookup server-side with no disambiguation when
 * multiple symbols share a name — same class of bug fixed for call/type
 * edges in v2.12.0, not yet fixed for get_symbol itself (separate, larger
 * scope). A hover over an ambiguous name may show the wrong symbol.
 */
export class TrelixHoverProvider implements vscode.HoverProvider {
  private readonly cache = new Map<string, Symbol | null>();

  constructor(
    private readonly getClient: () => Promise<TrelixMcpClient>,
    private readonly getRepoPath: () => string
  ) {}

  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    token: vscode.CancellationToken
  ): Promise<vscode.Hover | undefined> {
    const range = document.getWordRangeAtPosition(position);
    if (!range) return undefined;
    const word = document.getText(range);
    if (!word) return undefined;

    const repoPath = this.getRepoPath();
    const cacheKey = `${repoPath}::${word}`;

    let symbol: Symbol | null;
    if (this.cache.has(cacheKey)) {
      symbol = this.cache.get(cacheKey)!;
    } else {
      try {
        const client = await this.getClient();
        symbol = await client.getSymbol(word, repoPath);
      } catch {
        symbol = null; // silent no-op hover on any error
      }
      if (token.isCancellationRequested) return undefined;
      this.cache.set(cacheKey, symbol);
    }

    if (!symbol) return undefined;

    const md = new vscode.MarkdownString();
    md.appendCodeblock(symbol.signature || symbol.name, symbol.language || undefined);
    if (symbol.docstring) {
      md.appendMarkdown(`\n\n${symbol.docstring}`);
    }
    md.appendMarkdown(`\n\n---\n*${symbol.file}:${symbol.lineStart}-${symbol.lineEnd}*`);
    return new vscode.Hover(md, range);
  }

  /** Exposed for tests — cache has no TTL, cleared explicitly if ever needed. */
  clearCache(): void {
    this.cache.clear();
  }
}
