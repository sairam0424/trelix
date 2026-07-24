/**
 * Thin wrapper around the trelix MCP server (trelix-mcp) over stdio transport.
 * Spawns `trelix-mcp` as a child process and communicates via JSON-RPC stdio.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

export interface SearchResult {
  symbol: string;
  file: string;
  kind: string;
  lines: string;
  score: number;
  source: string;
  body: string;
  language: string;
}

export interface SearchPage {
  results: SearchResult[];
  nextCursor: number | null;
  totalAvailable: number;
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

  /**
   * Matches search_code's real response shape exactly (see
   * packages/trelix-mcp/src/trelix_mcp/server.py): {results, next_cursor,
   * total_available}, with each result keyed by symbol/file/kind/lines/
   * score/source/body/language — not symbol_name/file_path, which don't
   * exist on this tool's response at all.
   */
  async search(query: string, repoPath: string, k = 10, cursor = 0): Promise<SearchPage> {
    if (!this.client) throw new Error("Not connected");
    const result = await this.client.callTool({
      name: "search_code",
      arguments: { query, repo_path: repoPath, k, cursor },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const text = content.find((c) => c.type === "text")?.text ?? "{}";
    const parsed = JSON.parse(text) as {
      results?: Array<{
        symbol?: string;
        file?: string;
        kind?: string;
        lines?: string;
        score?: number;
        source?: string;
        body?: string;
        language?: string;
      }>;
      next_cursor?: number | null;
      total_available?: number;
    };
    const results: SearchResult[] = (parsed.results ?? []).map((r) => ({
      symbol: r.symbol ?? "",
      file: r.file ?? "",
      kind: r.kind ?? "",
      lines: r.lines ?? "",
      score: r.score ?? 0,
      source: r.source ?? "",
      body: r.body ?? "",
      language: r.language ?? "",
    }));
    return {
      results,
      nextCursor: parsed.next_cursor ?? null,
      totalAvailable: parsed.total_available ?? results.length,
    };
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
