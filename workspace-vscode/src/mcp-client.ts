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
    const parsed = JSON.parse(text) as { results?: Array<{
      symbol_name?: string;
      file_path?: string;
      score?: number;
      body?: string;
    }> };
    const results: SearchResult[] = (parsed.results ?? []).map((r) => ({
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
