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

export interface Symbol {
    name: string;
    qualifiedName: string;
    kind: string;
    file: string;
    lineStart: number;
    lineEnd: number;
    signature: string;
    docstring: string;
    body: string;
    language: string;
}

/** Result envelope of the ask_agent tool (see server.py:730). */
export interface AskResult {
    answer: string;
    sessionId: string;
    turnCount: number;
}

/** One dependent-symbol entry returned by the blast_radius tool (see server.py:381). */
export interface BlastRadiusEntry {
    file: string;
    symbol: string;
    kind: string;
    lineStart: number;
    language: string;
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
            { capabilities: {} },
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
    async search(
        query: string,
        repoPath: string,
        k = 10,
        cursor = 0,
    ): Promise<SearchPage> {
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

    /**
     * Matches get_symbol's real response shape exactly (see
     * packages/trelix-mcp/src/trelix_mcp/server.py): a single nullable dict
     * keyed by name/qualified_name/kind/file/line_start/line_end/signature/
     * docstring/body/language — not a paginated {results, next_cursor} page
     * like search_code. Returns null both when the server returns null (no
     * match) and when the JSON body itself parses to null/undefined.
     */
    async getSymbol(
        qualifiedName: string,
        repoPath: string,
    ): Promise<Symbol | null> {
        if (!this.client) throw new Error("Not connected");
        const result = await this.client.callTool({
            name: "get_symbol",
            arguments: { qualified_name: qualifiedName, repo_path: repoPath },
        });
        const content = result.content as Array<{ type: string; text: string }>;
        const text = content.find((c) => c.type === "text")?.text ?? "null";
        const parsed = JSON.parse(text) as {
            name?: string;
            qualified_name?: string;
            kind?: string;
            file?: string;
            line_start?: number;
            line_end?: number;
            signature?: string;
            docstring?: string;
            body?: string;
            language?: string;
        } | null;
        if (!parsed) return null;
        return {
            name: parsed.name ?? "",
            qualifiedName: parsed.qualified_name ?? "",
            kind: parsed.kind ?? "",
            file: parsed.file ?? "",
            lineStart: parsed.line_start ?? 0,
            lineEnd: parsed.line_end ?? 0,
            signature: parsed.signature ?? "",
            docstring: parsed.docstring ?? "",
            body: parsed.body ?? "",
            language: parsed.language ?? "",
        };
    }

    /**
     * Calls the ask_agent TOOL — the multi-turn ReAct agentic loop — not the
     * trelix-search PROMPT. getPrompt returns an interpolated prompt template,
     * NOT an answer; ask_agent actually runs the loop and returns
     * {answer, session_id, turn_count} (see server.py:730). Mirrors search()'s
     * callTool + JSON.parse pattern. Pass sessionId to resume a prior
     * conversation; the resolved session_id is surfaced back for follow-ups.
     */
    async ask(
        query: string,
        repoPath: string,
        sessionId?: string,
    ): Promise<AskResult> {
        if (!this.client) throw new Error("Not connected");
        const result = await this.client.callTool({
            name: "ask_agent",
            arguments: { query, repo_path: repoPath, session_id: sessionId },
        });
        const content = result.content as Array<{ type: string; text: string }>;
        const text = content.find((c) => c.type === "text")?.text ?? "{}";
        const parsed = JSON.parse(text) as {
            answer?: string;
            session_id?: string;
            turn_count?: number;
        };
        return {
            answer: parsed.answer ?? "",
            sessionId: parsed.session_id ?? "",
            turnCount: parsed.turn_count ?? 0,
        };
    }

    /**
     * Impact analysis: which symbols depend on (call/import) `symbolName`.
     * A dedicated blast_radius TOOL exists (server.py:381) — not just the
     * trelix-blast-radius PROMPT — so we call the tool directly and JSON.parse
     * its array of {file, symbol, kind, line_start, language} entries. Same
     * callTool pattern as search()/getSymbol().
     */
    async blastRadius(
        symbolName: string,
        repoPath: string,
    ): Promise<BlastRadiusEntry[]> {
        if (!this.client) throw new Error("Not connected");
        const result = await this.client.callTool({
            name: "blast_radius",
            arguments: { symbol_name: symbolName, repo_path: repoPath },
        });
        const content = result.content as Array<{ type: string; text: string }>;
        const text = content.find((c) => c.type === "text")?.text ?? "[]";
        const parsed = JSON.parse(text) as Array<{
            file?: string;
            symbol?: string;
            kind?: string;
            line_start?: number;
            language?: string;
        }> | null;
        return (parsed ?? []).map((r) => ({
            file: r.file ?? "",
            symbol: r.symbol ?? "",
            kind: r.kind ?? "",
            lineStart: r.line_start ?? 0,
            language: r.language ?? "",
        }));
    }

    async disconnect(): Promise<void> {
        await this.client?.close();
        this.client = null;
        this.transport = null;
    }
}
