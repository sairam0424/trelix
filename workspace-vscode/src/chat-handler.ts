import * as path from "path";
import * as vscode from "vscode";
import { TrelixMcpClient } from "./mcp-client";

/**
 * Minimal subset of vscode.ChatResponseStream this handler depends on.
 *
 * Deliberately NOT typed as vscode.ChatResponseStream: keeping a local
 * interface lets the handler be unit-tested with a plain fake stream (no
 * Extension Host, no proposed-API surface) while staying structurally
 * assignable — vscode.ChatResponseStream has all three of these methods, so a
 * real stream satisfies this interface at the registration site.
 */
export interface ChatStream {
    progress(message: string): void;
    markdown(value: string): void;
    reference(value: vscode.Uri | vscode.Location): void;
}

/** Minimal subset of vscode.ChatRequest we read. */
export interface ChatRequestLike {
    /** The user's prompt with any leading slash-command already stripped. */
    prompt: string;
    /** The slash command (e.g. "search"), or undefined for a plain @trelix ask. */
    command?: string;
}

/** Minimal subset of vscode.ChatContext — only the history length is used. */
export interface ChatContextLike {
    history: ReadonlyArray<unknown>;
}

export interface ChatHandlerDeps {
    getClient: () => Promise<TrelixMcpClient>;
    getRepoPath: () => string;
}

/** The pure handler signature — assignable to vscode.ChatRequestHandler. */
export type TrelixChatHandler = (
    request: ChatRequestLike,
    context: ChatContextLike,
    stream: ChatStream,
    token: vscode.CancellationToken,
) => Promise<void>;

const SEARCH_LIMIT = 10;

/**
 * Session-continuity limitation (documented honestly): the 1.95 chat API gives
 * the handler no stable thread/conversation id, so we key trelix sessions by a
 * single constant. Consequence — two chat threads open at the same time SHARE
 * one trelix agent session. The only per-conversation signal available is
 * context.history: an empty history means a fresh conversation (new session);
 * a non-empty history reuses the stored session id (threaded through ask()).
 */
const THREAD_KEY = "default";

function errorMessage(err: unknown): string {
    const msg = err instanceof Error ? err.message : String(err);
    return `⚠️ trelix could not complete that request: ${msg}`;
}

/** Text of the active editor's selection, or "" if there is none. */
function activeSelectionText(): string {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) return "";
    return editor.document.getText(editor.selection);
}

/**
 * Build a clickable reference (Location when we can parse a line range from the
 * "start-end" string search_code returns, else a bare file Uri). Line numbers
 * from the server are 1-indexed; vscode.Range is 0-indexed.
 */
function toReference(
    file: string,
    lineStart: number,
    lineEnd: number,
    repoPath: string,
): vscode.Uri | vscode.Location {
    const abs = path.isAbsolute(file) ? file : path.join(repoPath, file);
    const uri = vscode.Uri.file(abs);
    if (lineStart <= 0) return uri;
    const s = Math.max(0, lineStart - 1);
    const e = Math.max(s, lineEnd - 1);
    return new vscode.Location(uri, new vscode.Range(s, 0, e, 0));
}

function parseLineRange(lines: string): [number, number] {
    const m = /^(\d+)-(\d+)$/.exec(lines);
    if (!m) return [0, 0];
    return [parseInt(m[1], 10), parseInt(m[2], 10)];
}

async function handleAsk(
    client: TrelixMcpClient,
    repoPath: string,
    prompt: string,
    context: ChatContextLike,
    stream: ChatStream,
    sessions: Map<string, string>,
): Promise<void> {
    if (!prompt) {
        stream.markdown(
            "Ask me about this codebase — e.g. `@trelix how does authentication work?`\n\n" +
                "Or use `/search`, `/explain`, `/impact`.",
        );
        return;
    }
    // The agentic ReAct loop can take a while — show progress before the call.
    stream.progress("Retrieving relevant code…");

    const isNewThread = context.history.length === 0;
    const prior = isNewThread ? undefined : sessions.get(THREAD_KEY);
    const result = await client.ask(prompt, repoPath, prior);
    if (result.sessionId) sessions.set(THREAD_KEY, result.sessionId);

    stream.markdown(result.answer || "_trelix returned no answer._");
}

async function handleSearch(
    client: TrelixMcpClient,
    repoPath: string,
    prompt: string,
    stream: ChatStream,
): Promise<void> {
    if (!prompt) {
        stream.markdown("Usage: `/search <natural-language query>`");
        return;
    }
    stream.progress("Searching the codebase…");
    const page = await client.search(prompt, repoPath, SEARCH_LIMIT, 0);
    if (page.results.length === 0) {
        stream.markdown(`No results for _${prompt}_.`);
        return;
    }

    const lines: string[] = [`Found ${page.results.length} result(s):`, ""];
    for (const r of page.results) {
        const [start, end] = parseLineRange(r.lines);
        stream.reference(toReference(r.file, start, end, repoPath));
        lines.push(`- \`${r.symbol}\` — ${r.file}:${r.lines} (${r.kind})`);
    }
    stream.markdown(lines.join("\n"));
}

async function handleExplain(
    client: TrelixMcpClient,
    repoPath: string,
    prompt: string,
    stream: ChatStream,
): Promise<void> {
    const selection = activeSelectionText();
    const parts: string[] = [];
    if (prompt) parts.push(prompt);
    if (selection) parts.push("```\n" + selection + "\n```");

    if (parts.length === 0) {
        stream.markdown(
            "Select some code in the editor (or add a question) then run `/explain`.",
        );
        return;
    }
    stream.progress("Explaining the selection…");
    const query = `Explain the following in the context of this codebase:\n\n${parts.join("\n\n")}`;
    const result = await client.ask(query, repoPath);
    stream.markdown(result.answer || "_trelix returned no explanation._");
}

async function handleImpact(
    client: TrelixMcpClient,
    repoPath: string,
    prompt: string,
    stream: ChatStream,
): Promise<void> {
    const symbolName = prompt || activeSelectionText().trim();
    if (!symbolName) {
        stream.markdown(
            "Usage: `/impact <symbol name>` (or select a symbol, then run `/impact`).",
        );
        return;
    }
    stream.progress(`Analyzing the blast radius of ${symbolName}…`);
    const entries = await client.blastRadius(symbolName, repoPath);
    if (entries.length === 0) {
        stream.markdown(`Nothing depends on \`${symbolName}\`.`);
        return;
    }

    const lines: string[] = [
        `\`${symbolName}\` has ${entries.length} dependent(s):`,
        "",
    ];
    for (const e of entries) {
        stream.reference(
            toReference(e.file, e.lineStart, e.lineStart, repoPath),
        );
        lines.push(`- \`${e.symbol}\` — ${e.file}:${e.lineStart} (${e.kind})`);
    }
    stream.markdown(lines.join("\n"));
}

/**
 * Build the trelix chat request handler. Pure and dependency-injected: it never
 * touches vscode.chat, never throws to the caller (any client error is rendered
 * as markdown), and keeps its own per-instance session map so tests get an
 * isolated instance each time.
 */
export function createTrelixChatHandler(
    deps: ChatHandlerDeps,
): TrelixChatHandler {
    const { getClient, getRepoPath } = deps;
    const sessions = new Map<string, string>();

    return async function handler(request, context, stream, _token) {
        const command = request.command;
        const prompt = (request.prompt ?? "").trim();

        try {
            const client = await getClient();
            const repoPath = getRepoPath();

            switch (command) {
                case "search":
                    await handleSearch(client, repoPath, prompt, stream);
                    return;
                case "explain":
                    await handleExplain(client, repoPath, prompt, stream);
                    return;
                case "impact":
                    await handleImpact(client, repoPath, prompt, stream);
                    return;
                default:
                    await handleAsk(
                        client,
                        repoPath,
                        prompt,
                        context,
                        stream,
                        sessions,
                    );
            }
        } catch (err) {
            // Never throw/reject out of a chat handler — render the error.
            stream.markdown(errorMessage(err));
        }
    };
}
