import * as assert from "assert";
import * as vscode from "vscode";
import {
    createTrelixChatHandler,
    ChatStream,
    ChatRequestLike,
    ChatContextLike,
} from "../../chat-handler";
import { TrelixMcpClient } from "../../mcp-client";

type Recorded =
    | { kind: "progress"; value: string }
    | { kind: "markdown"; value: string }
    | { kind: "reference"; value: vscode.Uri | vscode.Location };

/** Fake stream that records every call in order — mirrors the real stream's shape. */
class FakeStream implements ChatStream {
    readonly calls: Recorded[] = [];
    progress(value: string): void {
        this.calls.push({ kind: "progress", value });
    }
    markdown(value: string): void {
        this.calls.push({ kind: "markdown", value });
    }
    reference(value: vscode.Uri | vscode.Location): void {
        this.calls.push({ kind: "reference", value });
    }
    kinds(): string[] {
        return this.calls.map((c) => c.kind);
    }
    references(): Recorded[] {
        return this.calls.filter((c) => c.kind === "reference");
    }
    markdownText(): string {
        return this.calls
            .filter((c) => c.kind === "markdown")
            .map((c) => c.value as string)
            .join("\n");
    }
}

const NO_CANCEL: vscode.CancellationToken = {
    isCancellationRequested: false,
    onCancellationRequested: () => ({ dispose: () => undefined }),
};

function request(overrides: Partial<ChatRequestLike> = {}): ChatRequestLike {
    return { prompt: "", command: undefined, ...overrides };
}

function ctx(history: unknown[] = []): ChatContextLike {
    return { history };
}

const DEPS_REPO = () => "/repo";

suite("createTrelixChatHandler", () => {
    test("plain ask emits progress BEFORE markdown and renders the answer", async () => {
        const fakeClient = {
            ask: async () => ({
                answer: "Auth is handled by AuthService.validate_token.",
                sessionId: "sess-1",
                turnCount: 2,
            }),
        } as unknown as TrelixMcpClient;

        const handler = createTrelixChatHandler({
            getClient: async () => fakeClient,
            getRepoPath: DEPS_REPO,
        });
        const stream = new FakeStream();

        await handler(
            request({ prompt: "how does auth work?" }),
            ctx([]),
            stream,
            NO_CANCEL,
        );

        const progressIdx = stream.calls.findIndex(
            (c) => c.kind === "progress",
        );
        const markdownIdx = stream.calls.findIndex(
            (c) => c.kind === "markdown",
        );
        assert.ok(progressIdx >= 0, "expected a progress call");
        assert.ok(markdownIdx >= 0, "expected a markdown call");
        assert.ok(
            progressIdx < markdownIdx,
            "progress must be emitted before markdown",
        );
        assert.ok(
            stream.markdownText().includes("AuthService.validate_token"),
            "answer should be rendered as markdown",
        );
    });

    test("/search emits exactly one reference per result plus a markdown list", async () => {
        const fakeClient = {
            search: async () => ({
                results: [
                    {
                        symbol: "validate_token",
                        file: "src/auth.py",
                        kind: "function",
                        lines: "10-25",
                        score: 0.9,
                        source: "vector",
                        body: "",
                        language: "python",
                    },
                    {
                        symbol: "issue_token",
                        file: "src/auth.py",
                        kind: "function",
                        lines: "30-40",
                        score: 0.8,
                        source: "bm25",
                        body: "",
                        language: "python",
                    },
                ],
                nextCursor: null,
                totalAvailable: 2,
            }),
        } as unknown as TrelixMcpClient;

        const handler = createTrelixChatHandler({
            getClient: async () => fakeClient,
            getRepoPath: DEPS_REPO,
        });
        const stream = new FakeStream();

        await handler(
            request({ prompt: "token", command: "search" }),
            ctx([]),
            stream,
            NO_CANCEL,
        );

        assert.strictEqual(
            stream.references().length,
            2,
            "one reference per result",
        );
        assert.ok(stream.markdownText().includes("validate_token"));
        assert.ok(stream.markdownText().includes("issue_token"));
    });

    test("a thrown client error is rendered as markdown and never rejects", async () => {
        const fakeClient = {
            ask: async () => {
                throw new Error("agent loop failed");
            },
        } as unknown as TrelixMcpClient;

        const handler = createTrelixChatHandler({
            getClient: async () => fakeClient,
            getRepoPath: DEPS_REPO,
        });
        const stream = new FakeStream();

        await assert.doesNotReject(() =>
            handler(
                request({ prompt: "explode please" }),
                ctx([]),
                stream,
                NO_CANCEL,
            ),
        );
        assert.ok(
            stream.markdownText().includes("agent loop failed"),
            "the error message should surface in markdown",
        );
    });

    test("empty prompt short-circuits with no MCP call", async () => {
        let askCalls = 0;
        const fakeClient = {
            ask: async () => {
                askCalls++;
                return { answer: "", sessionId: "", turnCount: 0 };
            },
        } as unknown as TrelixMcpClient;

        const handler = createTrelixChatHandler({
            getClient: async () => fakeClient,
            getRepoPath: DEPS_REPO,
        });
        const stream = new FakeStream();

        await handler(request({ prompt: "   " }), ctx([]), stream, NO_CANCEL);

        assert.strictEqual(
            askCalls,
            0,
            "ask() must not be called for an empty prompt",
        );
        assert.ok(
            stream.kinds().includes("markdown"),
            "a help/usage markdown should still be shown",
        );
        assert.ok(
            !stream.kinds().includes("progress"),
            "no progress spinner for an empty prompt",
        );
    });

    test("history.length===0 starts a new session; a non-empty history reuses it", async () => {
        const seenSessions: (string | undefined)[] = [];
        const fakeClient = {
            ask: async (_q: string, _repo: string, sessionId?: string) => {
                seenSessions.push(sessionId);
                return { answer: "ok", sessionId: "sess-1", turnCount: 1 };
            },
        } as unknown as TrelixMcpClient;

        // Same handler instance across both turns so the session map persists.
        const handler = createTrelixChatHandler({
            getClient: async () => fakeClient,
            getRepoPath: DEPS_REPO,
        });

        await handler(
            request({ prompt: "first question" }),
            ctx([]),
            new FakeStream(),
            NO_CANCEL,
        );
        await handler(
            request({ prompt: "follow-up" }),
            ctx([{ turn: 1 }]),
            new FakeStream(),
            NO_CANCEL,
        );

        assert.strictEqual(
            seenSessions[0],
            undefined,
            "first turn (empty history) must start with no session id",
        );
        assert.strictEqual(
            seenSessions[1],
            "sess-1",
            "second turn (non-empty history) must reuse the resolved session id",
        );
    });
});
