import * as assert from "assert";
import { TrelixMcpClient } from "../../mcp-client";

/** Injects a mocked MCP Client onto a TrelixMcpClient without a real stdio connection. */
function withMockedClient(
    client: TrelixMcpClient,
    mock: { callTool?: Function; getPrompt?: Function },
): void {
    (client as unknown as { client: unknown }).client = mock;
}

suite("TrelixMcpClient.search", () => {
    test("parses the real search_code envelope (symbol/file/kind/lines/... keys, not symbol_name/file_path)", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => ({
                content: [
                    {
                        type: "text",
                        text: JSON.stringify({
                            results: [
                                {
                                    file: "src/auth.py",
                                    symbol: "validate_token",
                                    kind: "function",
                                    lines: "10-25",
                                    score: 0.92,
                                    source: "vector",
                                    body: "def validate_token(): ...",
                                    language: "python",
                                },
                            ],
                            next_cursor: 10,
                            total_available: 42,
                        }),
                    },
                ],
            }),
        });

        const page = await client.search("auth", "/repo");

        assert.strictEqual(page.results.length, 1);
        assert.strictEqual(page.results[0].symbol, "validate_token");
        assert.strictEqual(page.results[0].file, "src/auth.py");
        assert.strictEqual(page.results[0].lines, "10-25");
        assert.strictEqual(page.nextCursor, 10);
        assert.strictEqual(page.totalAvailable, 42);
    });

    test("defaults nextCursor to null and totalAvailable to results.length when the server omits them", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => ({
                content: [
                    {
                        type: "text",
                        text: JSON.stringify({
                            results: [
                                {
                                    file: "a.py",
                                    symbol: "f",
                                    kind: "function",
                                    lines: "1-2",
                                    score: 1,
                                    source: "bm25",
                                    body: "",
                                    language: "python",
                                },
                            ],
                        }),
                    },
                ],
            }),
        });

        const page = await client.search("x", "/repo");

        assert.strictEqual(page.nextCursor, null);
        assert.strictEqual(page.totalAvailable, 1);
    });

    test("returns an empty page when the server returns no results", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => ({
                content: [
                    {
                        type: "text",
                        text: JSON.stringify({
                            results: [],
                            next_cursor: null,
                            total_available: 0,
                        }),
                    },
                ],
            }),
        });

        const page = await client.search("nothing", "/repo");

        assert.deepStrictEqual(page.results, []);
        assert.strictEqual(page.totalAvailable, 0);
    });

    test("forwards the cursor argument to callTool for pagination", async () => {
        const client = new TrelixMcpClient();
        let receivedArgs: unknown;
        withMockedClient(client, {
            callTool: async (call: { arguments: unknown }) => {
                receivedArgs = call.arguments;
                return {
                    content: [
                        {
                            type: "text",
                            text: JSON.stringify({
                                results: [],
                                next_cursor: null,
                                total_available: 0,
                            }),
                        },
                    ],
                };
            },
        });

        await client.search("q", "/repo", 10, 20);

        assert.deepStrictEqual(receivedArgs, {
            query: "q",
            repo_path: "/repo",
            k: 10,
            cursor: 20,
        });
    });
});

suite("TrelixMcpClient.getSymbol", () => {
    test("parses the real get_symbol envelope into a Symbol", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => ({
                content: [
                    {
                        type: "text",
                        text: JSON.stringify({
                            name: "validate_token",
                            qualified_name: "AuthService.validate_token",
                            kind: "method",
                            file: "src/auth.py",
                            line_start: 10,
                            line_end: 25,
                            signature:
                                "def validate_token(self, token: str) -> bool",
                            docstring: "Validate a JWT token.",
                            body: "def validate_token(self, token): ...",
                            language: "python",
                        }),
                    },
                ],
            }),
        });

        const symbol = await client.getSymbol(
            "AuthService.validate_token",
            "/repo",
        );

        assert.ok(symbol);
        assert.strictEqual(symbol!.name, "validate_token");
        assert.strictEqual(symbol!.qualifiedName, "AuthService.validate_token");
        assert.strictEqual(symbol!.lineStart, 10);
        assert.strictEqual(symbol!.lineEnd, 25);
        assert.strictEqual(symbol!.docstring, "Validate a JWT token.");
    });

    test("returns null when the server returns null (symbol not found)", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => ({
                content: [{ type: "text", text: "null" }],
            }),
        });

        const symbol = await client.getSymbol("does_not_exist", "/repo");

        assert.strictEqual(symbol, null);
    });

    test("forwards qualified_name and repo_path arguments to callTool", async () => {
        const client = new TrelixMcpClient();
        let receivedArgs: unknown;
        withMockedClient(client, {
            callTool: async (call: { arguments: unknown }) => {
                receivedArgs = call.arguments;
                return { content: [{ type: "text", text: "null" }] };
            },
        });

        await client.getSymbol("MyClass.my_method", "/repo/path");

        assert.deepStrictEqual(receivedArgs, {
            qualified_name: "MyClass.my_method",
            repo_path: "/repo/path",
        });
    });
});

suite("TrelixMcpClient.ask", () => {
    function askEnvelope(overrides: Record<string, unknown> = {}) {
        return {
            content: [
                {
                    type: "text",
                    text: JSON.stringify({
                        answer: "Auth is handled by AuthService.validate_token.",
                        session_id: "sess-123",
                        turn_count: 3,
                        ...overrides,
                    }),
                },
            ],
        };
    }

    test("calls the ask_agent TOOL (not the trelix-search prompt) and returns its answer", async () => {
        const client = new TrelixMcpClient();
        let toolCall: { name?: string } | undefined;
        let getPromptCalls = 0;
        withMockedClient(client, {
            callTool: async (call: { name: string }) => {
                toolCall = call;
                return askEnvelope();
            },
            getPrompt: async () => {
                getPromptCalls++;
                return { messages: [] };
            },
        });

        const result = await client.ask("how does auth work?", "/repo");

        // Regression guard: old impl used getPrompt("trelix-search") and would
        // fail both of these — it never touched callTool and returned a template.
        assert.strictEqual(toolCall?.name, "ask_agent");
        assert.strictEqual(getPromptCalls, 0, "ask() must not call getPrompt");
        assert.strictEqual(
            result.answer,
            "Auth is handled by AuthService.validate_token.",
        );
        assert.strictEqual(result.turnCount, 3);
    });

    test("round-trips session_id: forwards a passed-in id and surfaces the resolved id", async () => {
        const client = new TrelixMcpClient();
        let receivedArgs: unknown;
        withMockedClient(client, {
            callTool: async (call: { arguments: unknown }) => {
                receivedArgs = call.arguments;
                return askEnvelope({ session_id: "sess-resumed" });
            },
        });

        const result = await client.ask(
            "follow-up question",
            "/repo",
            "sess-existing",
        );

        assert.deepStrictEqual(receivedArgs, {
            query: "follow-up question",
            repo_path: "/repo",
            session_id: "sess-existing",
        });
        assert.strictEqual(result.sessionId, "sess-resumed");
    });

    test("propagates a tool error instead of silently falling back to a template", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => {
                throw new Error("agent loop failed");
            },
        });

        await assert.rejects(
            () => client.ask("q", "/repo"),
            /agent loop failed/,
            "ask() must surface tool errors, not swallow them",
        );
    });
});

suite("TrelixMcpClient.blastRadius", () => {
    test("parses the blast_radius tool's array into BlastRadiusEntry[] (line_start -> lineStart)", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => ({
                content: [
                    {
                        type: "text",
                        text: JSON.stringify([
                            {
                                file: "src/api.py",
                                symbol: "Handler.dispatch",
                                kind: "method",
                                line_start: 42,
                                language: "python",
                            },
                        ]),
                    },
                ],
            }),
        });

        const entries = await client.blastRadius("validate_token", "/repo");

        assert.strictEqual(entries.length, 1);
        assert.strictEqual(entries[0].symbol, "Handler.dispatch");
        assert.strictEqual(entries[0].file, "src/api.py");
        assert.strictEqual(entries[0].lineStart, 42);
        assert.strictEqual(entries[0].language, "python");
    });

    test("forwards symbol_name and repo_path to the blast_radius tool", async () => {
        const client = new TrelixMcpClient();
        let toolCall: { name?: string; arguments?: unknown } | undefined;
        withMockedClient(client, {
            callTool: async (call: { name: string; arguments: unknown }) => {
                toolCall = call;
                return { content: [{ type: "text", text: "[]" }] };
            },
        });

        await client.blastRadius("MyClass.my_method", "/repo/path");

        assert.strictEqual(toolCall?.name, "blast_radius");
        assert.deepStrictEqual(toolCall?.arguments, {
            symbol_name: "MyClass.my_method",
            repo_path: "/repo/path",
        });
    });

    test("returns an empty array when there are no dependents", async () => {
        const client = new TrelixMcpClient();
        withMockedClient(client, {
            callTool: async () => ({ content: [{ type: "text", text: "[]" }] }),
        });

        const entries = await client.blastRadius("orphan", "/repo");

        assert.deepStrictEqual(entries, []);
    });
});
