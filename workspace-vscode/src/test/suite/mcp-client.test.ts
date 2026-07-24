import * as assert from "assert";
import { TrelixMcpClient } from "../../mcp-client";

/** Injects a mocked MCP Client onto a TrelixMcpClient without a real stdio connection. */
function withMockedClient(client: TrelixMcpClient, mock: { callTool?: Function; getPrompt?: Function }): void {
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
              results: [{ file: "a.py", symbol: "f", kind: "function", lines: "1-2", score: 1, source: "bm25", body: "", language: "python" }],
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
        content: [{ type: "text", text: JSON.stringify({ results: [], next_cursor: null, total_available: 0 }) }],
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
        return { content: [{ type: "text", text: JSON.stringify({ results: [], next_cursor: null, total_available: 0 }) }] };
      },
    });

    await client.search("q", "/repo", 10, 20);

    assert.deepStrictEqual(receivedArgs, { query: "q", repo_path: "/repo", k: 10, cursor: 20 });
  });
});

suite("TrelixMcpClient.ask", () => {
  test("joins prompt message contents into a single string", async () => {
    const client = new TrelixMcpClient();
    withMockedClient(client, {
      getPrompt: async () => ({
        messages: [{ content: "First part." }, { content: "Second part." }],
      }),
    });

    const answer = await client.ask("how does auth work?", "/repo");

    assert.strictEqual(answer, "First part.\nSecond part.");
  });

  test("stringifies non-string message content", async () => {
    const client = new TrelixMcpClient();
    withMockedClient(client, {
      getPrompt: async () => ({
        messages: [{ content: { type: "text", text: "structured" } }],
      }),
    });

    const answer = await client.ask("q", "/repo");

    assert.strictEqual(answer, JSON.stringify({ type: "text", text: "structured" }));
  });
});
