import { describe, expect, it, vi } from "vitest";
import { askStream, TrelixAskError } from "../src/sse.js";

/** Builds a fetch Response whose body streams the given raw SSE text in one or more chunks. */
function sseResponse(chunks: string[], init: { status?: number; ok?: boolean } = {}): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  const status = init.status ?? 200;
  return new Response(stream, { status });
}

async function collect(gen: AsyncGenerator<string, void, void>): Promise<string[]> {
  const tokens: string[] = [];
  for await (const token of gen) tokens.push(token);
  return tokens;
}

describe("askStream", () => {
  it("yields each token frame and stops at [DONE]", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(sseResponse(["data: Hello\n\n", "data:  world\n\n", "data: [DONE]\n\n"]));

    const tokens = await collect(
      askStream("http://127.0.0.1:8765", { query: "hi", repo: "/repo" }, { fetch: fetchMock }),
    );

    expect(tokens).toEqual(["Hello", " world"]);
  });

  it("reassembles frames split across multiple network chunks", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(sseResponse(["data: Hel", "lo\n\n", "data: [DO", "NE]\n\n"]));

    const tokens = await collect(
      askStream("http://127.0.0.1:8765", { query: "hi", repo: "/repo" }, { fetch: fetchMock }),
    );

    expect(tokens).toEqual(["Hello"]);
  });

  it("rejects with TrelixAskError on a data: [ERROR: ...] frame instead of yielding it", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(sseResponse(["data: partial\n\n", "data: [ERROR: boom]\n\n"]));

    const gen = askStream(
      "http://127.0.0.1:8765",
      { query: "hi", repo: "/repo" },
      { fetch: fetchMock },
    );

    await expect(collect(gen)).rejects.toThrow(TrelixAskError);
  });

  it("throws TrelixAskError immediately on a non-2xx HTTP response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([], { status: 500 }));

    const gen = askStream(
      "http://127.0.0.1:8765",
      { query: "hi", repo: "/repo" },
      { fetch: fetchMock },
    );

    await expect(collect(gen)).rejects.toThrow(TrelixAskError);
  });

  it("passes query and repo as URL search params", async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse(["data: [DONE]\n\n"]));

    await collect(
      askStream(
        "http://127.0.0.1:8765",
        { query: "how does auth work?", repo: "/repo" },
        { fetch: fetchMock },
      ),
    );

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.pathname).toBe("/ask");
    expect(url.searchParams.get("query")).toBe("how does auth work?");
    expect(url.searchParams.get("repo")).toBe("/repo");
  });

  it.each([
    ["http://127.0.0.1:8765/", "http://127.0.0.1:8765/ask"],
    ["http://127.0.0.1:8765//", "http://127.0.0.1:8765/ask"],
    ["http://127.0.0.1:8765", "http://127.0.0.1:8765/ask"],
    ["http://127.0.0.1:8765/api/", "http://127.0.0.1:8765/api/ask"],
  ])("normalises the base URL %j for the /ask request", async (baseUrl, expected) => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse(["data: [DONE]\n\n"]));

    await collect(askStream(baseUrl, { query: "hi", repo: "/repo" }, { fetch: fetchMock }));

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.origin + url.pathname).toBe(expected);
  });

  it("normalises a base URL ending in a very long run of slashes", async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse(["data: [DONE]\n\n"]));

    const tokens = await collect(
      askStream(
        "http://127.0.0.1:8765" + "/".repeat(160_000),
        { query: "hi", repo: "/repo" },
        { fetch: fetchMock },
      ),
    );

    expect(tokens).toEqual([]);
    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.origin + url.pathname).toBe("http://127.0.0.1:8765/ask");
  });

  it("keeps a long run of slashes inside the path and does not slow down on it", async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse(["data: [DONE]\n\n"]));

    const started = performance.now();
    await collect(
      askStream(
        "http://127.0.0.1:8765" + "/".repeat(160_000) + "api",
        { query: "hi", repo: "/repo" },
        { fetch: fetchMock },
      ),
    );
    const elapsedMs = performance.now() - started;

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.pathname.length).toBe(160_000 + "api/ask".length);
    expect(url.pathname.endsWith("/api/ask")).toBe(true);
    expect(elapsedMs).toBeLessThan(2000);
  });
});
