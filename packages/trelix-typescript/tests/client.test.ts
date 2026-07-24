import { describe, expect, it, vi } from "vitest";
import { TrelixApiError, TrelixClient } from "../src/client.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("TrelixClient", () => {
  it("health() calls GET /health and parses the response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: "ok", version: "2.8.1" }));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    const result = await client.health();

    expect(result).toEqual({ status: "ok", version: "2.8.1" });
    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.toString()).toBe("http://127.0.0.1:8765/health");
  });

  it("search() sends cursor and k as query params and returns the pagination envelope", async () => {
    const envelope = {
      results: [
        {
          file: "src/foo.py",
          symbol: "foo",
          kind: "function",
          lines: "1-5",
          score: 0.9,
          source: "vector",
          body: "def foo(): ...",
          language: "python",
        },
      ],
      next_cursor: 10,
      total_available: 25,
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(envelope));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    const result = await client.search({ query: "auth", repo: "/repo", k: 10, cursor: 0 });

    expect(result).toEqual(envelope);
    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.get("query")).toBe("auth");
    expect(url.searchParams.get("repo")).toBe("/repo");
    expect(url.searchParams.get("k")).toBe("10");
    expect(url.searchParams.get("cursor")).toBe("0");
  });

  it("search() omits k/cursor from the query string when not provided", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ results: [], next_cursor: null, total_available: 0 }));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.search({ query: "auth", repo: "/repo" });

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.has("k")).toBe(false);
    expect(url.searchParams.has("cursor")).toBe(false);
  });

  it("search() follows next_cursor to fetch the second page", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ results: [], next_cursor: null, total_available: 25 }));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.search({ query: "auth", repo: "/repo", cursor: 10 });

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.get("cursor")).toBe("10");
  });

  it("index() POSTs repo_path as the JSON body", async () => {
    const indexResult = {
      files_found: 10,
      files_indexed: 9,
      files_skipped: 1,
      symbols_extracted: 42,
      chunks_total: 100,
      chunks_embedded: 100,
      errors: 0,
      elapsed_seconds: 1.23,
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(indexResult));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    const result = await client.index("/repo");

    expect(result).toEqual(indexResult);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:8765/index");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ repo_path: "/repo" });
  });

  it("stats() calls GET /stats with the repo query param", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ files: 1, symbols: 2, chunks: 3 }));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    const result = await client.stats("/repo");

    expect(result).toEqual({ files: 1, symbols: 2, chunks: 3 });
    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.get("repo")).toBe("/repo");
  });

  it("graphCommunities() returns a list of community summaries", async () => {
    const summaries = [
      { community_id: 0, size: 5, top_files: ["a.py"], top_symbols: ["foo"], label: "auth" },
    ];
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(summaries));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    const result = await client.graphCommunities("/repo");

    expect(result).toEqual(summaries);
  });

  it("graphVisualize() includes the optional output param only when provided", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(() =>
        Promise.resolve(jsonResponse({ path: "/repo/.trelix/graph.html", node_count: 3 })),
      );
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.graphVisualize("/repo");
    let url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.has("output")).toBe(false);

    await client.graphVisualize("/repo", "/repo/.trelix/custom.html");
    url = fetchMock.mock.calls[1][0] as URL;
    expect(url.searchParams.get("output")).toBe("/repo/.trelix/custom.html");
  });

  it("graphSearch() includes symbol_id and optional depth", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.graphSearch("/repo", 42, 3);

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.get("symbol_id")).toBe("42");
    expect(url.searchParams.get("depth")).toBe("3");
  });

  it("throws TrelixApiError with status and parsed body on a non-2xx response", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: "repo not found" }, 404));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await expect(client.stats("/missing")).rejects.toMatchObject({
      name: "TrelixApiError",
      status: 404,
      body: { detail: "repo not found" },
    });
    expect(TrelixApiError).toBeDefined();
  });

  it("strips a trailing slash from the base URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: "ok", version: "x" }));
    const client = new TrelixClient("http://127.0.0.1:8765/", { fetch: fetchMock });

    await client.health();

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.toString()).toBe("http://127.0.0.1:8765/health");
  });
});
