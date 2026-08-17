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

  // intent_hint/hyde_snippet_hint let a caller that already classified the query
  // (an agent) skip trelix's internal LLM intent classification. They shipped on
  // GET /search but were unreachable from this client while schema.ts was stale.
  it("search() forwards intent_hint and hyde_snippet_hint when provided", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ results: [], next_cursor: null, total_available: 0 }));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.search({
      query: "auth",
      repo: "/repo",
      intentHint: "symbol_lookup",
      hydeSnippetHint: "def authenticate(request): ...",
    });

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.get("intent_hint")).toBe("symbol_lookup");
    expect(url.searchParams.get("hyde_snippet_hint")).toBe("def authenticate(request): ...");
  });

  it("search() omits intent_hint/hyde_snippet_hint when not provided", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ results: [], next_cursor: null, total_available: 0 }));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.search({ query: "auth", repo: "/repo" });

    const url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.has("intent_hint")).toBe(false);
    expect(url.searchParams.has("hyde_snippet_hint")).toBe(false);
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

  // POST /parse is the editor/pre-commit route: structural info for content that
  // was never indexed. The server enforces "exactly one content source", so the
  // client's job is only to pass through whichever one the caller supplied.
  it("parseFile() POSTs a disk-backed file_path and returns the parse result", async () => {
    const parseResult = {
      symbols: [
        {
          name: "foo",
          qualified_name: "mod.foo",
          kind: "function",
          line_start: 1,
          line_end: 5,
          signature: "def foo()",
        },
      ],
      call_edge_count: 0,
      import_edge_count: 2,
      type_edge_count: 0,
      parse_errors: 0,
      note: "single-file parse; cross-file resolution skipped",
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(parseResult));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    const result = await client.parseFile({ repo_path: "/repo", file_path: "src/foo.py" });

    expect(result).toEqual(parseResult);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:8765/parse");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      repo_path: "/repo",
      file_path: "src/foo.py",
    });
  });

  it("parseFile() passes inline content + file_name through unchanged", async () => {
    const empty = {
      symbols: [],
      call_edge_count: 0,
      import_edge_count: 0,
      type_edge_count: 0,
      parse_errors: 0,
      note: "",
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(empty));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.parseFile({
      repo_path: "/repo",
      content: "def unsaved(): ...",
      file_name: "draft.py",
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual({
      repo_path: "/repo",
      content: "def unsaved(): ...",
      file_name: "draft.py",
    });
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

  // The server caps this route at the 50 largest communities of size >= 2 (on
  // trelix's own index the uncapped list is 6,497 entries / ~1.16 MB, 99.1% of
  // them singletons). min_community_size=1 + max_communities=0 is the documented
  // escape hatch back to the uncapped list, so it has to be expressible here.
  it("graphCommunities() forwards the size/count caps only when overridden", async () => {
    // mockImplementation, not mockResolvedValue: a single Response body can only
    // be read once, and this case issues two requests.
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse([])));
    const client = new TrelixClient("http://127.0.0.1:8765", { fetch: fetchMock });

    await client.graphCommunities("/repo");
    let url = fetchMock.mock.calls[0][0] as URL;
    expect(url.searchParams.has("min_community_size")).toBe(false);
    expect(url.searchParams.has("max_communities")).toBe(false);

    await client.graphCommunities("/repo", { minCommunitySize: 1, maxCommunities: 0 });
    url = fetchMock.mock.calls[1][0] as URL;
    expect(url.searchParams.get("min_community_size")).toBe("1");
    expect(url.searchParams.get("max_communities")).toBe("0");
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
