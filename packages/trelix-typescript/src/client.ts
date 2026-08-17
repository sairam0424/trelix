import type { components } from "./generated/schema.js";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type SearchResponse = components["schemas"]["SearchResponse"];
export type SearchResultModel = components["schemas"]["SearchResultModel"];
export type IndexResponse = components["schemas"]["IndexResponse"];
export type ParseRequest = components["schemas"]["ParseRequest"];
export type ParseResponse = components["schemas"]["ParseResponse"];
export type ParseSymbolModel = components["schemas"]["ParseSymbolModel"];
export type StatsResponse = components["schemas"]["StatsResponse"];
export type GraphStatsResponse = components["schemas"]["GraphStatsResponse"];
export type CommunitySummaryModel = components["schemas"]["CommunitySummaryModel"];
export type GraphVisualizeResponse = components["schemas"]["GraphVisualizeResponse"];
export type GraphSearchResultModel = components["schemas"]["GraphSearchResultModel"];

export interface TrelixClientOptions {
  /** Override the `fetch` implementation (e.g. for tests or non-global-fetch runtimes). */
  fetch?: typeof fetch;
}

export interface SearchParams {
  query: string;
  repo: string;
  /** Page size. Server default is 10. */
  k?: number;
  /** Pass the previous response's `next_cursor` to fetch the next page. */
  cursor?: number;
  /**
   * One of the server's IntentType values (symbol_lookup, file_overview,
   * feature_flow, project_overview, comparison, config_lookup, dependency_map,
   * blast_radius). Set it when the caller has already classified the query — the
   * server then skips its own LLM intent classification and routes straight to
   * that intent's strategy. An unrecognized value is never an error; the server
   * silently falls back to normal classification.
   */
  intentHint?: string;
  /** Only consulted by the server when `intentHint` is also a valid intent. */
  hydeSnippetHint?: string;
}

/** Server-side caps on `GET /graph/communities`; omit both for the server defaults
 * (largest 50 communities of size >= 2). `{ minCommunitySize: 1, maxCommunities: 0 }`
 * is the documented escape hatch back to the uncapped list. */
export interface GraphCommunitiesParams {
  minCommunitySize?: number;
  maxCommunities?: number;
}

/** Thrown when the trelix API responds with a non-2xx status. */
export class TrelixApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(message);
    this.name = "TrelixApiError";
  }
}

/**
 * Thin HTTP client for the trelix REST API (`trelix serve`).
 *
 * Every method maps 1:1 to a route in `src/trelix/api/app.py`. `/ask` is
 * intentionally excluded — it streams over SSE, so use `askStream` from
 * `./sse.js` instead of a single request/response method.
 */
export class TrelixClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string, options: TrelixClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? fetch;
  }

  async health(): Promise<HealthResponse> {
    return this.get<HealthResponse>("/health");
  }

  async search(params: SearchParams): Promise<SearchResponse> {
    const query: Record<string, string> = { query: params.query, repo: params.repo };
    if (params.k !== undefined) query.k = String(params.k);
    if (params.cursor !== undefined) query.cursor = String(params.cursor);
    if (params.intentHint !== undefined) query.intent_hint = params.intentHint;
    if (params.hydeSnippetHint !== undefined) query.hyde_snippet_hint = params.hydeSnippetHint;
    return this.get<SearchResponse>("/search", query);
  }

  async index(repoPath: string): Promise<IndexResponse> {
    return this.post<IndexResponse>("/index", { repo_path: repoPath });
  }

  /**
   * Parse a single file without touching the index — for editor / pre-commit
   * callers that need structural info on unsaved or not-yet-indexed content.
   *
   * The request is passed through verbatim so the server's "exactly one content
   * source" rule (`file_path` XOR `content` + `file_name`) stays enforced in one
   * place; violating it is a 422, surfaced as `TrelixApiError`. Cross-file call
   * and type resolution is skipped server-side, so edge counts reflect only what
   * Tree-sitter could determine from this file alone.
   */
  async parseFile(request: ParseRequest): Promise<ParseResponse> {
    return this.post<ParseResponse>("/parse", request);
  }

  async stats(repo: string): Promise<StatsResponse> {
    return this.get<StatsResponse>("/stats", { repo });
  }

  async graphStats(repo: string): Promise<GraphStatsResponse> {
    return this.get<GraphStatsResponse>("/graph", { repo });
  }

  async graphCommunities(
    repo: string,
    params: GraphCommunitiesParams = {},
  ): Promise<CommunitySummaryModel[]> {
    const query: Record<string, string> = { repo };
    if (params.minCommunitySize !== undefined) {
      query.min_community_size = String(params.minCommunitySize);
    }
    if (params.maxCommunities !== undefined) query.max_communities = String(params.maxCommunities);
    return this.get<CommunitySummaryModel[]>("/graph/communities", query);
  }

  async graphVisualize(repo: string, output?: string): Promise<GraphVisualizeResponse> {
    const query: Record<string, string> = { repo };
    if (output !== undefined) query.output = output;
    return this.get<GraphVisualizeResponse>("/graph/visualize", query);
  }

  async graphSearch(
    repo: string,
    symbolId: number,
    depth?: number,
  ): Promise<GraphSearchResultModel[]> {
    const query: Record<string, string> = { repo, symbol_id: String(symbolId) };
    if (depth !== undefined) query.depth = String(depth);
    return this.get<GraphSearchResultModel[]>("/graph/search", query);
  }

  private async get<T>(path: string, query?: Record<string, string>): Promise<T> {
    const url = new URL(this.baseUrl + path);
    if (query) {
      for (const [key, value] of Object.entries(query)) {
        url.searchParams.set(key, value);
      }
    }
    const res = await this.fetchImpl(url);
    return this.parse<T>(res);
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await this.fetchImpl(this.baseUrl + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return this.parse<T>(res);
  }

  private async parse<T>(res: Response): Promise<T> {
    if (!res.ok) {
      const body = await res.json().catch(() => undefined);
      throw new TrelixApiError(
        `trelix API request failed: ${res.status} ${res.statusText}`,
        res.status,
        body,
      );
    }
    return res.json() as Promise<T>;
  }
}
