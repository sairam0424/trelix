import type { components } from "./generated/schema.js";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type SearchResponse = components["schemas"]["SearchResponse"];
export type SearchResultModel = components["schemas"]["SearchResultModel"];
export type IndexResponse = components["schemas"]["IndexResponse"];
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
    return this.get<SearchResponse>("/search", query);
  }

  async index(repoPath: string): Promise<IndexResponse> {
    return this.post<IndexResponse>("/index", { repo_path: repoPath });
  }

  async stats(repo: string): Promise<StatsResponse> {
    return this.get<StatsResponse>("/stats", { repo });
  }

  async graphStats(repo: string): Promise<GraphStatsResponse> {
    return this.get<GraphStatsResponse>("/graph", { repo });
  }

  async graphCommunities(repo: string): Promise<CommunitySummaryModel[]> {
    return this.get<CommunitySummaryModel[]>("/graph/communities", { repo });
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
