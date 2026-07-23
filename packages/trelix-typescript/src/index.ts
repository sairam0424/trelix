export {
  TrelixClient,
  TrelixApiError,
  type TrelixClientOptions,
  type SearchParams,
  type HealthResponse,
  type SearchResponse,
  type SearchResultModel,
  type IndexResponse,
  type StatsResponse,
  type GraphStatsResponse,
  type CommunitySummaryModel,
  type GraphVisualizeResponse,
  type GraphSearchResultModel,
} from "./client.js";

export { askStream, TrelixAskError, type AskStreamOptions } from "./sse.js";
