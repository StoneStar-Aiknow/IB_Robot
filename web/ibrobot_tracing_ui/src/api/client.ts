import type {
  AnalysisSummary,
  ApiList,
  CallTreeProjection,
  CapabilitiesResponse,
  ComparisonJob,
  ComparisonJobCreate,
  ComparisonResult,
  ComparisonSummary,
  CriticalPathFilters,
  CriticalPathResponse,
  FlowRecord,
  GraphView,
  LatencyDistributionFilters,
  LatencyDistributionResponse,
  LoadJob,
  MetricName,
  RequestRecord,
  SpanProfileFilters,
  SpanProfileResponse,
  SpanRecord,
  SourceListResponse,
  TimelineProjection,
  TraceEvent,
  TracepointDefinition,
  TraceTopology,
} from "./types";

const API_ROOT = "/api/v1";

export interface PageFilters {
  offset?: number;
  limit?: number;
  requestId?: string;
  componentId?: string;
  name?: string;
}

export interface RequestFilters extends PageFilters {
  sortBy?: string;
  descending?: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  const body = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    const message = body && typeof body === "object" && "message" in body
      ? String(body.message)
      : body && typeof body === "object" && "detail" in body
        ? String(body.detail)
        : `请求失败（${response.status}）`;
    throw new ApiError(message, response.status, body);
  }
  return body as T;
}

function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") search.set(key, String(value));
  });
  const value = search.toString();
  return value ? `?${value}` : "";
}

export const tracingApi = {
  capabilities: (signal?: AbortSignal) => request<CapabilitiesResponse>("/capabilities", { signal }),
  sources: (signal?: AbortSignal) => request<SourceListResponse>("/sources", { signal }),
  refreshSources: (signal?: AbortSignal) => request<SourceListResponse>("/sources/refresh", { method: "POST", signal }),
  createAnalysis: (sourceId: string, signal?: AbortSignal) =>
    request<LoadJob>("/analysis-jobs", {
      method: "POST",
      body: JSON.stringify({ source_id: sourceId }),
      signal,
    }),
  job: (jobId: string, signal?: AbortSignal) => request<LoadJob>(`/analysis-jobs/${encodeURIComponent(jobId)}`, { signal }),
  createComparison: (payload: ComparisonJobCreate, signal?: AbortSignal) =>
    request<ComparisonJob>("/comparison-jobs", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }),
  comparisonJob: (jobId: string, signal?: AbortSignal) =>
    request<ComparisonJob>(`/comparison-jobs/${encodeURIComponent(jobId)}`, { signal }),
  comparisons: (signal?: AbortSignal) => request<{ items: ComparisonSummary[] }>("/comparisons", { signal }),
  comparison: (comparisonId: string, signal?: AbortSignal) =>
    request<ComparisonResult>(`/comparisons/${encodeURIComponent(comparisonId)}`, { signal }),
  deleteComparison: async (comparisonId: string, signal?: AbortSignal): Promise<void> => {
    await request<null>(`/comparisons/${encodeURIComponent(comparisonId)}`, { method: "DELETE", signal });
  },
  summary: (analysisId: string, signal?: AbortSignal) =>
    request<AnalysisSummary>(`/analyses/${encodeURIComponent(analysisId)}/summary`, { signal }),
  requests: (analysisId: string, signal?: AbortSignal, filters: RequestFilters = {}) =>
    request<ApiList<RequestRecord>>(
      `/analyses/${encodeURIComponent(analysisId)}/requests${query({ offset: filters.offset ?? 0, limit: filters.limit ?? 100, sort_by: filters.sortBy ?? "total_ms", descending: String(filters.descending ?? true), request_id: filters.requestId })}`,
      { signal },
    ),
  distribution: (analysisId: string, filters: LatencyDistributionFilters = {}, signal?: AbortSignal) =>
    request<LatencyDistributionResponse>(
      `/analyses/${encodeURIComponent(analysisId)}/distribution${query({
        metric: filters.metric,
        bins: filters.bins,
        outlier_limit: filters.outlierLimit,
        bucket_request_limit: filters.bucketRequestLimit,
      })}`,
      { signal },
    ),
  tracepoints: (
    analysisId: string,
    pagination: { offset?: number; limit?: number } = {},
    signal?: AbortSignal,
  ) =>
    request<ApiList<TracepointDefinition>>(
      `/analyses/${encodeURIComponent(analysisId)}/tracepoints${query({
        offset: pagination.offset ?? 0,
        limit: pagination.limit ?? 1000,
      })}`,
      { signal },
    ),
  topology: (
    analysisId: string,
    filters: { metric: MetricName; view: GraphView; requestId?: string },
    signal?: AbortSignal,
  ) =>
    request<TraceTopology>(
      `/analyses/${encodeURIComponent(analysisId)}/graph${query({
        metric: filters.metric,
        view: filters.view,
        request_id: filters.requestId,
      })}`,
      { signal },
    ),
  events: (analysisId: string, filters: PageFilters, signal?: AbortSignal) =>
    request<ApiList<TraceEvent>>(
      `/analyses/${encodeURIComponent(analysisId)}/events${query({
        request_id: filters.requestId,
        component_id: filters.componentId,
        offset: filters.offset ?? 0,
        limit: filters.limit ?? 100,
        event_name: filters.name,
      })}`,
      { signal },
    ),
  spans: (analysisId: string, filters: PageFilters, signal?: AbortSignal) =>
    request<ApiList<SpanRecord>>(
      `/analyses/${encodeURIComponent(analysisId)}/spans${query({
        request_id: filters.requestId,
        component_id: filters.componentId,
        offset: filters.offset ?? 0,
        limit: filters.limit ?? 100,
        name: filters.name,
      })}`,
      { signal },
    ),
  flows: (analysisId: string, filters: PageFilters, signal?: AbortSignal) =>
    request<ApiList<FlowRecord>>(
      `/analyses/${encodeURIComponent(analysisId)}/flows${query({
        request_id: filters.requestId,
        component_id: filters.componentId,
        offset: filters.offset ?? 0,
        limit: filters.limit ?? 100,
        edge_id: filters.name,
      })}`,
      { signal },
    ),
  timeline: (analysisId: string, filters: { requestId?: string; componentId?: string }, signal?: AbortSignal) =>
    request<TimelineProjection>(
      `/analyses/${encodeURIComponent(analysisId)}/timeline${query({
        request_id: filters.requestId,
        component_id: filters.componentId,
      })}`,
      { signal },
    ),
  callTree: (analysisId: string, filters: { requestId?: string; componentId?: string }, signal?: AbortSignal) =>
    request<CallTreeProjection>(
      `/analyses/${encodeURIComponent(analysisId)}/call-tree${query({
        request_id: filters.requestId,
        component_id: filters.componentId,
      })}`,
      { signal },
    ),
  spanProfile: (analysisId: string, filters: SpanProfileFilters, signal?: AbortSignal) =>
    request<SpanProfileResponse>(
      `/analyses/${encodeURIComponent(analysisId)}/span-profile${query({
        mode: filters.mode,
        request_id: filters.mode === "request" ? filters.requestId : undefined,
        component_id: filters.componentId,
        start_ns: filters.startNs,
        end_ns: filters.endNs,
        origin: filters.origin,
        status: filters.status,
        max_nodes: filters.maxNodes ?? 10_000,
      })}`,
      { signal },
    ),
  criticalPath: (analysisId: string, filters: CriticalPathFilters, signal?: AbortSignal) =>
    request<CriticalPathResponse>(
      `/analyses/${encodeURIComponent(analysisId)}/critical-path${query({
        request_id: filters.requestId,
        component_id: filters.componentId,
        start_ns: filters.startNs,
        end_ns: filters.endNs,
        include_flows: filters.includeFlows === false ? "false" : "true",
        max_segments: filters.maxSegments ?? 1_000,
      })}`,
      { signal },
    ),
  warnings: (analysisId: string, signal?: AbortSignal) =>
    request<{ items: string[]; total: number; limit: number; truncated: boolean }>(`/analyses/${encodeURIComponent(analysisId)}/warnings`, { signal }),
};

/** Resolve the exact occurrence ID, even when repeated names span several pages. */
export async function findSourceRecord(
  analysisId: string,
  kind: "span" | "flow",
  sourceId: string,
  filters: PageFilters,
  signal: AbortSignal,
): Promise<SpanRecord | FlowRecord | null> {
  let offset = 0;
  while (!signal.aborted) {
    const page = kind === "span"
      ? await tracingApi.spans(analysisId, { ...filters, offset, limit: 100 }, signal)
      : await tracingApi.flows(analysisId, { ...filters, offset, limit: 100 }, signal);
    const record = page.items.find((item) => item.id === sourceId);
    if (record) return record;
    if (page.next_offset == null || page.next_offset <= offset) return null;
    offset = page.next_offset;
  }
  return null;
}

export async function pollJob(
  initial: LoadJob,
  onUpdate: (job: LoadJob) => void,
  signal?: AbortSignal,
  intervalMs = 700,
): Promise<LoadJob> {
  let job = initial;
  onUpdate(job);
  while (job.status === "queued" || job.status === "running") {
    await waitForPoll(intervalMs, signal);
    job = await tracingApi.job(job.id, signal);
    onUpdate(job);
  }
  return job;
}

export async function pollComparisonJob(
  initial: ComparisonJob,
  onUpdate: (job: ComparisonJob) => void,
  signal?: AbortSignal,
  intervalMs = 700,
): Promise<ComparisonJob> {
  let job = initial;
  onUpdate(job);
  while (job.status === "queued" || job.status === "running") {
    await waitForPoll(intervalMs, signal);
    job = await tracingApi.comparisonJob(job.id, signal);
    onUpdate(job);
  }
  return job;
}

function waitForPoll(intervalMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const abort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("请求已取消", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, intervalMs);
    if (signal?.aborted) abort();
    else signal?.addEventListener("abort", abort, { once: true });
  });
}
