import type {
  ComparisonStatistic,
  DetailTab,
  GraphView,
  MetricName,
  SpanAnalysisMode,
} from "../api/types";

export const DETAIL_TABS: DetailTab[] = [
  "summary",
  "requests",
  "distribution",
  "comparison",
  "timeline",
  "span-profile",
  "critical-path",
  "events",
  "spans",
  "flows",
  "warnings",
];

const METRICS: MetricName[] = ["p50", "p95", "p99", "minimum", "maximum", "mean"];
const GRAPH_VIEWS: GraphView[] = ["nodes", "components", "tracepoints"];
const SPAN_ANALYSIS_MODES: SpanAnalysisMode[] = ["call-tree", "request", "aggregate"];
const COMPARISON_METRIC = /^[A-Za-z_][A-Za-z0-9_.-]*_ms$/;

export interface ProfilerRouteState {
  sourceId?: string;
  sourceVersion?: string;
  requestOffset?: number;
  requestSortBy?: string;
  requestDescending?: boolean;
  requestFilter?: string;
  recordOffset?: number;
  recordName?: string;
  activeTab: DetailTab;
  requestId: string;
  componentId: string;
  distributionMetric: string;
  metric: MetricName;
  graphView: GraphView;
  spanAnalysisMode: SpanAnalysisMode;
  searchQuery: string;
  selectedBaselineSourceId: string;
  comparisonStatistic: ComparisonStatistic;
  comparisonRelativeThreshold: number;
  comparisonAbsoluteThresholdMs: number;
  comparisonMetric: string;
  comparisonResultId?: string;
}

function value(input: unknown): string {
  return Array.isArray(input) ? String(input[0] ?? "") : String(input ?? "");
}

function nonNegativeNumber(input: unknown, fallback: number): number {
  const text = value(input).trim();
  if (!text) return fallback;
  const number = Number(text);
  return Number.isFinite(number) && number >= 0 ? number : fallback;
}

export function parseProfilerRoute(query: Record<string, unknown>): ProfilerRouteState {
  const requestedTab = value(query.tab);
  const tab = (requestedTab === "call-tree" ? "span-profile" : requestedTab) as DetailTab;
  const metric = value(query.metric) as MetricName;
  const graphView = value(query.view) as GraphView;
  const requestedSpanMode = (requestedTab === "call-tree" ? "call-tree" : value(query.mode)) as SpanAnalysisMode;
  const statistic = value(query.statistic) as ComparisonStatistic;
  const comparisonMetric = value(query.comparison_metric).trim();
  return {
    sourceId: value(query.source),
    sourceVersion: value(query.source_version),
    requestOffset: Math.min(10_000_000, Math.floor(nonNegativeNumber(query.request_offset, 0))),
    requestSortBy: ["total_ms", "inference_ms", "preprocess_ms", "postprocess_ms", "queue_refill_ms"].includes(value(query.request_sort)) ? value(query.request_sort) : "total_ms",
    requestDescending: value(query.request_order) !== "asc",
    requestFilter: value(query.request_filter ?? (tab === "requests" ? query.q : "")).slice(0, 512),
    recordOffset: Math.min(10_000_000, Math.floor(nonNegativeNumber(query.record_offset, 0))),
    recordName: value(query.record_name).slice(0, 512),
    activeTab: DETAIL_TABS.includes(tab) ? tab : "summary",
    requestId: value(query.request),
    componentId: value(query.component),
    distributionMetric: value(query.distribution_metric),
    metric: METRICS.includes(metric) ? metric : "p95",
    graphView: GRAPH_VIEWS.includes(graphView) ? graphView : "components",
    spanAnalysisMode: SPAN_ANALYSIS_MODES.includes(requestedSpanMode) ? requestedSpanMode : "request",
    searchQuery: value(query.q),
    selectedBaselineSourceId: value(query.baseline_source),
    comparisonStatistic: METRICS.includes(statistic) ? statistic : "p95",
    comparisonRelativeThreshold: nonNegativeNumber(query.relative_threshold, 10),
    comparisonAbsoluteThresholdMs: nonNegativeNumber(query.absolute_threshold_ms, 1),
    comparisonMetric: COMPARISON_METRIC.test(comparisonMetric) ? comparisonMetric : "",
    comparisonResultId: value(query.comparison_result).slice(0, 128),
  };
}

export function buildProfilerQuery(state: ProfilerRouteState): Record<string, string> {
  const query: Record<string, string> = {};
  if (state.sourceId) query.source = state.sourceId;
  if (state.sourceVersion) query.source_version = state.sourceVersion;
  if (state.requestOffset) query.request_offset = String(state.requestOffset);
  if (state.requestSortBy && state.requestSortBy !== "total_ms") query.request_sort = state.requestSortBy;
  if (state.requestDescending === false) query.request_order = "asc";
  if (state.requestFilter) query.request_filter = state.requestFilter;
  if (["events", "spans", "flows"].includes(state.activeTab)) {
    if (state.recordOffset) query.record_offset = String(state.recordOffset);
    if (state.recordName) query.record_name = state.recordName;
  }
  if (state.activeTab !== "summary") query.tab = state.activeTab;
  if (state.requestId) query.request = state.requestId;
  if (state.componentId) query.component = state.componentId;
  if (state.distributionMetric) query.distribution_metric = state.distributionMetric;
  if (state.metric !== "p95") query.metric = state.metric;
  if (state.graphView !== "components") query.view = state.graphView;
  if (state.spanAnalysisMode !== "request") query.mode = state.spanAnalysisMode;
  if (state.searchQuery) query.q = state.searchQuery;
  if (state.selectedBaselineSourceId) query.baseline_source = state.selectedBaselineSourceId;
  if (state.comparisonStatistic !== "p95") query.statistic = state.comparisonStatistic;
  if (state.comparisonRelativeThreshold !== 10) query.relative_threshold = String(state.comparisonRelativeThreshold);
  if (state.comparisonAbsoluteThresholdMs !== 1) query.absolute_threshold_ms = String(state.comparisonAbsoluteThresholdMs);
  if (state.comparisonMetric) query.comparison_metric = state.comparisonMetric;
  if (state.comparisonResultId) query.comparison_result = state.comparisonResultId;
  return query;
}
