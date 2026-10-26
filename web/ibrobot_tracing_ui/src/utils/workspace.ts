import type { DetailTab, SpanAnalysisMode } from "../api/types";

export const VIEW_LABELS: Record<DetailTab, string> = {
  summary: "概览", requests: "请求列表", distribution: "延迟分布", comparison: "实验对比",
  timeline: "时间线", "span-profile": "调用层级", "critical-path": "耗时拆分",
  events: "原始事件", spans: "Span 记录", flows: "Flow 记录", warnings: "数据质量",
};
export const REQUEST_VIEWS: DetailTab[] = ["timeline", "span-profile", "critical-path"];
export const RECORD_VIEWS: DetailTab[] = ["events", "spans", "flows"];

export function isRequestView(tab: DetailTab, mode: SpanAnalysisMode): boolean {
  return REQUEST_VIEWS.includes(tab) && !(tab === "span-profile" && mode === "aggregate");
}

export function scopeLabel(tab: DetailTab, requestId: string, componentId: string, mode: SpanAnalysisMode): string {
  if (["summary", "requests", "distribution", "comparison", "warnings"].includes(tab)) {
    return tab === "comparison" ? "两份 trace 文件 · 不使用请求筛选" : "全部请求";
  }
  const request = tab === "span-profile" && mode === "aggregate"
    ? "全部请求聚合"
    : requestId ? `请求 ${requestId}` : RECORD_VIEWS.includes(tab) ? "全部请求" : "尚未选择请求";
  return componentId ? `${request} · 组件 ${componentId}` : request;
}

export function filterHint(tab: DetailTab, mode: SpanAnalysisMode): string | null {
  if (tab === "span-profile" && mode !== "call-tree") return "高亮调用名称或组件";
  if (["distribution", "critical-path", "comparison"].includes(tab)) return null;
  if (["requests", "events", "spans", "flows"].includes(tab)) return "过滤当前页记录";
  return tab === "summary" ? "过滤下方统计表" : "过滤当前视图";
}
