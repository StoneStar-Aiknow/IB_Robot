const STAGE_LABELS: Record<string, string> = {
  obs_frame_ms: "观测采样",
  dispatch_to_infer_ms: "分发至推理",
  preprocess_ms: "预处理",
  inference_ms: "模型调用",
  postprocess_ms: "后处理",
  action_chunk_publish_ms: "动作块发布",
  dispatch_decode_ms: "分发解码",
  queue_refill_ms: "分发至队列补充",
  refill_to_execute_ms: "队列补充至执行",
  execute_publish_ms: "执行发布",
  cloud_roundtrip_ms: "云端往返",
  policy_total_reported_ms: "策略阶段报告耗时",
  total_ms: "分发至首次执行",
};

export function stageLabel(value: string): string {
  return STAGE_LABELS[value] ?? value;
}

export function metricLabel(value: string): string {
  return ({ minimum: "最小值", maximum: "最大值", mean: "均值", p50: "中位数 P50", p95: "高位 P95", p99: "高位 P99" })[value] ?? value;
}

export function boundaryLabel(value: string): string {
  return ({
    dispatch_request_to_first_action_execute: "请求分发 → 首次动作执行",
    "Dispatch Request -> First Action Execute": "请求分发 → 首次动作执行",
    partial_boundary_fallback: "部分业务端点缺失，使用观测记录补齐边界",
    selected_record_bounds_fallback: "业务端点缺失，使用当前观测范围",
    "fallback:selected_records": "当前选中记录的时间边界",
    "fallback:preferred_anchor": "可用的业务端点",
  })[value] ?? value;
}

const RECORD_STATUS_LABELS = new Map([
  ["ok", "完整记录"], ["complete", "完整"], ["error", "错误记录"],
  ["incomplete", "不完整"], ["invalid", "无效记录"], ["negative", "负耗时（检查时钟）"],
  ["ambiguous", "关联不唯一"], ["uncalibrated", "未校准"],
  ["cross_host_unsynchronized", "跨主机时钟未同步"], ["clock_mismatch", "时钟域不一致"],
  ["clock_discontinuity", "时钟不连续"], ["uncertainty_exceeded", "时钟误差超出范围"],
  ["calibrated", "已校准（条件工程估计）"],
]);

export function recordStatusLabel(value: unknown): string {
  const status = value === null || value === undefined ? "" : String(value);
  return RECORD_STATUS_LABELS.get(status) ?? (status || "未提供");
}

export function formatDuration(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (Math.abs(value) < 0.001) return `${(value * 1000).toFixed(1)} μs`;
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(digits)} ms`;
}

export function formatInteger(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : new Intl.NumberFormat("zh-CN").format(value);
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const units = ["字节", "千字节", "兆字节", "吉字节"];
  let amount = value;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024;
    index += 1;
  }
  return `${amount.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatTimestamp(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

export function shortId(value: string, length = 12): string {
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

export function eventComponent(event: { component_id?: string; fields: Record<string, unknown>; origin: { provider: string } }): string {
  return event.component_id || String(event.fields.component_id ?? "") || event.origin.provider || "—";
}

export function recordToData(value: object): Record<string, unknown> {
  return { ...value } as Record<string, unknown>;
}
