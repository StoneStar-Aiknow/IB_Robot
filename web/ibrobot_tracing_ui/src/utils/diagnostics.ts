import { recordStatusLabel, stageLabel } from "./format";

export interface DiagnosticExplanation {
  title: string;
  explanation: string;
  requestId: string;
  metric: string;
  raw: string;
}

/** Interpret known analyzer messages; retain unknown messages without guessing. */
export function explainDiagnostic(raw: string): DiagnosticExplanation {
  const metric = raw.match(/^request (\S+) metric (\S+): ([^;]+); excluded from request summary$/);
  if (metric) {
    const status = metric[3].split(" ")[0].split(",").map(recordStatusLabel).join("、");
    return { title: `${stageLabel(metric[2])}：${status}`, requestId: metric[1], metric: metric[2], raw,
      explanation: "此请求的这项指标未纳入汇总，原始记录仍可查看；这不表示 trace 文件中没有记录，也不等于业务失败。" };
  }
  const clocks = raw.match(/^request (\S+): (multiple host clocks without calibrated merge; metrics excluded|uncalibrated events; excluded from shared-clock request metrics)$/);
  if (clocks) return { title: "请求时钟尚未对齐", requestId: clocks[1], metric: "", raw,
    explanation: "该请求包含未校准的时钟，相关指标未纳入共享时钟统计。请查看原始记录的主机与校准信息。" };
  return { title: "分析器提示", requestId: "", metric: "", raw,
    explanation: "请查看原始诊断并核对 trace 文件和记录质量。这条提示尚无专用解释，不据此判断业务成功或失败。" };
}
