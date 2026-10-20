<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../stores/profiler";
import AppIcon from "./AppIcon.vue";
import { formatDuration, formatInteger, recordStatusLabel, recordToData, stageLabel } from "../utils/format";
import { findSourceRecord } from "../api/client";

const store = useProfilerStore();
const { selection, selectionDescription } = storeToRefs(store);
const sourceLoading = ref(false);
const sourceError = ref("");
let sourceController: AbortController | null = null;
const isAttribution = computed(() => selection.value?.kind === "critical-path");
const sourceKind = computed(() => {
  const kind = selection.value?.data.kind;
  return isAttribution.value && (kind === "span" || kind === "flow") ? kind : null;
});

watch(selection, () => {
  sourceController?.abort();
  sourceLoading.value = false;
  sourceError.value = "";
});
onBeforeUnmount(() => sourceController?.abort());

async function openSource(): Promise<void> {
  const segment = selection.value;
  const kind = sourceKind.value;
  if (!segment || !kind || !segment.data.source_id) return;
  sourceController?.abort();
  const controller = new AbortController();
  sourceController = controller;
  const analysis = store.analysisId;
  sourceLoading.value = true;
  sourceError.value = "";
  try {
    const record = await findSourceRecord(analysis, kind, String(segment.data.source_id), {
      requestId: String(segment.data.request_id || store.requestId),
      name: String(kind === "span" ? segment.data.label : segment.data.edge_id),
    }, controller.signal);
    if (controller.signal.aborted || selection.value !== segment || store.analysisId !== analysis) return;
    if (!record) throw new Error("当前分析中未找到该原始记录；可重试或重新载入 trace 文件。");
    store.selection = { kind, title: `原始 ${kind === "span" ? "Span" : "Flow"} · ${segment.title}`,
      subtitle: record.id, data: recordToData(record), previous: segment };
  } catch (error) {
    if (!controller.signal.aborted) sourceError.value = error instanceof Error ? error.message : "原始记录读取失败，请重试。";
  } finally {
    if (sourceController === controller) sourceLoading.value = false;
  }
}

const entries = computed(() => (selection.value ? Object.entries(selection.value.data).filter(([key]) => key !== "description") : []));
function asFields(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
const digest = computed<[string, unknown][]>(() => {
  if (!selection.value) return [];
  const data = selection.value.data;
  if (selection.value.kind === "warning") {
    return [["含义", data.explanation || "请查看原始诊断"], ["关联请求", data.request_id || "未提供"],
      ["涉及指标", data.metric ? stageLabel(String(data.metric)) : "未提供"]];
  }
  if (isAttribution.value) {
    return [
      ["计时含义", "本段是请求墙钟的归因片段，不是原始 Span / Flow 的完整观测区间。"],
      ["请求", data.request_id || "未提供"],
      ["本段归因时长", formatDuration(Number(data.duration_ns) / 1e6, 3)],
      ["占请求墙钟", `${Number(data.percentage).toFixed(2)}%`],
      ["本段起点（ns）", data.start_ns],
      ["本段终点（ns）", data.end_ns],
      ["原始记录 ID", data.source_id || "当前范围无归因记录"],
      ["原始记录状态", sourceKind.value ? recordStatusLabel(data.status) : "不适用"],
    ];
  }
  if (selection.value.kind === "span-summary") {
    return [
      ["范围", "全部请求 · 相同组件、名称、来源和记录状态的调用"],
      ["组件", data.component_id || "未标记"],
      ["记录状态", recordStatusLabel(data.status)],
      ["来源", data.origin === "user" ? "用户埋点" : data.origin === "built-in" ? "内置埋点" : data.origin],
      ["调用次数", formatInteger(data.count as number | undefined)],
      ...["minimum", "p50", "p95", "maximum", "mean"].map((key): [string, unknown] => [
        ({ minimum: "最小值", p50: "P50", p95: "P95", maximum: "最大值", mean: "均值" } as Record<string, string>)[key],
        formatDuration(data[key] as number | null),
      ]),
    ];
  }
  const mapping = asFields(data.time_mapping);
  const origin = typeof data.origin === "object" ? asFields(data.origin) : asFields(mapping.source_origin);
  const fields = asFields(data.fields);
  const rows: [string, unknown][] = [
    ["请求", data.request_id || data.trace_id || fields.request_id || "未提供"],
    ["组件 / 交接边", data.component_id || data.edge_id || "未提供"],
  ];
  if ("duration_ms" in data) rows.push(["观测耗时", formatDuration(data.duration_ms as number | null, 3)]);
  else if ("duration_ns" in data) rows.push(["观测耗时", data.duration_ns === null ? "未完成 / 未提供" : formatDuration(Number(data.duration_ns) / 1e6, 3)]);
  if (data.status) rows.push(["记录状态", recordStatusLabel(data.status)]);
  if (selection.value.kind === "flow") {
    const provenance = asFields(data.provenance);
    for (const [endpoint, label] of [["send", "发送"], ["receive", "接收"]]) {
      const point = asFields(provenance[endpoint]);
      const pointOrigin = asFields(point.origin);
      const pointMapping = asFields(point.time_mapping);
      rows.push([`${label}主机 / 机器标识`, pointOrigin.host || pointOrigin.machine_id || "未提供"]);
      rows.push([`${label}端校准`, recordStatusLabel(pointMapping.status)]);
    }
    if (data.uncertainty_ns !== null && data.uncertainty_ns !== undefined) rows.push(["时钟误差预算（ns）", data.uncertainty_ns]);
  } else {
    rows.push(["主机 / 机器标识", origin.host || origin.machine_id || "未提供"]);
    if (mapping.status) rows.push(["校准状态", recordStatusLabel(mapping.status)]);
  }
  if (data.clock) rows.push(["时钟域", data.clock]);
  if ("timestamp_ns" in data) rows.push(["事件时间（ns）", data.timestamp_ns]);
  if ("start_ns" in data || "send_ns" in data) rows.push(["起点（ns）", data.start_ns ?? data.send_ns]);
  if ("end_ns" in data || "receive_ns" in data) rows.push(["终点（ns）", data.end_ns ?? data.receive_ns ?? "未完成"]);
  return rows;
});

function display(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}
</script>

<template>
  <aside v-if="selection" class="inspector" aria-label="记录检查器">
    <header>
      <div><span>{{ isAttribution ? '归因分段详情' : '记录详情' }}</span><strong id="inspector-heading" tabindex="-1">{{ selection.title }}</strong><small v-if="selection.subtitle">{{ selection.subtitle }}</small></div>
      <button class="icon-button" type="button" title="关闭检查器" @click="store.selection = null"><AppIcon name="close" /><span class="sr-only">关闭检查器</span></button>
    </header>
    <div v-if="sourceKind || selection.previous" class="inspector-source-actions">
      <button v-if="sourceKind" class="button secondary" type="button" :disabled="sourceLoading" @click="openSource">{{ sourceLoading ? '正在读取原始记录…' : `打开原始 ${sourceKind === 'span' ? 'Span' : 'Flow'}` }}</button>
      <button v-if="selection.previous" class="text-button" type="button" @click="store.selection = selection.previous!">返回归因分段</button>
      <section v-if="sourceError" class="context-error" role="alert">
        <strong>原始记录暂时无法读取</strong>
        <p>当前归因分段已保留。重试后可继续核对原始证据。</p>
        <button class="button secondary" type="button" :disabled="sourceLoading" @click="openSource">重试原始记录</button>
        <details><summary>查看错误详情</summary><pre>{{ sourceError }}</pre></details>
      </section>
    </div>
    <section v-if="selectionDescription && !isAttribution" class="inspector-description">
      <strong>埋点说明</strong>
      <p>{{ selectionDescription }}</p>
    </section>
    <dl class="record-digest"><template v-for="([label, value]) in digest" :key="label"><dt>{{ label }}</dt><dd>{{ display(value) }}</dd></template></dl>
    <details class="raw-fields"><summary>{{ isAttribution ? '归因分段字段' : '全部原始字段' }}</summary><dl>
      <template v-for="([key, value]) in entries" :key="key">
        <dt>{{ key }}</dt>
        <dd :class="{ structured: typeof value === 'object' && value !== null }">{{ display(value) }}</dd>
      </template>
    </dl></details>
  </aside>
</template>
