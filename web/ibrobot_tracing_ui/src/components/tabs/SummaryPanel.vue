<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { boundaryLabel, formatDuration, formatInteger, formatTimestamp, metricLabel, recordStatusLabel, recordToData, stageLabel } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";

const store = useProfilerStore();
const { summary, tabState, tabError, searchQuery, analysisId } = storeToRefs(store);
const primary = computed(() => {
  const metrics = summary.value?.stages ?? {};
  const name = metrics.total_ms?.count ? "total_ms" : metrics.inference_ms?.count ? "inference_ms" : "";
  return { name, stats: name ? metrics[name] : null };
});
const excluded = computed(() => Math.max(0, (summary.value?.analysis.request_count ?? 0) - (primary.value.stats?.count ?? 0)));

async function openSlowest(): Promise<void> {
  if (!primary.value.name) return;
  const analysis = store.analysisId;
  store.requestSortBy = primary.value.name;
  store.requestDescending = true;
  store.requestsOffset = 0;
  await store.fetchRequests();
  if (store.analysisId !== analysis || store.tabState.requests !== "ready") return;
  const first = store.requests.find((request) => typeof request[primary.value.name] === "number");
  if (first) await store.navigateToRequest(first.request_id, "critical-path");
}

const stages = computed(() => {
  const rows = Object.entries(summary.value?.stages ?? {}).sort(([left], [right]) => Number(right === "total_ms") - Number(left === "total_ms"));
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return query ? rows.filter(([name]) => `${name} ${stageLabel(name)}`.toLocaleLowerCase("zh-CN").includes(query)) : rows;
});

const spanColumns: GridColumn[] = [
  { key: "name", label: "Span" },
  { key: "component_id", label: "组件", className: "mono-cell" },
  { key: "origin", label: "来源", format: (value) => value === "built-in" ? "内置埋点" : value === "user" ? "用户埋点" : String(value) },
  { key: "status", label: "记录状态", format: recordStatusLabel },
  ...["p50", "p95", "maximum", "mean"].map((key): GridColumn => ({
    key, label: metricLabel(key), align: "right", className: "numeric",
    format: (value) => formatDuration(value as number | null),
  })),
  { key: "count", label: "调用次数", align: "right", className: "numeric" },
];

const spanRows = computed(() => {
  const rows = summary.value?.span_summary ?? [];
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return rows
    .filter((item) => `${item.name} ${item.component_id} ${item.origin} ${item.status}`.toLocaleLowerCase("zh-CN").includes(query))
    .map((item) => ({ ...recordToData(item), id: JSON.stringify([item.component_id, item.name, item.origin, item.status]) }));
});

function inspectSummary(row: Record<string, unknown>): void {
  const { id: _displayId, ...data } = row;
  store.selection = {
    kind: "span-summary",
    title: `${String(row.name)} 汇总`,
    subtitle: `全部请求 · ${String(row.component_id || "未标记组件")}`,
    data,
  };
}

async function openSpanAnalysis(): Promise<void> {
  store.componentId = "";
  store.searchQuery = "";
  await store.setSpanAnalysisMode("aggregate");
  store.activeTab = "span-profile";
  await store.fetchTab("span-profile");
}
</script>

<template>
  <AsyncState
    context="trace 文件概览"
    :state="tabState.summary"
    :error="tabError.summary"
    :empty="!summary"
    empty-title="尚未载入分析"
    empty-message="从页首选择追踪源并载入。"
    @retry="analysisId && store.openAnalysis(analysisId)"
  >
    <div v-if="summary" class="tab-scroll summary-view">
      <section class="overview-intro" aria-label="trace 文件概况">
        <h2>{{ primary.name ? stageLabel(primary.name) : '这份 trace 文件的记录' }}</h2>
        <p>{{ boundaryLabel(summary.coverage.boundary || '') || '计时边界以具体记录为准' }} · 主机侧观测</p>
        <dl v-if="primary.stats" class="overview-metrics">
          <div><dt>典型耗时 · P50</dt><dd>{{ formatDuration(primary.stats.p50) }}</dd></div>
          <div><dt>高位耗时 · P95</dt><dd>{{ formatDuration(primary.stats.p95) }}</dd></div>
          <div><dt>最慢</dt><dd>{{ formatDuration(primary.stats.maximum) }}</dd></div>
        </dl>
        <p v-if="primary.stats">共 {{ formatInteger(summary.analysis.request_count) }} 个请求，{{ formatInteger(primary.stats.count) }} 个有此指标，{{ formatInteger(excluded) }} 个未纳入该项统计。<button class="text-button" type="button" @click="store.activeTab = 'warnings'">查看数据质量</button></p>
        <div class="state-actions">
          <button class="button primary" type="button" :disabled="!primary.stats || tabState.requests === 'loading'" @click="openSlowest">查看最慢请求</button>
          <button class="button secondary" type="button" @click="store.activeTab = 'requests'">浏览请求列表</button>
          <button class="text-button" type="button" @click="store.activeTab = 'distribution'; store.fetchTab('distribution')">查看延迟分布</button>
        </div>
        <p v-if="tabError.requests" role="alert">{{ tabError.requests }} <button class="text-button" type="button" @click="store.fetchRequests(true)">重试请求列表</button></p>
      </section>
      <header class="profile-semantic-banner">
        <strong>请求链投影</strong>
        <span>阶段延迟仅汇总可唯一关联的请求指标，不代表业务成功或可用性。歧义、错误、不完整记录不纳入该投影，不等于未采到，也不表示需要重试。</span>
      </header>
      <section class="summary-strip">
        <dl><dt>请求</dt><dd>{{ formatInteger(summary.analysis.request_count) }}</dd></dl>
        <dl><dt>事件</dt><dd>{{ formatInteger(summary.analysis.event_count) }}</dd></dl>
        <dl><dt>Span 记录</dt><dd>{{ formatInteger(summary.analysis.span_count) }}</dd></dl>
        <dl><dt>关联 Flow</dt><dd>{{ formatInteger(summary.analysis.flow_count) }}</dd></dl>
      </section>
      <details class="overview-metadata">
        <summary>投影指标覆盖 {{ summary.coverage.observed_metrics ?? 0 }} / {{ summary.coverage.expected_metrics ?? 0 }} · 查看含义与生成时间</summary>
        <p>有数值样本的预期指标数 / 预期指标总数。未形成投影可能是未采到、歧义、错误或不完整，不代表业务成功率。</p>
        <p>生成时间：{{ formatTimestamp(summary.analysis.created_at) }}</p>
      </details>

      <section class="matrix-section">
        <header><strong>各阶段耗时</strong><span>全部请求 · 阶段可能重叠，不直接相加</span></header>
        <div class="data-grid-wrap">
          <table class="data-grid summary-matrix">
            <thead><tr><th>阶段</th><th class="align-right">最小</th><th class="align-right">P50 中位</th><th class="align-right">P95 高位</th><th class="align-right">P99 高位</th><th class="align-right">最大</th><th class="align-right">均值</th><th class="align-right">样本</th></tr></thead>
            <tbody>
              <tr v-if="!stages.length"><td colspan="8" class="empty-cell">没有匹配的投影数值样本；请在请求视图查看指标状态，或在 Span 分析查看记录。</td></tr>
              <tr v-for="([name, stats]) in stages" :key="name" :class="{ emphasized: name === 'total_ms' }">
                <td><strong>{{ stageLabel(name) }}</strong><code>{{ name }}</code></td>
                <td class="align-right numeric">{{ formatDuration(stats.minimum) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.p50) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.p95) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.p99) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.maximum) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.mean) }}</td>
                <td class="align-right numeric">{{ formatInteger(stats.count) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="matrix-section">
        <header><strong>通用 Span 汇总</strong><button class="text-button" type="button" @click="openSpanAnalysis">查看全部调用聚合</button></header>
        <p class="grid-limit">按组件、名称、埋点来源和记录状态分组。统计完整且耗时有效的调用，包含错误及重复调用；不完整记录见调用层级。此处次数不是请求数或成功次数。</p>
        <div class="grid-limit">
          {{ summary.span_summary_truncated ? "已截断，返回前" : "返回" }} {{ summary.span_summary.length }} / {{ summary.span_summary_total }} 组，上限 {{ summary.span_summary_limit }} 组；搜索仅作用于此预览。
        </div>
        <DataGrid :rows="spanRows" :columns="spanColumns" row-key="id" empty-text="当前预览没有匹配的完整 Span 分组" @select="inspectSummary" />
      </section>

      <section v-if="summary.custom_mark_summary.length" class="mark-section">
        <header><strong>用户事件计数</strong></header>
        <button v-for="mark in summary.custom_mark_summary" :key="`${mark.component_id}:${mark.name}`" type="button" @click="store.componentId = mark.component_id">
          <span>{{ mark.name }}</span><code>{{ mark.component_id || "未绑定组件" }}</code><b>{{ mark.count }}</b>
        </button>
      </section>

      <section v-if="summary.coverage.missing_events?.length" class="coverage-warning">
        <strong>未形成投影数值</strong>
        <span>{{ summary.coverage.missing_events.map(stageLabel).join("、") }}。不等于事件未采到；请在请求视图查看 *_status 区分歧义、错误、不完整与未采到。</span>
      </section>
    </div>
  </AsyncState>
</template>
