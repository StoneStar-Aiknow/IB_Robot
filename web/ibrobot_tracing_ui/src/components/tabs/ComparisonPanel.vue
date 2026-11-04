<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import type { ComparisonMetricResult, FlameDiffEntry, FlameDiffStatus } from "../../api/types";
import { useProfilerStore } from "../../stores/profiler";
import {
  buildSharedHistogramGeometry,
  flameDiffBreadcrumb,
  flameDiffMatches,
  layoutFlameDiff,
} from "../../utils/comparison";
import { formatBytes, formatDuration, formatInteger, formatTimestamp, metricLabel, recordToData, shortId, stageLabel } from "../../utils/format";
import { numericNs } from "../../utils/spanProfile";
import AppIcon from "../AppIcon.vue";

const store = useProfilerStore();
const {
  analysisId,
  analysisState,
  baselineSources,
  comparisonAbsoluteThresholdMs,
  comparisonError,
  comparisonJob,
  comparisonMetric,
  comparisonRelativeThreshold,
  comparisonResult,
  comparisonResultId,
  comparisonState,
  comparisonStatistic,
  requests,
  searchQuery,
  selectedBaselineSource,
  selectedBaselineSourceId,
  selection,
  sourcesState,
  sourcesError,
  summary,
} = storeToRefs(store);

const histogramHost = ref<HTMLElement | null>(null);
const histogramWidth = ref(720);
const hoveredBucketIndex = ref<number | null>(null);
const flameHost = ref<HTMLElement | null>(null);
const flameWidth = ref(760);
const focusId = ref("");
const hoverEntry = ref<FlameDiffEntry | null>(null);
const hoverPoint = ref({ x: 0, y: 0 });
const deletePrompt = ref(false);
const deleting = ref(false);
const deleteError = ref("");
const resultNotice = ref("");
const deleteTrigger = ref<HTMLButtonElement | null>(null);
const deleteCancel = ref<HTMLButtonElement | null>(null);
let resizeObserver: ResizeObserver | null = null;

const HISTOGRAM_HEIGHT = 146;
const HISTOGRAM_INSET = 14;
const FLAME_LEFT = 58;
const FLAME_ROW_HEIGHT = 27;

const canCompare = computed(() => (
  analysisState.value === "ready"
  && sourcesState.value !== "error"
  && Boolean(selectedBaselineSource.value)
  && comparisonState.value !== "loading"
  && Number.isFinite(comparisonRelativeThreshold.value)
  && comparisonRelativeThreshold.value >= 0
  && Number.isFinite(comparisonAbsoluteThresholdMs.value)
  && comparisonAbsoluteThresholdMs.value >= 0
));
const metricOptions = computed(() => Array.from(new Set([
  comparisonMetric.value,
  ...(comparisonResult.value?.metrics.map((metric) => metric.metric) ?? []),
  ...requests.value.flatMap((request) => Object.keys(request).filter((key) => (
    /^[A-Za-z_][A-Za-z0-9_.-]*_ms$/.test(key) && !key.endsWith("_source")
  ))),
].filter((name) => /^[A-Za-z_][A-Za-z0-9_.-]*_ms$/.test(name)))).sort());
const resultMatchesControls = computed(() => {
  const result = comparisonResult.value;
  if (!result) return true;
  return result.baseline_source_id === selectedBaselineSourceId.value
    && result.baseline_source_version === selectedBaselineSource.value?.version
    && result.candidate_analysis_id === analysisId.value
    && result.statistic === comparisonStatistic.value
    && result.relative_threshold_percent === comparisonRelativeThreshold.value
    && result.absolute_threshold_ms === comparisonAbsoluteThresholdMs.value
    && (result.histogram.metric ?? "") === comparisonMetric.value;
});
const exceededMetrics = computed(() => comparisonResult.value?.metrics.filter((item) => item.regression) ?? []);

const histogram = computed(() => comparisonResult.value?.histogram ?? null);
const histogramPlotWidth = computed(() => Math.max(1, histogramWidth.value - HISTOGRAM_INSET * 2));
const histogramGeometry = computed(() => buildSharedHistogramGeometry(
  histogram.value?.buckets ?? [],
  histogramPlotWidth.value,
  HISTOGRAM_HEIGHT,
));
const hoveredBucket = computed(() => histogram.value?.buckets.find((bucket) => bucket.index === hoveredBucketIndex.value) ?? null);
const hoveredBar = computed(() => histogramGeometry.value.bars.find((bar) => bar.index === hoveredBucketIndex.value) ?? null);

const flameChartWidth = computed(() => Math.max(520, flameWidth.value));
const flamePlotWidth = computed(() => Math.max(1, flameChartWidth.value - FLAME_LEFT - 10));
const flameLayout = computed(() => layoutFlameDiff(comparisonResult.value?.flame_diff ?? [], flamePlotWidth.value, focusId.value, FLAME_ROW_HEIGHT));
const flamePlotHeight = computed(() => Math.max(110, flameLayout.value.contentHeight));
const flameRects = computed(() => flameLayout.value.rects);
const flameHeight = computed(() => flamePlotHeight.value + 8);
const flameDepths = computed(() => Array.from({ length: flameLayout.value.depthCount }, (_, depth) => depth));
const breadcrumbs = computed(() => flameDiffBreadcrumb(comparisonResult.value?.flame_diff ?? [], focusId.value));

const hoverInfo = computed(() => {
  const entry = hoverEntry.value;
  if (!entry) return null;
  return {
    title: entry.frame[1],
    subtitle: `${entry.frame[0] || "未标记组件"} · ${entry.frame[2] || "未知来源"}`,
    lines: [
      `候选 ${formatNs(entry.candidate_value_ns)} · 基线 ${formatNs(entry.baseline_value_ns)}`,
      `包含差值 ${formatSignedNs(entry.absolute_delta_ns)} · ${formatPercent(entry.percent_delta)}`,
      `候选自身 ${formatNs(entry.candidate_self_value_ns)} · 基线自身 ${formatNs(entry.baseline_self_value_ns)}`,
      `自身差值 ${formatSignedNs(entry.self_absolute_delta_ns)} · ${formatPercent(entry.self_percent_delta)}`,
      `状态 ${statusLabel(entry.status)}`,
    ],
  };
});

function metricValue(metric: ComparisonMetricResult, side: "baseline" | "candidate"): number | null {
  return metric[side][comparisonResult.value?.statistic ?? "p95"];
}

function metricDelta(metric: ComparisonMetricResult): { absolute_delta_ms: number | null; percent_delta: number | null } {
  return metric.deltas[comparisonResult.value?.statistic ?? "p95"];
}

function metricStatus(metric: ComparisonMetricResult): "regression" | "ok" | "unavailable" {
  if (metricValue(metric, "baseline") === null || metricValue(metric, "candidate") === null) return "unavailable";
  return metric.regression ? "regression" : "ok";
}

function metricStatusLabel(metric: ComparisonMetricResult): string {
  return ({ regression: "超过阈值", ok: "未超阈值", unavailable: "指标样本不足" })[metricStatus(metric)];
}

function formatSignedDuration(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${formatDuration(value, 3)}`;
}

function formatPercent(value: number | null, baseline?: number | null, candidate?: number | null): string {
  if (value !== null && Number.isFinite(value)) return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
  if (baseline === 0 && candidate !== null && candidate !== undefined && candidate > 0) return "无上界";
  return "—";
}

function formatNs(value: number | string): string {
  return formatDuration(numericNs(value) / 1_000_000, 3);
}

function formatSignedNs(value: number | string): string {
  const number = numericNs(value) / 1_000_000;
  return `${number > 0 ? "+" : ""}${formatDuration(number, 3)}`;
}

function statusLabel(status: FlameDiffStatus): string {
  return ({
    improved: "墙钟减少",
    unchanged: "不变",
    regressed: "墙钟增加",
    added: "新增",
    removed: "移除",
  })[status];
}

function selectBaseline(event: Event): void {
  selectedBaselineSourceId.value = (event.target as HTMLSelectElement).value;
}

function chooseMetric(metric: string): void {
  comparisonMetric.value = metric;
}

function runComparison(): void {
  resultNotice.value = "";
  void store.compareWithSourceBaseline();
}

async function askDelete(): Promise<void> {
  deleteError.value = "";
  deletePrompt.value = true;
  await nextTick();
  deleteCancel.value?.focus();
}

async function cancelDelete(): Promise<void> {
  if (deleting.value) return;
  deletePrompt.value = false;
  await nextTick();
  deleteTrigger.value?.focus();
}

async function confirmDelete(): Promise<void> {
  if (deleting.value) return;
  deleting.value = true;
  deleteError.value = "";
  const candidate = analysisId.value;
  const resultId = comparisonResult.value?.id;
  try {
    if (await store.deleteComparisonResult()) {
      deletePrompt.value = false;
      resultNotice.value = `已删除缓存比较结果 ${resultId}。原始 trace 文件保留，可按当前参数重新比较。`;
      await nextTick();
      if (store.activeTab === "comparison" && analysisId.value === candidate) document.querySelector<HTMLElement>(".compare-button")?.focus();
    }
  } catch (error) {
    if (analysisId.value === candidate && comparisonResult.value?.id === resultId) {
      deleteError.value = error instanceof Error ? error.message : "删除请求失败";
    }
  } finally {
    deleting.value = false;
    await nextTick();
    if (deleteError.value && store.activeTab === "comparison" && analysisId.value === candidate) deleteCancel.value?.focus();
  }
}

function bucketLabel(index: number): string {
  const bucket = histogram.value?.buckets.find((item) => item.index === index);
  if (!bucket) return "直方图区间";
  return `区间 ${formatDuration(bucket.start)} 至 ${formatDuration(bucket.end)}，基线 ${bucket.baseline_count} 个样本，候选 ${bucket.candidate_count} 个样本`;
}

function matches(entry: FlameDiffEntry): boolean {
  return flameDiffMatches(entry, searchQuery.value);
}

function flameLabel(entry: FlameDiffEntry): string {
  return `${entry.frame[1]}，组件 ${entry.frame[0] || "未标记"}，状态 ${statusLabel(entry.status)}，候选 ${formatNs(entry.candidate_value_ns)}，基线 ${formatNs(entry.baseline_value_ns)}。按 Enter 检查，按 Shift+Enter 聚焦子树`;
}

function inspectFlame(entry: FlameDiffEntry): void {
  const path = flameDiffBreadcrumb(comparisonResult.value?.flame_diff ?? [], entry.path_id)
    .map((item) => item.frame.join(" / "));
  selection.value = {
    kind: "comparison",
    title: entry.frame[1],
    subtitle: `${entry.frame[0]} · ${statusLabel(entry.status)}`,
    data: { ...recordToData(entry), full_path: path },
  };
}

async function focusFlame(entry: FlameDiffEntry): Promise<void> {
  focusId.value = entry.path_id;
  await nextTick();
  flameHost.value?.scrollTo({ left: 0, top: 0 });
}

async function navigateFocus(id: string): Promise<void> {
  focusId.value = id;
  await nextTick();
  flameHost.value?.scrollTo({ left: 0, top: 0 });
}

function handleFlameKey(event: KeyboardEvent, entry: FlameDiffEntry): void {
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  if (event.shiftKey && event.key === "Enter") void focusFlame(entry);
  else inspectFlame(entry);
}

function showFlameTooltip(event: PointerEvent, entry: FlameDiffEntry): void {
  const bounds = flameHost.value?.getBoundingClientRect();
  hoverEntry.value = entry;
  hoverPoint.value = {
    x: (flameHost.value?.scrollLeft ?? 0) + Math.min((bounds?.width ?? 300) - 282, Math.max(8, event.clientX - (bounds?.left ?? 0) + 12)),
    y: (flameHost.value?.scrollTop ?? 0) + Math.max(8, event.clientY - (bounds?.top ?? 0) + 12),
  };
}

function updateWidths(): void {
  if (histogramHost.value) histogramWidth.value = Math.max(320, Math.floor(histogramHost.value.getBoundingClientRect().width));
  if (flameHost.value) flameWidth.value = Math.max(520, Math.floor(flameHost.value.getBoundingClientRect().width));
}

watch(comparisonResult, (result) => {
  if (result) resultNotice.value = "";
  deletePrompt.value = false;
  deleteError.value = "";
  if (!result?.flame_diff.some((entry) => entry.path_id === focusId.value)) focusId.value = "";
  hoveredBucketIndex.value = null;
  hoverEntry.value = null;
  void nextTick(updateWidths);
});

watch([histogramHost, flameHost], ([histogramElement, flameElement], [oldHistogram, oldFlame]) => {
  if (oldHistogram) resizeObserver?.unobserve(oldHistogram);
  if (oldFlame) resizeObserver?.unobserve(oldFlame);
  if (histogramElement) resizeObserver?.observe(histogramElement);
  if (flameElement) resizeObserver?.observe(flameElement);
  updateWidths();
});

onMounted(() => {
  resizeObserver = new ResizeObserver(updateWidths);
  if (histogramHost.value) resizeObserver.observe(histogramHost.value);
  if (flameHost.value) resizeObserver.observe(flameHost.value);
});

onBeforeUnmount(() => resizeObserver?.disconnect());
</script>

<template>
  <section class="comparison-panel">
    <header class="comparison-semantic-banner">
      <div><strong>追踪数值比较</strong><span>选择另一份 trace 文件作为基线，当前分析作为候选。可比较仅表示样本足以计算差值，不代表实验条件一致。</span></div>
      <button class="icon-button" type="button" title="刷新追踪目录" :disabled="sourcesState === 'loading'" @click="store.refreshSources"><AppIcon name="refresh" /><span class="sr-only">刷新追踪目录</span></button>
    </header>

    <section v-if="sourcesState === 'error'" class="context-error" role="alert">
      <strong>trace 文件列表暂时无法读取</strong>
      <p>这不表示没有可比较的 trace 文件。请检查追踪服务连接，再重试读取；已有选择和比较结果会保留。</p>
      <button class="button secondary" type="button" @click="store.loadSources">重试读取 trace 文件</button>
      <details><summary>查看错误详情</summary><pre>{{ sourcesError }}</pre></details>
    </section>
    <div v-if="sourcesState === 'loading' && !baselineSources.length" class="comparison-catalog-state" aria-live="polite">
      正在读取追踪目录…
    </div>
    <div v-else-if="sourcesState !== 'error' && !baselineSources.length" class="comparison-empty state-view">
      <AppIcon name="trace" :size="24" />
      <strong>没有另一份可比较的 trace</strong>
      <span>追踪目录中至少需要两份 trace：当前分析作为候选，另一份作为基线。</span>
      <span>将另一份 CTF、日志或 .ibtrace 格式的 trace 文件放入服务配置的追踪目录，再刷新列表。</span>
      <button class="button secondary" type="button" @click="store.refreshSources">刷新追踪目录</button>
    </div>

    <template v-else-if="baselineSources.length || comparisonResult">
      <form class="comparison-controls" @submit.prevent="runComparison">
        <label class="comparison-control baseline-control">
          <span>基线</span>
          <select :value="selectedBaselineSourceId" aria-label="比较基线 trace" @change="selectBaseline">
            <option v-for="source in baselineSources" :key="source.id" :value="source.id">
              {{ source.name }} · {{ source.kind }} · {{ formatBytes(source.size_bytes) }}
            </option>
          </select>
        </label>
        <details class="comparison-options">
          <summary>统计与判定条件 · {{ metricLabel(comparisonStatistic) }}</summary>
          <div class="comparison-options-grid">
        <label class="comparison-control">
          <span>统计值</span>
          <select v-model="comparisonStatistic" aria-label="比较统计值">
            <option v-for="name in ['minimum', 'p50', 'p95', 'p99', 'maximum', 'mean']" :key="name" :value="name">{{ metricLabel(name) }}</option>
          </select>
        </label>
        <label class="comparison-control threshold-control">
          <span>相对阈值</span>
          <span class="input-suffix"><input v-model.number="comparisonRelativeThreshold" type="number" min="0" step="0.1" aria-label="相对回归阈值百分比" /><i>%</i></span>
        </label>
        <label class="comparison-control threshold-control">
          <span>绝对阈值</span>
          <span class="input-suffix"><input v-model.number="comparisonAbsoluteThresholdMs" type="number" min="0" step="0.1" aria-label="绝对回归阈值毫秒" /><i>ms</i></span>
        </label>
        <label class="comparison-control metric-control">
          <span>直方图指标</span>
          <select v-model="comparisonMetric" aria-label="比较直方图指标">
            <option value="">自动选择</option>
            <option v-for="name in metricOptions" :key="name" :value="name">{{ stageLabel(name) }} · {{ name }}</option>
          </select>
        </label>
          </div>
        </details>
        <button class="button primary compare-button" type="submit" :disabled="!canCompare">
          <AppIcon :name="comparisonState === 'loading' ? 'refresh' : 'play'" :size="14" />
          {{ comparisonState === "loading" ? "正在比较" : "执行比较" }}
        </button>
      </form>

      <div class="comparison-threshold-note" role="note">
        <strong>AND 判定：</strong>仅当候选的 {{ metricLabel(comparisonStatistic) }} 同时严格超过基线
        <b>{{ comparisonRelativeThreshold }}%</b> 和 <b>{{ comparisonAbsoluteThresholdMs }} ms</b>，才标记为差值超阈值。
      </div>

      <div class="comparison-identity-strip">
        <dl><dt>基线 trace</dt><dd><strong>{{ selectedBaselineSource?.name || "—" }}</strong><code>{{ selectedBaselineSource ? shortId(selectedBaselineSource.id, 18) : "无来源标识" }}</code></dd></dl>
        <dl><dt>基线版本</dt><dd><strong>{{ selectedBaselineSource?.kind || "—" }}</strong><code>{{ selectedBaselineSource ? shortId(selectedBaselineSource.version, 18) : "无版本标识" }}</code></dd></dl>
        <dl><dt>当前候选</dt><dd><strong>{{ summary?.analysis.source_name || "尚未载入" }}</strong><code>{{ analysisId ? shortId(analysisId, 18) : "无分析标识" }}</code></dd></dl>
        <dl><dt>追踪目录</dt><dd><strong>{{ baselineSources.length }} 份可选基线</strong><span v-if="selectedBaselineSource">{{ formatBytes(selectedBaselineSource.size_bytes) }} · {{ formatTimestamp(selectedBaselineSource.modified_at) }}</span></dd></dl>
      </div>

      <div v-if="!resultMatchesControls" class="comparison-message dirty" role="status">
        控件已改变；下方仍是上次结果。点击“执行比较”后才会按新参数重新计算。
      </div>
      <p v-if="resultNotice" class="comparison-result-link" role="status">{{ resultNotice }}</p>
      <div v-if="comparisonState === 'loading'" class="comparison-running" aria-live="polite">
        <span class="comparison-running-line"><i /></span>
        <strong>{{ comparisonResultId ? '正在读取已保存的比较结果' : comparisonJob?.status === "queued" ? `比较任务排队中${comparisonJob.queue_position ? `（第 ${comparisonJob.queue_position} 位）` : ""}` : "正在分析所选基线 trace 并比较" }}</strong>
        <span>可停止页面等待；后台任务可能继续，不会删除 trace 文件。</span>
        <button class="button secondary" type="button" @click="store.clearComparison">停止等待</button>
      </div>
      <div v-else-if="comparisonState === 'error'" class="comparison-message error" role="alert">
        <AppIcon name="warning" :size="16" /><div><strong>比较失败</strong><span>{{ comparisonError }}</span></div>
        <button v-if="comparisonResultId" class="button secondary" type="button" @click="store.openComparisonResult(comparisonResultId)">重读结果</button>
        <button class="button secondary" type="button" :disabled="!canCompare" @click="runComparison">{{ comparisonResultId ? '重新执行比较' : '重试' }}</button>
      </div>
      <div v-else-if="comparisonState === 'idle' || !comparisonResult" class="comparison-start state-view">
        <AppIcon name="trace" :size="23" /><strong>配置阈值后显式执行比较</strong>
        <span>服务端会只读分析所选 trace，并与当前候选分析比较；不会接收浏览器文件路径。</span>
      </div>

      <div v-else class="comparison-results">
        <section class="comparison-verdict" :class="comparisonResult.comparable ? (comparisonResult.has_regression ? 'regression' : 'ok') : 'blocked'">
          <div>
            <span>{{ comparisonResult.comparable ? "所选指标数据足够比较" : "所选指标数据不足以比较" }}</span>
            <strong v-if="!comparisonResult.comparable">无法比较所选指标</strong>
            <strong v-else-if="comparisonResult.has_regression">{{ exceededMetrics.length }} 项指标差值超过阈值</strong>
            <strong v-else>未发现超过双阈值的差异</strong>
          </div>
          <dl class="saved-result-settings" aria-label="当前结果使用的参数">
            <div><dt>结果统计值</dt><dd>{{ metricLabel(comparisonResult.statistic) }}</dd></div>
            <div><dt>结果双阈值</dt><dd>&gt;{{ comparisonResult.relative_threshold_percent }}% 且 &gt;{{ comparisonResult.absolute_threshold_ms }} ms</dd></div>
            <div><dt>结果来源</dt><dd>基线 <code>{{ comparisonResult.baseline_source_id }}</code> · 候选 <code>{{ comparisonResult.candidate_analysis_id }}</code></dd></div>
          </dl>
          <button ref="deleteTrigger" class="text-button" type="button" :disabled="deleting" @click="askDelete">删除此结果</button>
        </section>
        <section v-if="deletePrompt" class="delete-result-confirmation" role="group" aria-labelledby="delete-result-title" @keydown.esc.stop.prevent="cancelDelete">
          <h2 id="delete-result-title">删除这个缓存结果？</h2>
          <p>结果 <code>{{ comparisonResult.id }}</code> 的现有链接将失效，原始 trace 文件保留。重新比较会生成或返回可用的结果。</p>
          <div class="state-actions"><button ref="deleteCancel" class="button secondary" type="button" :disabled="deleting" @click="cancelDelete">取消删除</button><button class="button secondary" type="button" :disabled="deleting" @click="confirmDelete">{{ deleting ? '正在删除…' : '确认删除结果' }}</button></div>
          <div v-if="deleteError" class="context-error" role="alert"><strong>删除失败，当前结果和链接已保留</strong><p>可重新确认删除，或取消并继续查看。</p><details><summary>查看错误详情</summary><pre>{{ deleteError }}</pre></details></div>
        </section>
        <div class="comparison-conditions" role="note">
          <strong>实验条件未核对；数值差异不证明改动导致性能退化。</strong>
          <span>请核对两份 trace 文件对应的硬件、模型、工作负载与埋点配置。本工具未验证这些条件一致，也不据此判断业务成功。</span>
          <ul v-if="exceededMetrics.length"><li v-for="item in exceededMetrics" :key="item.metric">{{ stageLabel(item.metric) }}：基线 {{ formatDuration(metricValue(item, 'baseline'), 3) }}（{{ item.baseline_count }} 样本）→ 候选 {{ formatDuration(metricValue(item, 'candidate'), 3) }}（{{ item.candidate_count }} 样本），差值 {{ formatSignedDuration(metricDelta(item).absolute_delta_ms) }}。</li></ul>
        </div>
        <p class="comparison-result-link">结果 <code>{{ comparisonResult.id }}</code> · 当前地址可重新打开此结果；缓存过期后可按保存参数显式重算。</p>

        <section v-if="comparisonResult.blocking_reasons.length" class="comparison-diagnostics blocking" aria-labelledby="comparison-blocking-title">
          <header><strong id="comparison-blocking-title">指标数据不足原因</strong><span>{{ comparisonResult.blocking_reasons.length }}</span></header>
          <ul><li v-for="reason in comparisonResult.blocking_reasons" :key="reason">{{ reason }}</li></ul>
        </section>
        <section v-if="comparisonResult.coverage_warnings.length || comparisonResult.count_warnings.length" class="comparison-diagnostics warnings" aria-labelledby="comparison-warning-title">
          <header><strong id="comparison-warning-title">覆盖率与样本数警告</strong><span>{{ comparisonResult.coverage_warnings.length + comparisonResult.count_warnings.length }}</span></header>
          <p>{{ comparisonResult.coverage_warnings.length }} 项观测覆盖差异、{{ comparisonResult.count_warnings.length }} 项样本数量差异。请结合 trace 文件的记录范围解释结果。</p>
          <details><summary>查看原始诊断</summary>
          <ul>
            <li v-for="warning in comparisonResult.coverage_warnings" :key="`coverage:${warning}`"><b>覆盖率</b>{{ warning }}</li>
            <li v-for="warning in comparisonResult.count_warnings" :key="`count:${warning}`"><b>计数</b>{{ warning }}</li>
          </ul>
          </details>
        </section>

        <section class="comparison-metrics-section" aria-labelledby="comparison-metrics-title">
          <header><div><strong id="comparison-metrics-title">指标差值</strong><span>选定统计值：{{ metricLabel(comparisonResult.statistic) }}</span></div><span>点击指标可设为下次直方图指标</span></header>
          <div class="comparison-table-wrap">
            <table class="comparison-metrics-table">
              <thead><tr><th>指标</th><th>基线 {{ metricLabel(comparisonResult.statistic) }}</th><th>候选 {{ metricLabel(comparisonResult.statistic) }}</th><th>绝对差值</th><th>百分比差值</th><th>基线样本</th><th>候选样本</th><th>状态</th></tr></thead>
              <tbody>
                <tr v-if="!comparisonResult.metrics.length"><td colspan="8" class="empty-cell">没有请求级延迟指标。</td></tr>
                <tr v-for="item in comparisonResult.metrics" v-else :key="item.metric" :class="[`metric-${metricStatus(item)}`, { selected: comparisonMetric === item.metric }]">
                  <td><button type="button" class="metric-selector" :title="`将 ${item.metric} 设为下次直方图指标`" @click="chooseMetric(item.metric)"><i />{{ stageLabel(item.metric) }}<code>{{ item.metric }}</code></button></td>
                  <td class="numeric">{{ formatDuration(metricValue(item, "baseline"), 3) }}</td>
                  <td class="numeric">{{ formatDuration(metricValue(item, "candidate"), 3) }}</td>
                  <td class="numeric delta-cell">{{ formatSignedDuration(metricDelta(item).absolute_delta_ms) }}</td>
                  <td class="numeric delta-cell">{{ formatPercent(metricDelta(item).percent_delta, metricValue(item, "baseline"), metricValue(item, "candidate")) }}</td>
                  <td class="numeric">{{ formatInteger(item.baseline_count) }}</td>
                  <td class="numeric">{{ formatInteger(item.candidate_count) }}</td>
                  <td><span class="metric-status" :class="metricStatus(item)">{{ metricStatusLabel(item) }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="comparison-histogram-section" aria-labelledby="comparison-histogram-title">
          <header>
            <div><strong id="comparison-histogram-title">共享边界延迟直方图</strong><span>{{ histogram?.metric ? `${stageLabel(histogram.metric)} · ${histogram.metric}` : "没有选定指标" }}</span></div>
            <span class="comparison-chart-legend"><i class="baseline" />基线 {{ formatInteger(histogram?.baseline_count) }}<i class="candidate" />候选 {{ formatInteger(histogram?.candidate_count) }}</span>
          </header>
          <div v-if="!histogram?.buckets.length" class="comparison-chart-empty">该指标没有可绘制的共同分桶样本。</div>
          <div v-else ref="histogramHost" class="comparison-histogram-host">
            <svg class="comparison-histogram" :viewBox="`0 0 ${histogramWidth} ${HISTOGRAM_HEIGHT + 30}`" role="group" :aria-label="`${histogram.metric} 共享边界直方图，基线 ${histogram.baseline_count} 个样本，候选 ${histogram.candidate_count} 个样本`">
              <line class="comparison-axis" :x1="HISTOGRAM_INSET" :x2="histogramWidth - HISTOGRAM_INSET" :y1="HISTOGRAM_HEIGHT" :y2="HISTOGRAM_HEIGHT" />
              <g v-for="bar in histogramGeometry.bars" :key="bar.index" class="comparison-histogram-bar" role="group" tabindex="0" :aria-label="bucketLabel(bar.index)" @pointerenter="hoveredBucketIndex = bar.index" @pointerleave="hoveredBucketIndex = null" @focus="hoveredBucketIndex = bar.index" @blur="hoveredBucketIndex = null">
                <title>{{ bucketLabel(bar.index) }}</title>
                <rect class="bar-hit" :x="HISTOGRAM_INSET + bar.x" y="0" :width="bar.width" :height="HISTOGRAM_HEIGHT" />
                <rect class="bar-baseline" :x="HISTOGRAM_INSET + bar.x" :y="bar.baselineY" :width="bar.width" :height="Math.max(bar.baselineHeight, histogram.buckets.find((item) => item.index === bar.index)?.baseline_count ? 1 : 0)" />
                <rect class="bar-candidate" :x="HISTOGRAM_INSET + bar.candidateX" :y="bar.candidateY" :width="bar.candidateWidth" :height="Math.max(bar.candidateHeight, histogram.buckets.find((item) => item.index === bar.index)?.candidate_count ? 1 : 0)" />
              </g>
              <text class="comparison-axis-label" :x="HISTOGRAM_INSET" :y="HISTOGRAM_HEIGHT + 18">{{ formatDuration(histogram.minimum) }}</text>
              <text class="comparison-axis-label end" :x="histogramWidth - HISTOGRAM_INSET" :y="HISTOGRAM_HEIGHT + 18">{{ formatDuration(histogram.maximum) }}</text>
            </svg>
            <aside v-if="hoveredBucket && hoveredBar" class="comparison-histogram-tooltip" :style="{ left: `${Math.min(94, Math.max(6, (HISTOGRAM_INSET + hoveredBar.x + hoveredBar.width / 2) * 100 / histogramWidth))}%` }" role="status">
              <strong>{{ formatDuration(hoveredBucket.start) }} 至 {{ formatDuration(hoveredBucket.end) }}</strong>
              <span>基线 {{ hoveredBucket.baseline_count }} · 候选 {{ hoveredBucket.candidate_count }}</span>
            </aside>
          </div>
        </section>

        <section class="flame-diff-section" aria-labelledby="flame-diff-title">
          <header class="flame-diff-heading">
            <div><strong id="flame-diff-title">聚合 Span 墙钟差异</strong><span>总墙钟受调用数量和工作负载影响，不是单次请求时延；包含等待、I/O 与异步暂停，不是 CPU。</span></div>
          </header>
          <div class="flame-diff-legend" role="group" aria-label="差异状态图例">
            <span v-for="status in ['improved', 'unchanged', 'regressed', 'added', 'removed'] as FlameDiffStatus[]" :key="status"><i :class="status" />{{ statusLabel(status) }}</span>
            <small>宽度按 max(基线, 候选) 包含墙钟权重分区</small>
          </div>
          <nav v-if="focusId" class="profile-breadcrumb" aria-label="差异图焦点路径">
            <button type="button" @click="navigateFocus('')">全部根</button>
            <template v-for="item in breadcrumbs" :key="item.path_id"><i>/</i><button type="button" :class="{ current: item.path_id === focusId }" @click="navigateFocus(item.path_id)">{{ item.frame[1] }}</button></template>
          </nav>
          <div v-if="!comparisonResult.flame_diff.length" class="comparison-chart-empty">没有可重建的完整 Span 路径。</div>
          <div v-else ref="flameHost" class="flame-diff-scroll">
            <svg class="flame-diff-chart" :width="flameChartWidth" :height="flameHeight" :viewBox="`0 0 ${flameChartWidth} ${flameHeight}`" role="group" aria-label="按完整组件、名称和来源路径重建的工具化 Span 墙钟差异图">
              <defs><pattern id="flame-diff-removed-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="rgba(var(--inverse-rgb), .75)" stroke-width="2" /></pattern></defs>
              <rect class="flame-diff-background" :width="flameChartWidth" :height="flameHeight" />
              <g class="flame-diff-depths"><g v-for="depth in flameDepths" :key="depth"><rect :x="FLAME_LEFT" :y="depth * FLAME_ROW_HEIGHT" :width="flamePlotWidth" :height="FLAME_ROW_HEIGHT" :class="{ alternate: depth % 2 }" /><text x="6" :y="depth * FLAME_ROW_HEIGHT + 17">D{{ depth }}</text></g></g>
              <g v-for="rect in flameRects" :key="rect.entry.path_id" class="flame-diff-node" :class="[`status-${rect.entry.status}`, { selected: selection?.kind === 'comparison' && selection.data.path_id === rect.entry.path_id, 'search-match': searchQuery && matches(rect.entry), 'search-dim': searchQuery && !matches(rect.entry) }]" role="button" tabindex="0" :aria-label="flameLabel(rect.entry)" @click.stop="inspectFlame(rect.entry)" @dblclick.stop="focusFlame(rect.entry)" @keydown="handleFlameKey($event, rect.entry)" @pointerenter="showFlameTooltip($event, rect.entry)" @pointermove="showFlameTooltip($event, rect.entry)" @pointerleave="hoverEntry = null">
                <title>{{ flameLabel(rect.entry) }}</title>
                <rect class="flame-diff-shape" :x="FLAME_LEFT + rect.x" :y="rect.y" :width="rect.width" :height="rect.height" />
                <rect v-if="rect.entry.status === 'removed'" class="flame-diff-removed-pattern" :x="FLAME_LEFT + rect.x" :y="rect.y" :width="rect.width" :height="rect.height" />
                <rect class="flame-self-baseline" :x="FLAME_LEFT + rect.x" :y="rect.y + rect.height - 5" :width="Math.min(rect.width, rect.baselineSelfWidth)" height="2" />
                <rect class="flame-self-candidate" :x="FLAME_LEFT + rect.x" :y="rect.y + rect.height - 2" :width="Math.min(rect.width, rect.candidateSelfWidth)" height="2" />
                <svg v-if="rect.width > 22" class="profile-clipped-label" :x="FLAME_LEFT + rect.x + 4" :y="rect.y" :width="Math.max(0, rect.width - 8)" :height="rect.height - 5" overflow="hidden"><text x="0" y="15">{{ rect.entry.frame[1] }} · {{ statusLabel(rect.entry.status) }}</text></svg>
              </g>
            </svg>
            <aside v-if="hoverInfo" class="profile-tooltip flame-diff-tooltip" :style="{ left: `${hoverPoint.x}px`, top: `${hoverPoint.y}px` }" role="status">
              <strong>{{ hoverInfo.title }}</strong><code>{{ hoverInfo.subtitle }}</code><span v-for="line in hoverInfo.lines" :key="line">{{ line }}</span><small>双击或 Shift+Enter 聚焦 · 单击或 Enter 检查</small>
            </aside>
          </div>
        </section>
      </div>
    </template>
  </section>
</template>
