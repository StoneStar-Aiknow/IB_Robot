<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import type { CriticalPathBottleneck, CriticalPathSegment, JsSafeInteger } from "../../api/types";
import { useProfilerStore } from "../../stores/profiler";
import { buildCriticalPathGeometry } from "../../utils/criticalPath";
import { boundaryLabel, formatDuration, formatInteger, recordStatusLabel, recordToData, shortId } from "../../utils/format";
import { componentColor, numericNs } from "../../utils/spanProfile";
import AppIcon from "../AppIcon.vue";
import AsyncState from "../AsyncState.vue";

const store = useProfilerStore();
const { criticalPath, requestId, tabError, tabState } = storeToRefs(store);
const selectedSegmentId = ref("");
const chartCursor = ref(0);
const showAllBottlenecks = ref(false);
const segmentDetails = ref<HTMLDetailsElement | null>(null);
const chart = ref<SVGSVGElement | null>(null);
const restrictedScope = computed(() => Boolean(criticalPath.value?.component_id));
const gapLabel = computed(() => restrictedScope.value ? "当前范围未归因" : "未归因间隙");

function segmentLabel(segment: CriticalPathSegment): string {
  return segment.kind === "unattributed" ? gapLabel.value : segment.label || "未命名记录";
}

const PLOT_LEFT = 68;
const PLOT_RIGHT = 14;
const svgWidth = ref(1_000);
const TRACK_Y = 34;
const TRACK_HEIGHT = 34;
const plotWidth = computed(() => svgWidth.value - PLOT_LEFT - PLOT_RIGHT);
let resizeObserver: ResizeObserver | null = null;

function updateWidth(): void {
  if (chart.value) svgWidth.value = Math.max(620, Math.floor(chart.value.parentElement?.clientWidth ?? 1_000));
}
watch(chart, (element, previous) => {
  if (previous?.parentElement) resizeObserver?.unobserve(previous.parentElement);
  if (element?.parentElement) resizeObserver?.observe(element.parentElement);
  updateWidth();
});
onMounted(() => {
  if (typeof ResizeObserver !== "undefined") resizeObserver = new ResizeObserver(updateWidth);
  if (chart.value?.parentElement) resizeObserver?.observe(chart.value.parentElement);
  updateWidth();
});
onBeforeUnmount(() => resizeObserver?.disconnect());

const geometry = computed(() => buildCriticalPathGeometry(
  criticalPath.value?.segments ?? [],
  criticalPath.value?.duration_ns ?? 0,
  criticalPath.value?.returned_duration_ns ?? 0,
  criticalPath.value?.omitted_duration_ns ?? 0,
  plotWidth.value,
));
const orderedSegments = computed(() => geometry.value.rects.map((rect) => rect.segment));
const bottlenecks = computed(() => [...(criticalPath.value?.bottlenecks ?? [])].sort((left, right) => left.rank - right.rank));
const visibleBottlenecks = computed(() => showAllBottlenecks.value ? bottlenecks.value : bottlenecks.value.slice(0, 3));

async function openSegments(): Promise<void> {
  if (segmentDetails.value) segmentDetails.value.open = true;
  await nextTick();
  segmentDetails.value?.querySelector<HTMLButtonElement>(".segment-inspect")?.focus();
}
const diagnostics = computed(() => [...(criticalPath.value?.diagnostics ?? [])].sort((left, right) => right.count - left.count || left.code.localeCompare(right.code)));
const axisTicks = computed(() => Array.from({ length: 5 }, (_, index) => {
  const ratio = index / 4;
  return {
    ratio,
    label: formatNs(numericNs(criticalPath.value?.duration_ns) * ratio),
  };
}));

function formatNs(value: JsSafeInteger | null | undefined): string {
  return formatDuration(numericNs(value) / 1_000_000, 3);
}

function kindLabel(kind: CriticalPathSegment["kind"] | CriticalPathBottleneck["kind"]): string {
  if (kind === "span") return "Span";
  if (kind === "flow") return "Flow";
  return gapLabel.value;
}

function owner(segment: CriticalPathSegment | CriticalPathBottleneck): string {
  return segment.component_id || segment.edge_id || "—";
}

function kindTotal(kind: CriticalPathSegment["kind"]): string {
  return formatNs(criticalPath.value?.totals.by_kind_ns[kind] ?? 0);
}

function segmentTitle(segment: CriticalPathSegment): string {
  return `${kindLabel(segment.kind)} · ${segmentLabel(segment)} · ${formatNs(segment.duration_ns)} · ${segment.percentage.toFixed(2)}% · ${owner(segment)}`;
}

function inspect(segment: CriticalPathSegment): void {
  selectedSegmentId.value = segment.id;
  store.selection = {
    kind: "critical-path",
    title: segmentLabel(segment),
    subtitle: `本段归因 ${formatNs(segment.duration_ns)} · ${owner(segment)}`,
    data: { ...recordToData(segment), request_id: criticalPath.value?.request_id },
  };
}

function moveSegment(event: KeyboardEvent): void {
  const count = orderedSegments.value.length;
  if (!count || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  chartCursor.value = event.key === "Home" ? 0 : event.key === "End" ? count - 1
    : (chartCursor.value + (event.key === "ArrowRight" ? 1 : -1) + count) % count;
  chart.value?.querySelectorAll<SVGElement>(".critical-path-segment")[chartCursor.value]?.focus();
}

function inspectBottleneck(item: CriticalPathBottleneck): void {
  const segment = orderedSegments.value.find((segment) => segment.kind === item.kind && segment.source_id === item.source_id);
  if (segment) inspect(segment);
}

function segmentFill(segment: CriticalPathSegment): string {
  return segment.kind === "span" ? componentColor(segment.component_id) : `url(#critical-path-${segment.kind})`;
}

watch(criticalPath, () => {
  selectedSegmentId.value = "";
  chartCursor.value = 0;
  showAllBottlenecks.value = false;
});
</script>

<template>
  <section class="critical-path-panel">
    <header class="critical-path-semantic-banner">
      <div><strong>请求耗时拆分</strong><b>主机侧墙钟</b></div>
      <p>墙钟归因包含等待、I/O 与异步暂停；它不是 CPU 时间，也不是跨主机调度 DAG。</p>
    </header>

    <div v-if="!requestId" class="critical-path-prompt state-view">
      <AppIcon name="trace" :size="24" />
      <strong>请选择一个请求以生成关键墙钟路径</strong>
      <span>从请求列表打开一条记录，或在上方粘贴完整请求 ID。</span>
      <button class="button secondary" type="button" @click="store.activeTab = 'requests'">打开请求列表</button>
    </div>

    <AsyncState
      context="耗时拆分"
      v-else
      class="critical-path-async"
      :state="tabState['critical-path']"
      :error="tabError['critical-path']"
      :empty="tabState['critical-path'] === 'ready' && !criticalPath"
      empty-title="没有关键路径结果"
      empty-message="该请求没有可用的墙钟边界或归因记录。"
      loading-message="正在计算请求关键墙钟路径…"
      @retry="store.fetchTab('critical-path', true)"
    >
      <div v-if="criticalPath" class="critical-path-scroll">
        <div class="critical-path-stat-strip" role="group" aria-label="关键路径摘要">
          <dl><dt>请求总墙钟</dt><dd>{{ formatNs(criticalPath.duration_ns) }}</dd></dl>
          <dl><dt>{{ restrictedScope ? '当前范围归因覆盖' : '归因覆盖' }}</dt><dd>{{ criticalPath.totals.coverage_percent.toFixed(2) }}%</dd></dl>
          <dl><dt>{{ gapLabel }} · Gap</dt><dd>{{ kindTotal("unattributed") }}</dd></dl>
          <dl><dt>返回分段</dt><dd>{{ formatInteger(criticalPath.returned_segments) }} / {{ formatInteger(criticalPath.total_segments) }}</dd></dl>
        </div>
        <details class="measurement-details">
          <summary>计时边界：{{ boundaryLabel(criticalPath.boundary_source) || '未提供' }} · 查看归因组成</summary>
          <div class="measurement-groups">
            <section aria-label="计时边界"><h2>计时边界</h2><dl>
              <div><dt>起点</dt><dd>{{ boundaryLabel(criticalPath.boundary_start_source) || '未提供' }}<code>{{ criticalPath.start_ns ?? '未提供' }} ns</code></dd></div>
              <div><dt>终点</dt><dd>{{ boundaryLabel(criticalPath.boundary_end_source) || '未提供' }}<code>{{ criticalPath.end_ns ?? '未提供' }} ns</code></dd></div>
              <div><dt>原始边界代码</dt><dd><code>{{ criticalPath.boundary_source || '未提供' }}</code></dd></div>
            </dl></section>
            <section aria-label="归因组成"><h2>归因组成</h2><dl>
              <div><dt>操作片段 · Span</dt><dd>{{ kindTotal('span') }}</dd></div>
              <div><dt>交接片段 · Flow</dt><dd>{{ kindTotal('flow') }}</dd></div>
              <div><dt>{{ gapLabel }} · Gap</dt><dd>{{ kindTotal('unattributed') }}</dd></div>
            </dl><p>这三类归因片段互不重叠，合起来是请求总墙钟；原始操作记录可能重叠，不能直接相加。</p></section>
          </div>
        </details>

        <div v-if="restrictedScope" class="scope-explanation" role="note">
          <span>请求总时间边界不变；这里只归因于选中组件及关联 Flow。其他组件已有的观测也可能落入当前未归因区间，不能据此认定整份请求缺少观测或发生了等待。</span>
          <button class="text-button" type="button" @click="store.componentId = ''">查看全部组件归因</button>
        </div>

        <div v-if="criticalPath.truncated" class="critical-path-truncation" role="status">
          <AppIcon name="warning" :size="14" />
          <strong>仅显示按时间排序的返回前缀</strong>
          <span>已返回 {{ formatInteger(criticalPath.returned_segments) }} 个分段（{{ formatNs(criticalPath.returned_duration_ns) }}）；尾部省略 {{ formatInteger(criticalPath.omitted_segments) }} 个分段，精确时长 {{ formatNs(criticalPath.omitted_duration_ns) }}。原因：<code>{{ criticalPath.truncation_reason }}</code></span>
        </div>

        <div v-if="diagnostics.length" class="critical-path-diagnostic-strip" role="region" tabindex="0" aria-label="关键路径诊断摘要">
          <strong>诊断</strong>
          <span v-for="item in diagnostics" :key="item.code" :title="item.message"><code>{{ item.code }}</code> {{ formatInteger(item.count) }}</span>
        </div>

        <section class="critical-path-chart-section" aria-labelledby="critical-path-chart-heading">
          <header>
            <div>
              <strong id="critical-path-chart-heading">墙钟分区时间线</strong>
            </div>
            <span class="critical-path-legend"><i class="span" />Span 组件 <i class="flow" />Flow <i class="gap" />Gap <template v-if="criticalPath.truncated"><i class="omitted" />省略尾部</template></span>
            <button class="text-button" type="button" @click="openSegments">查看分段列表</button>
          </header>
          <div v-if="numericNs(criticalPath.duration_ns) > 0" class="critical-path-chart-wrap">
            <svg
              ref="chart"
              class="critical-path-chart"
              :viewBox="`0 0 ${svgWidth} 94`"
              role="group"
              aria-labelledby="critical-path-chart-title critical-path-chart-description"
              @keydown="moveSegment"
            >
              <title id="critical-path-chart-title">请求关键墙钟路径</title>
              <desc id="critical-path-chart-description">按请求完整边界比例显示不重叠的 Span、Flow 和未归因墙钟分段；截断时单独标出未返回的尾部。</desc>
              <defs>
                <pattern id="critical-path-flow" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="8" height="8" fill="var(--chart-flow-surface)" /><line x1="0" y1="0" x2="0" y2="8" stroke="var(--chart-flow)" stroke-width="4" />
                </pattern>
                <pattern id="critical-path-unattributed" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="8" height="8" fill="var(--chart-gap-surface)" /><line x1="0" y1="0" x2="0" y2="8" stroke="var(--chart-gap)" stroke-width="3" />
                </pattern>
                <pattern id="critical-path-omitted" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="7" height="7" fill="var(--chart-omitted-surface)" /><line x1="0" y1="0" x2="0" y2="7" stroke="var(--chart-omitted)" stroke-width="2" />
                </pattern>
              </defs>
              <g class="critical-path-axis">
                <line :x1="PLOT_LEFT" :x2="svgWidth - PLOT_RIGHT" y1="74" y2="74" />
                <g v-for="tick in axisTicks" :key="tick.ratio" :transform="`translate(${PLOT_LEFT + tick.ratio * plotWidth}, 0)`">
                  <line y1="25" y2="78" />
                  <text :text-anchor="tick.ratio === 0 ? 'start' : tick.ratio === 1 ? 'end' : 'middle'" y="89">{{ tick.label }}</text>
                </g>
              </g>
              <rect class="critical-path-track" :x="PLOT_LEFT" :y="TRACK_Y" :width="plotWidth" :height="TRACK_HEIGHT" />
              <g
                v-for="(rect, index) in geometry.rects"
                :key="rect.segment.id"
                class="critical-path-segment"
                :class="{ selected: selectedSegmentId === rect.segment.id }"
                role="button"
                :tabindex="chartCursor === index ? 0 : -1"
                @focus="chartCursor = index"
                :aria-label="`${segmentTitle(rect.segment)}。按回车或空格检查。`"
                @click="inspect(rect.segment)"
                @keydown.enter.prevent="inspect(rect.segment)"
                @keydown.space.prevent="inspect(rect.segment)"
              >
                <title>{{ segmentTitle(rect.segment) }}</title>
                <rect
                  :x="PLOT_LEFT + rect.x"
                  :y="TRACK_Y"
                  :width="rect.width"
                  :height="TRACK_HEIGHT"
                  :fill="segmentFill(rect.segment)"
                />
                <svg
                  v-if="rect.width > 54"
                  class="critical-path-clipped-label"
                  :x="PLOT_LEFT + rect.x + 4"
                  :y="TRACK_Y"
                  :width="Math.max(0, rect.width - 8)"
                  :height="TRACK_HEIGHT"
                  overflow="hidden"
                ><rect v-if="rect.segment.kind !== 'span'" x="0" y="7" width="100%" height="20" rx="2" fill="var(--chart-label-surface)" /><text x="0" y="21">{{ segmentLabel(rect.segment) }}</text></svg>
              </g>
              <g v-if="criticalPath.truncated && geometry.omitted" class="critical-path-omitted-tail">
                <title>省略尾部：{{ formatInteger(criticalPath.omitted_segments) }} 个分段，{{ formatNs(criticalPath.omitted_duration_ns) }}</title>
                <rect :x="PLOT_LEFT + geometry.omitted.x" :y="TRACK_Y" :width="geometry.omitted.width" :height="TRACK_HEIGHT" />
                <text v-if="geometry.omitted.width > 84" :x="PLOT_LEFT + geometry.omitted.x + 6" :y="TRACK_Y + 21">OMITTED · {{ formatNs(criticalPath.omitted_duration_ns) }}</text>
              </g>
            </svg>
          </div>
          <div v-else class="critical-path-zero" role="status">请求边界时长为 0；没有可按比例绘制的墙钟区间。</div>
        </section>

        <div class="critical-path-detail-grid">
          <section class="critical-path-bottlenecks">
            <header><div><strong>耗时最多的记录</strong><span>按归因墙钟排序 · 默认显示前三项</span></div><b>{{ formatInteger(bottlenecks.length) }} 项</b></header>
            <div class="critical-path-table-wrap">
              <table>
                <caption class="sr-only">已归因记录按耗时排序，点击名称查看原始证据入口</caption>
                <thead><tr><th>排名</th><th>记录 / 组件</th><th>归因时长</th><th>占比</th></tr></thead>
                <tbody>
                  <tr v-if="!bottlenecks.length"><td colspan="4" class="empty-cell">没有可排序的已归因瓶颈。</td></tr>
                  <tr v-for="item in visibleBottlenecks" v-else :key="`${item.rank}:${item.kind}:${item.source_id}`">
                    <td class="numeric">#{{ item.rank }}</td>
                    <td><button class="text-button evidence-button" type="button" :title="item.label" @click="inspectBottleneck(item)">{{ item.label || "—" }}</button><code>{{ owner(item) }}</code><small>{{ kindLabel(item.kind) }} · {{ recordStatusLabel(item.status) }} · {{ formatInteger(item.segment_count) }} 段</small></td>
                    <td class="numeric">{{ formatNs(item.duration_ns) }}</td>
                    <td class="numeric">{{ item.percentage.toFixed(2) }}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <button v-if="bottlenecks.length > 3" class="text-button ranking-toggle" type="button" :aria-expanded="showAllBottlenecks" @click="showAllBottlenecks = !showAllBottlenecks">{{ showAllBottlenecks ? '收起为前三项' : `查看全部 ${bottlenecks.length} 项` }}</button>
          </section>

          <details ref="segmentDetails" class="critical-path-segments-section">
            <summary>按时间分段 · {{ formatInteger(orderedSegments.length) }} 段 · 查看完整明细</summary>
            <div class="critical-path-table-wrap segment-table-wrap">
              <table class="critical-path-segment-table">
                <thead><tr><th>#</th><th>偏移</th><th>记录</th><th>时长</th><th>占比</th><th>类型</th><th>组件 / 边</th><th>状态</th></tr></thead>
                <tbody>
                  <tr v-if="!orderedSegments.length"><td colspan="8" class="empty-cell">边界内没有可返回的墙钟分段；请检查诊断。</td></tr>
                  <tr
                    v-for="segment in orderedSegments"
                    v-else
                    :key="segment.id"
                    :class="{ selected: selectedSegmentId === segment.id }"
                    @click="inspect(segment)"
                  >
                    <td class="numeric">{{ segment.index + 1 }}</td>
                    <td class="numeric">{{ formatNs(segment.offset_ns) }}</td>
                    <td><button class="text-button evidence-button segment-inspect" type="button" :aria-label="`${segmentTitle(segment)}。打开归因详情。`" @click.stop="inspect(segment)">{{ segmentLabel(segment) }}</button><small v-if="segment.diagnostics.length" :title="segment.diagnostics.join(', ')">{{ segment.diagnostics.join(", ") }}</small></td>
                    <td class="numeric">{{ formatNs(segment.duration_ns) }}</td>
                    <td class="numeric">{{ segment.percentage.toFixed(2) }}%</td>
                    <td><span class="critical-path-kind" :class="segment.kind">{{ kindLabel(segment.kind) }}</span></td>
                    <td><code :title="owner(segment)">{{ owner(segment) }}</code></td>
                    <td>{{ segment.kind === 'unattributed' ? '未归因' : recordStatusLabel(segment.status) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </details>

          <section v-if="diagnostics.length" class="critical-path-diagnostics">
            <header><div><strong>诊断详情</strong><span>边界、记录排除与时钟完整性</span></div><b>{{ formatInteger(diagnostics.length) }}</b></header>
            <ul>
              <li v-for="item in diagnostics" :key="item.code">
                <code>{{ item.code }}</code><strong>{{ formatInteger(item.count) }}</strong><span>{{ item.message }}</span><small v-if="item.source_ids.length" :title="item.source_ids.join(', ')">{{ item.source_ids.map((id) => shortId(id, 16)).join(", ") }}</small>
              </li>
            </ul>
          </section>
        </div>
      </div>
    </AsyncState>
  </section>
</template>
