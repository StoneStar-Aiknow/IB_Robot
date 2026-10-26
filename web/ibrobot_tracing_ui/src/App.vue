<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { RouterView, useRoute, useRouter } from "vue-router";
import ProfilerHeader from "./components/ProfilerHeader.vue";
import ScopeBar from "./components/ScopeBar.vue";
import AsyncState from "./components/AsyncState.vue";
import WelcomePanel from "./components/WelcomePanel.vue";
import { useProfilerStore } from "./stores/profiler";
import { formatInteger, shortId } from "./utils/format";
import { DETAIL_TABS } from "./utils/routeState";
import { useProfilerNavigation } from "./useProfilerNavigation";
import { dismissOutsidePopovers, dismissTopPopover, keepOnePopover } from "./utils/popovers";
import { asyncView } from "./utils/asyncView";
import { routeLoadError } from "./router";

const DetailPane = asyncView(() => import("./components/DetailPane.vue"), "分析工作区");
const TopologyGraph = asyncView(() => import("./components/TopologyGraph.vue"), "拓扑视图");

function retryRoute(): void {
  if (routeLoadError.value) void router.push(routeLoadError.value.target).catch(() => { /* Keep the recovery panel. */ });
}
function reloadRoute(): void {
  if (routeLoadError.value) window.location.assign(router.resolve(routeLoadError.value.target).href);
}

async function focusWorkspace(): Promise<void> {
  await nextTick();
  if (!isGuide.value && store.analysisState === "ready") document.getElementById("workspace-view-heading")?.focus({ preventScroll: true });
}

const store = useProfilerStore();
const route = useRoute();
const router = useRouter();
const isGuide = computed(() => route.name === "guide");
const { analyze, recover } = useProfilerNavigation(store);
const {
  activeTab,
  analysisId,
  analysisState,
  analysisError,
  analysisMissing,
  topologyOpen,
  capabilities,
  summary,
  requestId,
  componentId,
  spanProfile,
  criticalPath,
  spanAnalysisMode,
  searchQuery,
  selection,
  comparisonJob,
  comparisonResult,
} = storeToRefs(store);
const topPercent = ref(35);
const resizing = ref(false);
const workspace = ref<HTMLElement | null>(null);
const recoverySource = computed(() => store.sources.find((item) => item.id === store.restoreSourceId));
const canRecover = computed(() => recoverySource.value && (!store.restoreSourceVersion || recoverySource.value.version === store.restoreSourceVersion));
let inspectorTrigger: HTMLElement | null = null;

const statusText = computed(() => {
  if (analysisState.value === "loading") return "正在解析追踪";
  if (analysisState.value === "error") return "分析不可用";
  if (analysisState.value === "ready") return "分析就绪";
  return "等待载入追踪";
});

function chooseSource(): void {
  store.requestId = "";
  store.componentId = "";
  store.searchQuery = "";
  store.activeTab = "summary";
  store.closeAnalysis();
  void router.push({ name: "analysis" }).then(() => nextTick(() => document.querySelector<HTMLElement>(".welcome-panel h1")?.focus()));
}

function beginResize(event: PointerEvent): void {
  if (!workspace.value || window.matchMedia("(max-width: 900px)").matches) return;
  resizing.value = true;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}

function resize(event: PointerEvent): void {
  if (!resizing.value || !workspace.value) return;
  const bounds = workspace.value.getBoundingClientRect();
  topPercent.value = Math.min(65, Math.max(25, ((event.clientY - bounds.top) / bounds.height) * 100));
}

function endResize(): void {
  resizing.value = false;
}

function handleKeyboard(event: KeyboardEvent): void {
  if (isGuide.value) {
    if (event.key === "Escape" && dismissTopPopover()) event.preventDefault();
    return;
  }
  const target = event.target as HTMLElement;
  const editing = target.matches("input, select, textarea, [contenteditable='true']");
  if (event.key === "/" && !editing) {
    event.preventDefault();
    const search = document.getElementById("global-trace-search") ?? document.getElementById("request-id-search");
    if (search && !search.getClientRects().length) {
      document.getElementById(search.id === "global-trace-search" ? "view-tools-toggle" : "scope-tools-toggle")?.click();
    }
    void nextTick(() => search?.focus());
  } else if (event.key === "Escape") {
    if (dismissTopPopover()) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (selection.value) store.selection = null;
    else if (activeTab.value === "requests" && store.requestListFilter) store.requestListFilter = "";
    else if (searchQuery.value) searchQuery.value = "";
  } else if (event.altKey && /^[1-9]$/.test(event.key)) {
    event.preventDefault();
    const tabs = DETAIL_TABS.filter((tab) => tab !== "comparison" || capabilities.value?.baseline_compare);
    const tab = tabs[Number(event.key) - 1];
    if (!tab) return;
    activeTab.value = tab;
    store.fetchTab(activeTab.value);
  }
}

watch(capabilities, (value) => {
  if (value && !value.baseline_compare && activeTab.value === "comparison") activeTab.value = "summary";
});
watch(selection, async (value, previous) => {
  if (value && !previous) inspectorTrigger = document.activeElement as HTMLElement;
  await nextTick();
  if (value) {
    document.getElementById("inspector-heading")?.focus({ preventScroll: true });
  } else if (!value && previous && document.activeElement === document.body) {
    (inspectorTrigger?.isConnected ? inspectorTrigger : document.getElementById("request-id-search"))?.focus();
  }
});
watch(analysisState, async (value, previous) => {
  if (!isGuide.value && value === "ready" && previous !== "ready") {
    await nextTick();
    document.getElementById("workspace-view-heading")?.focus({ preventScroll: true });
  }
});
watch(isGuide, async (value, previous) => {
  if (!value && previous) {
    await nextTick();
    (document.getElementById("workspace-view-heading") ?? document.querySelector<HTMLElement>(".welcome-panel h1"))?.focus();
  }
});
watch(topologyOpen, (value) => {
  try { localStorage.setItem("ibrobot.tracing.topology", String(value)); } catch { /* Optional layout preference. */ }
});

onMounted(() => {
  store.loadSources();
  try { topologyOpen.value = localStorage.getItem("ibrobot.tracing.topology") === "true"; } catch { /* Default is collapsed. */ }
  window.addEventListener("keydown", handleKeyboard);
  document.addEventListener("pointerdown", dismissOutsidePopovers, true);
  document.addEventListener("toggle", keepOnePopover, true);
});
onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleKeyboard);
  document.removeEventListener("pointerdown", dismissOutsidePopovers, true);
  document.removeEventListener("toggle", keepOnePopover, true);
});
</script>

<template>
  <div class="profiler-app" :class="{ resizing }">
    <ProfilerHeader @change-source="chooseSource" />
    <section v-if="routeLoadError" class="context-error" role="alert">
      <strong>目标页面暂时无法载入</strong><p>当前分析仍保留。请检查连接后重试；资源已更新时可刷新打开目标页面。</p>
      <div class="state-actions"><button class="button secondary" type="button" @click="retryRoute">重试打开页面</button><button class="button secondary" type="button" @click="reloadRoute">刷新打开目标页面</button><button class="text-button" type="button" @click="routeLoadError = null">继续当前分析</button></div>
    </section>
    <ScopeBar v-else-if="!isGuide && analysisState === 'ready'" />
    <div v-else aria-hidden="true" />
    <main ref="workspace" class="workspace" :class="{ 'with-topology': !isGuide && topologyOpen && analysisState === 'ready' }" :style="{ '--top-pane': `${topPercent}%` }">
      <RouterView v-if="isGuide" />
      <WelcomePanel v-else-if="analysisState === 'idle'" @analyze="analyze" />
      <section v-else-if="analysisState === 'loading'" class="analysis-loading">
        <AsyncState state="loading" loading-message="正在解析 trace 文件并准备分析…" />
        <button class="button secondary" type="button" @click="chooseSource">停止等待，返回首页</button>
        <p>仅停止当前页面等待，后台解析可能继续；稍后可重新打开同一 trace 文件。</p>
      </section>
      <section v-else-if="analysisState === 'error'" class="state-view recovery-view" role="alert">
        <h1>{{ analysisMissing ? "分析结果已过期或不存在" : "暂时无法读取分析" }}</h1>
        <p>{{ analysisMissing ? "原始 trace 文件可能仍在。重新分析后可继续查看原请求和视图。" : analysisError }}</p>
        <p v-if="analysisMissing && recoverySource && !canRecover">原始 trace 文件已变化，请选择 trace 文件重新开始，避免混用旧请求。</p>
        <p v-else-if="analysisMissing && !recoverySource">这个链接没有可恢复的原始 trace 文件，请重新选择。</p>
        <div class="state-actions">
          <button v-if="analysisMissing && canRecover" class="button primary" type="button" @click="recover">重新分析原 trace 文件</button>
          <button v-else-if="!analysisMissing" class="button primary" type="button" @click="store.openAnalysis(analysisId)">重试</button>
          <button class="button secondary" type="button" @click="chooseSource">选择 trace 文件</button>
        </div>
      </section>
      <template v-else>
      <TopologyGraph v-if="topologyOpen" />
      <div
        v-if="topologyOpen"
        class="pane-resizer"
        role="separator"
        aria-label="调整拓扑与详情高度"
        aria-orientation="horizontal"
        :aria-valuenow="Math.round(topPercent)"
        tabindex="0"
        @pointerdown="beginResize"
        @pointermove="resize"
        @pointerup="endResize"
        @pointercancel="endResize"
        @lostpointercapture="endResize"
        @keydown.up.prevent="topPercent = Math.max(30, topPercent - 2)"
        @keydown.down.prevent="topPercent = Math.min(70, topPercent + 2)"
      ><i /></div>
      <Suspense @resolve="focusWorkspace"><DetailPane /><template #fallback><AsyncState state="loading" loading-message="正在载入分析工作区…" /></template></Suspense>
      </template>
    </main>
    <footer class="status-bar">
      <template v-if="isGuide"><span>使用指南 · 命令在对应机器的终端执行</span></template>
      <template v-else>
      <span class="status-state"><i :class="analysisState" />{{ statusText }}</span>
      <span v-if="summary">{{ formatInteger(summary.analysis.request_count) }} 个请求 · {{ formatInteger(summary.analysis.event_count) }} 个事件</span>
      <span v-if="requestId">请求 {{ shortId(requestId) }}</span>
      <span v-if="componentId">组件 {{ componentId }}</span>
      <span v-if="activeTab === 'span-profile'">{{ spanAnalysisMode === "call-tree" ? "调用树" : spanAnalysisMode === "request" ? "请求 Span 区间" : "聚合 Span 剖析" }} · 冰柱</span>
      <span v-if="activeTab === 'span-profile' && spanProfile">{{ formatInteger(spanProfile.returned_nodes) }}/{{ formatInteger(spanProfile.total_nodes) }} 个节点<span v-if="spanProfile.truncated">（已截断）</span></span>
      <span v-if="activeTab === 'critical-path' && criticalPath">{{ formatInteger(criticalPath.returned_segments) }}/{{ formatInteger(criticalPath.total_segments) }} 个墙钟分段<span v-if="criticalPath.truncated">（返回前缀）</span></span>
      <span v-if="activeTab === 'comparison' && comparisonJob">比较任务 {{ comparisonJob.status }}</span>
      <span v-if="activeTab === 'comparison' && comparisonResult">{{ comparisonResult.comparable ? "可计算差值" : "数据不足" }} · {{ comparisonResult.has_regression ? "有指标超阈值" : "未超阈值" }}</span>
      <span v-if="capabilities?.authentication === 'none'" class="security-warning">局域网无认证模式</span>
      <span class="status-spacer" />
      <span class="keyboard-hint">/ 过滤 · Esc 关闭检查器 · Alt+1…9 切换视图</span>
      </template>
    </footer>
  </div>
</template>
