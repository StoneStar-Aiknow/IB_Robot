import { nextTick, onBeforeUnmount, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useProfilerStore, type RecordTab } from "./stores/profiler";
import { buildProfilerQuery, parseProfilerRoute } from "./utils/routeState";

/** Task navigation gets history; edits to a filter replace the current task URL. */
export function useProfilerNavigation(store: ReturnType<typeof useProfilerStore>) {
  const route = useRoute();
  const router = useRouter();
  let applying = false;
  let routeGeneration = 0;

  const routeState = () => ({
    activeTab: store.activeTab, requestId: store.requestId, componentId: store.componentId,
    distributionMetric: store.distributionMetric, metric: store.metric, graphView: store.graphView,
    spanAnalysisMode: store.spanAnalysisMode, searchQuery: store.activeTab === "requests" ? store.requestListFilter : store.searchQuery,
    selectedBaselineSourceId: store.selectedBaselineSourceId, comparisonStatistic: store.comparisonStatistic,
    comparisonRelativeThreshold: store.comparisonRelativeThreshold, comparisonAbsoluteThresholdMs: store.comparisonAbsoluteThresholdMs,
    comparisonMetric: store.comparisonMetric, sourceId: store.restoreSourceId, sourceVersion: store.restoreSourceVersion,
    comparisonResultId: store.comparisonResultId,
    requestOffset: store.requestsOffset, requestSortBy: store.requestSortBy, requestDescending: store.requestDescending,
    requestFilter: store.requestListFilter,
    recordOffset: store.recordPages[store.activeTab as RecordTab]?.offset ?? 0,
    recordName: store.recordPages[store.activeTab as RecordTab]?.name ?? "",
  });

  async function applyRoute(): Promise<void> {
    const generation = ++routeGeneration;
    if (route.name !== "analysis") { applying = false; return; }
    const returningHome = !route.params.analysisId && store.analysisState !== "idle";
    applying = true;
    if (store.sourcesState !== "ready") await store.loadSources();
    if (generation !== routeGeneration) return;
    const state = parseProfilerRoute(route.query);
    store.activeTab = state.activeTab;
    store.requestId = state.requestId;
    store.componentId = state.componentId;
    store.distributionMetric = state.distributionMetric;
    store.metric = state.metric;
    store.graphView = state.graphView;
    store.spanAnalysisMode = state.spanAnalysisMode;
    store.requestListFilter = state.requestFilter ?? "";
    store.searchQuery = state.activeTab === "requests" ? "" : state.searchQuery;
    store.selectedBaselineSourceId = state.selectedBaselineSourceId;
    store.comparisonStatistic = state.comparisonStatistic;
    store.comparisonRelativeThreshold = state.comparisonRelativeThreshold;
    store.comparisonAbsoluteThresholdMs = state.comparisonAbsoluteThresholdMs;
    store.comparisonMetric = state.comparisonMetric;
    store.requestsOffset = state.requestOffset ?? 0;
    store.requestSortBy = state.requestSortBy ?? "total_ms";
    store.requestDescending = state.requestDescending ?? true;
    void store.refreshScopedData(false);
    if (["events", "spans", "flows"].includes(state.activeTab)) {
      const page = store.recordPages[state.activeTab as RecordTab];
      page.offset = state.recordOffset ?? 0;
      page.name = state.recordName ?? "";
    }
    const id = String(route.params.analysisId ?? "");
    if (id !== store.analysisId || state.sourceId) {
      store.restoreSourceId = state.sourceId || "";
      store.restoreSourceVersion = state.sourceId ? state.sourceVersion || "" : "";
    }
    await nextTick();
    if (generation !== routeGeneration) return;
    if (!id) {
      store.closeAnalysis();
      store.restoreSourceId = "";
      store.restoreSourceVersion = "";
      await nextTick();
      if (returningHome && generation === routeGeneration) document.querySelector<HTMLElement>(".welcome-panel h1")?.focus({ preventScroll: true });
    }
    else if (id !== store.analysisId) await store.openAnalysis(id);
    else if (store.analysisState === "ready") await store.fetchTab(store.activeTab);
    if (generation !== routeGeneration) return;
    const comparisonId = state.comparisonResultId ?? "";
    if (store.analysisState === "ready" && (comparisonId !== store.comparisonResultId
      || (comparisonId && !store.comparisonResult && store.comparisonState === "idle"))) {
      await store.openComparisonResult(comparisonId);
    }
    if (generation !== routeGeneration) return;
    applying = false;
    await syncRoute(routeState());
  }

  watch(() => route.fullPath, applyRoute, { immediate: true });
  async function syncRoute(state: ReturnType<typeof routeState>): Promise<void> {
    if (applying || route.name !== "analysis") return;
    const target = { name: "analysis", params: store.analysisId ? { analysisId: store.analysisId } : {}, query: buildProfilerQuery(state) };
    if (router.resolve(target).fullPath === route.fullPath) return;
    const previous = parseProfilerRoute(route.query);
    const taskChanged = String(route.params.analysisId ?? "") !== store.analysisId
      || previous.activeTab !== state.activeTab || previous.requestId !== state.requestId
      || previous.spanAnalysisMode !== state.spanAnalysisMode;
    await (taskChanged ? router.push(target) : router.replace(target));
  }
  watch(routeState, syncRoute);
  watch(() => [store.activeTab, store.analysisState, store.requestId, store.componentId, store.spanAnalysisMode], () => {
    if (store.analysisState === "ready") void store.refreshScopedData().then(() => store.fetchTab(store.activeTab));
  });
  watch(() => [store.requestsOffset, store.requestSortBy, store.requestDescending], () => {
    if (store.analysisState === "ready") void store.fetchRequests();
  });
  watch(() => [store.metric, store.graphView, store.requestId], () => {
    if (store.analysisState === "ready") void store.refreshTopology();
  });
  watch(() => store.graphView, () => {
    if (!applying) store.componentId = "";
  });
  watch(() => store.activeTab, (tab, previous) => {
    if (tab !== previous) {
      store.selection = null;
      if (!applying) store.searchQuery = "";
    }
  }, { flush: "sync" });
  watch(() => [store.activeTab, store.requestId] as const, async ([, request], [, previousRequest]) => {
    // A tablist owns its arrow-key focus. Task entry from another surface owns
    // heading/row focus; do not let the global fallback undo the local action.
    const tablist = request === previousRequest ? document.activeElement?.closest<HTMLElement>('[role="tablist"]') : null;
    await nextTick();
    if (store.analysisState !== "ready") return;
    const selectedTab = tablist?.isConnected ? tablist.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]') : null;
    if (selectedTab) {
      selectedTab.focus({ preventScroll: true });
      return;
    }
    const row = store.activeTab === "requests" ? document.getElementById(`grid-row-${store.requestId}`) : null;
    (row ?? document.getElementById("workspace-view-heading"))?.focus({ preventScroll: true });
  });
  onBeforeUnmount(() => { routeGeneration += 1; });

  async function analyze(): Promise<void> {
    try { await store.analyzeSelectedSource(); } catch { /* The store owns actionable error state. */ }
  }

  async function recover(): Promise<void> {
    const source = store.sources.find((item) => item.id === store.restoreSourceId);
    if (!source || (store.restoreSourceVersion && source.version !== store.restoreSourceVersion)) return;
    store.sourceId = source.id;
    try { await store.analyzeSelectedSource(true); } catch { /* Keep the original request URL for retry. */ }
  }

  return { analyze, recover };
}
