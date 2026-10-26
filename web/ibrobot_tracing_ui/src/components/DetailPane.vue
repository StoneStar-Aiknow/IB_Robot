<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import { storeToRefs } from "pinia";
import type { DetailTab, SpanAnalysisMode } from "../api/types";
import { useProfilerStore } from "../stores/profiler";
import InspectorPanel from "./InspectorPanel.vue";
import ComparisonPanel from "./tabs/ComparisonPanel.vue";
import CriticalPathPanel from "./tabs/CriticalPathPanel.vue";
import DistributionPanel from "./tabs/DistributionPanel.vue";
import EventsPanel from "./tabs/EventsPanel.vue";
import FlowsPanel from "./tabs/FlowsPanel.vue";
import RequestsPanel from "./tabs/RequestsPanel.vue";
import SpansPanel from "./tabs/SpansPanel.vue";
import SummaryPanel from "./tabs/SummaryPanel.vue";
import SpanProfilePanel from "./tabs/SpanProfilePanel.vue";
import TimelinePanel from "./tabs/TimelinePanel.vue";
import WarningsPanel from "./tabs/WarningsPanel.vue";
import RequestContext from "./RequestContext.vue";
import { filterHint, isRequestView, RECORD_VIEWS, scopeLabel, VIEW_LABELS } from "../utils/workspace";
import { shortId } from "../utils/format";

const store = useProfilerStore();
const toolsOpen = ref(false);
const { activeTab, capabilities, requestId, componentId, components, searchQuery, spanAnalysisMode, selection, topologyOpen } = storeToRefs(store);
const requestView = computed(() => isRequestView(activeTab.value, spanAnalysisMode.value));
const overview = computed(() => ["summary", "distribution"].includes(activeTab.value) || (activeTab.value === "span-profile" && spanAnalysisMode.value === "aggregate"));
const scope = computed(() => scopeLabel(activeTab.value, requestId.value, componentId.value, spanAnalysisMode.value));
const hint = computed(() => filterHint(activeTab.value, spanAnalysisMode.value));
const viewFilter = computed({
  get: () => activeTab.value === "requests" ? store.requestListFilter : searchQuery.value,
  set: (value: string) => { if (activeTab.value === "requests") store.requestListFilter = value; else searchQuery.value = value; },
});
const scoped = computed(() => requestView.value || RECORD_VIEWS.includes(activeTab.value) || activeTab.value === "span-profile");
const hasTools = computed(() => Boolean(hint.value) || scoped.value);
const scopeComponents = computed(() => components.value.filter((item) => !["operation", "instant_event"].includes(item.kind)));
const tabs = computed<DetailTab[]>(() => requestView.value ? ["critical-path", "timeline", "span-profile"] : []);

function selectTab(tab: DetailTab, mode?: SpanAnalysisMode): void {
  store.cancelLookup();
  if (tab === "span-profile") void store.setSpanAnalysisMode(mode ?? (spanAnalysisMode.value === "aggregate" ? "request" : spanAnalysisMode.value));
  store.searchQuery = "";
  store.selection = null;
  activeTab.value = tab;
  store.fetchTab(tab);
}

async function finishFilters(): Promise<void> {
  if (!window.matchMedia("(max-width: 600px)").matches) return;
  toolsOpen.value = false;
  await nextTick();
  document.getElementById("workspace-view-heading")?.focus({ preventScroll: true });
}

function closeMenu(event: Event): void {
  const menu = (event.target as HTMLElement).closest("details");
  if (!menu) return;
  menu.open = false;
  menu.querySelector<HTMLElement>(":scope > summary")?.focus();
}

function moveTab(delta: number): void {
  const next = (tabs.value.indexOf(activeTab.value) + delta + tabs.value.length) % tabs.value.length;
  selectTab(tabs.value[next]);
  document.getElementById(`detail-tab-${tabs.value[next]}`)?.focus();
}
</script>

<template>
  <section class="detail-pane" :class="{ 'request-workspace': requestView && requestId }" aria-label="追踪详情">
    <nav class="workspace-nav" aria-label="主导航">
      <button class="nav-button" type="button" :aria-current="overview ? 'page' : undefined" @click="selectTab('summary')">概览</button>
      <button class="nav-button" type="button" :aria-current="activeTab === 'requests' || requestView ? 'page' : undefined" @click="selectTab('requests')">请求分析</button>
      <button v-if="capabilities?.baseline_compare" id="detail-tab-comparison" class="nav-button" type="button" :aria-current="activeTab === 'comparison' ? 'page' : undefined" @click="selectTab('comparison')">实验对比</button>
      <details class="more-views" data-popover>
        <summary>更多视图</summary>
        <div>
          <fieldset><legend>原始记录</legend>
            <button v-for="tab in (['events', 'spans', 'flows'] as DetailTab[])" :id="`detail-tab-${tab}`" :key="tab" class="text-button" type="button" @click="closeMenu($event); selectTab(tab)">{{ VIEW_LABELS[tab] }}</button>
          </fieldset>
          <fieldset><legend>分析与布局</legend>
            <button id="detail-tab-warnings" class="text-button" type="button" @click="closeMenu($event); selectTab('warnings')">数据质量</button>
            <button class="text-button" type="button" @click="closeMenu($event); selectTab('span-profile', 'aggregate')">全部调用聚合</button>
            <button class="text-button" type="button" :aria-expanded="topologyOpen" @click="topologyOpen = !topologyOpen; closeMenu($event)">{{ topologyOpen ? "收起拓扑" : "查看拓扑" }}</button>
          </fieldset>
        </div>
      </details>
    </nav>
    <header class="view-context">
      <div><h1 id="workspace-view-heading" tabindex="-1">{{ activeTab === 'span-profile' && spanAnalysisMode === 'aggregate' ? '全部调用聚合' : VIEW_LABELS[activeTab] }}</h1><span class="view-scope">{{ scope }}</span></div>
      <button v-if="hasTools" id="view-tools-toggle" class="text-button compact-only" type="button" :aria-expanded="toolsOpen" aria-controls="view-tools" :title="viewFilter || componentId" @click="toolsOpen = !toolsOpen">{{ toolsOpen ? '收起筛选' : viewFilter ? `筛选：${shortId(viewFilter, 12)}` : componentId ? '组件已筛选' : '筛选' }}</button>
      <div v-if="hasTools || activeTab === 'distribution'" id="view-tools" class="view-tools" :class="{ expanded: toolsOpen || !hasTools }">
        <details v-if="scoped" class="component-scope" data-popover><summary>{{ componentId ? "更改组件范围" : "筛选组件" }}</summary><label>组件<select v-model="componentId" aria-label="当前详情组件"><option value="">全部组件</option><option v-for="component in scopeComponents" :key="component.component_id" :value="component.component_id">{{ component.name }}</option></select></label></details>
        <button v-if="activeTab === 'distribution'" class="text-button" type="button" @click="selectTab('summary')">返回概览</button>
        <form v-if="hint" class="local-filter" @submit.prevent="finishFilters">
          <label for="global-trace-search">{{ hint }}</label>
          <div class="filter-entry"><input id="global-trace-search" v-model="viewFilter" type="search" :placeholder="hint" :aria-label="hint" /><button v-if="viewFilter" class="text-button" type="button" aria-label="清除过滤" @click="viewFilter = ''">清除</button><button class="text-button compact-only" type="submit">完成</button></div>
        </form>
        <button v-if="RECORD_VIEWS.includes(activeTab) && requestId" class="text-button" type="button" @click="store.requestId = ''; store.componentId = ''; store.searchQuery = ''">查看全部请求记录</button>
      </div>
    </header>
    <RequestContext v-if="requestView && requestId" />
    <nav v-if="tabs.length" class="detail-tabs" role="tablist" aria-label="当前任务视图" @keydown.left.prevent="moveTab(-1)" @keydown.right.prevent="moveTab(1)">
      <button v-for="tab in tabs" :id="`detail-tab-${tab}`" :key="tab" type="button" role="tab" :aria-selected="activeTab === tab" aria-controls="analysis-view" :tabindex="activeTab === tab ? 0 : -1" :class="{ active: activeTab === tab }" @click="selectTab(tab)">{{ VIEW_LABELS[tab] }}</button>
    </nav>
    <div class="detail-workarea" :class="{ 'with-inspector': selection }">
      <div id="analysis-view" class="detail-panel" :role="tabs.length ? 'tabpanel' : 'region'" :aria-labelledby="tabs.length ? `detail-tab-${activeTab}` : 'workspace-view-heading'">
        <SummaryPanel v-if="activeTab === 'summary'" />
        <RequestsPanel v-else-if="activeTab === 'requests'" />
        <DistributionPanel v-else-if="activeTab === 'distribution'" />
        <ComparisonPanel v-else-if="activeTab === 'comparison'" />
        <TimelinePanel v-else-if="activeTab === 'timeline'" />
        <SpanProfilePanel v-else-if="activeTab === 'span-profile'" />
        <CriticalPathPanel v-else-if="activeTab === 'critical-path'" />
        <EventsPanel v-else-if="activeTab === 'events'" />
        <SpansPanel v-else-if="activeTab === 'spans'" />
        <FlowsPanel v-else-if="activeTab === 'flows'" />
        <WarningsPanel v-else />
      </div>
      <InspectorPanel />
    </div>
  </section>
</template>
