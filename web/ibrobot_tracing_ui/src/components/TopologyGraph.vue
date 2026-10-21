<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { MarkerType, Panel, VueFlow, useVueFlow, type Edge, type Node, type NodeMouseEvent } from "@vue-flow/core";
import { Background } from "@vue-flow/background";
import { MiniMap } from "@vue-flow/minimap";
import { useProfilerStore } from "../stores/profiler";
import { formatDuration, recordToData, shortId } from "../utils/format";
import AppIcon from "./AppIcon.vue";
import AsyncState from "./AsyncState.vue";
import TopologyEdge from "./TopologyEdge.vue";
import TopologyNode from "./TopologyNode.vue";

const store = useProfilerStore();
const { topology, topologyState, topologyError, componentId, motionPaused, requestId, metric, graphView } = storeToRefs(store);
const graphSearch = ref("");
const nodes = ref<Node[]>([]);
const edges = ref<Edge[]>([]);
const layoutPending = ref(false);
const layoutError = ref("");
const { fitView, zoomIn, zoomOut } = useVueFlow({ id: "ibrobot-topology" });
const motionDuration = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 160;
const fit = () => fitView({ padding: 0.16, minZoom: 0.75, duration: motionDuration() });
let elkPromise: ReturnType<typeof createElk> | null = null;
let layoutGeneration = 0;
const flowColor = "var(--oe-success)";

async function createElk() {
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  return new ELK();
}

const shownComponents = computed(() => {
  if (!topology.value) return [];
  const all = topology.value.nodes;
  const byId = new Map(all.map((component) => [component.component_id, component]));
  let allowed = new Set(all.map((component) => component.component_id));
  if (componentId.value) {
    allowed = new Set([componentId.value]);
    let changed = true;
    while (changed) {
      changed = false;
      all.forEach((component) => {
        if (allowed.has(component.parent_id) && !allowed.has(component.component_id)) {
          allowed.add(component.component_id);
          changed = true;
        }
      });
    }
    let parent = byId.get(componentId.value)?.parent_id;
    while (parent) {
      allowed.add(parent);
      parent = byId.get(parent)?.parent_id;
    }
    topology.value.edges.forEach((edge) => {
      if (edge.source_id === componentId.value) allowed.add(edge.target_id);
      if (edge.target_id === componentId.value) allowed.add(edge.source_id);
    });
  }
  const term = graphSearch.value.trim().toLocaleLowerCase("zh-CN");
  if (term) {
    const matches = new Set(
      all
        .filter((component) => `${component.name} ${component.component_id} ${component.kind}`.toLocaleLowerCase("zh-CN").includes(term))
        .map((component) => component.component_id),
    );
    for (const id of [...matches]) {
      let parent = byId.get(id)?.parent_id;
      while (parent) {
        matches.add(parent);
        parent = byId.get(parent)?.parent_id;
      }
    }
    allowed = new Set([...allowed].filter((id) => matches.has(id)));
  }
  return all.filter((component) => allowed.has(component.component_id));
});

async function layoutGraph(): Promise<void> {
  const generation = ++layoutGeneration;
  layoutError.value = "";
  const topologyValue = topology.value;
  const components = shownComponents.value;
  if (!topologyValue || !components.length) {
    nodes.value = [];
    edges.value = [];
    layoutPending.value = false;
    return;
  }
  layoutPending.value = true;
  const shown = new Set(components.map((component) => component.component_id));
  const graphEdges = topologyValue.edges.filter((edge) => shown.has(edge.source_id) && shown.has(edge.target_id));
  try {
    const elk = await (elkPromise ??= createElk());
    const result = await elk.layout({
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.spacing.nodeNode": "46",
        "elk.layered.spacing.nodeNodeBetweenLayers": "196",
        "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      },
      children: components.map((component) => ({ id: component.component_id, width: 206, height: 88 })),
      edges: graphEdges.map((edge) => ({ id: edge.edge_id, sources: [edge.source_id], targets: [edge.target_id] })),
    });
    if (generation !== layoutGeneration) return;
    const byId = new Map(topologyValue.nodes.map((component) => [component.component_id, component]));
    const nextNodes: Node[] = (result.children ?? []).map((child) => {
      const component = byId.get(child.id)!;
      return {
        id: component.component_id,
        type: "topology",
        position: { x: child.x ?? 0, y: child.y ?? 0 },
        data: {
          label: component.name,
          componentId: component.component_id,
          kind: component.kind,
          parentName: component.parent_id ? byId.get(component.parent_id)?.name ?? component.parent_id : "",
          provenance: component.provenance,
          processingMs: component.metric_value_ms,
          scoped: component.component_id === componentId.value,
        },
      };
    });
    nodes.value = nextNodes;
    const nextEdges: Edge[] = graphEdges.map((edge) => {
      const contains = edge.kind === "contains";
      return {
        id: edge.edge_id,
        source: edge.source_id,
        target: edge.target_id,
        type: "topology",
        markerEnd: edge.directed ? { type: MarkerType.ArrowClosed, color: flowColor } : undefined,
        animated: !contains,
        class: contains ? "contains-edge" : "flow-edge",
        data: {
          ...edge,
          displayLabel: contains ? "" : edge.name,
          displayMetric: !contains && edge.metric_value_ms !== null ? formatDuration(edge.metric_value_ms) : "",
        },
      };
    });
    if (generation !== layoutGeneration) return;
    edges.value = nextEdges;
    layoutPending.value = false;
    await nextTick();
    if (generation === layoutGeneration) {
      window.requestAnimationFrame(() => {
        if (generation === layoutGeneration) fit();
      });
    }
  } catch (error) {
    if (generation !== layoutGeneration) return;
    nodes.value = [];
    edges.value = [];
    elkPromise = null;
    layoutError.value = error instanceof Error ? error.message : "无法完成拓扑布局";
  } finally {
    if (generation === layoutGeneration) layoutPending.value = false;
  }
}

function reloadLayout(): void {
  // A failed dynamic import can remain in the browser's module map. Reload
  // restores the URL-backed task while retrying the original asset cleanly.
  window.location.reload();
}

function onNodeClick(event: NodeMouseEvent): void {
  store.selectComponent(event.node.id);
}

function onEdgeClick(event: { edge: Edge }): void {
  const edge = topology.value?.edges.find((item) => item.edge_id === event.edge.id);
  if (edge) store.selection = { kind: "edge", title: edge.name, subtitle: edge.edge_id, data: recordToData(edge) };
}

function onGraphKey(event: KeyboardEvent): void {
  if (event.key !== "Enter" && event.key !== " ") return;
  const target = event.target as HTMLElement;
  if (target.closest("button, input, select")) return;
  const node = target.closest<HTMLElement>(".vue-flow__node");
  const edge = target.closest<HTMLElement>(".vue-flow__edge");
  if (node?.dataset.id) {
    event.preventDefault();
    store.selectComponent(node.dataset.id);
  } else if (edge?.dataset.id) {
    event.preventDefault();
    onEdgeClick({ edge: { id: edge.dataset.id } as Edge });
  }
}

watch([topology, shownComponents, requestId, metric], layoutGraph, { immediate: true });
onBeforeUnmount(() => { layoutGeneration += 1; });
</script>

<template>
  <section class="topology-panel" :class="{ 'motion-paused': motionPaused }" aria-label="调用拓扑">
    <header class="panel-heading">
      <div>
        <strong>调用拓扑</strong>
      </div>
      <span v-if="requestId" class="scope-indicator" :title="requestId">请求 {{ shortId(requestId, 20) }} · 实际值</span>
      <label>层级<select v-model="graphView" aria-label="拓扑层级"><option value="nodes">ROS 节点</option><option value="components">组件</option><option value="tracepoints">埋点</option></select></label>
      <label v-if="!requestId">统计值<select v-model="metric" aria-label="拓扑统计值"><option value="p50">P50</option><option value="p95">P95</option><option value="p99">P99</option><option value="maximum">最大值</option><option value="mean">均值</option><option value="minimum">最小值</option></select></label>
      <input v-model="graphSearch" type="search" placeholder="查找拓扑节点" aria-label="查找拓扑节点" />
      <button v-if="componentId" class="text-button" type="button" @click="componentId = ''">清除组件范围</button>
      <button class="icon-button" type="button" :aria-label="motionPaused ? '恢复流向动画' : '暂停流向动画'" :aria-pressed="motionPaused" @click="motionPaused = !motionPaused"><AppIcon :name="motionPaused ? 'play' : 'pause'" /></button>
      <button class="icon-button" type="button" title="适配可读视图" :disabled="!nodes.length" @click="fit">
        <AppIcon name="fit" /><span class="sr-only">适配全部节点</span>
      </button>
    </header>
    <div class="topology-canvas" @keydown.capture="onGraphKey">
      <section v-if="layoutError" class="state-view" role="alert">
        <strong>拓扑布局暂时不可用</strong>
        <span>trace 文件中的数据仍可在其他视图中查看。刷新页面重试，或收起拓扑继续分析。</span>
        <button class="button secondary" type="button" @click="reloadLayout">刷新页面重试</button>
        <details><summary>技术详情</summary><p>{{ layoutError }}</p></details>
      </section>
      <AsyncState
        v-else
        :state="layoutPending && topologyState === 'ready' ? 'loading' : topologyState"
        :error="topologyError"
        :loading-message="layoutPending ? '正在准备拓扑布局…' : '正在读取拓扑…'"
        :empty="!layoutPending && !nodes.length"
        empty-title="没有可显示的拓扑"
        empty-message="当前追踪可能缺少拓扑清单，或过滤条件未命中节点。"
        @retry="store.refreshTopology()"
      >
        <VueFlow
          id="ibrobot-topology"
          v-model:nodes="nodes"
          v-model:edges="edges"
          :nodes-draggable="true"
          :nodes-connectable="false"
          :elements-selectable="true"
          :min-zoom="0.32"
          :max-zoom="2.2"
          :default-viewport="{ x: 20, y: 20, zoom: 0.8 }"
          @node-click="onNodeClick"
          @edge-click="onEdgeClick"
        >
          <Background pattern-color="var(--oe-divider)" :gap="24" :size="1" />
          <Panel class="topology-legend" position="top-right" role="note" aria-label="拓扑关系图例">
            <strong>关系图例</strong>
            <span><i class="directed" aria-hidden="true" /><span><b>有向箭头</b> 调用或数据流方向</span></span>
            <span><i class="contains" aria-hidden="true" /><span><b>灰色虚线</b> 层级包含关系</span></span>
          </Panel>
          <Panel position="bottom-left" class="topology-controls" aria-label="拓扑缩放">
            <button class="icon-button" type="button" aria-label="放大拓扑" @click="zoomIn({ duration: motionDuration() })"><AppIcon name="plus" /></button>
            <button class="icon-button" type="button" aria-label="缩小拓扑" @click="zoomOut({ duration: motionDuration() })"><AppIcon name="minus" /></button>
            <button class="icon-button" type="button" aria-label="适配拓扑视图" @click="fit"><AppIcon name="fit" /></button>
          </Panel>
          <MiniMap position="bottom-right" :pannable="true" :zoomable="true" node-color="var(--chart-overview-node)" mask-color="rgba(var(--oe-canvas-rgb), .78)" />
          <template #node-topology="props"><TopologyNode v-bind="props" /></template>
          <template #edge-topology="props"><TopologyEdge v-bind="props" /></template>
        </VueFlow>
      </AsyncState>
    </div>
  </section>
</template>
