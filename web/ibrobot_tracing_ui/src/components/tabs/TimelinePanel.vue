<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";

const store = useProfilerStore();
const { timeline, requestId, tabState, tabError, searchQuery } = storeToRefs(store);
const durationNs = computed(() => Math.max(0, timeline.value?.duration_ns ?? 0));

const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  const laneNames = new Map((timeline.value?.lanes ?? []).map((lane) => [lane.id, lane.label]));
  const items = timeline.value?.items ?? [];
  const filtered = query
    ? items.filter((item) => `${item.label} ${item.component_id} ${laneNames.get(item.lane_id)} ${JSON.stringify(item.fields)}`.toLocaleLowerCase("zh-CN").includes(query))
    : items;
  if (!filtered.length) return [];
  const range = Math.max(1, durationNs.value);
  let previous = filtered[0].start_offset_ns;
  return filtered.map((item) => {
    const row = {
      item,
      key: item.id,
      relative: item.start_offset_ns / 1_000_000,
      delta: (item.start_offset_ns - previous) / 1_000_000,
      position: (item.start_offset_ns / range) * 100,
      width: Math.max(0, (item.duration_ns / range) * 100),
      component: laneNames.get(item.lane_id) || item.component_id || item.lane_id,
    };
    previous = item.start_offset_ns;
    return row;
  });
});

function inspect(index: number): void {
  const row = rows.value[index];
  store.selection = {
    kind: row.item.kind,
    title: row.item.label,
    subtitle: `${row.relative.toFixed(3)} ms · ${row.component}`,
    data: recordToData(row.item),
  };
}
</script>

<template>
  <AsyncState
    context="时间线"
    :state="tabState.timeline"
    :error="tabError.timeline"
    :empty="!requestId || !timeline?.items.length"
    :empty-title="requestId ? '该请求没有事件' : '时间线需要请求范围'"
    :empty-message="requestId ? '调整组件范围或检查追踪覆盖。' : '从请求列表打开记录，或在上方粘贴完整请求 ID。'"
    @retry="store.fetchTab('timeline', true)"
  >
    <div v-if="!rows.length" class="state-view timeline-empty" role="status">
      <strong>没有匹配的记录</strong>
      <span>当前请求有 {{ timeline?.items.length ?? 0 }} 条记录，时间范围仍为 0–{{ (durationNs / 1_000_000).toFixed(3) }} ms。</span>
      <button class="button secondary" type="button" @click="store.searchQuery = ''">清除过滤</button>
    </div>
    <div v-else class="timeline-view">
      <div class="timeline-ruler"><span class="timeline-filter-note">{{ searchQuery ? '文字过滤仅隐藏记录，时间尺度不变' : '当前观测范围' }}</span><div class="timeline-axis"><span>0 ms</span><i /><span>{{ (durationNs / 1_000_000).toFixed(3) }} ms</span></div></div>
      <div class="timeline-header"><span>相对时间</span><span title="与上一条显示记录的起点之差">显示记录间隔</span><span>泳道</span><span>事件、Span 与 Flow</span></div>
      <div class="timeline-scroll">
        <button v-for="(row, index) in rows" :key="row.key" class="timeline-row" type="button" @click="inspect(index)">
          <span class="numeric">{{ row.relative.toFixed(3) }} ms</span>
          <span class="numeric muted">+{{ row.delta.toFixed(3) }}</span>
          <code :title="row.component">{{ row.component }}</code>
          <span class="timeline-track"><span class="timeline-plot"><i :class="`kind-${row.item.kind}`" :style="{ left: `${row.position}%`, width: `${row.width}%` }" /></span><b>{{ row.item.label }}</b></span>
        </button>
      </div>
    </div>
  </AsyncState>
</template>
