<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, recordStatusLabel, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";
import RecordPagination from "../RecordPagination.vue";

const store = useProfilerStore();
const { spans, tabState, tabError, searchQuery } = storeToRefs(store);
const columns: GridColumn[] = [
  { key: "name", label: "Span", width: "16%", className: "mono-cell" },
  { key: "component_id", label: "组件", width: "19%", className: "mono-cell" },
  { key: "span_id", label: "Span 标识", width: "13%", className: "mono-cell" },
  { key: "parent_span_id", label: "父标识", width: "13%", className: "mono-cell" },
  { key: "duration_ms", label: "耗时", width: "11%", align: "right", className: "numeric", format: (value) => formatDuration(value as number | null, 3) },
  { key: "origin", label: "来源", width: "8%", format: (value) => value === "user" ? "用户" : "内置" },
  { key: "status", label: "记录状态", width: "8%", format: recordStatusLabel },
  { key: "trace_id", label: "请求标识", width: "12%", className: "mono-cell" },
];
const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return spans.value
    .filter((span) => !query || `${span.name} ${span.component_id} ${span.span_id} ${span.status}`.toLocaleLowerCase("zh-CN").includes(query))
    .map(recordToData);
});
function select(row: Record<string, unknown>): void {
  store.selection = { kind: "span", title: String(row.name), subtitle: String(row.component_id || ""), data: row };
}
</script>

<template>
  <div class="record-page">
    <RecordPagination tab="spans" :count="spans.length" />
    <AsyncState :state="tabState.spans" :error="tabError.spans" :empty="!spans.length" empty-title="没有匹配的 Span" empty-message="可清除名称查询，或调整请求与组件范围。" @retry="store.fetchTab('spans', true)">
    <div class="table-panel"><div class="table-toolbar"><span>本页匹配 {{ rows.length }} / {{ spans.length }} 条 Span</span><span class="toolbar-help">单击查看记录与起终点</span></div><DataGrid :rows="rows" :columns="columns" row-key="id" @select="select" /></div>
  </AsyncState>
  </div>
</template>
