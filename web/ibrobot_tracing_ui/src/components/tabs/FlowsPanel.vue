<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, recordStatusLabel, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";
import RecordPagination from "../RecordPagination.vue";

const store = useProfilerStore();
const { flows, tabState, tabError, searchQuery } = storeToRefs(store);
const columns: GridColumn[] = [
  { key: "edge_id", label: "边标识", width: "18%", className: "mono-cell" },
  { key: "flow_id", label: "Flow 标识", width: "13%", className: "mono-cell" },
  { key: "trace_id", label: "请求标识", width: "13%", className: "mono-cell" },
  { key: "send_ns", label: "发送（纳秒）", width: "16%", align: "right", className: "numeric" },
  { key: "receive_ns", label: "接收（纳秒）", width: "16%", align: "right", className: "numeric" },
  { key: "duration_ms", label: "耗时", width: "9%", align: "right", className: "numeric", format: (value) => formatDuration(value as number | null, 3) },
  { key: "status", label: "记录状态", width: "15%", format: recordStatusLabel },
];
const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return flows.value
    .filter((flow) => !query || `${flow.edge_id} ${flow.flow_id} ${flow.trace_id} ${flow.status}`.toLocaleLowerCase("zh-CN").includes(query))
    .map(recordToData);
});
function select(row: Record<string, unknown>): void {
  store.selection = { kind: "flow", title: String(row.edge_id), subtitle: String(row.flow_id), data: row };
}
</script>

<template>
  <div class="record-page">
    <RecordPagination tab="flows" :count="flows.length" />
    <AsyncState :state="tabState.flows" :error="tabError.flows" :empty="!flows.length" empty-title="没有匹配的 Flow" empty-message="可清除边 ID 查询，或调整请求与组件范围。" @retry="store.fetchTab('flows', true)">
    <div class="table-panel"><div class="table-toolbar"><span>本页匹配 {{ rows.length }} / {{ flows.length }} 条 Flow</span><span class="toolbar-help">交接区间可能包含调度、编码和传输；不等同纯网络耗时</span></div><DataGrid :rows="rows" :columns="columns" row-key="id" @select="select" /></div>
  </AsyncState>
  </div>
</template>
