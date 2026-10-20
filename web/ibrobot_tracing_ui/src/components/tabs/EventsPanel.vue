<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { eventComponent, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";
import RecordPagination from "../RecordPagination.vue";

const store = useProfilerStore();
const { events, tabState, tabError, searchQuery } = storeToRefs(store);
const columns: GridColumn[] = [
  { key: "timestamp_ns", label: "时间戳（纳秒）", width: "18%", align: "right", className: "numeric" },
  { key: "name", label: "事件", width: "18%", className: "mono-cell" },
  { key: "component", label: "组件", width: "18%", className: "mono-cell" },
  { key: "provider", label: "提供者", width: "14%", className: "mono-cell" },
  { key: "schema_version", label: "模式", width: "7%", align: "right", className: "numeric" },
  { key: "fields", label: "字段", width: "25%", className: "mono-cell", format: (value) => JSON.stringify(value) },
];

const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return events.value
    .filter((event) => !query || `${event.name} ${eventComponent(event)} ${JSON.stringify(event.fields)}`.toLocaleLowerCase("zh-CN").includes(query))
    .map((event) => ({ ...recordToData(event), component: eventComponent(event), provider: event.origin.provider }));
});

function select(row: Record<string, unknown>): void {
  store.selection = { kind: "event", title: String(row.name), subtitle: String(row.component), data: row };
}
</script>

<template>
  <div class="record-page">
    <RecordPagination tab="events" :count="events.length" />
    <AsyncState :state="tabState.events" :error="tabError.events" :empty="!events.length" empty-title="没有匹配的事件" empty-message="可清除名称查询，或调整请求与组件范围。" @retry="store.fetchTab('events', true)">
      <div class="table-panel"><div class="table-toolbar"><span>本页匹配 {{ rows.length }} / {{ events.length }} 条事件</span></div><DataGrid :rows="rows" :columns="columns" row-key="id" empty-text="本页没有匹配记录；名称查询可搜索当前范围全部事件" @select="select" /></div>
  </AsyncState>
  </div>
</template>
