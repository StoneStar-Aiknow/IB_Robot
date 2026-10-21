<script setup lang="ts">
import { computed, nextTick, onMounted, watch } from "vue";
import { storeToRefs } from "pinia";
import { PAGE_SIZE, useProfilerStore } from "../../stores/profiler";
import { formatDuration, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";

const store = useProfilerStore();
const { requests, requestId, tabState, tabError, requestListFilter, requestsTotal, requestsOffset, requestsNextOffset, requestSortBy, requestDescending } = storeToRefs(store);
async function restoreScroll(): Promise<void> {
  await nextTick();
  const scrollHost = document.querySelector<HTMLElement>("#analysis-view .data-grid-wrap");
  if (scrollHost) scrollHost.scrollTop = store.requestListScroll;
}
onMounted(restoreScroll);
watch(() => tabState.value.requests, (state) => { if (state === "ready") void restoreScroll(); });

function changeSort(): void {
  requestsOffset.value = 0;
  store.requestListScroll = 0;
  void store.fetchRequests();
}

const columns: GridColumn[] = [
  { key: "request_id", label: "请求标识", width: "180px", className: "mono-cell" },
  ...[
    ["total_ms", "分发→首次执行", "130px"],
    ["inference_ms", "模型调用", "130px"],
    ["preprocess_ms", "预处理", "130px"],
    ["postprocess_ms", "后处理", "130px"],
    ["queue_refill_ms", "分发→入队", "130px"],
  ].map(([key, label, width]): GridColumn => ({
    key, label, width, align: "right", className: "numeric",
    format: (value, row) => {
      const status = String(row[`${key}_status`] ?? "");
      if (status && status !== "ok") {
        const labels: Record<string, string> = { ambiguous: "歧义", error: "错误记录", incomplete: "记录不完整", invalid: "无效数据" };
        return status.split(",").map((part) => labels[part] ? `${labels[part]} (${part})` : part).join(" / ");
      }
      return typeof value === "number" && Number.isFinite(value) ? formatDuration(value) : "未采到";
    },
  })),
  { key: "total_ms_source", label: "端到端来源", width: "120px" },
  { key: "inference_ms_source", label: "推理来源", width: "120px" },
  { key: "actions", label: "打开请求视图", width: "240px", className: "request-actions" },
];

const rows = computed(() => {
  const query = requestListFilter.value.trim().toLocaleLowerCase("zh-CN");
  const filtered = query
    ? requests.value.filter((request) => Object.values(request).some((value) => String(value ?? "").toLocaleLowerCase("zh-CN").includes(query)))
    : [...requests.value];
  return filtered.map((row) => recordToData(row));
});

function select(row: Record<string, unknown>): void {
  void store.navigateToRequest(String(row.request_id), "critical-path");
}

function openRequest(id: string, tab: "timeline" | "span-profile" | "critical-path"): void {
  void store.navigateToRequest(id, tab);
}
</script>

<template>
  <AsyncState :state="tabState.requests" :error="tabError.requests" :empty="!requests.length && requestsOffset === 0" empty-title="没有请求记录" empty-message="追踪中没有可关联的 request_id。" @retry="store.fetchRequests(true)">
    <div class="table-panel paged-table">
      <div class="table-toolbar">
        <span>全部请求 · 点击一行查看耗时拆分</span>
        <label>排序<select v-model="requestSortBy" @change="changeSort"><option value="total_ms">分发→首次执行</option><option value="inference_ms">模型调用</option><option value="preprocess_ms">预处理</option><option value="queue_refill_ms">分发→入队</option></select></label>
        <button class="text-button" type="button" @click="requestDescending = !requestDescending; changeSort()">{{ requestDescending ? "耗时从高到低" : "耗时从低到高" }}</button>
      </div>
      <DataGrid class="request-grid" :rows="rows" :columns="columns" row-key="request_id" :selected-key="requestId" @select="select" @scroll="store.requestListScroll = ($event.target as HTMLElement).scrollTop">
        <template #empty-actions><button v-if="requestListFilter" class="button secondary" type="button" @click="requestListFilter = ''">清除过滤</button></template>
        <template #cell-actions="{ row }">
          <button type="button" class="text-button" :aria-label="`在时间线打开请求 ${row.request_id}`" @click.stop="openRequest(String(row.request_id), 'timeline')">时间线</button>
          <button type="button" class="text-button" :aria-label="`在调用层级打开请求 ${row.request_id}`" @click.stop="openRequest(String(row.request_id), 'span-profile')">调用层级</button>
          <button type="button" class="text-button" :aria-label="`在耗时拆分打开请求 ${row.request_id}`" @click.stop="openRequest(String(row.request_id), 'critical-path')">耗时拆分</button>
        </template>
      </DataGrid>
      <nav class="page-controls" aria-label="请求分页">
        <span aria-live="polite">{{ requests.length ? requestsOffset + 1 : 0 }}–{{ requestsOffset + requests.length }} / {{ requestsTotal || requests.length }} 条<span v-if="requestListFilter"> · 本页匹配 {{ rows.length }} 条</span></span>
        <button class="button secondary" type="button" :disabled="requestsOffset === 0" @click="requestsOffset = Math.max(0, requestsOffset - PAGE_SIZE); store.requestListScroll = 0; store.fetchRequests()">上一页</button>
        <button class="button secondary" type="button" :disabled="requestsNextOffset === null" @click="requestsOffset = requestsNextOffset!; store.requestListScroll = 0; store.fetchRequests()">下一页</button>
      </nav>
    </div>
  </AsyncState>
</template>
