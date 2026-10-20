<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { PAGE_SIZE, useProfilerStore } from "../stores/profiler";
import AppIcon from "./AppIcon.vue";

const store = useProfilerStore();
const { requestId, requests, requestsOffset, requestsNextOffset, tabState } = storeToRefs(store);
const index = computed(() => requests.value.findIndex((request) => request.request_id === requestId.value));
const previous = computed(() => index.value >= 0 && (index.value > 0 || requestsOffset.value > 0));
const next = computed(() => index.value >= 0 && (index.value < requests.value.length - 1 || requestsNextOffset.value !== null));
async function adjacent(direction: number): Promise<void> {
  let position = index.value + direction;
  if (position < 0 || position >= requests.value.length) {
    store.requestsOffset = direction > 0 ? requestsNextOffset.value! : Math.max(0, requestsOffset.value - PAGE_SIZE);
    store.requestListScroll = 0;
    await store.fetchRequests();
    if (store.tabState.requests !== "ready") return;
    position = direction > 0 ? 0 : store.requests.length - 1;
  }
  const request = store.requests[position];
  if (request) await store.navigateToRequest(request.request_id, "critical-path");
}
</script>

<template>
  <section class="request-context" aria-label="当前请求">
    <div class="request-context-actions">
      <button class="text-button" type="button" aria-label="返回请求列表" @click="store.selection = null; store.activeTab = 'requests'; store.searchQuery = ''"><span class="request-full-label">返回请求列表</span><span class="request-compact-label">返回列表</span></button>
      <code :title="requestId">{{ requestId }}</code>
      <div class="adjacent-requests" role="group" aria-label="相邻请求，按列表排序">
        <button class="button secondary" type="button" aria-label="上一请求" title="上一请求，按列表排序" :disabled="!previous || tabState.requests === 'loading'" @click="adjacent(-1)"><span class="request-step-icon previous"><AppIcon name="chevron" /></span><span class="request-full-label">上一请求</span></button>
        <button class="button secondary" type="button" aria-label="下一请求" title="下一请求，按列表排序" :disabled="!next || tabState.requests === 'loading'" @click="adjacent(1)"><span class="request-step-icon"><AppIcon name="chevron" /></span><span class="request-full-label">下一请求</span></button>
      </div>
    </div>
    <p v-if="index < 0">当前请求不在已加载的列表页。返回列表后可按排序选择相邻请求。</p>
    <p v-else>上一/下一按请求列表排序。详细视图使用原始记录边界，可能与日志汇总值有精度差异。</p>
  </section>
</template>
