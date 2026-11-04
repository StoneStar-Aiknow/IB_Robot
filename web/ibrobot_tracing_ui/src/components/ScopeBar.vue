<script setup lang="ts">
import { storeToRefs } from "pinia";
import { computed, ref, watch } from "vue";
import { useProfilerStore } from "../stores/profiler";
import RequestFinder from "./RequestFinder.vue";
import { boundaryLabel } from "../utils/format";

const store = useProfilerStore();
const toolsOpen = ref(false);
watch(() => store.requestId, () => { toolsOpen.value = false; });
const { summary } = storeToRefs(store);
const machineIds = computed(() => {
  const sources = summary.value?.metadata.sources;
  if (!Array.isArray(sources)) return [];
  return [...new Set(sources.map((source) => source?.calibration_identity?.machine_id).filter((id): id is string => typeof id === "string" && Boolean(id)))];
});
const reference = computed(() => summary.value?.metadata.reference_id);
</script>

<template>
  <section class="scope-bar" aria-label="请求查找与 trace 文件信息">
    <button id="scope-tools-toggle" class="text-button compact-only" type="button" :aria-expanded="toolsOpen" aria-controls="scope-tools" @click="toolsOpen = !toolsOpen">查找请求与 trace 文件信息</button>
    <div id="scope-tools" class="scope-tools" :class="{ expanded: toolsOpen }">
    <RequestFinder />
    <details class="capture-info" data-popover>
      <summary>trace 文件信息与计时边界</summary>
      <dl v-if="summary">
        <dt>已载入 trace 文件</dt><dd>{{ summary.analysis.source_name }}</dd>
        <dt>主要计时边界</dt><dd>{{ boundaryLabel(summary.coverage.boundary || '') || "未提供；请查看具体记录起终点" }}</dd>
        <dt>trace 文件来源主机</dt><dd>{{ machineIds.length ? machineIds.join("、") : "汇总元数据未提供；记录来源可在详情查看" }}</dd>
        <dt>跨主机校准</dt><dd>{{ reference ? `PTP 参考 ${JSON.stringify(reference)}；条件工程估计，以记录误差为准` : "未提供校准信息" }}</dd>
        <dt>来源版本</dt><dd><code>{{ summary.analysis.source_version }}</code></dd>
      </dl>
    </details>
    </div>
  </section>
</template>
