<script setup lang="ts">
import { storeToRefs } from "pinia";
import { RouterLink, useRoute } from "vue-router";
import { useProfilerStore } from "../stores/profiler";
import AppIcon from "./AppIcon.vue";
import HelpGuide from "./HelpGuide.vue";

const store = useProfilerStore();
const route = useRoute();
const { summary, analysisState, loadJob } = storeToRefs(store);

const emit = defineEmits<{ 'change-source': [] }>();

function homeClicked(event: MouseEvent): void {
  // The URL is already home while a new file is being parsed. Router navigation
  // is then a no-op, so also stop the current page wait through the existing action.
  if (route.name === "analysis" && !store.analysisId && event.button === 0 && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) emit("change-source");
}
</script>

<template>
  <header class="profiler-header">
    <RouterLink class="brand" :to="{ name: 'analysis' }" title="返回首页" aria-label="IB-Robot Trace，返回首页" @click="homeClicked">
      <span class="brand-mark"><AppIcon name="trace" :size="18" /></span>
      <strong>IB-Robot Trace</strong>
    </RouterLink>
    <div class="header-context" aria-live="polite">
      <template v-if="route.name === 'guide'"><span>使用指南</span></template>
      <template v-else-if="loadJob && analysisState === 'loading'">
        <span>{{ loadJob.status === "queued" ? `排队等待${loadJob.queue_position ? `（第 ${loadJob.queue_position} 位）` : ""}` : "正在解析追踪" }}</span>
        <span class="job-progress indeterminate"><i /></span>
      </template>
      <template v-else-if="summary">
        <span class="path-text" :title="summary.analysis.source_name">已载入 · {{ summary.analysis.source_name }}</span>
      </template>
    </div>
    <div class="header-actions"><HelpGuide /><button v-if="route.name !== 'guide' && analysisState !== 'idle' && analysisState !== 'loading'" class="button secondary" type="button" aria-label="切换 trace 文件" title="切换 trace 文件" @click="$emit('change-source')"><span class="source-switch-full">切换 trace 文件</span><span class="source-switch-compact">切换文件</span></button></div>
  </header>
</template>
