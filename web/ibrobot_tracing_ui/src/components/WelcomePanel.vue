<script setup lang="ts">
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../stores/profiler";
import { formatBytes, formatTimestamp } from "../utils/format";

const store = useProfilerStore();
const { sources, sourcesState, sourcesError, sourceId, selectedSource } = storeToRefs(store);
defineEmits<{ analyze: [] }>();
</script>

<template>
  <section class="welcome-panel">
    <h1 tabindex="-1">IB-Robot Trace</h1>
    <p>先看整体耗时，再打开一个请求，追到具体阶段和原始记录。</p>
    <form class="welcome-source" @submit.prevent="$emit('analyze')">
      <label for="trace-source">选择 trace 文件</label>
      <select id="trace-source" v-model="sourceId" :disabled="sourcesState === 'loading' || !sources.length">
        <option v-if="!sources.length" value="">{{ sourcesState === 'loading' ? '正在查找 trace 文件…' : sourcesState === 'error' ? 'trace 文件列表暂时无法读取' : '还没有可用的 trace 文件' }}</option>
        <option v-for="source in sources" :key="source.id" :value="source.id">{{ source.name }} · {{ source.kind.toUpperCase() }}</option>
      </select>
      <span v-if="selectedSource" class="source-description">{{ formatTimestamp(selectedSource.modified_at) }} · {{ formatBytes(selectedSource.size_bytes) }}</span>
      <div class="state-actions">
        <button class="button primary" type="submit" :disabled="!sourceId || sourcesState === 'loading'">打开分析</button>
        <button class="text-button" type="button" :disabled="sourcesState === 'loading'" @click="store.refreshSources">刷新 trace 文件列表</button>
      </div>
    </form>
    <section v-if="sourcesState === 'error'" class="context-error" role="alert">
      <strong>trace 文件列表暂时无法读取</strong>
      <p>这不表示没有 trace 文件。请检查追踪服务连接，再重试读取；已有选择会保留。</p>
      <button class="button secondary" type="button" @click="store.loadSources">重试读取 trace 文件</button>
      <details><summary>查看错误详情</summary><pre>{{ sourcesError }}</pre></details>
    </section>
    <details v-else-if="sourcesError" class="catalog-notice"><summary>trace 文件列表已载入，查看扫描提示</summary><pre>{{ sourcesError }}</pre></details>
    <p v-else-if="sourcesState === 'ready' && !sources.length">将 CTF、日志或 .ibtrace 文件放入服务配置的追踪目录，然后刷新列表。</p>
    <details class="welcome-help"><summary>如何使用</summary><ol><li>在概览查看延迟分布和慢请求。</li><li>打开请求查看耗时拆分，或粘贴完整请求 ID。</li><li>在实验对比选择另一份 trace 文件，查看指标变化。</li></ol></details>
  </section>
</template>
