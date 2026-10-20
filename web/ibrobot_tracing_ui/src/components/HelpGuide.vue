<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import { useProfilerStore } from "../stores/profiler";
import { searchHelp } from "../utils/help";
import AppIcon from "./AppIcon.vue";

const store = useProfilerStore();
const query = ref("");
const panel = ref<HTMLDetailsElement | null>(null);
const input = ref<HTMLInputElement | null>(null);
const topics = computed(() => searchHelp(query.value, store.activeTab));
async function opened(): Promise<void> {
  if (panel.value?.open) { await nextTick(); input.value?.focus(); }
}
function close(): void {
  if (!panel.value) return;
  panel.value.open = false;
  panel.value.querySelector<HTMLElement>(":scope > summary")?.focus();
}
</script>

<template>
  <details ref="panel" class="help-guide" data-popover @toggle="opened">
    <summary title="使用帮助" aria-label="使用帮助"><AppIcon name="help" :size="18" /><span class="help-label">使用帮助</span></summary>
    <section class="help-content" aria-label="使用帮助内容">
      <header><h2>使用帮助</h2><button class="icon-button" type="button" aria-label="关闭帮助" @click="close"><AppIcon name="close" /></button></header>
      <label for="help-search">搜索操作或术语</label>
      <input id="help-search" ref="input" v-model="query" type="search" placeholder="如：P95、Gap、原始记录、过期" />
      <p class="help-status" role="status">{{ query ? `找到 ${topics.length} 条说明` : '优先显示与当前视图相关的说明' }}</p>
      <div v-if="!topics.length" class="help-empty"><p>没有匹配说明。试试更短的关键词，或查看全部说明。</p><button class="button secondary" type="button" @click="query = ''">清除帮助搜索</button></div>
      <details v-for="topic in topics" :key="`${topic.id}:${query.trim()}`" class="help-topic" :open="Boolean(query.trim()) || topic.id === topics[0]?.id">
        <summary>{{ topic.title }}</summary><ol><li v-for="step in topic.steps" :key="step">{{ step }}</li></ol>
      </details>
    </section>
  </details>
</template>
