<script setup lang="ts">
import type { LoadState } from "../api/types";
import AppIcon from "./AppIcon.vue";

// Classes belong to the state placeholder; a ready slot owns its own layout.
defineOptions({ inheritAttrs: false });

withDefaults(
  defineProps<{
    state: LoadState;
    empty?: boolean;
    error?: string;
    emptyTitle?: string;
    emptyMessage?: string;
    loadingMessage?: string;
    context?: string;
  }>(),
  {
    empty: false,
    error: "",
    emptyTitle: "没有匹配数据",
    emptyMessage: "调整当前范围或载入其他追踪源。",
    loadingMessage: "正在读取追踪数据…",
    context: "当前视图",
  },
);

defineEmits<{ retry: [] }>();
</script>

<template>
  <div v-if="state === 'loading'" v-bind="$attrs" class="state-view state-loading" aria-live="polite">
    <div class="loading-line" /><div class="loading-line short" /><div class="loading-line" />
    <span>{{ loadingMessage }}</span>
  </div>
  <div v-else-if="state === 'error'" v-bind="$attrs" class="state-view" role="alert">
    <AppIcon name="warning" :size="22" />
    <strong>{{ context }}暂时无法读取</strong>
    <span>{{ error }}</span>
    <p class="state-recovery-hint">当前请求和筛选条件已保留。重试后可继续；若仍失败，请检查追踪服务连接或切换 trace 文件重新打开。</p>
    <button class="button secondary" type="button" @click="$emit('retry')">重试</button>
  </div>
  <div v-else-if="empty || state === 'idle'" v-bind="$attrs" class="state-view">
    <AppIcon name="trace" :size="24" />
    <strong>{{ emptyTitle }}</strong>
    <span>{{ emptyMessage }}</span>
  </div>
  <slot v-else />
</template>
