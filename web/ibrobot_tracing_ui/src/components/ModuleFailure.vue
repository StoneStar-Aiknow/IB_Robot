<script setup lang="ts">
defineProps<{ context: string }>();
defineEmits<{ retry: [] }>();
function reload(): void { window.location.reload(); }
</script>

<template>
  <section class="state-view" role="alert">
    <h2>{{ context }}暂时无法载入</h2>
    <p>可能是连接中断或页面资源已更新。当前分析地址和筛选保留，请重试；仍失败时刷新页面。</p>
    <div class="state-actions"><button class="button primary" type="button" @click="$emit('retry')">重试加载</button><button class="button secondary" type="button" @click="reload">刷新当前页面</button></div>
  </section>
</template>
