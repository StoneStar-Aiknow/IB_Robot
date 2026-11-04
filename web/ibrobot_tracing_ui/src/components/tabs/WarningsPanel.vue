<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import AsyncState from "../AsyncState.vue";
import AppIcon from "../AppIcon.vue";
import { explainDiagnostic, type DiagnosticExplanation } from "../../utils/diagnostics";

const store = useProfilerStore();
const { warnings, warningTotal, warningsTruncated, tabState, tabError, searchQuery } = storeToRefs(store);
const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return warnings.value.map(explainDiagnostic).filter((warning) => !query || `${warning.raw} ${warning.title} ${warning.explanation}`.toLocaleLowerCase("zh-CN").includes(query));
});
function inspect(warning: DiagnosticExplanation): void {
  store.selection = { kind: "warning", title: warning.title, subtitle: "分析器诊断", data: {
    message: warning.raw, explanation: warning.explanation, request_id: warning.requestId, metric: warning.metric,
  } };
}
</script>

<template>
  <AsyncState :state="tabState.warnings" :error="tabError.warnings" :empty="!warnings.length" :empty-title="tabState.warnings === 'idle' ? '尚未载入诊断' : '分析器未报告警告'" :empty-message="tabState.warnings === 'idle' ? '载入 trace 文件后查看记录质量。' : '这仅表示分析器未发现已知诊断，不代表业务成功或所有观测完整。'" @retry="store.fetchTab('warnings', true)">
    <div class="warnings-view">
      <p v-if="warningsTruncated" role="status">共 {{ warningTotal }} 条诊断，仅展示前 {{ warnings.length }} 条；筛选仅作用于已展示记录。</p>
      <header><strong>分析器警告</strong><span>这些记录不会阻止查看已有数据，但可能影响统计完整性。</span></header>
      <article v-for="(warning, index) in rows" :key="`${index}:${warning.raw}`" class="warning-entry">
        <button class="warning-open" type="button" @click="inspect(warning)"><AppIcon name="warning" /><strong>{{ warning.title }}</strong><span v-if="warning.requestId">请求 {{ warning.requestId }}</span></button>
        <p>{{ warning.explanation }}</p>
        <div class="warning-actions"><button v-if="warning.requestId" class="text-button" type="button" @click="store.findRequest(warning.requestId)">查看关联请求 {{ warning.requestId }}</button><details><summary>原始诊断</summary><pre>{{ warning.raw }}</pre></details></div>
      </article>
      <p v-if="store.lookupError" role="alert">{{ store.lookupError }}</p>
      <div v-if="!rows.length" class="empty-cell">过滤条件未命中警告</div>
    </div>
  </AsyncState>
</template>
