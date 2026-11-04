<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { PAGE_SIZE, useProfilerStore, type RecordTab } from "../stores/profiler";
import { formatInteger } from "../utils/format";

const props = defineProps<{ tab: RecordTab; count: number }>();
const store = useProfilerStore();
const page = computed(() => store.recordPages[props.tab]);
const name = ref(page.value.name);
watch(() => page.value.name, (value) => { name.value = value; });
const busy = computed(() => store.tabState[props.tab] === "loading");
</script>

<template>
  <div class="record-pagination">
    <form @submit.prevent="store.setRecordPage(tab, 0, name)">
      <label :for="`${tab}-name-filter`">{{ tab === "flows" ? "边 ID" : "名称" }}</label>
      <input :id="`${tab}-name-filter`" v-model="name" placeholder="完整名称，查询当前范围全部记录" maxlength="512" />
      <button class="button secondary" type="submit" :disabled="busy">查询</button>
      <button v-if="page.name" class="text-button" type="button" @click="store.setRecordPage(tab, 0, '')">清除名称</button>
    </form>
    <nav aria-label="记录分页">
      <span v-if="busy" aria-live="polite">正在读取记录…</span>
      <span v-else aria-live="polite">{{ count ? formatInteger(page.offset + 1) : 0 }}–{{ count ? formatInteger(page.offset + count) : 0 }} / {{ formatInteger(page.total) }} 条</span>
      <button class="button secondary" type="button" :disabled="busy || page.offset === 0" @click="store.setRecordPage(tab, Math.max(0, page.offset - PAGE_SIZE))">上一页</button>
      <button v-if="!busy && !count && page.offset" class="button secondary" type="button" @click="store.setRecordPage(tab, 0)">返回第一页</button>
      <button class="button secondary" type="button" :disabled="busy || page.nextOffset === null" @click="store.setRecordPage(tab, page.nextOffset!)">下一页</button>
    </nav>
  </div>
</template>
