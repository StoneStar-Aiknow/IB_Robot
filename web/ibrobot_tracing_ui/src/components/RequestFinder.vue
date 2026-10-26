<script setup lang="ts">
import { ref } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../stores/profiler";
import AppIcon from "./AppIcon.vue";

const store = useProfilerStore();
const { lookupState, lookupError } = storeToRefs(store);
const query = ref("");
</script>

<template>
  <form class="request-finder" role="search" aria-label="在整份 trace 文件中查找请求" @submit.prevent="store.findRequest(query)">
    <label for="request-id-search">查找请求</label>
    <div class="finder-input">
      <AppIcon name="search" :size="16" />
      <input id="request-id-search" v-model="query" type="search" placeholder="粘贴完整请求 ID" maxlength="512" :aria-invalid="lookupState === 'error'" aria-describedby="request-finder-feedback" @input="store.cancelLookup()" />
      <button class="button secondary" type="submit" :disabled="!query.trim() || lookupState === 'loading'">{{ lookupState === "loading" ? "查找中…" : "打开" }}</button>
    </div>
    <span id="request-finder-feedback" :role="lookupError ? 'alert' : 'status'">{{ lookupError }}</span>
  </form>
</template>
