<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{ label: string; code: string }>();
const status = ref("");
const block = ref<HTMLElement | null>(null);

async function copy(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.code);
    status.value = "已复制";
  } catch {
    // LAN HTTP may not provide the Clipboard API. Leave a selected, focusable
    // code block for the browser's native keyboard/context-menu copy action.
    block.value?.focus();
    if (block.value) {
      const range = document.createRange();
      range.selectNodeContents(block.value);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
    status.value = "自动复制不可用，已选中代码。请按 Ctrl+C / ⌘C，或使用系统复制菜单。";
  }
}
</script>

<template>
  <div class="guide-code">
    <div class="guide-code-toolbar"><span>{{ label }}</span><button class="button secondary" type="button" :aria-label="`复制${label}`" @click="copy">复制</button></div>
    <pre ref="block" tabindex="0" role="region" :aria-label="label"><code>{{ code }}</code></pre>
    <p v-if="status" class="guide-copy-status" role="status">{{ status }}</p>
  </div>
</template>
