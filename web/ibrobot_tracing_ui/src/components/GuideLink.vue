<script setup lang="ts">
import { RouterLink, useRoute } from "vue-router";
import { guideLocation, type GuideTopic } from "../utils/guide";

defineProps<{ topic: GuideTopic; section?: "manual" }>();
const route = useRoute();
</script>

<template>
  <RouterLink :to="{ ...guideLocation(topic, route.fullPath, route.query.from), hash: section ? `#${section}` : '' }"><slot /></RouterLink>
</template>
