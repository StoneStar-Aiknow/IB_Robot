import { defineComponent, h, shallowRef } from "vue";
import { createRouter, createWebHistory } from "vue-router";

const RouteOutlet = defineComponent(() => () => h("span"));
export const routeLoadError = shallowRef<{ target: string } | null>(null);

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: "/", redirect: "/analysis" },
    { path: "/analysis/:analysisId?", name: "analysis", component: RouteOutlet },
    { path: "/guide/:topic(capture|instrument)", name: "guide", component: () => import("./components/TraceGuide.vue") },
  ],
});

router.onError((_error, to) => { routeLoadError.value = { target: to.fullPath }; });
router.afterEach((_to, _from, failure) => { if (!failure) routeLoadError.value = null; });
