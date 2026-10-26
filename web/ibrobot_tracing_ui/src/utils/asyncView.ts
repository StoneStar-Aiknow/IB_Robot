import { defineAsyncComponent, defineComponent, h, shallowRef, type Component } from "vue";
import ModuleFailure from "../components/ModuleFailure.vue";

/** User-controlled retry; persistent offline/404 errors never auto-reload. */
export function asyncView(loader: () => Promise<{ default: Component }>, context: string) {
  return defineComponent({
    setup() {
      const create = (): Component => defineAsyncComponent({
        loader: () => loader().then(module => module.default),
        errorComponent: defineComponent(() => () => h(ModuleFailure, { context, onRetry: () => { view.value = create(); } })),
      });
      const view = shallowRef(create());
      return () => h(view.value);
    },
  });
}
