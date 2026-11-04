import { createPinia, setActivePinia } from "pinia";
import { createApp, nextTick, type App } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import RequestsPanel from "../src/components/tabs/RequestsPanel.vue";
import WarningsPanel from "../src/components/tabs/WarningsPanel.vue";
import { useProfilerStore } from "../src/stores/profiler";

let app: App | undefined;
let host: HTMLDivElement | undefined;
afterEach(() => { app?.unmount(); host?.remove(); vi.restoreAllMocks(); });

describe("native keyboard and empty-state semantics", () => {
  it("does not intercept Enter on a request action or toggle the selected request", async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const store = useProfilerStore();
    store.tabState.requests = "ready";
    store.requests = [{ request_id: "selected", total_ms: 12 }];
    store.requestId = "selected";
    const navigate = vi.spyOn(store, "navigateToRequest").mockResolvedValue();
    host = document.createElement("div");
    document.body.appendChild(host);
    app = createApp(RequestsPanel).use(pinia);
    app.mount(host);
    const button = host.querySelector<HTMLButtonElement>('button[aria-label="在时间线打开请求 selected"]')!;
    button.focus();
    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    button.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(store.requestId).toBe("selected");
    button.click();
    await nextTick();
    expect(navigate).toHaveBeenCalledWith("selected", "timeline");
  });

  it("does not claim complete boundaries before diagnostics have loaded", () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    host = document.createElement("div");
    app = createApp(WarningsPanel).use(pinia);
    app.mount(host);
    expect(host.textContent).toContain("尚未载入诊断");
    expect(host.textContent).not.toContain("边界完整");
    expect(host.textContent).not.toContain("未报告警告");
  });
});
