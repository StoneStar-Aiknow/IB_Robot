import { createApp, h, nextTick, Suspense, type App } from "vue";
import { createPinia, setActivePinia } from "pinia";
import { createRouter, createMemoryHistory } from "vue-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { asyncView } from "../src/utils/asyncView";
import { useProfilerStore } from "../src/stores/profiler";
import { useProfilerNavigation } from "../src/useProfilerNavigation";
import RequestsPanel from "../src/components/tabs/RequestsPanel.vue";
import { findSourceRecord, tracingApi } from "../src/api/client";

let app: App | undefined;
let host: HTMLElement | undefined;
afterEach(() => { app?.unmount(); host?.remove(); vi.restoreAllMocks(); });
function container() { host = document.createElement("div"); document.body.appendChild(host); return host; }

describe("PR review recovery contracts", () => {
  it("shows lazy workspace failure inside Suspense and retries only on request", async () => {
    const loader = vi.fn().mockRejectedValueOnce(new Error("chunk offline"))
      .mockResolvedValueOnce({ default: { render: () => h("p", "workspace ready") } });
    const View = asyncView(loader, "分析工作区");
    app = createApp({ render: () => h(Suspense, null, { default: () => h(View) }) });
    app.config.errorHandler = () => {};
    app.mount(container());
    await vi.waitFor(() => expect(host!.textContent).toContain("分析工作区暂时无法载入"));
    expect(loader).toHaveBeenCalledTimes(1);
    host!.querySelector<HTMLButtonElement>("button")!.click();
    await vi.waitFor(() => expect(host!.textContent).toContain("workspace ready"));
    expect(loader).toHaveBeenCalledTimes(2);
  });

  it("clears obsolete recovery source before opening a different expired legacy URL", async () => {
    const pinia = createPinia(); setActivePinia(pinia);
    const store = useProfilerStore();
    store.sourcesState = "ready"; store.analysisId = "old";
    store.restoreSourceId = "old-source"; store.restoreSourceVersion = "old-version";
    const open = vi.spyOn(store, "openAnalysis").mockImplementation(async id => { store.analysisId = id; store.analysisState = "error"; store.analysisMissing = true; });
    const router = createRouter({ history: createMemoryHistory(), routes: [{ path: "/analysis/:analysisId?", name: "analysis", component: {} }] });
    await router.push("/analysis/expired");
    app = createApp({ setup() { useProfilerNavigation(store); return () => h("p"); } }).use(pinia).use(router);
    app.mount(container());
    await vi.waitFor(() => expect(open).toHaveBeenCalledWith("expired"));
    expect(store.restoreSourceId).toBe(""); expect(store.restoreSourceVersion).toBe("");
    await router.push("/analysis/second?source=new-source&source_version=v2");
    await vi.waitFor(() => expect(open).toHaveBeenCalledWith("second"));
    expect(store.restoreSourceId).toBe("new-source"); expect(store.restoreSourceVersion).toBe("v2");
  });

  it("preserves server order with missing and negative metrics, including filtering", async () => {
    const pinia = createPinia(); setActivePinia(pinia);
    const store = useProfilerStore(); store.tabState.requests = "ready";
    store.requests = [{ request_id: "missing" }, { request_id: "negative", total_ms: -2 }];
    app = createApp(RequestsPanel).use(pinia); app.mount(container());
    const ids = () => Array.from(host!.querySelectorAll("tbody tr")).map(row => row.id);
    expect(ids()).toEqual(["grid-row-missing", "grid-row-negative"]);
    store.requestListFilter = "negative"; await nextTick();
    expect(ids()).toEqual(["grid-row-negative"]);
  });

  it("resolves a suffixed occurrence without choosing another same-ID span", async () => {
    vi.spyOn(tracingApi, "spans")
      .mockResolvedValueOnce({ items: [{ id: "span:same", name: "same" }], next_offset: 100 } as any)
      .mockResolvedValueOnce({ items: [{ id: "span:same:101", name: "same", duration_ms: 7 }], next_offset: null } as any);
    const record = await findSourceRecord("analysis", "span", "span:same:101", { requestId: "request", name: "same" }, new AbortController().signal);
    expect(record?.duration_ms).toBe(7);
  });
});
