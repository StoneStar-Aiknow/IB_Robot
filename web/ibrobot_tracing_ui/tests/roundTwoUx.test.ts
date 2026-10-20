import { createPinia, getActivePinia, setActivePinia } from "pinia";
import { createApp, nextTick, type App, type Component } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TimelinePanel from "../src/components/tabs/TimelinePanel.vue";
import CriticalPathPanel from "../src/components/tabs/CriticalPathPanel.vue";
import SummaryPanel from "../src/components/tabs/SummaryPanel.vue";
import InspectorPanel from "../src/components/InspectorPanel.vue";
import { useProfilerStore } from "../src/stores/profiler";
import { findSourceRecord, tracingApi } from "../src/api/client";
import { dismissOutsidePopovers, dismissTopPopover, keepOnePopover } from "../src/utils/popovers";

let app: App | undefined;
let host: HTMLDivElement;
function mount(component: Component) {
  host = document.createElement("div");
  document.body.appendChild(host);
  app = createApp(component).use(getActivePinia()!);
  app.mount(host);
}
beforeEach(() => setActivePinia(createPinia()));
afterEach(() => { app?.unmount(); host?.remove(); app = undefined; vi.restoreAllMocks(); document.body.innerHTML = ""; });

describe("measurement and interaction contracts", () => {
  it("keeps attributed and original intervals distinct and preserves the return context", async () => {
    const store = useProfilerStore();
    store.analysisId = "analysis";
    store.selection = { kind: "critical-path", title: "policy_pipeline", data: {
      kind: "span", source_id: "span:occurrence", label: "policy_pipeline", request_id: "one", duration_ns: 46_084,
      percentage: 0.091, start_ns: "1000", end_ns: "47084", status: "ok", fields: { duration_ns: 47_502_061 },
    } };
    vi.spyOn(tracingApi, "spans").mockResolvedValue({ items: [{ id: "span:occurrence", name: "policy_pipeline", duration_ms: 47.502061,
      start_ns: "1000", end_ns: "47503061", status: "ok" }], total: 1, next_offset: null } as any);
    mount(InspectorPanel);
    expect(host.querySelector(".record-digest")?.textContent).toContain("本段归因时长0.046 ms");
    expect(host.querySelector(".record-digest")?.textContent).not.toContain("观测耗时");
    expect(host.querySelector(".raw-fields summary")?.textContent).toBe("归因分段字段");
    host.querySelector<HTMLButtonElement>(".inspector-source-actions button")!.click();
    await vi.waitFor(() => expect(store.selection?.kind).toBe("span"));
    await nextTick();
    expect(host.querySelector(".record-digest")?.textContent).toContain("观测耗时47.502 ms");
    host.querySelector<HTMLButtonElement>(".inspector-source-actions button")!.click();
    await nextTick();
    expect(store.selection?.kind).toBe("critical-path");
    expect(host.querySelector(".record-digest")?.textContent).toContain("本段归因时长0.046 ms");
  });

  it("resolves original occurrence identity beyond a page of repeated names", async () => {
    const fetch = vi.spyOn(tracingApi, "spans")
      .mockResolvedValueOnce({ items: [{ id: "other", name: "same" }], next_offset: 100 } as any)
      .mockResolvedValueOnce({ items: [{ id: "wanted", name: "same" }], next_offset: null } as any);
    const signal = new AbortController().signal;
    const record = await findSourceRecord("analysis", "span", "wanted", { requestId: "one", name: "same" }, signal);
    expect(record?.id).toBe("wanted");
    expect(fetch.mock.calls[1][1]).toMatchObject({ requestId: "one", name: "same", offset: 100 });
  });

  it("keeps one timeline domain when filtering, including an interval ending after its start", async () => {
    const store = useProfilerStore();
    store.requestId = "one";
    store.tabState.timeline = "ready";
    store.timeline = {
      start_ns: 0, end_ns: 50_000_000, duration_ns: 50_000_000, lanes: [{ id: "worker", label: "Worker" }],
      items: [
        { id: "one", kind: "span", lane_id: "worker", label: "model_call", start_offset_ns: 5_000_000, duration_ns: 20_000_000, component_id: "worker", fields: {} },
        { id: "two", kind: "event", lane_id: "worker", label: "end", start_offset_ns: 50_000_000, duration_ns: 0, component_id: "worker", fields: {} },
      ],
    } as any;
    mount(TimelinePanel);
    const before = host.querySelector<HTMLElement>(".kind-span")!.getAttribute("style");
    store.searchQuery = "model_call";
    await nextTick();
    expect(host.querySelectorAll(".timeline-row")).toHaveLength(1);
    expect(host.querySelector(".timeline-axis")?.textContent).toContain("50.000 ms");
    expect(host.querySelector(".kind-span")?.getAttribute("style")).toBe(before);
    expect(host.querySelector<HTMLElement>(".kind-span")!.style.width).toBe("40%");
    store.searchQuery = "no-match";
    await nextTick();
    expect(host.querySelector(".timeline-empty")?.textContent).toContain("50.000 ms");
    expect(host.textContent).toContain("没有匹配的记录");
  });

  it("explains component-scoped gaps without changing the response's boundary or values", async () => {
    const store = useProfilerStore();
    store.requestId = "one";
    store.componentId = "worker";
    store.tabState["critical-path"] = "ready";
    store.criticalPath = {
      request_id: "one", component_id: "worker", duration_ns: 50_000_000, segments: [], bottlenecks: [], diagnostics: [],
      boundary_source: "dispatch_request_to_first_action_execute", returned_segments: 0, total_segments: 0,
      returned_duration_ns: 50_000_000, omitted_duration_ns: 0, truncated: false,
      totals: { coverage_percent: 60, by_kind_ns: { span: 30_000_000, flow: 0, unattributed: 20_000_000 } },
    } as any;
    mount(CriticalPathPanel);
    expect(host.textContent).toContain("当前范围归因覆盖");
    expect(host.textContent).toContain("其他组件已有的观测");
    expect(host.textContent).toContain("50.000 ms");
    expect(store.criticalPath!.totals.by_kind_ns.unattributed).toBe(20_000_000);
    host.querySelector<HTMLButtonElement>(".scope-explanation button")!.click();
    expect(store.componentId).toBe("");
    store.criticalPath!.component_id = "";
    await nextTick();
    expect(host.querySelector(".scope-explanation")).toBeNull();
  });

  it("opens the actual aggregate summary row without inventing a single request", async () => {
    const store = useProfilerStore();
    store.tabState.summary = "ready";
    store.summary = {
      analysis: { id: "one", request_count: 10 }, metadata: {}, stages: {}, coverage: {},
      span_summary: [{ name: "work", component_id: "worker", origin: "user", status: "error", count: 3, p50: 4, p95: 9, minimum: 2, maximum: 10, mean: 6 }],
      span_summary_total: 1, span_summary_limit: 100, span_summary_truncated: false, custom_mark_summary: [],
    } as any;
    mount(SummaryPanel);
    host.querySelector<HTMLTableRowElement>("tr[data-grid-row]")!.click();
    expect(store.selection?.kind).toBe("span-summary");
    expect(store.selection?.data).toMatchObject({ count: 3, p95: 9, status: "error" });
    expect(store.selection?.data).not.toHaveProperty("id");
    app!.unmount(); host.remove(); app = undefined;
    mount(InspectorPanel);
    await nextTick();
    expect(host.querySelector(".record-digest")?.textContent).toContain("全部请求");
    expect(host.querySelector(".record-digest")?.textContent).toContain("9.00 ms");
    expect(host.querySelector(".record-digest")?.textContent).not.toContain("主机 / 机器标识未提供");
  });
});

describe("floating disclosure cancellation", () => {
  it("closes only the active popup and returns focus without changing a filter", () => {
    document.body.innerHTML = '<input id="filter" value="model_call"><details data-popover open><summary>More</summary><button>Item</button></details><details class="raw-fields" open><summary>Raw</summary><p>Evidence</p></details>';
    document.querySelector<HTMLButtonElement>("button")!.focus();
    expect(dismissTopPopover()).toBe(true);
    expect(document.querySelector<HTMLDetailsElement>("[data-popover]")!.open).toBe(false);
    expect(document.activeElement).toBe(document.querySelector("[data-popover] summary"));
    expect(document.querySelector<HTMLInputElement>("#filter")!.value).toBe("model_call");
    expect(document.querySelector<HTMLDetailsElement>(".raw-fields")!.open).toBe(true);
    expect(dismissTopPopover()).toBe(false);
  });

  it("dismisses outside clicks and keeps only one floating popup open", () => {
    document.body.innerHTML = '<button id="outside">Outside</button><details data-popover open><summary>A</summary><button>Item</button></details><details data-popover open><summary>B</summary></details>';
    const popovers = document.querySelectorAll<HTMLDetailsElement>("details");
    const toggle = new Event("toggle");
    Object.defineProperty(toggle, "target", { value: popovers[1] });
    keepOnePopover(toggle);
    expect(popovers[0].open).toBe(false);
    expect(popovers[1].open).toBe(true);
    const pointer = new Event("pointerdown");
    Object.defineProperty(pointer, "target", { value: document.querySelector("#outside") });
    dismissOutsidePopovers(pointer as PointerEvent);
    expect(popovers[1].open).toBe(false);
  });
});
