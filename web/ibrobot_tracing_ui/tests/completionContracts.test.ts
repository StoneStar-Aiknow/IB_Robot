import { createPinia, setActivePinia } from "pinia";
import { createApp, nextTick, type App } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import InspectorPanel from "../src/components/InspectorPanel.vue";
import TopologyGraph from "../src/components/TopologyGraph.vue";
import { useProfilerStore } from "../src/stores/profiler";

vi.mock("@vue-flow/core", async (original) => ({
  ...await original<typeof import("@vue-flow/core")>(),
  useVueFlow: () => ({ fitView: vi.fn(), zoomIn: vi.fn(), zoomOut: vi.fn() }),
}));
vi.mock("elkjs/lib/elk.bundled.js", () => ({
  default: class { constructor() { throw new Error("simulated layout resource failure"); } },
}));

let app: App | undefined;
let host: HTMLDivElement | undefined;
afterEach(() => { app?.unmount(); host?.remove(); vi.restoreAllMocks(); });

describe("failure and provenance presentation", () => {
  it("shows quantitative aggregate meaning instead of a missing single-request identity", async () => {
    const pinia = createPinia(); setActivePinia(pinia);
    const store = useProfilerStore();
    store.selection = { kind: "span-aggregate", title: "model_call", data: {
      value_ns: "871686000", self_value_ns: "800000000", percentage: 25.23,
      occurrence_count: 100, request_count: 80, error_count: 2, incomplete_count: 1, invalid_count: 0,
      component_id: "policy.inference", path: [{ name: "model_call", component_id: "policy.inference" }],
    } };
    host = document.createElement("div"); app = createApp(InspectorPanel).use(pinia); app.mount(host);
    await nextTick();
    const digest = host.querySelector(".record-digest")!.textContent;
    expect(digest).toContain("871.686 ms");
    expect(digest).toContain("25.23%");
    expect(digest).toContain("100 / 80");
    expect(digest).toContain("2 / 1 / 0");
    expect(digest).not.toContain("主机 / 机器标识");
    expect(digest).not.toContain("请求未提供");
  });
  it("distinguishes layout failure from absent trace data and offers page recovery", async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const store = useProfilerStore();
    store.analysisState = "ready";
    store.topologyState = "ready";
    store.topology = {
      metric: "p95", view: "components", request_id: "", topology_source: "observed",
      nodes: [{ id: "worker", component_id: "worker", name: "worker", kind: "observed_component", parent_id: "", child_ids: [], depth: 0, provenance: "observed", trace_name: "", metrics: {}, metric_value_ms: 1, description: "" }],
      edges: [],
    };
    host = document.createElement("div");
    app = createApp(TopologyGraph).use(pinia);
    app.mount(host);
    await vi.waitFor(() => expect(host!.querySelector('[role="alert"]')?.textContent).toContain("拓扑布局暂时不可用"));
    expect(host.textContent).not.toContain("当前追踪可能缺少拓扑清单");
    expect(host.querySelector('[role="alert"] button')?.textContent).toBe("刷新页面重试");
    expect(store.analysisState).toBe("ready");
    expect(store.topology?.nodes).toHaveLength(1);
  });

  it("shows the actual endpoints and missing duration without inventing a host or synchronization", async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const store = useProfilerStore();
    store.selection = {
      kind: "flow", title: "return", subtitle: "request",
      data: { trace_id: "request", edge_id: "return", duration_ms: null, status: "cross_host_unsynchronized",
        provenance: { send: { origin: { host: "cloud-host" }, time_mapping: {} }, receive: { origin: {}, time_mapping: {} } } },
    };
    host = document.createElement("div");
    app = createApp(InspectorPanel).use(pinia);
    app.mount(host);
    await nextTick();
    const digest = Object.fromEntries(Array.from(host.querySelectorAll(".record-digest dt"), (term) => [term.textContent, term.nextElementSibling?.textContent]));
    expect(digest["发送主机 / 机器标识"]).toBe("cloud-host");
    expect(digest["接收主机 / 机器标识"]).toBe("未提供");
    expect(digest["记录状态"]).toBe("跨主机时钟未同步");
    expect(digest["观测耗时"]).toBe("—");
    expect(digest["发送端校准"]).toBe("未提供");
  });
});
