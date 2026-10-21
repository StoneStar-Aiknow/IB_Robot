import { createApp, h, nextTick, type App } from "vue";
import { createMemoryHistory, createRouter } from "vue-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import GuideCode from "../src/components/GuideCode.vue";
import GuideLink from "../src/components/GuideLink.vue";
import { guideReturnPath } from "../src/utils/guide";

let app: App | undefined;
let host: HTMLDivElement | undefined;
afterEach(() => { app?.unmount(); host?.remove(); vi.unstubAllGlobals(); });

function mount(component: any, props: Record<string, unknown>) {
  host = document.createElement("div"); document.body.appendChild(host);
  app = createApp(component, props); app.mount(host);
}

describe("trace onboarding", () => {
  it("returns only to local analysis routes, retaining their exact query", () => {
    expect(guideReturnPath("/analysis/abc?tab=timeline&request=r1&q=model_call")).toBe("/analysis/abc?tab=timeline&request=r1&q=model_call");
    for (const value of ["https://other.test/", "//other.test", "/analysis/../guide/capture", "/analysis-elsewhere", ["/analysis"], null]) {
      expect(guideReturnPath(value)).toBe("/analysis");
    }
  });

  it("preserves the originating analysis when switching between guide topics", async () => {
    const router = createRouter({ history: createMemoryHistory(), routes: [
      { path: "/analysis/:analysisId?", name: "analysis", component: {} },
      { path: "/guide/:topic", name: "guide", component: {} },
    ] });
    const from = "/analysis/abc?tab=timeline&request=r1&q=model_call";
    await router.push({ name: "guide", params: { topic: "instrument" }, query: { from } });
    host = document.createElement("div"); document.body.appendChild(host);
    app = createApp({ render: () => h(GuideLink, { topic: "capture", section: "manual" }, () => "手动记录") }).use(router); app.mount(host);
    host.querySelector("a")!.click();
    await vi.waitFor(() => expect(router.currentRoute.value.hash).toBe("#manual"));
    expect(router.currentRoute.value.query.from).toBe(from);
  });

  it("copies exactly the displayed snippet", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    mount(GuideCode, { label: "代码", code: "source .shrc_local\nlttng --version" });
    host!.querySelector("button")!.click();
    await vi.waitFor(() => expect(host!.querySelector('[role="status"]')?.textContent).toBe("已复制"));
    expect(writeText).toHaveBeenCalledWith("source .shrc_local\nlttng --version");
  });

  it("offers native selection on LAN HTTP instead of falsely reporting copy success", async () => {
    vi.stubGlobal("navigator", {});
    mount(GuideCode, { label: "代码", code: "print('trace')" });
    host!.querySelector("button")!.click(); await nextTick();
    expect(document.activeElement).toBe(host!.querySelector("pre"));
    expect(window.getSelection()?.toString()).toBe("print('trace')");
    expect(host!.querySelector('[role="status"]')?.textContent).toContain("自动复制不可用");
  });
});
