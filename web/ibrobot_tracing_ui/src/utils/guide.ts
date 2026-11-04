import type { LocationQueryValue } from "vue-router";

export type GuideTopic = "capture" | "instrument";

/** A guide can return only to a local analysis route, including its filters. */
export function guideReturnPath(value: LocationQueryValue | LocationQueryValue[] | undefined): string {
  return typeof value === "string" && /^\/analysis(?:\/[a-zA-Z0-9_-]+)?(?:\?[^#]*)?$/.test(value)
    ? value : "/analysis";
}

export function guideLocation(topic: GuideTopic, currentPath: string, from?: LocationQueryValue | LocationQueryValue[]) {
  return { name: "guide", params: { topic }, query: { from: guideReturnPath(currentPath.startsWith("/guide/") ? from : currentPath) } };
}
