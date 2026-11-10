"use strict";

// Deterministic Node + HTTP recovery test: the browser script is executed in a
// VM with a real loopback HTTP server, not a mocked api().  A page refresh must
// resume the saved execution reference by polling the read-only GET endpoint;
// it must never POST, replay the goal, or wait for the user to find the task.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../robot_skill_web/static/app.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../robot_skill_web/static/index.html"), "utf8");
const template = html.match(/<template\b[^>]*>([\s\S]*?)<\/template>/)[1];
const elementIds = markup => [...markup.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

function harness(fetchImpl) {
  const nodes = new Map();
  const mountedIds = new Set(elementIds(html.replace(template, "")));
  function node() {
    return {
      value: "", textContent: "", children: [], disabled: false, hidden: false, dataset: {}, attributes: {},
      content: { cloneNode: () => node() },
      append(...items) { this.children.push(...items); },
      replaceChildren(...items) {
        this.children = items;
        if (this.id === "app") for (const id of elementIds(template)) mountedIds.add(id);
      },
      querySelectorAll: () => [],
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener() {}, focus() {},
      after() {}, remove() {},
      checkValidity() { return true; }, reportValidity: () => true,
    };
  }
  const context = vm.createContext({
    document: {
      cookie: "",
      getElementById(id) {
        if (!mountedIds.has(id)) return null;
        if (!nodes.has(id)) nodes.set(id, {...node(), id});
        return nodes.get(id);
      },
      createElement: node, querySelector: () => null,
    },
    fetch: fetchImpl,
    requestAnimationFrame: callback => callback(),
    setTimeout, clearTimeout,
    FormData: class {
      constructor(form) { this.form = form; }
      entries() { return (this.form.values || [])[Symbol.iterator](); }
    },
  });
  vm.runInContext(source, context);
  return { run: code => vm.runInContext(code, context), nodes };
}

const CONNECTION = {
  connection_id: "connection-refresh",
  connection_version: 1,
  robot_id: "g1",
  robot_name: "g1",
  caller_scope: "robot_skill_cli",
  target_key: "scope:refresh",
  read_only: false,
};
const REFERENCE = {
  target_key: CONNECTION.target_key,
  robot_id: CONNECTION.robot_id,
  caller_scope: CONNECTION.caller_scope,
  request_id: "request-refresh",
  execution_id: "",
  legacy_task_id: "",
};

test("refresh resumes the saved execution over HTTP and never replays it", async () => {
  const requests = [];
  const executionPolls = [];
  let executionCount = 0;
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    requests.push({ method: request.method, path: url.pathname });
    const send = (status, payload) => {
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(JSON.stringify(payload));
    };
    if (request.method === "GET" && url.pathname === "/api/session") {
      return send(200, {
        session_active: true,
        connection: CONNECTION,
        unresolved_executions: [{ preview_id: "preview-refresh", reference: REFERENCE, phase: "pending" }],
      });
    }
    if (request.method === "POST" && url.pathname === "/api/session") {
      return send(201, { connection: CONNECTION, unresolved_executions: [] });
    }
    if (request.method === "GET" && url.pathname === "/api/status") {
      return send(200, { motion_authorized: false, control_plane_ready: true, busy: true });
    }
    if (request.method === "GET" && url.pathname === "/api/skills") return send(200, { skills: [] });
    if (request.method === "GET" && url.pathname === "/api/history") return send(200, { records: [], total: 0 });
    if (request.method === "GET" && url.pathname.startsWith("/api/executions/")) {
      const reference = JSON.parse(decodeURIComponent(url.pathname.slice("/api/executions/".length)));
      executionPolls.push(reference);
      executionCount += 1;
      const state = executionCount === 1 ? "RUNNING" : "SUCCEEDED";
      return send(200, {
        execution: { found: true, state, execution_id: "execution-refresh", observation_unknown: false },
        events: { found: true, execution_id: "execution-refresh", events: [], last_event_sequence: 0, has_more: false },
      });
    }
    return send(404, { error: { code: "NOT_FOUND", message: url.pathname } });
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const { run, nodes } = harness((url, options) => fetch(new URL(url, base), options));
  try {
    for (let attempt = 0; attempt < 200 && !run("state.terminal"); attempt += 1) await sleep(10);

    assert.equal(run("state.terminal"), true);
    assert.equal(run("state.execution.request_id"), REFERENCE.request_id);
    assert.equal(nodes.get("execution-state").textContent, "SUCCEEDED");
    assert.match(nodes.get("notification").textContent, /已观测到任务终态/);

    // The page resumed the known reference by itself and polled only GETs.
    assert.ok(executionPolls.length >= 2, `expected a repeated poll, got ${executionPolls.length}`);
    assert.deepEqual(executionPolls[0], REFERENCE);
    const executionRequests = requests.filter(item => item.path.startsWith("/api/executions"));
    assert.equal(executionRequests.length, executionPolls.length);
    assert.ok(executionRequests.every(item => item.method === "GET"));
    assert.equal(requests.some(item => item.method === "POST" && item.path.startsWith("/api/executions")), false);
    assert.equal(requests.some(item => item.method === "POST" && item.path === "/api/previews"), false);
  } finally {
    run("clearTimeout(state.statusTimer); clearTimeout(state.pollTimer); clearTimeout(state.recoveryTimer);");
    await new Promise(resolve => server.close(resolve));
  }
});
