"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../robot_skill_web/static/app.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../robot_skill_web/static/index.html"), "utf8");
const template = html.match(/<template\b[^>]*>([\s\S]*?)<\/template>/)[1];
const elementIds = markup => [...markup.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);

function harness({ mountWorkbench = true, missingWorkbenchIds = [], renderSkillList = false, withHistory = false } = {}) {
  const nodes = new Map();
  const mountedIds = new Set(elementIds(html.replace(template, "")));
  const timers = new Map();
  let nextTimer = 0;
  let displayedPlan = null;
  function node() {
    return {
      value: "", children: [], disabled: false, dataset: {}, attributes: {},
      content: { cloneNode: () => node() },
      append(...items) { this.children.push(...items); },
      replaceChildren(...items) {
        this.children = items;
        if (this.id === "app") {
          for (const id of elementIds(template)) if (!missingWorkbenchIds.includes(id)) mountedIds.add(id);
        }
      },
      querySelectorAll: () => [],
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener() {}, focus() {},
      after(item) { displayedPlan = item; },
      remove() { if (displayedPlan === this) displayedPlan = null; },
      checkValidity() { return this.valid !== false; }, reportValidity: () => true,
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
      createElement: node, querySelector: () => displayedPlan,
    },
    fetch: () => new Promise(() => {}),
    requestAnimationFrame: callback => callback(),
    setTimeout(callback, delay) { const id = ++nextTimer; timers.set(id, {callback, delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
    FormData: class {
      constructor(form) { this.form = form; }
      entries() { return (this.form.values || [])[Symbol.iterator](); }
    },
  });
  vm.runInContext(source, context);
  const run = code => vm.runInContext(code, context);
  if (mountWorkbench) run("showWorkbench()");
  run(`
    state.connection = {connection_version: 1, read_only: true};
    renderForm = () => {};
    globalThis.httpApi = api;
    globalThis.calls = [];
    api = async (path, options) => {
      calls.push({path, body: options?.body ? JSON.parse(options.body) : null});
      if (path === "/api/previews") return {preview_id: "preview-1", plan: {steps: []}};
      return {name: decodeURIComponent(path.split("/").pop()), parameters: {properties: {}}};
    };
  `);
  if (!renderSkillList) run("renderSkills = () => {}");
  if (!withHistory) run("refreshHistory = async () => {}");
  const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
  const flushTimers = async () => {
    for (const [id, timer] of [...timers]) {
      if (!timers.delete(id)) continue;
      timer.callback();
    }
    await settle();
  };
  return { run, nodes, input: nodes.get("raw-command"), execute: nodes.get("execute"), flushTimers, settle };
}

test("the real HTML template contains every workbench control bound by the script", () => {
  const { run } = harness();
  assert.equal(run("document.getElementById('login')"), null);
  assert.equal(run("document.getElementById('deployment-key')"), null);
  assert.equal(run("document.getElementById('connect-form')"), null);
  assert.equal(run("document.getElementById('preview')"), null);
  assert.deepEqual(JSON.parse(run("JSON.stringify(ids.filter(id => !elements[id]))")), []);
  assert.doesNotThrow(() => run("updateButtons()"));
});

test("collapsing and expanding the sidebar preserves the plan, history, and execution without requests", () => {
  const {run, nodes} = harness({withHistory: true});
  run(`
    state.preview = {preview_id: "prepared-plan"};
    state.execution = {request_id: "live-task"};
    state.connection.read_only = false;
    state.historySelected = "saved-task";
    elements["execution-state"].textContent = "ACTIVE";
    elements["history-detail"].hidden = false;
    elements["history-detail"].append(element("p", "", "saved detail"));
    updateButtons();
  `);
  nodes.get("history-list").scrollTop = 160;
  nodes.get("skill-list").scrollTop = 240;
  assert.equal(nodes.get("execution-sidebar").hidden, false);
  assert.equal(nodes.get("toggle-sidebar").attributes["aria-expanded"], "true");
  nodes.get("toggle-sidebar").onclick();
  assert.equal(nodes.get("execution-sidebar").hidden, true);
  assert.match(nodes.get("work-layout").className, /is-sidebar-collapsed/);
  assert.equal(nodes.get("toggle-sidebar").textContent, "展开侧栏");
  assert.equal(nodes.get("toggle-sidebar").attributes["aria-expanded"], "false");
  assert.equal(nodes.get("execution-summary").hidden, false);
  assert.equal(nodes.get("execution-summary").textContent, "执行：ACTIVE");
  assert.equal(nodes.get("cancel-collapsed").hidden, false);
  assert.equal(nodes.get("cancel-collapsed").disabled, false);
  nodes.get("toggle-sidebar").onclick();
  assert.equal(nodes.get("execution-sidebar").hidden, false);
  assert.equal(nodes.get("work-layout").className, "work-layout");
  assert.equal(nodes.get("toggle-sidebar").textContent, "收起侧栏");
  assert.equal(nodes.get("execution-summary").hidden, true);
  assert.equal(nodes.get("cancel-collapsed").hidden, true);
  assert.equal(nodes.get("history-detail").hidden, false);
  assert.equal(resultText(nodes.get("history-detail")).trim(), "saved detail");
  assert.equal(nodes.get("history-list").scrollTop, 160);
  assert.equal(nodes.get("skill-list").scrollTop, 240);
  assert.equal(run("state.historySelected"), "saved-task");
  assert.equal(run("state.preview.preview_id"), "prepared-plan");
  assert.equal(run("state.execution.request_id"), "live-task");
  assert.equal(run("calls.length"), 0);
});

test("folded execution status keeps polling and retains unknown and terminal outcomes", async () => {
  const {run, nodes} = harness();
  run(`
    state.execution = {request_id: "live-task"};
    state.connection.read_only = false;
    toggleSidebar();
    api = async () => {throw new Error("observation unavailable");};
  `);
  await run("pollExecution()");
  assert.equal(nodes.get("execution-sidebar").hidden, true);
  assert.match(nodes.get("execution-summary").textContent, /观测失败，结果未知/);
  assert.equal(nodes.get("cancel-collapsed").disabled, false);
  assert.equal(run("active()"), true);
  run('api = async () => ({execution: {found: true, state: "SUCCEEDED"}, events: {events: []}})');
  await run("pollExecution()");
  assert.equal(nodes.get("execution-sidebar").hidden, true);
  assert.match(nodes.get("execution-summary").textContent, /SUCCEEDED/);
  assert.equal(nodes.get("cancel-collapsed").hidden, true);
  assert.equal(nodes.get("cancel-collapsed").disabled, true);
  assert.equal(run("active()"), false);
});

test("folded cancel targets only the active task and cannot duplicate an in-flight cancel", async () => {
  const {run, nodes} = harness();
  run(`
    state.execution = {request_id: "live-task"};
    state.connection.read_only = false;
    state.historySelected = "old-task";
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET"});
      if (path.endsWith("/cancel")) return new Promise(resolve => {globalThis.finishCancel = resolve;});
      return {execution: {found: true, state: "CANCELED"}, events: {events: []}};
    };
    updateButtons();
    toggleSidebar();
  `);
  const pending = nodes.get("cancel-collapsed").onclick();
  assert.equal(nodes.get("cancel-collapsed").disabled, true);
  nodes.get("toggle-sidebar").onclick();
  assert.equal(nodes.get("cancel").disabled, true);
  await nodes.get("cancel").onclick();
  assert.equal(run("calls.length"), 1);
  assert.match(decodeURIComponent(run("calls[0].path")), /live-task/);
  assert.doesNotMatch(decodeURIComponent(run("calls[0].path")), /old-task/);
  run("finishCancel({accepted: true})");
  await pending;
  assert.equal(run("state.canceling"), false);
  assert.equal(run("state.terminal"), true);
  assert.equal(nodes.get("cancel").disabled, true);
  run("state.terminal = false; state.connection.read_only = true; updateButtons()");
  const count = run("calls.length");
  await nodes.get("cancel-collapsed").onclick();
  assert.equal(run("calls.length"), count);
  assert.equal(nodes.get("cancel-collapsed").disabled, true);
});

test("opening the page establishes the bound session without sending an action", async () => {
  const { run, nodes } = harness({ mountWorkbench: false });
  run(`
    state.connection = null;
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET", body: options?.body ? JSON.parse(options.body) : null});
      if (path === "/api/session" && (options?.method || "GET") === "GET") {
        return {session_active: false, connection: null, unresolved_executions: []};
      }
      if (path === "/api/session") {
        return {session_active: true, connection: {
          connection_id: "connection-1", connection_version: 1, robot_id: "unitree_g1",
          robot_name: "unitree_g1", caller_scope: "robot_skill_cli",
          target_key: "scope:binding", read_only: false}, unresolved_executions: []};
      }
      if (path === "/api/status") return {motion_authorized: false, control_plane_ready: true};
      if (path === "/api/skills") return {skills: []};
      throw new Error("Unexpected API request: " + path);
    };
  `);
  await run("bootstrap()");
  assert.equal(run("document.getElementById('login')"), null);
  assert.equal(run("document.getElementById('discovery-controls')"), null);
  assert.equal(nodes.get("connection-title").textContent, "已绑定部署配置");
  assert.equal(nodes.get("connection-summary").textContent, "unitree_g1 · 由启动配置绑定");
  assert.equal(resultText(nodes.get("session-panel")).trim(), "已绑定部署配置");
  assert.equal(nodes.get("execute").disabled, true);
  assert.match(resultText(nodes.get("target-chip")), /unitree_g1/);
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), [
    {path: "/api/session", method: "GET", body: null},
    {path: "/api/session", method: "POST", body: {}},
    {path: "/api/status", method: "GET", body: null},
    {path: "/api/skills", method: "GET", body: null},
  ]);
});

test("session establishment sends no target, credential, or motion request", async () => {
  assert.doesNotMatch(html, /type="password"/);
  const { run, nodes, settle } = harness();
  run(`
    state.connection = null;
    adoptConnection = async connection => {state.connection = connection;};
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET", body: options?.body ? JSON.parse(options.body) : null});
      if (path === "/api/session" && (options?.method || "GET") === "GET") return {session_active: false};
      if (path === "/api/session") return {session_active: true, connection: {
        connection_id: "connection-1", connection_version: 1, robot_id: "g1", robot_name: "g1",
        caller_scope: "robot_skill_cli", target_key: "scope:binding", read_only: false},
        unresolved_executions: []};
      throw new Error("Unexpected API request: " + path);
    };
  `);
  await run("establishSession()");
  await settle();
  assert.equal(run("state.connection.connection_id"), "connection-1");
  assert.equal(nodes.get("execute").disabled, true);
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), [
    {path: "/api/session", method: "POST", body: {}},
  ]);
});

test("a rejected establishment leaves execution disabled and reports the error", async () => {
  const { run, nodes } = harness();
  run(`
    state.connection = null;
    notice = (message, error) => {globalThis.lastNotice = {message, error};};
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET"});
      if (path === "/api/session" && !options?.method) return {session_active: false};
      throw new Error("deployment is not bound");
    };
  `);
  await run("bootstrap()");
  assert.equal(run("state.connection"), null);
  assert.equal(nodes.get("execute").disabled, true);
  assert.equal(run("lastNotice.error"), true);
  assert.match(run("lastNotice.message"), /deployment is not bound/);
});

test("the bound target never offers discovery, selection, or disconnect controls", async () => {
  const { run, nodes } = harness();
  for (const id of ["robot-list", "refresh-robots", "toggle-discovery", "discovery-controls", "disconnect"]) {
    assert.equal(run(`document.getElementById(${JSON.stringify(id)})`), null, id);
  }
  run(`
    api = async path => {
      calls.push({path});
      if (path === "/api/status") return {motion_authorized: false, control_plane_ready: true};
      if (path === "/api/skills") return {skills: []};
      throw new Error("Unexpected API request: " + path);
    };
  `);
  await run(`adoptConnection({connection_id: "connection-1", connection_version: 1,
    robot_id: "unitree_g1", robot_name: "unitree_g1", target_key: "scope:binding", read_only: false})`);
  assert.equal(nodes.get("connection-title").textContent, "已绑定部署配置");
  assert.equal(nodes.get("connection-summary").textContent, "unitree_g1 · 由启动配置绑定");
  assert.equal(nodes.get("connection-summary").hidden, false);
  assert.equal(nodes.get("execute").disabled, true);
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), [{path: "/api/status"}, {path: "/api/skills"}]);
});

test("before adoption the workbench waits and keeps actions disabled", () => {
  const { run, nodes } = harness();
  run("state.connection = null; updateButtons();");
  assert.equal(nodes.get("connection-title").textContent, "等待会话");
  assert.equal(nodes.get("connection-summary").hidden, true);
  assert.equal(nodes.get("execute").disabled, true);
  assert.equal(nodes.get("retry-preview").disabled, true);
});

test("incompatible HTML fails closed with refresh guidance instead of a null property error", async () => {
  const { run, nodes } = harness({mountWorkbench: false, missingWorkbenchIds: ["retry-preview"]});
  run(`
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET"});
      return {session_active: true};
    };
  `);
  await run("bootstrap()");
  assert.match(nodes.get("notification").textContent, /Ctrl\+Shift\+R/);
  assert.match(nodes.get("notification").textContent, /retry-preview/);
  assert.doesNotMatch(nodes.get("notification").textContent, /Cannot set properties of null/);
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), [{path: "/api/session", method: "GET"}]);
});

test("selecting skills generates and updates meaningful default audit text", async () => {
  const { run, input } = harness();
  await run('selectSkill({name: "wave_hand"})');
  assert.equal(input.value, "通过 Web 请求执行技能：wave_hand");
  await run('selectSkill({name: "arms_horizontal"})');
  assert.equal(input.value, "通过 Web 请求执行技能：arms_horizontal");
});

test("a manual purpose is retained instead of silently overwritten", async () => {
  const { run, input } = harness();
  await run('selectSkill({name: "wave_hand"})');
  input.value = "Operator supervised single gesture trial";
  input.oninput();
  await run('selectSkill({name: "arms_horizontal"})');
  assert.equal(input.value, "Operator supervised single gesture trial");
});

test("blank text is restored before preview, without unlocking execution", async () => {
  const { run, input, execute } = harness();
  await run('selectSkill({name: "wave_hand"})');
  input.value = "   ";
  run(`
    globalThis.calls = [];
    api = async (path, options) => {
      calls.push({path, body: JSON.parse(options.body)});
      return {preview_id: "preview-1", plan: {steps: []}};
    };
  `);
  await run("createPreview()");
  assert.equal(input.value, "通过 Web 请求执行技能：wave_hand");
  const calls = JSON.parse(run("JSON.stringify(calls)"));
  assert.equal(calls[0].path, "/api/previews");
  assert.equal(calls[0].body.raw_command, input.value);
  assert.equal(calls[0].body.skill_name, "wave_hand");
  assert.equal(execute.disabled, true);
  await run("executePreview()");
  assert.equal(run("calls.length"), 1);
});

test("manual edits invalidate the displayed plan and require a new preview", async () => {
  const { run, input, execute } = harness();
  await run('selectSkill({name: "wave_hand"})');
  run("state.preview = {preview_id: 'old'}; state.connection.read_only = false; state.motionAuthorized = true; updateButtons()");
  assert.equal(execute.disabled, false);
  input.value = "Revised trial purpose";
  input.oninput();
  assert.equal(run("state.preview"), null);
  assert.equal(execute.disabled, true);
});

test("a new connection does not inherit the previous robot's audit text", async () => {
  const { run, input } = harness();
  await run('selectSkill({name: "wave_hand"})');
  input.value = "Purpose for the previous robot";
  run("refreshStatus = async () => {}; api = async () => ({skills: []})");
  await run('adoptConnection({connection_version: 2, instance_id: "robot-2", read_only: true})');
  assert.equal(input.value, "");
  assert.equal(run("state.defaultRawCommand"), "");
});

test("selecting a skill automatically prepares but never submits a motion", async () => {
  const { run, execute, flushTimers } = harness();
  run("state.connection.read_only = false; state.motionAuthorized = true");
  await run('selectSkill({name: "wave_hand"})');
  assert.equal(execute.disabled, true);
  await flushTimers();
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls.map(call => call.path))")),
    ["/api/skills/wave_hand", "/api/previews"]);
  assert.equal(run("state.preview.preview_id"), "preview-1");
  assert.equal(execute.disabled, false);
  assert.equal(execute.textContent, "执行 wave_hand");
});

test("rapid parameter and purpose edits discard the old plan and debounce preparation", async () => {
  const { run, input, execute, nodes, flushTimers } = harness();
  run("state.connection.read_only = false; state.motionAuthorized = true");
  await run('selectSkill({name: "wave_hand"})');
  await flushTimers();
  run("calls = []; state.skill.parameters.properties.distance = {type: 'number'}");
  const form = nodes.get("parameters-form");
  form.values = [["distance", "1"]];
  form.oninput();
  form.values = [["distance", "2"]];
  form.oninput();
  input.value = "Updated purpose";
  input.oninput();
  assert.equal(run("state.preview"), null);
  assert.equal(execute.disabled, true);
  assert.equal(run("calls.length"), 0);
  await flushTimers();
  const calls = JSON.parse(run("JSON.stringify(calls)"));
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, "/api/previews");
  assert.equal(calls[0].body.parameters.distance, 2);
  assert.equal(calls[0].body.raw_command, "Updated purpose");
});

test("a slow skill description cannot restore an old selection or executable plan", async () => {
  const { run, execute, flushTimers, settle } = harness();
  run("state.connection.read_only = false; state.motionAuthorized = true");
  await run('selectSkill({name: "wave_hand"})');
  await flushTimers();
  run(`
    const originalApi = api;
    api = (path, options) => path === "/api/skills/slow" ?
      new Promise(resolve => {globalThis.resolveSlow = resolve;}) : originalApi(path, options);
    globalThis.selectSlow = selectSkill({name: "slow"});
  `);
  assert.equal(run("state.preview"), null);
  assert.equal(execute.disabled, true);
  await run('selectSkill({name: "new_skill"})');
  await flushTimers();
  run('resolveSlow({name: "slow", parameters: {properties: {}}})');
  await settle();
  assert.equal(run("state.skill.name"), "new_skill");
  assert.equal(execute.textContent, "执行 new_skill");
  assert.equal(run("calls.filter(call => call.path === '/api/previews').length"), 2);
});

test("in-flight preparations are serialized and stale replies never enable execution", async () => {
  const { run, input, execute, flushTimers, settle } = harness();
  run("state.connection.read_only = false; state.motionAuthorized = true");
  await run('selectSkill({name: "wave_hand"})');
  run(`
    calls = []; globalThis.replies = [];
    api = (path, options) => {
      calls.push({path, body: JSON.parse(options.body)});
      return new Promise(resolve => replies.push(resolve));
    };
  `);
  await flushTimers();
  assert.equal(run("calls.length"), 1);
  input.value = "Latest purpose";
  input.oninput();
  await flushTimers();
  assert.equal(run("calls.length"), 1);
  run('replies[0]({preview_id: "obsolete", plan: {steps: []}})');
  await settle();
  assert.equal(run("calls.length"), 2);
  assert.equal(run("state.preview"), null);
  assert.equal(execute.disabled, true);
  assert.equal(run("calls[1].body.raw_command"), "Latest purpose");
  run('replies[1]({preview_id: "latest", plan: {steps: []}})');
  await settle();
  assert.equal(run("state.preview.preview_id"), "latest");
  assert.equal(execute.disabled, false);
  assert.equal(run("calls.some(call => call.path === '/api/executions')"), false);
});

test("editing during the presentation flush cannot unlock a stale plan", async () => {
  const { run, input, execute, flushTimers, settle } = harness();
  run(`
    state.connection.read_only = false; state.motionAuthorized = true;
    globalThis.frames = []; requestAnimationFrame = callback => frames.push(callback);
  `);
  await run('selectSkill({name: "wave_hand"})');
  await flushTimers();
  assert.equal(run("frames.length"), 1);
  assert.equal(execute.disabled, true);
  input.value = "Changed while rendering";
  input.oninput();
  run("frames.shift()(); frames.shift()()");
  await settle();
  assert.equal(run("state.preview"), null);
  assert.equal(execute.disabled, true);
});

test("switching connection invalidates in-flight preparation", async () => {
  const { run, execute, flushTimers, settle } = harness();
  await run('selectSkill({name: "wave_hand"})');
  run('api = () => new Promise(resolve => {globalThis.reply = resolve;})');
  await flushTimers();
  run("refreshStatus = async () => {}; api = async () => ({skills: []})");
  await run('adoptConnection({connection_version: 2, instance_id: "robot-2", read_only: false})');
  run('reply({preview_id: "old-robot", plan: {steps: []}})');
  await settle();
  assert.equal(run("state.preview"), null);
  assert.equal(run("state.skill"), null);
  assert.equal(execute.disabled, true);
});

test("adopting a session invalidates the plan before the status request completes", async () => {
  const { run, execute, flushTimers, settle } = harness();
  run("state.connection.read_only = false; state.motionAuthorized = true");
  await run('selectSkill({name: "wave_hand"})');
  await flushTimers();
  assert.notEqual(run("state.preview"), null);
  run(`
    api = () => new Promise((resolve, reject) => {globalThis.rejectAdopt = reject;});
    globalThis.adopting = adoptConnection({connection_id: "connection-2", connection_version: 1,
      robot_id: "g1", robot_name: "g1", target_key: "scope:binding", read_only: false});
  `);
  assert.equal(run("state.preview"), null);
  assert.equal(execute.disabled, true);
  run('rejectAdopt(new Error("status failed"))');
  await settle();
  assert.equal(run("state.preview"), null);
  assert.equal(execute.disabled, true);
});

test("incomplete or unsupported parameters do not prepare a plan", async () => {
  const { run, nodes, flushTimers } = harness();
  await run('selectSkill({name: "wave_hand"})');
  nodes.get("parameters-form").valid = false;
  await flushTimers();
  assert.equal(run("calls.filter(call => call.path === '/api/previews').length"), 0);
  nodes.get("parameters-form").valid = true;
  run("state.unsupportedType = true; schedulePreview()");
  await flushTimers();
  assert.equal(run("calls.filter(call => call.path === '/api/previews').length"), 0);
  run("state.unsupportedType = false");
  nodes.get("parameters-form").oninput();
  await flushTimers();
  assert.equal(run("calls.filter(call => call.path === '/api/previews').length"), 1);
});

test("preparation failure offers an explicit retry without an automatic request loop", async () => {
  const { run, nodes, execute, flushTimers } = harness();
  await run('selectSkill({name: "wave_hand"})');
  run('calls = []; api = async path => {calls.push(path); throw new Error("unreachable");}');
  await flushTimers();
  assert.equal(execute.disabled, true);
  assert.equal(nodes.get("retry-preview").hidden, false);
  assert.match(nodes.get("plan-status").textContent, /unreachable/);
  await flushTimers();
  assert.equal(run("calls.length"), 1);
  nodes.get("retry-preview").onclick();
  await flushTimers();
  assert.equal(run("calls.length"), 2);
});

test("only an explicit execute click submits, and duplicates or active edits cannot resubmit", async () => {
  const { run, nodes, input, execute, flushTimers } = harness();
  run("state.connection.read_only = false; state.motionAuthorized = true");
  await run('selectSkill({name: "wave_hand"})');
  await flushTimers();
  run(`
    calls = []; startPolling = () => {};
    api = async (path, options) => {
      calls.push({path, body: JSON.parse(options.body)});
      return {execution: {request_id: "explicit-request"}};
    };
  `);
  await execute.onclick();
  await execute.onclick();
  input.oninput();
  nodes.get("parameters-form").oninput();
  await flushTimers();
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), [{
    path: "/api/executions", body: {preview_id: "preview-1", connection_version: 1},
  }]);
  assert.equal(execute.disabled, true);
  assert.equal(input.disabled, true);
  assert.equal(run("state.preview.preview_id"), "preview-1");
});

test("authorization loss prevents execution of an automatically prepared plan", async () => {
  const { run, execute, flushTimers } = harness();
  run("state.connection.read_only = false; state.motionAuthorized = true");
  await run('selectSkill({name: "wave_hand"})');
  await flushTimers();
  run("state.motionAuthorized = false; updateButtons(); calls = []");
  await execute.onclick();
  assert.equal(execute.disabled, true);
  assert.equal(run("calls.length"), 0);
});

function resultText(node) {
  if (typeof node === "string") return node;
  return [node.textContent || "", ...node.children.map(resultText)].join(" ");
}

async function observeFailure(events) {
  const result = harness();
  result.run(`
    state.connection.read_only = false;
    state.motionAuthorized = true;
    state.execution = {request_id: "request-1"};
    globalThis.calls = [];
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET"});
      return {
        execution: {found: true, state: "FAILED", error_code: "NAMED_MOTION_REJECTED", message: "service ownership query failed"},
        events: ${JSON.stringify(events)}
      };
    };
  `);
  await result.run("pollExecution()");
  assert.equal(result.run("state.terminal"), true);
  assert.equal(result.nodes.get("execution-state").textContent, "FAILED");
  assert.equal(result.nodes.get("cancel").disabled, true);
  assert.equal(result.execute.disabled, true);
  assert.deepEqual(JSON.parse(result.run("JSON.stringify(calls.map(call => call.method))")), ["GET"]);
  return resultText(result.nodes.get("execution-result"));
}

test("authoritative event states appear without a false gap", async () => {
  const output = await observeFailure({
    events: [{sequence: 1, state: "RUNNING"}, {sequence: 2, state: "FAILED"}], event_gap: false,
  });
  assert.match(output, /#1 RUNNING/);
  assert.match(output, /#2 FAILED/);
  assert.doesNotMatch(output, /事件存在缺口|事件查询失败/);
});

test("event query errors are distinct from missing event sequences", async () => {
  const output = await observeFailure({
    events: [], unavailable: true, error_code: "INVALID_RESPONSE", message: "invalid event response",
  });
  assert.match(output, /事件查询失败（INVALID_RESPONSE）/);
  assert.doesNotMatch(output, /事件存在缺口/);
});

test("a genuine sequence gap retains the warning and terminal result", async () => {
  const output = await observeFailure({events: [{sequence: 2, state: "FAILED"}], event_gap: true});
  assert.match(output, /事件存在缺口/);
  assert.doesNotMatch(output, /事件查询失败/);
});

test("workbench has a persistent readiness-scope note", () => {
  const note = html.match(/<p id="verification-note">([\s\S]*?)<\/p>/);
  assert.ok(note, "verification note is missing");
  assert.match(note[1], /就绪/);
  assert.match(note[1], /真机验收/);
});

test("compact skill rows show the supplied title, preset tag and readiness with technical details in the tooltip", () => {
  const { run, nodes } = harness({renderSkillList: true});
  run(`
    state.skills = [{name: "g1_blow_kiss_with_both_hands", display_name: "双手飞吻",
      summary: "双手飞吻", tags: ["preset"], ready: true, verification_status: "unverified"}];
    state.skill = state.skills[0];
    renderSkills();
  `);
  const item = nodes.get("skill-list").children[0];
  assert.equal(item.children[0].children[0].textContent, "双手飞吻");
  assert.equal(item.children[0].children[1].textContent, "预置");
  assert.equal(item.children[0].children[1].className, "skill-tag");
  assert.equal(item.children[0].children[2].textContent, "就绪");
  assert.equal(item.children.length, 1);
  assert.equal(item.title, "双手飞吻\ng1_blow_kiss_with_both_hands");
  assert.equal(item.attributes["aria-selected"], "true");
  const output = resultText(item);
  assert.equal(output.match(/双手飞吻/g).length, 1);
  assert.match(output, /就绪/);
  assert.doesNotMatch(output, /g1_blow_kiss_with_both_hands|READY|未验证|尚未经过验证/);
});

test("selecting a skill preserves list scroll position while changing the search resets it", async () => {
  const { run, nodes } = harness({renderSkillList: true});
  run(`state.skills = Array.from({length: 23}, (_, index) => ({name: "motion_" + index})); renderSkills()`);
  const list = nodes.get("skill-list");
  list.scrollTop = 420;
  await list.children[20].onclick();
  assert.equal(run("state.skill.name"), "motion_20");
  assert.equal(list.scrollTop, 420);
  nodes.get("skill-search").value = "motion_20";
  run("renderSkills()");
  assert.equal(list.children.length, 1);
  assert.equal(list.scrollTop, 0);
});

test("skill search matches Chinese labels, stable identifiers, and summaries", () => {
  const { run, nodes } = harness({renderSkillList: true});
  run(`state.skills = [{name: "vendor_kiss", display_name: "双手飞吻", summary: "双臂问候", tags: ["preset"]}]`);
  for (const query of ["飞吻", "VENDOR_KISS", "双臂"]) {
    nodes.get("skill-search").value = query;
    run("renderSkills()");
    assert.equal(nodes.get("skill-list").children[0].children[0].children[0].textContent, "双手飞吻");
  }
  nodes.get("skill-search").value = "不存在";
  run("renderSkills()");
  assert.equal(nodes.get("skill-list").children[0].textContent, "没有匹配技能");
});

test("legacy skills retain their original name without invented preset or verification badges", () => {
  const { run, nodes } = harness({renderSkillList: true});
  run(`state.skills = [{name: "new_vendor_motion", summary: "preset action", ready: false, verification_status: "unverified"}]; renderSkills()`);
  const item = nodes.get("skill-list").children[0];
  assert.equal(item.children[0].children[0].textContent, "new_vendor_motion");
  assert.equal(item.children[0].children.length, 2);
  assert.equal(item.children[0].children.some(child => child.className === "skill-tag"), false);
  assert.equal(item.title, "new_vendor_motion\npreset action");
  const output = resultText(item);
  assert.match(output, /未就绪/);
  assert.doesNotMatch(output, /preset action|预置|未验证/);
});

test("Chinese skill details and buttons keep the stable execution identifier and admission gates", async () => {
  const { run, nodes, input, execute, flushTimers, settle } = harness({renderSkillList: true});
  run(`
    state.skills = [{name: "vendor_kiss", display_name: "双手飞吻", summary: "双手飞吻", tags: ["preset"], ready: true}];
    api = async (path, options) => {
      calls.push({path, body: options?.body ? JSON.parse(options.body) : null});
      if (path === "/api/skills/vendor_kiss") return {...state.skills[0], parameters: {properties: {}}, verification_status: "unverified"};
      if (path === "/api/previews") return {preview_id: "preview-1", plan: {steps: []}};
      throw new Error("Unexpected API request: " + path);
    };
    renderSkills();
  `);
  await nodes.get("skill-list").children[0].onclick();
  await settle();
  const detail = resultText(nodes.get("skill-detail"));
  assert.match(detail, /双手飞吻/);
  assert.match(detail, /预置/);
  assert.match(detail, /vendor_kiss/);
  assert.equal(detail.match(/双手飞吻/g).length, 1);
  assert.doesNotMatch(detail, /未验证|尚未经过验证/);
  assert.equal(input.value, "通过 Web 请求执行技能：双手飞吻（vendor_kiss）");
  assert.equal(execute.textContent, "执行 双手飞吻");
  assert.equal(execute.disabled, true);
  await flushTimers();
  const calls = JSON.parse(run("JSON.stringify(calls)"));
  assert.deepEqual(calls.map(call => call.path), ["/api/skills/vendor_kiss", "/api/previews"]);
  assert.equal(calls[1].body.skill_name, "vendor_kiss");
  assert.equal(calls[1].body.raw_command, input.value);
  assert.match(nodes.get("plan-status").textContent, /「执行 双手飞吻」/);
  assert.equal(execute.disabled, true);
  await run("executePreview()");
  assert.equal(run("calls.length"), 2);
  run("state.connection.read_only = false; updateButtons()");
  assert.equal(execute.disabled, true);
  run("state.motionAuthorized = true; updateButtons()");
  assert.equal(execute.disabled, false);
});

test("connecting loads local history without submitting or querying old tasks", async () => {
  const {run, nodes, settle} = harness({withHistory: true});
  run(`api = async path => {
    calls.push(path);
    if (path === "/api/status") return {motion_authorized: true};
    if (path === "/api/skills") return {skills: []};
    if (path === "/api/history") return {records: [], total: 0, limit: 100};
    throw new Error("Unexpected API: " + path);
  }`);
  await run(`adoptConnection({connection_id: "new-connection", instance_id: "instance", robot_name: "g1", read_only: false})`);
  await settle();
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), ["/api/status", "/api/skills", "/api/history"]);
  assert.match(nodes.get("history-status").textContent, /暂无本机 Web 提交记录/);
  assert.equal(nodes.get("history-detail").hidden, true);
  assert.equal(nodes.get("refresh-history").disabled, false);
  assert.equal(run("state.execution"), null);
});

test("history inspection stays separate from a running task and its cancellation target", async () => {
  const {run, nodes, execute} = harness({withHistory: true});
  run(`
    state.connection.read_only = false;
    state.execution = {request_id: "live-request"};
    state.preview = {preview_id: "live-preview"};
    state.motionAuthorized = true;
    globalThis.originalExecution = state.execution;
    globalThis.originalPreview = state.preview;
    globalThis.record = {
      preview_id: "old-preview", submitted_at: 1760000000, state: "SUCCEEDED", observed_at: 1760000010,
      reference: {request_id: "old-request", target_key: "scope:old-instance", robot_id: "g1"},
      command: {skill_name: "old_skill", display_name: "previous motion", raw_command: "<img src=x onerror=alert(1)>", parameters: {distance: 1}},
      observation: {execution: {state: "SUCCEEDED", message: "finished"}, events: {events: [{sequence: 1, state: "SUCCEEDED", message: "done"}]}}
    };
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET"});
      if (path === "/api/history") return {records: [record], total: 1};
      if (path === "/api/history/old-preview") return {record};
      if (path.endsWith("/cancel")) return {accepted: true};
      throw new Error("Unexpected API: " + path);
    };
    updateButtons();
  `);
  await run("refreshHistory()");
  const item = nodes.get("history-list").children[0];
  assert.match(resultText(item), /previous motion.*成功/);
  await item.onclick();
  const detail = resultText(nodes.get("history-detail"));
  assert.match(detail, /old-request/);
  assert.match(detail, /old-inst/);
  assert.match(detail, /<img src=x onerror=alert\(1\)>/);
  assert.match(detail, /不是实时查询/);
  assert.equal(run("state.execution === originalExecution"), true);
  assert.equal(run("state.preview === originalPreview"), true);
  assert.equal(run("state.terminal"), false);
  assert.equal(execute.disabled, true);
  assert.equal(nodes.get("cancel").disabled, false);
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), [
    {path: "/api/history", method: "GET"}, {path: "/api/history/old-preview", method: "GET"},
  ]);
  run("pollExecution = async () => {}");
  await nodes.get("cancel").onclick();
  const cancel = JSON.parse(run("JSON.stringify(calls.at(-1))"));
  assert.equal(cancel.method, "POST");
  assert.match(decodeURIComponent(cancel.path), /live-request/);
  assert.doesNotMatch(decodeURIComponent(cancel.path), /old-request/);
});

test("legacy and unresolved history entries never invent command names or successful outcomes", async () => {
  const {run, nodes} = harness({withHistory: true});
  run(`
    globalThis.record = {preview_id: "legacy", state: "unknown", reference: {request_id: "legacy-request"}};
    api = async path => path === "/api/history" ? {records: [record], total: 102} : {record};
  `);
  await run("refreshHistory()");
  assert.match(nodes.get("history-status").textContent, /最近 1 条 \/ 共 102 条/);
  assert.match(resultText(nodes.get("history-list")), /请求 legacy-r.*未确认终态/);
  await run('showHistory("legacy")');
  assert.match(resultText(nodes.get("history-detail")), /旧记录未保存/);
  assert.match(resultText(nodes.get("history-detail")), /不代表动作已停止/);
  assert.equal(run("state.execution"), null);
  assert.equal(run("state.terminal"), false);
});

test("rapid history selections and closing details discard stale responses", async () => {
  const {run, nodes} = harness({withHistory: true});
  run(`
    globalThis.resolvers = {};
    api = path => new Promise(resolve => {resolvers[path] = resolve;});
    globalThis.first = showHistory("first");
    globalThis.second = showHistory("second");
    resolvers["/api/history/second"]({record: {preview_id: "second", state: "FAILED", reference: {request_id: "second-request"}}});
  `);
  await run("second");
  run('resolvers["/api/history/first"]({record: {preview_id: "first", state: "SUCCEEDED", reference: {request_id: "first-request"}}})');
  await run("first");
  assert.match(resultText(nodes.get("history-detail")), /second-request/);
  assert.doesNotMatch(resultText(nodes.get("history-detail")), /first-request/);
  run('globalThis.third = showHistory("third")');
  nodes.get("history-detail").children[1].onclick();
  run('resolvers["/api/history/third"]({record: {preview_id: "third", reference: {request_id: "third-request"}}})');
  await run("third");
  assert.equal(nodes.get("history-detail").hidden, true);
  assert.equal(nodes.get("history-detail").children.length, 0);
});

test("a late history listing cannot populate a different connection", async () => {
  const {run, nodes} = harness({withHistory: true});
  run(`
    api = () => new Promise(resolve => {globalThis.finishHistory = resolve;});
    globalThis.pendingHistory = refreshHistory();
    state.connection = {connection_id: "other-robot"};
    finishHistory({records: [{preview_id: "stale", state: "SUCCEEDED"}], total: 1});
  `);
  await run("pendingHistory");
  assert.equal(nodes.get("history-list").children.length, 0);
  assert.equal(run("state.history.length"), 0);
});

test("history failures do not change execution guards or overwrite live-task warnings", async () => {
  const {run, nodes} = harness({withHistory: true});
  run(`
    state.execution = {request_id: "unknown-task"};
    notice("live-task warning", true);
    api = async () => {throw new Error("archive unavailable");};
  `);
  await run("refreshHistory()");
  assert.match(nodes.get("history-status").textContent, /archive unavailable/);
  assert.equal(nodes.get("notification").textContent, "live-task warning");
  assert.equal(run("active()"), true);
});

test("an old backend history 404 explains restarting only Web without touching the active task", async () => {
  const {run, nodes} = harness({withHistory: true});
  run(`
    state.execution = {request_id: "current-task"};
    notice("live-task warning", true);
    api = httpApi;
    fetch = async (path, options) => {
      calls.push({path, method: options.method || "GET"});
      return {ok: false, status: 404, json: async () => {throw new SyntaxError("HTML error page");}};
    };
  `);
  await run("refreshHistory()");
  const message = nodes.get("history-status").textContent;
  assert.match(message, /HTTP 404/);
  assert.match(message, /请重启 Web/);
  assert.match(message, /无需重启 Gateway 或机器人/);
  assert.match(message, /确认没有活动或结果未知的任务/);
  assert.equal(nodes.get("notification").textContent, "live-task warning");
  assert.equal(run("state.execution.request_id"), "current-task");
  assert.equal(run("active()"), true);
  assert.equal(nodes.get("refresh-history").disabled, false);
  assert.deepEqual(JSON.parse(run("JSON.stringify(calls)")), [{path: "/api/history", method: "GET"}]);
});

test("observing a terminal task refreshes history without resubmission", async () => {
  const {run, nodes, settle} = harness({withHistory: true});
  run(`
    state.execution = {request_id: "completed-request"};
    api = async (path, options) => {
      calls.push({path, method: options?.method || "GET"});
      if (path === "/api/history") return {records: [{preview_id: "completed", state: "SUCCEEDED", reference: {request_id: "completed-request"}}], total: 1};
      return {execution: {found: true, state: "SUCCEEDED"}, events: {events: []}};
    };
  `);
  await run("pollExecution()");
  await settle();
  assert.equal(run("state.terminal"), true);
  assert.equal(run("calls.length"), 2);
  assert.equal(run("calls.every(call => call.method === 'GET')"), true);
  assert.match(resultText(nodes.get("history-list")), /成功/);
  assert.equal(nodes.get("cancel").disabled, true);
});

const VENDOR_SKILLS = [
  {name: "release_arm", display_name: "释放手臂控制", summary: "action 99 control ownership",
    tags: ["preset", "control_ownership", "unitree_action_id_99"], ready: true},
  {name: "arms_horizontal", display_name: "双臂平举", summary: "both hands up",
    tags: ["preset", "unitree_action_id_15"], ready: true},
  {name: "wave_hand", display_name: "挥手", summary: "播放 99 次提示", ready: true},
];

test("compact rows show the vendor action id and the control-ownership badge", () => {
  const { run, nodes } = harness({renderSkillList: true});
  run(`state.skills = ${JSON.stringify(VENDOR_SKILLS)}; renderSkills();`);
  const rows = nodes.get("skill-list").children;
  const release = rows[0].children[0];
  assert.deepEqual(release.children.map(node => node.textContent),
    ["释放手臂控制", "预置", "控制权操作", "ID99", "就绪"]);
  assert.equal(release.children[1].className, "skill-tag");
  assert.equal(release.children[2].className, "skill-tag");
  assert.equal(release.children[3].className, "skill-tag vendor-action-id");
  assert.match(release.children[3].title, /宇树动作ID：99/);
  assert.match(release.children[3].title, /不是列表序号、请求 ID 或工作流步骤/);
  const arms = rows[1].children[0];
  assert.deepEqual(arms.children.map(node => node.textContent), ["双臂平举", "预置", "ID15", "就绪"]);
  assert.deepEqual(rows[2].children[0].children.map(node => node.textContent), ["挥手", "就绪"]);
});

test("skill detail names the vendor action id and calls it display-only", async () => {
  const { run, nodes } = harness();
  run(`api = async () => (${JSON.stringify(VENDOR_SKILLS[1])});`);
  await run('selectSkill({name: "arms_horizontal"})');
  const detail = resultText(nodes.get("skill-detail"));
  assert.match(detail, /宇树动作ID：15/);
  assert.match(detail, /厂商固件动作编号/);
  assert.match(detail, /不是列表序号、请求 ID 或工作流步骤/);
  assert.equal(nodes.get("skill-detail").children.length >= 2, true);
});

test("digit searches match the vendor action id exactly instead of unrelated numbers", () => {
  const { run, nodes } = harness({renderSkillList: true});
  run(`state.skills = ${JSON.stringify(VENDOR_SKILLS)}; renderSkills();`);
  const search = nodes.get("skill-search");
  const names = () => nodes.get("skill-list").children.map(row => row.children[0].children[0].textContent);
  search.value = "99";
  run("renderSkills()");
  assert.deepEqual(names(), ["释放手臂控制"]);
  search.value = "15";
  run("renderSkills()");
  assert.deepEqual(names(), ["双臂平举"]);
  search.value = "9";
  run("renderSkills()");
  assert.match(resultText(nodes.get("skill-list")), /没有匹配技能/);
  search.value = "015";
  run("renderSkills()");
  assert.match(resultText(nodes.get("skill-list")), /没有匹配技能/);
  search.value = "平举";
  run("renderSkills()");
  assert.deepEqual(names(), ["双臂平举"]);
  search.value = "ownership";
  run("renderSkills()");
  assert.deepEqual(names(), ["释放手臂控制"]);
});

test("skills without a vendor action id show no id badge or id line", async () => {
  const { run, nodes } = harness({renderSkillList: true});
  run(`
    state.skills = [{name: "wave_hand", display_name: "挥手", summary: "挥手致意", tags: ["preset"], ready: false}];
    renderSkills();
    api = async () => ({name: "wave_hand", display_name: "挥手", summary: "挥手致意", tags: ["preset"]});
  `);
  await run('selectSkill({name: "wave_hand"})');
  assert.doesNotMatch(resultText(nodes.get("skill-list")), /ID\d/);
  assert.doesNotMatch(resultText(nodes.get("skill-detail")), /动作ID/);
  assert.doesNotMatch(resultText(nodes.get("skill-detail")), /厂商固件动作编号/);
});
