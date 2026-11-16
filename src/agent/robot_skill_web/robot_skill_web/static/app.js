"use strict";

const state = {
  connection: null, skills: [], skill: null, preview: null, execution: null,
  recovery: [], terminal: false, submitting: false, motionAuthorized: false,
  lastEventSequence: 0, pollTimer: null, statusTimer: null, recoveryTimer: null,
  polling: false, statusPolling: false, pollDelay: 1000, statusDelay: 2000,
  defaultRawCommand: "", selectionVersion: 0, previewVersion: 0,
  selecting: false, preparing: false, previewPending: false, previewTimer: null,
  sidebarCollapsed: false, canceling: false,
  history: [], historyLoading: false, historyVersion: 0, historyDetailVersion: 0, historySelected: null,
};
const elements = {};
const ids = ["app", "notification", "session-panel", "workbench-template",
  "connection", "connection-title", "connection-summary",
  "target-chip", "skill-search", "skill-list",
  "skill-detail", "parameters-form", "raw-command", "plan-status", "retry-preview", "execute", "execution-state",
  "cancel", "execution-result", "motion-chip", "refresh-history", "history-status", "history-list", "history-detail",
  "work-layout", "execution-sidebar", "toggle-sidebar", "execution-summary", "cancel-collapsed"];
function bindElements() { for (const id of ids) elements[id] = document.getElementById(id); }
bindElements();
function text(value) { return value == null ? "" : String(value); }
function short(value) { return text(value).slice(0, 8); }
function skillDisplayName(skill) { return text(skill.display_name).trim() || text(skill.name); }
// A vendor firmware action number arrives as an optional catalog display tag
// (for example "unitree_action_id_99"). This page only formats it: the number
// never selects a firmware action, and it is not a list index, request id or
// workflow step.
const VENDOR_ACTION_TAG = /^([a-z][a-z0-9]*)_action_id_([0-9]{1,6})$/;
const TAG_LABELS = {preset: "预置", control_ownership: "控制权操作"};
const VENDOR_LABELS = {unitree: "宇树"};
function vendorAction(skill) {
  for (const tag of Array.isArray(skill.tags) ? skill.tags : []) {
    const match = VENDOR_ACTION_TAG.exec(text(tag));
    if (match) return {vendor: VENDOR_LABELS[match[1]] || match[1], id: match[2]};
  }
  return null;
}
function tagLabel(tag) {
  if (TAG_LABELS[tag]) return TAG_LABELS[tag];
  const match = VENDOR_ACTION_TAG.exec(tag);
  return match ? `ID${match[2]}` : tag;
}
function vendorActionText(action) { return `${action.vendor}动作ID：${action.id}`; }
function skillMatchesQuery(skill, query) {
  if (!query) return true;
  // A digits-only search matches the vendor action id exactly. Unrelated
  // numbers in a summary must never match, so the free-text haystack is not
  // consulted for that form.
  if (/^[0-9]+$/.test(query)) {
    const action = vendorAction(skill);
    return action !== null && action.id === query;
  }
  return `${skillDisplayName(skill)} ${skill.name} ${skill.summary || ""}`.toLowerCase().includes(query);
}
function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content != null) node.textContent = text(content);
  return node;
}
function notice(message = "", error = false) {
  elements.notification.hidden = !message;
  elements.notification.className = error ? "notice text-error" : "notice";
  elements.notification.textContent = message;
}
function active() { return state.submitting || Boolean(state.execution && !state.terminal); }
function updateConnectionPanel() {
  const bound = Boolean(state.connection);
  elements.connection.className = bound ? "panel connection is-connected" : "panel connection";
  elements["connection-title"].textContent = bound ? "已绑定部署配置" : "等待会话";
  elements["connection-summary"].hidden = !bound;
  elements["connection-summary"].textContent = bound ?
    `${state.connection.robot_name || state.connection.robot_id} · 由启动配置绑定` : "";
}
function updateSidebar() {
  const collapsed = state.sidebarCollapsed;
  elements["work-layout"].className = collapsed ? "work-layout is-sidebar-collapsed" : "work-layout";
  elements["execution-sidebar"].hidden = collapsed;
  elements["toggle-sidebar"].textContent = collapsed ? "展开侧栏" : "收起侧栏";
  elements["toggle-sidebar"].setAttribute("aria-expanded", String(!collapsed));
  elements["execution-summary"].hidden = !collapsed;
  elements["execution-summary"].textContent = `执行：${state.submitting ? "提交中，结果待确认" : text(elements["execution-state"].textContent) || "未执行"}`;
  elements["cancel-collapsed"].hidden = !collapsed || !state.execution || state.terminal;
  elements["cancel-collapsed"].disabled = elements.cancel.disabled;
}
function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  updateSidebar();
}
function updateButtons() {
  if (!elements.execute) return;
  updateConnectionPanel();
  elements.execute.disabled = !state.preview || !state.connection || state.connection.read_only ||
    !state.motionAuthorized || state.preparing || state.selecting || active();
  elements.execute.textContent = state.skill ? `执行 ${skillDisplayName(state.skill)}` : "执行技能";
  elements["retry-preview"].disabled = !state.connection || !state.skill || state.unsupportedType ||
    state.preparing || state.selecting || active();
  elements["raw-command"].disabled = active() || state.selecting;
  for (const input of elements["parameters-form"].querySelectorAll("input, select, textarea")) {
    input.disabled = active() || state.selecting;
  }
  for (const button of elements["skill-list"].querySelectorAll("button")) {
    button.disabled = active();
  }
  elements.cancel.disabled = state.canceling || !state.execution || state.terminal || !state.connection || state.connection.read_only;
  updateSidebar();
  elements["refresh-history"].disabled = !state.connection || state.historyLoading;
}
async function api(path, options = {}) {
  const headers = {};
  const csrf = document.cookie.match(/(?:^|; )_robot_skill_web_csrf=([^;]+)/);
  if (options.method && options.method !== "GET") headers["X-CSRF-Token"] = csrf ? decodeURIComponent(csrf[1]) : "";
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error ? payload.error.message : `HTTP ${response.status}`);
    error.status = response.status;
    error.code = payload.error && payload.error.code;
    error.execution = payload.execution;
    throw error;
  }
  return payload;
}
function showWorkbench() {
  elements.app.replaceChildren(elements["workbench-template"].content.cloneNode(true));
  bindElements();
  const missing = ids.filter(id => !elements[id]);
  if (missing.length) {
    throw new Error(`页面资源不完整，请按 Ctrl+Shift+R 强制刷新后重试（缺少：${missing.join("、")}）。未提交机器人动作。`);
  }
  elements["skill-search"].oninput = renderSkills;
  elements["retry-preview"].onclick = () => schedulePreview(0);
  elements.execute.onclick = executePreview;
  elements.cancel.onclick = cancelExecution;
  elements["cancel-collapsed"].onclick = cancelExecution;
  elements["toggle-sidebar"].onclick = toggleSidebar;
  elements["refresh-history"].onclick = refreshHistory;
  elements["raw-command"].oninput = () => schedulePreview();
  elements["parameters-form"].oninput = () => schedulePreview();
  elements["parameters-form"].onchange = () => schedulePreview();
  elements["parameters-form"].onsubmit = event => event.preventDefault();
  updateButtons();
}
async function restoreSession(session) {
  state.recovery = session.unresolved_executions || [];
  if (session.connection) await adoptConnection(session.connection);
  else if (state.recovery.length) notice("存在未确认结果的任务。请先查询原任务；不要重复执行。", true);
}
async function establishSession() {
  // GET issues the CSRF and session cookies; this POST only binds the browser
  // session to the deployment the server was started with. No action is sent.
  const session = await api("/api/session", { method: "POST", body: "{}" });
  await restoreSession(session);
}
async function bootstrap() {
  try {
    const result = await api("/api/session");
    showWorkbench();
    if (result.session_active) await restoreSession(result);
    else await establishSession();
  } catch (error) { notice(error.message, true); }
}
async function adoptConnection(connection) {
  state.selectionVersion += 1;
  state.selecting = false;
  state.connection = connection;
  state.historyVersion += 1;
  state.historyDetailVersion += 1;
  state.historyLoading = false;
  state.history = [];
  state.historySelected = null;
  elements["history-list"].replaceChildren();
  elements["history-detail"].replaceChildren();
  elements["history-detail"].hidden = true;
  elements["history-status"].textContent = "正在读取本机历史…";
  elements["execution-state"].textContent = "未执行";
  elements["execution-result"].replaceChildren();
  elements["session-panel"].replaceChildren(element("span", "chip", "已绑定部署配置"));
  state.preview = state.skill = state.execution = null;
  state.motionAuthorized = false;
  invalidatePreview();
  state.skills = [];
  state.terminal = false;
  state.lastEventSequence = 0;
  state.defaultRawCommand = "";
  elements["raw-command"].value = "";
  elements["skill-detail"].replaceChildren();
  elements["parameters-form"].replaceChildren();
  document.querySelector(".plan")?.remove();
  elements["target-chip"].textContent = `${connection.robot_name || connection.robot_id} · 本机操作入口`;
  const saved = state.recovery.find(record => record.reference &&
    record.reference.target_key === connection.target_key);
  if (saved) {
    state.execution = saved.reference;
    notice("已恢复执行引用。仅查询原任务，不会重新提交。");
    startPolling();
  }
  await refreshStatus();
  try {
    const listing = await api("/api/skills");
    state.skills = listing.skills || [];
    renderSkills();
  } catch (error) { notice(`目录暂不可用；状态和任务查询仍可使用。${error.message}`, true); }
  updateButtons();
  void refreshHistory();
}

function historyState(record) {
  if (record.state === "UNKNOWN" && record.reconciled) return "已核对 · 结果未知";
  return {SUCCEEDED: "成功", FAILED: "失败", CANCELED: "已取消", REJECTED: "已拒绝"}[record.state] || "未确认终态";
}
function historyName(record) {
  const command = record.command || {};
  return command.display_name || command.skill_name || `请求 ${short(record.reference?.request_id)}`;
}
function historyTime(value) {
  return typeof value === "number" && Number.isFinite(value) ? new Date(value * 1000).toLocaleString("zh-CN", {hour12: false}) : "时间未记录";
}
function renderHistory() {
  const list = elements["history-list"];
  const scroll = list.scrollTop || 0;
  list.replaceChildren();
  for (const record of state.history) {
    const button = element("button", "history-item");
    button.type = "button";
    button.setAttribute("aria-pressed", String(state.historySelected === record.preview_id));
    const heading = element("div", "history-heading");
    heading.append(element("span", "history-name", historyName(record)),
      element("span", `status ${record.state === "SUCCEEDED" ? "ready" : "not-ready"}`, historyState(record)));
    button.append(heading, element("div", "meta", historyTime(record.submitted_at)));
    button.title = `${historyName(record)}\n请求 ${text(record.reference?.request_id)}`;
    button.onclick = () => showHistory(record.preview_id);
    list.append(button);
  }
  list.scrollTop = scroll;
}
async function refreshHistory() {
  if (!state.connection) return;
  const connection = state.connection;
  const version = ++state.historyVersion;
  const current = () => version === state.historyVersion && connection === state.connection;
  state.historyLoading = true;
  elements["history-status"].textContent = "正在读取本机历史…";
  updateButtons();
  try {
    const result = await api("/api/history");
    if (!current()) return;
    state.history = result.records || [];
    renderHistory();
    elements["history-status"].textContent = state.history.length ?
      `本机 Web · ${result.total > state.history.length ? `最近 ${state.history.length} 条 / 共 ${result.total} 条` : `共 ${state.history.length} 条`} · 点击查看记录，不会执行` :
      "当前机器人暂无本机 Web 提交记录。";
  } catch (error) {
    if (current()) elements["history-status"].textContent = error.status === 404 ?
      "历史接口不可用（HTTP 404）。页面与 Web 后端可能不是同一版本；确认没有活动或结果未知的任务后，请重启 Web，再刷新页面。无需重启 Gateway 或机器人。" :
      `历史读取失败：${error.message}`;
  } finally {
    if (current()) { state.historyLoading = false; updateButtons(); }
  }
}
async function showHistory(previewId) {
  if (!state.connection) return;
  const connection = state.connection;
  const version = ++state.historyDetailVersion;
  const current = () => version === state.historyDetailVersion && connection === state.connection;
  const detail = elements["history-detail"];
  state.historySelected = previewId;
  renderHistory();
  detail.hidden = false;
  detail.replaceChildren(element("p", "meta", "正在读取记录…"));
  const close = element("button", "ghost", "关闭详情");
  close.type = "button";
  close.onclick = () => {
    state.historyDetailVersion += 1;
    state.historySelected = null;
    detail.hidden = true;
    detail.replaceChildren();
    renderHistory();
  };
  detail.append(close);
  try {
    const {record} = await api(`/api/history/${encodeURIComponent(previewId)}`);
    if (!current()) return;
    const command = record.command;
    const observation = record.observation || {};
    const execution = observation.execution || {};
    detail.replaceChildren(element("h3", "", historyName(record)), close,
      element("p", "meta", `提交：${historyTime(record.submitted_at)} · ${historyState(record)}`),
      element("p", "meta", `机器人：${text(record.reference?.robot_id)} · 部署 ${short(text(record.reference?.target_key).split(":").pop())}`),
      element("p", "meta", `请求：${text(record.reference?.request_id)}`));
    if (command) {
      detail.append(element("p", "meta", `技能：${text(command.skill_name)}`),
        element("p", "history-command", `执行说明：${text(command.raw_command)}`),
        element("div", "event", command.parameters_omitted ? "参数过大，未存入历史记录。" : `参数：${JSON.stringify(command.parameters || {}, null, 2)}`));
    } else detail.append(element("p", "meta", "此旧记录未保存技能名称、说明和参数。"));
    detail.append(element("p", "meta", record.observed_at ? `最后观测：${historyTime(record.observed_at)}。这是保存的记录，不是实时查询。` :
      "未保存详细观测结果；未确认终态不代表动作已停止。"));
    if (execution.message || execution.error_code) detail.append(element("div", "event", `${text(execution.error_code)} ${text(execution.message)}`));
    const events = observation.events || {};
    for (const event of events.events || []) {
      detail.append(element("div", "event", `#${event.sequence} ${text(event.state || event.event_type || event.type)} ${text(event.message)}`));
    }
    if (events.unavailable || events.event_gap || events.truncated) {
      detail.append(element("p", "text-error", "保存的事件记录不完整；执行结果以服务端账本为准。"));
    }
    detail.scrollIntoView?.({block: "nearest"});
  } catch (error) {
    if (current()) detail.replaceChildren(element("p", "text-error", `记录读取失败：${error.message}`), close);
  }
}
async function refreshStatus() {
  clearTimeout(state.statusTimer);
  if (!state.connection || state.statusPolling) return;
  const connection = state.connection;
  state.statusPolling = true;
  try {
    const status = await api("/api/status");
    if (connection !== state.connection) return;
    state.motionAuthorized = Boolean(status.motion_authorized);
    elements["motion-chip"].textContent = `运动授权：${status.motion_authorized ? "开" : "关"} · 控制面：${status.control_plane_ready ? "就绪" : "未就绪"} · ${status.busy ? "忙碌" : "空闲"}`;
    state.statusDelay = 2000;
  } catch (error) {
    if (connection !== state.connection) return;
    state.motionAuthorized = false;
    elements["motion-chip"].textContent = `状态不可达：${error.message}`;
    state.statusDelay = Math.min(15000, state.statusDelay * 2);
  } finally {
    state.statusPolling = false;
    updateButtons();
    if (state.connection) state.statusTimer = setTimeout(refreshStatus, state.statusDelay);
  }
}
function appendSkillPresentation(container, skill, titleTag = "div", compact = false) {
  const title = skillDisplayName(skill);
  const action = vendorAction(skill);
  const heading = element("div", "skill-heading");
  heading.append(element(titleTag, "skill-name", title));
  for (const tag of Array.isArray(skill.tags) ? skill.tags : []) {
    const match = VENDOR_ACTION_TAG.exec(text(tag));
    const badge = element("span", match ? "skill-tag vendor-action-id" : "skill-tag", tagLabel(tag));
    if (match) {
      const label = `${VENDOR_LABELS[match[1]] || match[1]}动作ID：${match[2]}`;
      badge.title = `${label}（厂商固件动作编号，不是列表序号、请求 ID 或工作流步骤）`;
    }
    heading.append(badge);
  }
  container.append(heading);
  if (!compact && title !== skill.name) container.append(element("div", "meta skill-id", skill.name));
  if (!compact && action) {
    container.append(element("div", "meta vendor-action-id",
      `${vendorActionText(action)}（厂商固件动作编号，不是列表序号、请求 ID 或工作流步骤）`));
  }
  const summary = text(skill.summary).trim();
  if (!compact && summary && summary !== title) container.append(element("p", "meta skill-summary", summary));
  return heading;
}
function renderSkills() {
  const query = elements["skill-search"].value.trim().toLowerCase();
  const previousScroll = query === (elements["skill-list"].dataset.query || "") ? elements["skill-list"].scrollTop || 0 : 0;
  elements["skill-list"].replaceChildren();
  for (const skill of state.skills) {
    if (!skillMatchesQuery(skill, query)) continue;
    const item = element("button", "skill-item");
    item.type = "button";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(state.skill && state.skill.name === skill.name));
    const action = vendorAction(skill);
    item.title = [...new Set([skillDisplayName(skill), skill.name, skill.summary,
      action ? vendorActionText(action) : null].filter(Boolean))].join("\n");
    const heading = appendSkillPresentation(item, skill, "span", true);
    heading.append(element("span", `status ${skill.ready ? "ready" : "not-ready"}`, skill.ready ? "就绪" : "未就绪"));
    item.onclick = () => selectSkill(skill);
    elements["skill-list"].append(item);
  }
  if (!elements["skill-list"].children.length) elements["skill-list"].append(element("p", "meta", "没有匹配技能"));
  elements["skill-list"].dataset.query = query;
  elements["skill-list"].scrollTop = previousScroll;
}
async function selectSkill(skill) {
  if (active() || !state.connection) return;
  const selectionVersion = ++state.selectionVersion;
  const connection = state.connection;
  state.selecting = true;
  state.skill = null;
  invalidatePreview();
  elements["skill-detail"].replaceChildren();
  elements["parameters-form"].replaceChildren();
  elements["plan-status"].textContent = "正在读取技能…";
  try {
    const description = await api(`/api/skills/${encodeURIComponent(skill.name)}`);
    if (selectionVersion !== state.selectionVersion || connection !== state.connection) return;
    state.skill = description;
    syncDefaultRawCommand();
    const detail = elements["skill-detail"];
    detail.replaceChildren();
    appendSkillPresentation(detail, state.skill, "h3");
    const readiness = state.skill.ready ? "就绪" : "未就绪";
    detail.append(element("p", "meta", state.skill.reason ? `${readiness}：${state.skill.reason}` : readiness));
    renderForm();
    renderSkills();
    state.selecting = false;
    schedulePreview(0);
  } catch (error) {
    if (selectionVersion === state.selectionVersion && connection === state.connection) {
      elements["plan-status"].textContent = "技能读取失败，请重新选择。";
      notice(error.message, true);
    }
  } finally {
    if (selectionVersion === state.selectionVersion) state.selecting = false;
    updateButtons();
  }
}
function renderForm() {
  const form = elements["parameters-form"];
  form.replaceChildren();
  state.unsupportedType = false;
  const schema = state.skill.parameters || {};
  const required = new Set(schema.required || []);
  for (const [name, property] of Object.entries(schema.properties || {})) {
    const field = element("label", "field");
    field.append(element("span", "", `${name}${required.has(name) ? " *" : ""}`));
    let input;
    if (property.type === "string") {
      input = document.createElement(Array.isArray(property.enum) ? "select" : "input");
      if (Array.isArray(property.enum)) {
        input.append(new Option("请选择", ""));
        for (const value of property.enum) input.append(new Option(text(value), text(value)));
      }
    } else if (property.type === "number") {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
    } else {
      field.append(element("small", "text-error", "当前界面不支持此参数类型；禁止提交此技能"));
      form.append(field);
      state.unsupportedType = true;
      continue;
    }
    input.name = name;
    input.required = required.has(name);
    if (property.default != null) input.value = text(property.default);
    field.append(input, element("small", "", property.description || ""));
    form.append(field);
  }
}
function parametersFromForm() {
  const values = {};
  for (const [key, value] of new FormData(elements["parameters-form"]).entries()) {
    if (value === "") continue;
    const property = (state.skill.parameters.properties || {})[key] || {};
    values[key] = property.type === "number" ? Number(value) : value;
  }
  return values;
}
function invalidatePreview() {
  clearTimeout(state.previewTimer);
  state.previewTimer = null;
  state.previewPending = false;
  state.previewVersion += 1;
  state.preview = null;
  document.querySelector(".plan")?.remove();
  elements["retry-preview"].hidden = true;
  elements["plan-status"].textContent = "选择技能后自动准备计划，不会执行动作。";
  updateButtons();
}
function schedulePreview(delay = 300) {
  if (active()) return;
  invalidatePreview();
  if (!state.connection || !state.skill || state.selecting) return;
  if (state.unsupportedType) {
    elements["plan-status"].textContent = "此技能包含不支持的参数类型，无法准备计划。";
    return;
  }
  state.previewPending = true;
  elements["plan-status"].textContent = "正在准备计划，不会执行动作…";
  state.previewTimer = setTimeout(() => {
    state.previewTimer = null;
    void createPreview();
  }, delay);
}
function syncDefaultRawCommand() {
  const input = elements["raw-command"];
  const label = state.skill ? skillDisplayName(state.skill) : "";
  const identity = state.skill && label !== state.skill.name ? `（${state.skill.name}）` : "";
  const next = state.skill ? `通过 Web 请求执行技能：${label}${identity}` : "";
  if (!input.value.trim() || input.value === state.defaultRawCommand) input.value = next;
  state.defaultRawCommand = next;
}
async function createPreview() {
  if (!state.connection || !state.skill || active() || state.unsupportedType || state.selecting) return;
  // Only one preparation may be in flight: the backend retains just its latest plan.
  if (state.preparing) { state.previewPending = true; return; }
  clearTimeout(state.previewTimer);
  state.previewTimer = null;
  state.previewPending = false;
  syncDefaultRawCommand();
  if (!elements["parameters-form"].checkValidity() || !elements["raw-command"].checkValidity()) {
    elements["plan-status"].textContent = "请补全有效参数和执行说明，填写后会自动准备计划。";
    return;
  }
  invalidatePreview();
  const version = state.previewVersion;
  const connection = state.connection;
  const skill = state.skill;
  const current = () => version === state.previewVersion && connection === state.connection &&
    skill === state.skill && !state.selecting && !active();
  state.preparing = true;
  elements["plan-status"].textContent = "正在准备计划，不会执行动作…";
  updateButtons();
  try {
    const preview = await api("/api/previews", { method: "POST", body: JSON.stringify({
      connection_version: connection.connection_version, skill_name: skill.name,
      raw_command: elements["raw-command"].value.trim(), parameters: parametersFromForm(),
    }) });
    if (!current()) return;
    const plan = preview.plan || {};
    const box = element("div", "plan");
    box.append(element("h3", "", "执行计划"), element("p", "meta", `目标：${text(preview.target && preview.target.robot_id)} · 计划：${text(plan.plan_id)} · 预算：${text(plan.proposed_task_budget_sec)} 秒`));
    for (const step of plan.steps || []) box.append(element("div", "plan-step", JSON.stringify(step, null, 2)));
    box.append(element("div", "meta", `目录身份：${JSON.stringify(plan.registry_identity || {})}`));
    elements["parameters-form"].after(box);
    // Permit submission only after this exact plan is rendered for the operator.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    if (!current()) { box.remove(); return; }
    state.preview = preview;
    elements["plan-status"].textContent = `计划已准备。核对目标、步骤和参数后，点击「执行 ${skillDisplayName(skill)}」。`;
    notice();
  } catch (error) {
    if (current()) {
      elements["plan-status"].textContent = `计划准备失败：${error.message}`;
      elements["retry-preview"].hidden = false;
      notice(error.message, true);
    }
  } finally {
    state.preparing = false;
    updateButtons();
    if (state.previewPending && state.previewTimer === null) void createPreview();
  }
}
async function executePreview() {
  if (!state.preview || !state.connection || state.preparing || state.selecting ||
    active() || state.connection.read_only || !state.motionAuthorized) return;
  state.submitting = true;
  updateButtons();
  try {
    const result = await api("/api/executions", { method: "POST", body: JSON.stringify({
      preview_id: state.preview.preview_id, connection_version: state.connection.connection_version,
    }) });
    state.execution = result.execution;
    state.terminal = false;
    state.lastEventSequence = 0;
    state.submitting = false;
    startPolling();
  } catch (error) {
    if (error.execution) {
      state.execution = error.execution;
      state.submitting = false;
      startPolling();
    } else if (error.code === "SUBMISSION_PENDING" || !error.code) {
      notice("提交结果待确认。正在恢复执行引用，不会自动重发。", true);
      state.recoveryTimer = setTimeout(recoverSubmission, 1000);
      return;
    } else {
      state.submitting = false;
      invalidatePreview();
      elements["plan-status"].textContent = "提交被拒绝。核对错误后可重新准备计划，不会自动重发。";
      elements["retry-preview"].hidden = false;
    }
    notice(error.message, true);
  }
  updateButtons();
}
async function recoverSubmission() {
  try {
    const session = await api("/api/session");
    const record = (session.unresolved_executions || []).find(item => item.preview_id === state.preview?.preview_id);
    if (record) {
      state.execution = record.reference;
      state.submitting = false;
      startPolling();
      return;
    }
  } catch (_) { /* Observation never resends a goal. */ }
  state.recoveryTimer = setTimeout(recoverSubmission, 5000);
}
function executionPath() { return `/api/executions/${encodeURIComponent(JSON.stringify(state.execution))}`; }
function startPolling() {
  clearTimeout(state.pollTimer);
  state.pollDelay = 1000;
  elements["execution-state"].textContent = "正在查询执行状态";
  void pollExecution();
  void refreshHistory();
  updateButtons();
}
async function pollExecution() {
  if (!state.execution || state.polling) return;
  state.polling = true;
  try {
    const result = await api(`${executionPath()}?after_sequence=${state.lastEventSequence}`);
    const execution = result.execution;
    state.terminal = Boolean(execution.submission_rejected || (execution.found &&
      (["SUCCEEDED", "FAILED", "CANCELED"].includes(execution.state) || (execution.state === "UNKNOWN" && execution.reconciled))));
    elements["execution-state"].textContent = execution.submission_rejected ? "提交已明确拒绝" :
      (execution.found ? text(execution.state) : "结果尚未观测到");
    const box = element("div", "execution-observation");
    box.append(element("p", "", `request_id：${state.execution.request_id}`),
      element("p", "", `${text(execution.error_code)} ${text(execution.message)}`));
    for (const event of (result.events && result.events.events) || []) {
      box.append(element("div", "event", `#${event.sequence} ${text(event.state || event.event_type || event.type)} ${text(event.message || "")}`));
      state.lastEventSequence = Math.max(state.lastEventSequence, Number(event.sequence || 0));
    }
    if (result.events && (result.events.unavailable || result.events.error_code)) {
      box.append(element("p", "text-error", `事件查询失败（${text(result.events.error_code || "UNAVAILABLE")}）：${text(result.events.message)}；执行结果以服务端账本为准。`));
    } else if (result.events && result.events.event_gap) {
      box.append(element("p", "text-error", "事件存在缺口；执行结果以服务端账本为准。"));
    }
    elements["execution-result"].replaceChildren(box);
    state.pollDelay = 1000;
    if (state.terminal) {
      invalidatePreview();
      elements["plan-status"].textContent = "任务已结束。重新选择技能可准备新计划，不会自动执行。";
      notice("已观测到任务终态。可准备新任务或刷新页面。");
      void refreshHistory();
    }
  } catch (error) {
    elements["execution-state"].textContent = "观测失败，结果未知";
    notice(error.message, true);
    state.pollDelay = Math.min(15000, state.pollDelay * 2);
  } finally {
    state.polling = false;
    updateButtons();
    if (!state.terminal) state.pollTimer = setTimeout(pollExecution, state.pollDelay);
  }
}
async function cancelExecution() {
  if (state.canceling || !state.execution || state.terminal || !state.connection || state.connection.read_only) return;
  state.canceling = true;
  updateButtons();
  try {
    await api(`${executionPath()}/cancel`, { method: "POST", body: JSON.stringify({ connection_version: state.connection.connection_version }) });
    notice("取消请求已发送；等待服务端确认终态。这不是硬件急停。");
    await pollExecution();
  } catch (error) { notice(error.message, true); }
  finally { state.canceling = false; updateButtons(); }
}
void bootstrap();
