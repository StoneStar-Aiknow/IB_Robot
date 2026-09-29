---
name: mermaid-syntax-validation
description: "检查、修复并验证 Markdown/Sphinx 文档中的 Mermaid 图。用户遇到 docs Mermaid 渲染失败、浏览器显示 'Syntax error in text' / 'Parse error' / 'mermaid version'，或要求在不改变图内容的前提下进行 Mermaid 语法检查、Mermaid 修复、图表渲染验证时使用。触发词包括 Mermaid、mermaid、diagram render failure、图表渲染失败、Mermaid 语法检查、Mermaid 修复、docs HTML validation。"
---

# Mermaid 语法验证 Skill

用于验证 Markdown 文档中的 Mermaid 图、执行仅限语法层面的修复，并证明生成后的 HTML 不再渲染 Mermaid 错误 SVG。

## 何时使用

- 生成后的文档页面显示 `Syntax error in text`、`Parse error`、`Diagram error` 或 `mermaid version`。
- 用户要求检查 `docs/` 下所有 Markdown Mermaid 图。
- 用户要求在保留图内容的前提下修复 Mermaid 语法。
- Sphinx/MyST 文档流程使用 ```` ```mermaid ```` fenced block，并依赖浏览器端 Mermaid 渲染。
- 静态扫描通过，但渲染后的 HTML 仍出现 Mermaid 错误 SVG。
- 仓库使用本地 Mermaid runtime，例如 `mermaid.min.js` + `mermaid-run.js`，或 `mermaid.esm.min.mjs` + `chunks/` 这类本地 ESM runtime。

## 核心规则

只改 Mermaid 语法。必须保留图的含义、节点文本、边含义、顺序和周围正文。

不要重写架构、重命名概念实体、简化图，或为了让解析通过而删除 label。如果可见文本必须转义，应使用 Mermaid 兼容语法并保留显示含义。

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| 上下文检查的 WARNING 处置与人工核对清单、5 项检查的手工定义（文档栈、runtime、fence、HTML 页面、script 标签） | `references/context-checks.md` |
| 语法修复的条件性门禁规则、高风险模式表、flowchart/state diagram 修复规则、静态语法初筛回退 | `references/syntax-scan-and-fix-rules.md` |

Do not expose these references as separate skills.

## 必做上下文检查

执行任何修复前，先运行技能脚本完成 5 项上下文检查（文档栈、本地 runtime 资源、fence 统计、含图 HTML 页面、script 标签/CDN 引用）。命令中的 `<skill_dir>` 指本技能目录（本 SKILL.md 所在目录，即 `scripts/`、`references/` 的父目录），执行时替换为实际路径；`--docs-root` 等目标路径参数以当前工作目录为基准：

```bash
python -X utf8 <skill_dir>/scripts/context_check.py --docs-root docs [--build-dir docs/build/html]
```

门禁规则：

1. 没有脚本输出，不得开始编辑任何 Markdown 文件。
2. 最终报告的「已检查的 Mermaid runtime 文件」「docs 构建结果」「浏览器验证统计」等字段必须取自脚本输出或其后的构建/浏览器验证，不得凭记忆或推测填写。
3. 脚本报告 `WARNING`（如 CDN 引用），或存在脚本未覆盖的人工判断项（conf.py monkey patch / override 语义、Sphinx 配置组合）时，必须阅读 `references/context-checks.md` 并按其规则处理。

## Runtime 预期

对于当前本地 runtime 方案，项目 runner（例如 `mermaid-run.js`）应当：

- 等待 `window.mermaid` 存在
- 调用 `window.mermaid.initialize({ startOnLoad: false, ... })`
- 在页面加载后调用 `window.mermaid.run()`
- 绑定项目自定义交互，例如点击放大 modal

当前 Mermaid 版本应使用 `mermaid.run()`。把 `mermaid.init()` 示例视为旧写法，因为 Mermaid v10+ 已废弃该 API。

如果存在可运行的 Mermaid JS 环境，语法级检查优先使用 `mermaid.parse(text, { suppressErrors: true })`。静态 grep 适合初筛，但 `parse()` 和浏览器渲染是更强的证据。

## 语法风险扫描与修复

编辑前完成「诊断 → 预览 → 修复 → 复验」循环（幂等、仅改语法；`parse()` 与浏览器渲染使用同一解析器，复验全绿即等价于渲染通过）：

```bash
node <skill_dir>/scripts/parse_check.mjs <target_dir>                   # 1. 诊断：定位失败块
python -X utf8 <skill_dir>/scripts/fix_labels.py <target_dir> --dry-run  # 2. 预览将修改的内容
python -X utf8 <skill_dir>/scripts/fix_labels.py <target_dir>            # 3. 应用自动修复（graph/flowchart 标签语法：引号翻倍、内部引号转义、特殊字符未加引号、匿名节点分配 ID、边标签加引号）
node <skill_dir>/scripts/parse_check.mjs <target_dir>                   # 4. 复验：必须全绿
```

依赖 `mermaid`、`jsdom` 从当前工作目录解析，缺失时必须站在目标目录之外（如仓库根）执行 `npm install mermaid jsdom`——npm 会在执行目录创建 `node_modules`，站在目标内执行会给目标目录新增文件。

门禁规则：

1. 循环顺序固定为 1→4；第 4 步不全绿不得进入构建验证。
2. 第 3 步应用后 `parse_check.mjs` 仍有失败块时，禁止凭经验直接手工编辑——必须阅读 `references/syntax-scan-and-fix-rules.md`，按其高风险模式表定位失败模式并按其修复规则处理。
3. 无 JS runtime（无法运行 `parse_check.mjs`）时，按该 reference 的「静态语法初筛」回退诊断，并在报告中说明未做 parse 验证的原因。
4. 存在手工修复时，最终报告的「修复的具体语法类别」必须注明依据（`references/syntax-scan-and-fix-rules.md` 中高风险模式表的行或其「修复规则」章节的条目）。

该环节不能替代浏览器验证，复验全绿后再走构建与浏览器验证。

## 构建验证

修复后运行真实文档构建。项目有文档化命令时优先使用项目命令。常见 Sphinx 命令包括：

```bash
sphinx-build -M html source build
```

或从文档项目根目录运行：

```bash
python3 -m sphinx -b html docs/source docs/build/html
```

用技能脚本扫描生成 HTML（bash 与 Windows PowerShell 均可直接运行；匹配文件逐行输出到 stdout，扫描统计输出到 stderr）：

```bash
python -X utf8 <skill_dir>/scripts/html_scan.py --build-dir docs/build/html --scan errors
python -X utf8 <skill_dir>/scripts/html_scan.py --build-dir docs/build/html --scan cdn
```

- `--scan errors`：扫描 Mermaid 错误文本（`Syntax error in text` / `Parse error` / `mermaid version` / `Diagram error`）。期望退出码 0 且 stdout 无匹配文件；出现任何匹配都是失败。
- `--scan cdn`：扫描是否意外引入在线 runtime 依赖（`cdn.jsdelivr.net` / `unpkg.com`）。对要求离线可用的文档，期望退出码 0 且 stdout 无匹配文件；允许在线的文档出现匹配时，须在报告中说明在线依赖及处置。
- 构建目录缺失或其中没有 HTML 文件时，脚本以退出码 2 报错——此时不得视为验证通过，应先修正 `--build-dir` 或重新构建。

用同一脚本确认本地 runtime 文件已复制到生成产物的 `_static` 目录（stdout 逐项列出找到的资产，可直接作为报告「已检查的 Mermaid runtime 文件」字段的证据）：

```bash
python -X utf8 <skill_dir>/scripts/html_scan.py --build-dir docs/build/html --scan runtime
```

- 使用本地浏览器 runtime 的项目（UMD：`mermaid.min.js` + runner；ESM：`.mjs` 入口 + `chunks/`）期望退出码 0。
- ESM 入口存在但旁边没有 `chunks/` 目录时，脚本输出 WARNING 并以退出码 1 提示核查 ESM runtime 是否完整。
- 服务端渲染（png/svg 直出）的项目没有浏览器 runtime 属正常，退出码 1 不代表失败，但须在报告中说明渲染方式。

## 浏览器验证

Sphinx 成功并不够。Mermaid 语法通常是在浏览器端才真正解析。

从生成的 HTML 根目录启动临时本地服务：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs/build/html &
SERVER_PID=$!
```

使用浏览器自动化打开每个包含 `.mermaid` 的 HTML 页面，并检查每个 Mermaid 容器：

- 包含 `svg`
- SVG 的 `aria-roledescription` 不是 `error`
- 文本不匹配 `Syntax error in text|Parse error|Diagram error|mermaid version`

如果项目实现了点击放大功能，也要验证交互：

- 点击渲染后的 `.mermaid` 图会打开 `.mermaid-modal` 或项目自定义 modal
- 滚轮输入会改变克隆 SVG 的 transform 或缩放状态
- 支持拖拽时，拖拽会移动放大后的 SVG
- `Escape`、关闭按钮、点击背景均可关闭 modal

验证后关闭临时服务：

```bash
kill "$SERVER_PID"
```

## 报告格式

最终报告必须包含：

- `context_check.py` 的输出摘要（文档栈、runtime 文件、fence 数、含图页面数）——缺失该项即视为未执行门禁步骤
- 修改的文件
- 修复的具体语法类别
- `parse_check.mjs` 结果（N/N 块通过）——缺失该项即视为未完成语法风险扫描与修复的复验；存在手工修复时，语法类别须注明依据（`references/syntax-scan-and-fix-rules.md` 中高风险模式表的行或其「修复规则」章节的条目）
- 已检查的 Mermaid runtime 文件，包括本地/ CDN 结果
- docs 构建结果
- 浏览器验证统计：检查页数、Mermaid 图数量、失败数
- 如果项目实现点击放大，包含点击放大验证结果
- docs 构建中剩余的非 Mermaid warning
- 确认临时服务已关闭

示例：

```text
Context check: myst_parser + sphinxcontrib.mermaid, 211 fences, local runtime mermaid.min.js + mermaid-run.js, 56 HTML pages with mermaid, no CDN.
Parse check: 211/211 mermaid blocks passed (3 manual fixes per pattern table rows: subgraph legacy syntax, stateDiagram colon).
Changed 4 Markdown files, syntax-only Mermaid edits.
Runtime check: local mermaid.min.js + mermaid-run.js copied, no CDN runtime references.
Sphinx build: passed.
Browser Mermaid QA: 56 pages, 142 diagrams, 0 failures.
Click-to-zoom QA: modal open, wheel zoom, drag pan, ESC close passed.
Generated HTML error text scan: no output.
Temporary HTTP server stopped.
```

## 禁止事项

- 不要只依赖静态 grep 或 `sphinx-build`。
- 不要为了避免语法错误而删除 label。
- 除非用户明确要求，不要改变图语义、来源引用、标题或正文。
- 除非用户明确要求，不要修改 `sphinxcontrib-mermaid` 等依赖版本号。
- 对要求离线可用的文档，不要重新引入 CDN runtime 依赖。
- 不要让临时 HTTP 服务残留运行。
