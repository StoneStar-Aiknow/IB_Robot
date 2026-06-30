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

## 必做上下文检查

1. 确认文档栈：
   - `docs/source/conf.py`
   - `extensions` 包含 `myst_parser` 和/或 `sphinxcontrib.mermaid`
   - 如果 Markdown 使用 ```` ```mermaid ```` fence，应保留 `myst_fence_as_directive = ["mermaid"]`
   - `mermaid_output_format`、`mermaid_version`、`mermaid_include_elk`、`mermaid_fullscreen`、`mermaid_init_config`
   - `html_js_files` 中加载浏览器 runtime 的条目，例如 `js/mermaid.min.js`、`js/mermaid-run.js`、`js/mermaid.esm.min.mjs`，或旧的自定义文件如 `js/mermaid-init.js`
   - 是否有项目自定义 monkey patch 或 override，用于禁用 `sphinxcontrib-mermaid` 自动注入 JavaScript
2. 确认本地 runtime 资源：
   - UMD/全局对象 runtime：`docs/source/_static/js/mermaid.min.js` 以及 runner，例如 `docs/source/_static/js/mermaid-run.js`
   - ESM runtime：`docs/source/_static/js/mermaid.esm.min.mjs` 以及必需的相对 `docs/source/_static/js/chunks/` 目录树
   - 通过 Sphinx 配置项指定的可选本地依赖，例如 `mermaid_use_local`、`mermaid_elk_use_local`、`mermaid_zenuml_use_local`、`d3_use_local`
   - 除非用户明确接受在线文档，否则 Mermaid/Panzoom/D3 runtime 不应从 `cdn.jsdelivr.net`、`unpkg.com` 或其他 CDN 拉取
3. 统计 Mermaid fence（Linux/bash 主路径）：
   ```bash
   python3 - <<'PY'
   from pathlib import Path

   print(sum(1 for path in Path("docs").rglob("*.md") for line in path.read_text(encoding="utf-8").splitlines() if line.strip() == "```mermaid"))
   PY
   ```
   <details><summary>Windows PowerShell 可选扩展</summary>

   ```powershell
   Get-ChildItem -LiteralPath "docs" -Recurse -File -Filter "*.md" |
     Select-String -Pattern '^```mermaid\s*$' |
     Measure-Object |
     Select-Object -ExpandProperty Count
   ```

   </details>
4. 找出生成 HTML 中包含 Mermaid 的页面（Linux/bash 主路径）：
   ```bash
   python3 - <<'PY'
   from pathlib import Path
   import re

   pattern = re.compile(r'class="mermaid"|class="[^" ]*mermaid')
   for path in sorted(Path("docs/build/html").rglob("*.html")):
       if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
           print(path)
   PY
   ```
   <details><summary>Windows PowerShell 可选扩展</summary>

   ```powershell
   Get-ChildItem -LiteralPath "docs/build/html" -Recurse -File -Filter "*.html" |
     Select-String -Pattern 'class="mermaid"|class="[^" ]*mermaid' |
     ForEach-Object { $_.Path } |
     Sort-Object -Unique
   ```

   </details>
5. 至少检查一个 Mermaid 较多页面的生成 HTML script 标签（Linux/bash 主路径）：
   ```bash
   python3 - <<'PY'
   from pathlib import Path
   import re

   pattern = re.compile(r'mermaid\.min\.js|mermaid-run\.js|mermaid\.esm\.min\.mjs|cdn\.jsdelivr\.net|unpkg\.com')
   for path in sorted(Path("docs/build/html").rglob("*.html")):
       text = path.read_text(encoding="utf-8", errors="ignore")
       if pattern.search(text):
           print(path)
   PY
   ```
   <details><summary>Windows PowerShell 可选扩展</summary>

   ```powershell
   Select-String -Path "docs/build/html/**/*.html" `
     -Pattern 'mermaid\.min\.js|mermaid-run\.js|mermaid\.esm\.min\.mjs|cdn\.jsdelivr\.net|unpkg\.com'
   ```

   </details>

## Runtime 预期

对于当前本地 runtime 方案，项目 runner（例如 `mermaid-run.js`）应当：

- 等待 `window.mermaid` 存在
- 调用 `window.mermaid.initialize({ startOnLoad: false, ... })`
- 在页面加载后调用 `window.mermaid.run()`
- 绑定项目自定义交互，例如点击放大 modal

当前 Mermaid 版本应使用 `mermaid.run()`。把 `mermaid.init()` 示例视为旧写法，因为 Mermaid v10+ 已废弃该 API。

如果存在可运行的 Mermaid JS 环境，语法级检查优先使用 `mermaid.parse(text, { suppressErrors: true })`。静态 grep 适合初筛，但 `parse()` 和浏览器渲染是更强的证据。

## 静态风险扫描

编辑前先做源文件扫描。该扫描不能替代浏览器验证。

```bash
python3 - <<'PY'
from pathlib import Path
import re

patterns = [
    r'^\s*\["',
    r'--?>\s*\["',
    r'\.->\s*\["',
    r'^\s*subgraph\s+"[^"]+"\s+\[',
    r'\s--\s*"',
    r'\s--"',
]
compiled = [re.compile(pattern) for pattern in patterns]
for path in sorted(Path("docs").rglob("*.md")):
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if any(pattern.search(line) for pattern in compiled):
            print(f"{path}:{line_number}:{line}")
PY
```

<details><summary>Windows PowerShell 可选扩展</summary>

```powershell
Get-ChildItem -LiteralPath "docs" -Recurse -File -Filter "*.md" |
  Select-String -Pattern '^\s*\["','--?>\s*\["','\.->\s*\["','^\s*subgraph\s+"[^"]+"\s+\[','\s--\s*"','\s--"'
```

</details>

高风险模式：

| 模式 | 失败原因 | 仅语法修复 |
|---|---|---|
| `A --> ["label"]` | 匿名节点作为边端点 | 创建稳定节点 id：`A --> node_id["label"]` |
| `["label"]` 独立声明 | 匿名节点声明 | 创建稳定节点 id：`node_id["label"]` |
| `subgraph "ID" ["Title"]` | subgraph 写法不兼容 | `subgraph ID["Title"]` |
| `A -- "label" --> B` | 旧式边 label 可能失败 | `A -->|"label"| B` |
| `A --> B : "label"` | flowchart 旧式尾随 label | `A -->|"label"| B` |
| `A -->|/topic (type)| B` | edge label 含未加引号的括号 | `A -->|"/topic (type)"| B` |
| `B{func(arg)}` | diamond 节点 label 含未加引号的括号 | `B{"func(arg)"}` |
| `A --> B: file.py:1-2` in `stateDiagram-v2` | transition label 内含额外冒号 | `A --> B: file.py#58;1-2` |

## 修复规则

### Flowchart 边 Label

使用 pipe label 语法。

```mermaid
A -->|"label with (parentheses), /slashes, or :colon"| B
```

当 label 含有 `(`、`)`、`/`、`:`、`<`、`>`、`,` 或其他标点时，优先加引号。

### Flowchart 节点 Label

节点形状内的 label 如果包含标点，应加引号。

```mermaid
B{"get_scene_file(scene_name, platform)"}
C["/base_velocity_controller/commands (std_msgs/Float64MultiArray)"]
```

### State Diagram Transition Label

`stateDiagram-v2` 使用 `A --> B: label`。如果 label 自身还需要额外冒号，应把内部冒号转成 Mermaid entity 文本：

```mermaid
Idle --> GoalEvaluation: handle_goal()<br/>episode_recorder.py#58;277-288
```

### 保留原内容

除 Mermaid 语法转义外，保持可见 label 不变。例如 `file.py#58;277-288` 应在 Mermaid 中显示为预期的 `file.py:277-288`。

## 构建验证

修复后运行真实文档构建。项目有文档化命令时优先使用项目命令。常见 Sphinx 命令包括：

```bash
sphinx-build -M html source build
```

或从仓库根目录运行：

```bash
python3 -m sphinx -b html docs/source docs/build/html
```

同时扫描生成 HTML 中的 Mermaid 错误文本：

```bash
python3 - <<'PY'
from pathlib import Path
import re

pattern = re.compile(r"Syntax error in text|Parse error|mermaid version|Diagram error")
for path in sorted(Path("docs/build/html").rglob("*.html")):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if pattern.search(text):
        print(path)
PY
```

期望无输出。

扫描生成 HTML 是否意外引入在线 runtime 依赖：

```bash
python3 - <<'PY'
from pathlib import Path
import re

pattern = re.compile(r"cdn\.jsdelivr\.net|unpkg\.com")
for path in sorted(Path("docs/build/html").rglob("*.html")):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if pattern.search(text):
        print(path)
PY
```

对于要求离线可用的文档，期望无输出。

确认本地 runtime 文件已复制到生成产物的 `_static` 目录，例如：

```bash
test -f docs/build/html/_static/js/mermaid.min.js
test -f docs/build/html/_static/js/mermaid-run.js
```

如果项目使用 ESM Mermaid，还要确认生成产物中存在 `.mjs` 入口和 `chunks/` 目录。

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

- 修改的文件
- 修复的具体语法类别
- 已检查的 Mermaid runtime 文件，包括本地/ CDN 结果
- docs 构建结果
- 浏览器验证统计：检查页数、Mermaid 图数量、失败数
- 如果项目实现点击放大，包含点击放大验证结果
- docs 构建中剩余的非 Mermaid warning
- 确认临时服务已关闭

示例：

```text
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
