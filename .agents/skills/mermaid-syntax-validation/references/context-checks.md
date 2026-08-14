# 必做上下文检查脚本

## When to Read

- 执行"必做上下文检查"环节时
- 需要统计 Mermaid fence 数量时
- 需要找出生成 HTML 中包含 Mermaid 的页面时
- 需要检查 HTML script 标签引用的 runtime 时

## 1. 确认文档栈

- `docs/source/conf.py`
- `extensions` 包含 `myst_parser` 和/或 `sphinxcontrib.mermaid`
- 如果 Markdown 使用 ```` ```mermaid ```` fence，应保留 `myst_fence_as_directive = ["mermaid"]`
- `mermaid_output_format`、`mermaid_version`、`mermaid_include_elk`、`mermaid_fullscreen`、`mermaid_init_config`
- `html_js_files` 中加载浏览器 runtime 的条目，例如 `js/mermaid.min.js`、`js/mermaid-run.js`、`js/mermaid.esm.min.mjs`，或旧的自定义文件如 `js/mermaid-init.js`
- 是否有项目自定义 monkey patch 或 override，用于禁用 `sphinxcontrib-mermaid` 自动注入 JavaScript

## 2. 确认本地 runtime 资源

- UMD/全局对象 runtime：`docs/source/_static/js/mermaid.min.js` 以及 runner，例如 `docs/source/_static/js/mermaid-run.js`
- ESM runtime：`docs/source/_static/js/mermaid.esm.min.mjs` 以及必需的相对 `docs/source/_static/js/chunks/` 目录树
- 通过 Sphinx 配置项指定的可选本地依赖，例如 `mermaid_use_local`、`mermaid_elk_use_local`、`mermaid_zenuml_use_local`、`d3_use_local`
- 除非用户明确接受在线文档，否则 Mermaid/Panzoom/D3 runtime 不应从 `cdn.jsdelivr.net`、`unpkg.com` 或其他 CDN 拉取

## 3. 统计 Mermaid fence（Linux/bash 主路径）

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

## 4. 找出生成 HTML 中包含 Mermaid 的页面（Linux/bash 主路径）

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

## 5. 检查生成 HTML 的 Mermaid script 标签（Linux/bash 主路径）

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
