# 语法风险扫描与修复——条件性规则

命令循环与无条件门禁（顺序 1→4、复验全绿）见 SKILL.md「语法风险扫描与修复」；本文件只保留按条件触发的规则。高风险模式表中「自动修复」列标注 `fix_labels.py` 的行已由脚本循环覆盖；标注「手工」的行按下方「修复规则」处理。静态 grep 只能初筛部分模式；`[]`、`{}`、`""` 翻倍这类问题只有 `parse()` 能稳定暴露，因此修复循环必须以 `parse_check.mjs` 为准。

## When to Read

- `fix_labels.py` 应用后 `parse_check.mjs` 仍有失败块，需要定位失败模式时
- 无 JS runtime，需要静态初筛回退时
- 需要判断某段 Mermaid 是否属于高风险模式时
- 需要参考 flowchart 边/节点 label、state diagram transition label 的修复规则时

## 门禁规则（触发上述场景时必须满足）

1. `fix_labels.py` 应用后 `parse_check.mjs` 仍有失败块时，禁止凭经验直接手工编辑——必须先按下文「高风险模式」表逐行定位失败模式，按「仅语法修复」列处理；表中未覆盖的失败模式，按下方「修复规则」章节的一般原则处理（含标点的 label 加引号、state diagram 冒号转义、保留可见文本）。
2. 无 JS runtime（无法运行 `parse_check.mjs`）时，必须按「静态语法初筛」节的命令初筛并按高风险模式表处理，且在报告中说明未做 parse 验证的原因。
3. 每一处手工修复必须在最终报告的「修复的具体语法类别」中注明依据（高风险模式表的行或「修复规则」章节的具体条目），作为修复依据的审计记录。

## 静态语法初筛（无 JS runtime 时的回退诊断）

仅当无法运行 `parse_check.mjs`（无 JS runtime，见门禁规则 2）时，用下面的静态 grep 初筛定位可疑行，再按「高风险模式」表处理；有 JS runtime 时跳过本节，直接走 SKILL.md 的修复循环。静态 grep 只覆盖部分模式（如 `[]`、`{}`、`""` 翻倍无法稳定匹配），其结果不能替代 `parse()` 与浏览器验证。

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

## 高风险模式

| 模式 | 失败原因 | 仅语法修复 | 自动修复 |
|---|---|---|---|
| `A --> ["label"]` | 匿名节点作为边端点 | 创建稳定节点 id：`A --> node_id["label"]` | 手工（行中括号不处理，避免误伤边标签内括号） |
| `["label"]` 独立声明 | 匿名节点声明 | 创建稳定节点 id：`node_id["label"]` | `fix_labels.py`（分配 `mmd_anon_N`） |
| `subgraph "ID" ["Title"]` | subgraph 写法不兼容 | `subgraph ID["Title"]` | 手工（subgraph 语句行跳过） |
| `A -- "label" --> B` | 旧式边 label 可能失败 | `A -->|"label"| B` | 手工 |
| `A --> B : "label"` | flowchart 旧式尾随 label | `A -->|"label"| B` | 手工 |
| `A -->|/topic (type)| B` | edge label 含未加引号的括号 | `A -->|"/topic (type)"| B` | `fix_labels.py` |
| `B{func(arg)}` | diamond 节点 label 含未加引号的括号 | `B{"func(arg)"}` | `fix_labels.py` |
| `A --> B: file.py:1-2` in `stateDiagram-v2` | transition label 内含额外冒号 | `A --> B: file.py#58;1-2` | 手工（非 graph/flowchart，脚本跳过） |
| `A[""label""]` | 标签引号翻倍（DeepWiki 导出/翻译引入），空标签边界 | 去重外层引号并转义内部引号：`A["label"]` | `fix_labels.py` |
| `A[k: {v}]` | 方括号标签内未加引号的 `{` 被解析为菱形开始 | `A["k: {v}"]` | `fix_labels.py` |
| `A -->|Returns X[]| B` | edge label 内未加引号的 `[]` 触发节点形状解析 | `A -->|"Returns X[]"| B` | `fix_labels.py` |

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
